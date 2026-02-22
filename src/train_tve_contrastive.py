import os
import subprocess
import signal

import rootutils

import torch
from torch import optim, nn

import warnings
# Suppress all warnings (including experimental ones)
warnings.filterwarnings("ignore")

import wandb

from relbench.base import EntityTask, RecommendationTask

rootutils.setup_root('.', indicator='.project-root', pythonpath=True)

from src.helpers.dataset_helper import get_data, get_loaders
from src.helpers.loss_helper import get_loss
from src.helpers.lr_scheduler import get_lr_scheduler, step_scheduler
from src.helpers.model_helper import get_hetero_encoder, get_conv_layer
from src.helpers.task_helper import get_task_metadata
from src.helpers.save_helper import save_backbone_checkpoint, save_checkpoint
from src.helpers.tve_contrastive_train_helper import train, test

from src.models.tve_contrastive import TVEContrastive

from src.utils.io_utils import (
    parse_cfg_args,
    parse_train_args, 
    load_config,
    is_main_process, 
    get_device,  
    create_log_dir,
    print_config_tree,
    synchronize,
    compute_params,
    get_world_size,
    handle_exit,
    init_process_group,
    destroy_process_group,
    setup_cfg,
    setup_wandb,
    infer_dataset_task,
    sanity_check
)
from src.utils.io_utils import rank_zero_print as print
from src.utils.io_utils import rank_zero_pprint as pprint
from src.utils.log_utils import print_epoch_metrics, CurvesLogger

from src.utils.random_utils import set_random_seed
from src.utils.wandb_utils import set_wandb_directories

set_wandb_directories()

def main(cfg, cfg_file, sweep_file, sweep_folder_name=None):
    # Set device
    device = get_device(cfg)
    
    # Init Debug Mode
    setup_cfg(cfg)
    
    # Create log directory.
    log_dir = create_log_dir(cfg, cfg_file, sweep_file, sweep_folder_name)
    
    # Init MLOps
    setup_wandb(cfg, log_dir)
    
    # Print the configuration.
    print_config_tree(cfg, save_to_file=True, output_dir=log_dir)
    
    # Load the dataset
    data_dict = get_data(cfg, device, print_stype=True)
    
    # Make sure single task
    assert 'task' in data_dict, "Only single task training is supported."
    
    # Load data loaders
    if issubclass(data_dict['task'].__class__, EntityTask):
        loader_dict = get_loaders(cfg, data_dict)
        train_dst_sparse_tensor = None
    
    # Load task metadata
    task: EntityTask | RecommendationTask = data_dict["task"]
    task_name = data_dict["task_name"]
    metadata = get_task_metadata(task, cfg.dataset.cls, task_name)
    pprint({task_name: metadata})
    
    for run in range(cfg.num_runs):
        print(f"================== Run {run+1}/{cfg.num_runs} ==================")
        # Set random seed (default 43)
        set_random_seed(cfg.seed + run)
        
        # Load the model
        model = TVEContrastive(
            data=data_dict["data"],
            schema_graph=data_dict["schema_graph"],
            schema_line_graph=data_dict["schema_line_graph"],
            out_channels=metadata["out_channels"],
            col_stats_dict=data_dict["col_stats_dict"],
            hetero_encoder_cls=get_hetero_encoder(cfg),
            hetero_encoder_kwargs={
                "out_channels": cfg.model.conv.kwargs.channels,
                **cfg.model.hetero_encoder.kwargs
            },
            temporal_encoder_kwargs={
                "out_channels": cfg.model.conv.kwargs.channels,
                **cfg.model.temporal_encoder.kwargs
            },
            conv_cls=get_conv_layer(cfg),
            conv_kwargs={
                "num_layers": len(cfg.train.num_neighbors),
                **cfg.model.conv.kwargs
            },
            shallow_list=cfg.model.shallow_list,
            task_names=None,
            mask_rate=cfg.train.mask_rate
        ).to(device)
        print(f"Loaded TVEContrastive({', '.join([f'{k}={v}' for k, v in cfg.model.items()])})")
        
        # Load the optimizer
        cls = cfg.optimizer.pop("cls")
        optimizer = getattr(optim, cls)(model.parameters(), **cfg.optimizer)
        print(f"Loaded Optimizer: {cls}({', '.join([f'{k}={v}' for k, v in cfg.optimizer.items()])})")
        cfg.optimizer.cls = cls                # restore the class name

        # Load the scheduler
        scheduler = get_lr_scheduler(cfg, optimizer, metadata)
        if scheduler:
            cls = cfg.lr_scheduler.pop("cls")
            print(f"Loaded Scheduler: {cls}({', '.join([f'{k}={v}' for k, v in cfg.lr_scheduler.items()])})")
            cfg.lr_scheduler.cls = cls
        else:
            print("No Scheduler Loaded.")
        
        # Print model parameters
        if not cfg.debug:
            print(model)
        num_trainable_params, num_total_params = compute_params(model, print_params=False)
            
        # Data Parallel
        if get_world_size() > 1:
            parallel_model = nn.parallel.DistributedDataParallel(model, device_ids=[device], find_unused_parameters=True)
        else:
            parallel_model = model
        
        # Train the model
        print("Training the model...")
        loss_fn = get_loss(device=device, **cfg.train.loss_fn)
        curves_logger = CurvesLogger(task_names=[task_name], metadata_dict={task_name: metadata})
        best_val_loss = float('inf')
        for epoch in range(0, cfg.train.num_epochs):
            if hasattr(loader_dict['train'], 'sampler') and isinstance(loader_dict['train'].sampler, torch.utils.data.distributed.DistributedSampler):
                loader_dict['train'].sampler.set_epoch(epoch + run * cfg.train.num_epochs)
            train_loss_dict = train(cfg, parallel_model, task, task_name, loader_dict['train'], optimizer, loss_fn, data_dict['col_stats_dict'], device)
            
            # Print epoch metrics if evaluation epoch is reached
            if epoch % cfg.train.eval_every == 0:
                val_loss_dict = test(cfg, parallel_model, task, task_name, loader_dict['val'], loss_fn, data_dict['col_stats_dict'], device)
                test_loss_dict = test(cfg, parallel_model, task, task_name, loader_dict['test'], loss_fn, data_dict['col_stats_dict'], device)
            else:
                pass
            
            # Log loss curves
            curves_logger.log_loss(train_loss_dict=train_loss_dict, val_loss_dict=val_loss_dict, test_loss_dict=test_loss_dict)
            
            # Print epoch metrics
            print_epoch_metrics(run, epoch, curves_logger, optimizer)

            if is_main_process():
                # Save the model every eval_every epochs
                if epoch % cfg.train.eval_every == 0:
                    save_backbone_checkpoint(model, optimizer, epoch, os.path.join(log_dir, f'epoch{epoch+1}.pth'))
                    save_checkpoint(model, optimizer, epoch, os.path.join(log_dir, f'model-epoch{epoch+1}.pth'))
                    if val_loss_dict[task_name]['total_loss'] < best_val_loss:
                        best_val_loss = val_loss_dict[task_name]['total_loss']
                        save_backbone_checkpoint(model, optimizer, epoch, os.path.join(log_dir, f'best.pth'))
                        save_checkpoint(model, optimizer, epoch, os.path.join(log_dir, f'best-model.pth'))
            synchronize()
            
            # Step the scheduler
            break_signal = step_scheduler(cfg, scheduler, optimizer, None, metadata)
            if break_signal:
                break
        
        print(f"================== End of Run {run+1}/{cfg.num_runs} ==================")
        synchronize()
        
    wandb.finish()
    synchronize()      

if __name__ == '__main__':
    # Register the signal handler to handle Ctrl-C across processes
    signal.signal(signal.SIGINT, handle_exit)
    
    args, unparsed = parse_cfg_args()
    
    if os.path.isdir(args.sweep_config):        # run sweep files
        raise NotImplementedError("Sweep files are not supported for MAE training.")
    else:                       # run single experiment
        args, vars = parse_train_args(args, unparsed)
        cfg = load_config(args.config, context=vars)

        # Extract the commit sha so we can check the code that was used for each experiment
        # sha = subprocess.check_output(["git", "describe", "--always"]).strip().decode()
        # cfg.sha = str(sha)
        
        # Sanity check
        sanity_check(cfg)
        
        # Set sweep name
        cfg.sweep = False
        
        # import easydict
        # def print_cfg_data_types(cfg, indent=0):
        #     for key, value in cfg.items():
        #         if isinstance(value, easydict.EasyDict):
        #             print(" " * indent + f"{key}: {type(value), value}")
        #             print_cfg_data_types(value, indent + 2)
        #         else:
        #             print(" " * indent + f"{key}: {type(value), value}")
        # print_cfg_data_types(cfg)
        
        # Infer dataset class and task
        cfg.dataset.cls, cfg.dataset.task = infer_dataset_task(cfg, args.config)
        cfg.model.conv.kwargs.num_layers = len(cfg.train.num_neighbors)
        
        init_process_group(cfg)
        main(cfg, args.config, args.sweep_config)
        destroy_process_group()