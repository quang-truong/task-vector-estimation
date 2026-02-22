import os
import subprocess
import signal
import copy
import datetime

from collections import defaultdict
from typing import Dict

import rootutils

import torch
from torch import optim, nn

import warnings
# Suppress all warnings (including experimental ones)
warnings.filterwarnings("ignore")

import wandb

from relbench.base import EntityTask, RecommendationTask
from relbench.modeling.loader import SparseTensor

rootutils.setup_root('.', indicator='.project-root', pythonpath=True)

from src.definitions import LOG_DIR
from src.helpers.dataset_helper import get_data, get_loaders
from src.helpers.loss_helper import get_loss
from src.helpers.lr_scheduler import get_lr_scheduler, step_scheduler
from src.helpers.model_helper import get_hetero_encoder, get_conv_layer
from src.helpers.task_helper import get_task_metadata
from src.helpers.save_helper import save_checkpoint, load_checkpoint, load_backbone_checkpoint
from src.helpers.single_task_train_helper import train, test

from src.models.baseline import Baseline

from src.utils.io_utils import (
    parse_cfg_args,
    parse_train_args, 
    load_config,
    literal_eval, 
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
from src.utils.log_utils import print_epoch_metrics, print_final_results, print_runs_stats, CurvesLogger

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
    else:
        raise ValueError(f"Unknown task type: {data_dict['task']}")
    
    # Load task metadata
    task: EntityTask | RecommendationTask = data_dict["task"]
    task_name = data_dict["task_name"]
    metadata = get_task_metadata(task, cfg.dataset.cls, task_name)
    pprint({task_name: metadata})
    all_perf = {"val": defaultdict(lambda: []), "test": defaultdict(lambda: [])}
    
    for run in range(cfg.num_runs):
        print(f"================== Run {run+1}/{cfg.num_runs} ==================")
        # Set random seed (default 43)
        set_random_seed(cfg.seed + run)
        
        # Load the model
        model = Baseline(
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
                **cfg.model.conv.kwargs,
            },
            shallow_list=cfg.model.shallow_list,
            task_names=None,
        ).to(device)
        print(f"Loaded Baseline({', '.join([f'{k}={v}' for k, v in cfg.model.items()])})")
        
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
            
        # Load checkpoint if available
        if cfg.checkpoint is not None:
            ckpt_dir = cfg.checkpoint
            if 'ssl' in ckpt_dir or 'tve' in ckpt_dir:
                load_backbone_checkpoint(model, optimizer, cfg.checkpoint, device)
            else:
                load_checkpoint(model, optimizer, cfg.checkpoint, device)
        
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
        
        for epoch in range(0, cfg.train.num_epochs):
            if hasattr(loader_dict['train'], 'sampler') and isinstance(loader_dict['train'].sampler, torch.utils.data.distributed.DistributedSampler):
                loader_dict['train'].sampler.set_epoch(epoch + run * cfg.train.num_epochs)
            train_loss_dict = train(cfg, parallel_model, task, task_name, loader_dict['train'], optimizer, loss_fn, device, train_dst_sparse_tensor=train_dst_sparse_tensor)
            
            # Print epoch metrics if evaluation epoch is reached
            if epoch % cfg.train.eval_every == 0:
                val_perf_dict = test(cfg, parallel_model, task, task_name, loader_dict['val'], data_dict, device, is_test=False)
                test_perf_dict = test(cfg, parallel_model, task, task_name, loader_dict['test'], data_dict, device, is_test=True)
            else:
                pass
            
            # Log loss and performance curves
            curves_logger.log_loss(train_loss_dict=train_loss_dict)
            curves_logger.log_perf(val_perf_dict=val_perf_dict, test_perf_dict=test_perf_dict)
            
            # Print epoch metrics
            print_epoch_metrics(run, epoch, curves_logger, optimizer)

            # Save the model if it's the best
            if is_main_process():
                best_epoch = curves_logger.best_epoch_dict[task_name]
                if epoch == 0 or best_epoch == epoch:
                    save_checkpoint(model, optimizer, best_epoch, os.path.join(log_dir, f'run{run+1}-best_model.pth'))
            synchronize()
            
            # Step the scheduler
            break_signal = step_scheduler(cfg, scheduler, optimizer, val_perf_dict[task_name], metadata)
            if break_signal:
                break
        
        # Print final results
        print_final_results(cfg, run, curves_logger, log_dir)
        # Keep track of the best performance per run
        best_epoch = curves_logger.best_epoch_dict[task_name]
        for metric, values in curves_logger.perf_curves_dict[task_name]['val'].items():
            all_perf['val'][metric].append(values[best_epoch])
        for metric, values in curves_logger.perf_curves_dict[task_name]['test'].items():
            all_perf['test'][metric].append(values[best_epoch])
        print(f"================== End of Run {run+1}/{cfg.num_runs} ==================")
        synchronize()
    
    log_dict = print_runs_stats(cfg, all_perf, task_name, log_dir, (num_trainable_params, num_total_params))
    if is_main_process():
        if cfg.get("sweep", False):
            job_type = "debug" if cfg.debug else "train"
            # Create directory for sweep results
            if isinstance(cfg.dataset.task, str):
                res_dir = os.path.join(LOG_DIR, job_type, cfg.dataset.cls, cfg.dataset.task, "-".join([cfg.model.hetero_encoder.cls, cfg.model.conv.cls]), sweep_folder_name)
            else:
                raise ValueError("Invalid dataset task.")
            # Retrieve the sweep results
            res = log_dict[f'{task_name}/mean_best_val_{metadata["tune_metric"]}']
            # Save the sweep results
            print(f"Saving sweep results to {res_dir}/sweep_results.txt")
            with open(os.path.join(res_dir, "sweep_results.txt"), "a") as f:
                f.write(f"{sweep_file}: {res}\n")
            
        wandb.finish()
    synchronize()

if __name__ == '__main__':
    # Register the signal handler to handle Ctrl-C across processes
    signal.signal(signal.SIGINT, handle_exit)
    
    args, unparsed = parse_cfg_args()
    
    if os.path.isdir(args.sweep_config):        # run sweep files
        sweep_dir = args.sweep_config
        sweep_files = sorted([f for f in os.listdir(sweep_dir) if f.endswith(".yaml")])
        sweep_only_for = literal_eval(args.sweep_only_for)
        if sweep_only_for:
            sweep_files = [sweep_files[i-1] for i in sweep_only_for]
        sweep_folder_name = f"sweep-{datetime.datetime.now().isoformat(timespec='seconds').replace('T', '_').replace(':','-')}"
        for i, sweep_file in enumerate(sweep_files):
            print(f"================== Sweep {sweep_file} ==================")
            # copy args and unparsed to avoid overwriting for each sweep
            copied_args = copy.deepcopy(args)
            copied_unparsed = copy.deepcopy(unparsed)
            
            # update the sweep config file
            copied_args.sweep_config = os.path.join(sweep_dir, sweep_file)
            copied_args, vars = parse_train_args(copied_args, copied_unparsed)
            cfg = load_config(copied_args.config, context=vars)
            assert cfg.name is not None, "Name is required for the sweep."
            assert cfg.note is not None, "Note is required for the sweep."

            # Extract the commit sha so we can check the code that was used for each experiment
            # sha = subprocess.check_output(["git", "describe", "--always"]).strip().decode()
            # cfg.sha = str(sha)
            
            # Sanity check
            sanity_check(cfg)
            
            # Infer dataset class and task
            cfg.dataset.cls, cfg.dataset.task = infer_dataset_task(cfg, args.config)
            cfg.model.conv.kwargs.num_layers = len(cfg.train.num_neighbors)
            
            if i == 0:
                init_process_group(cfg)
            try:
                tmp_sweep = cfg.name + '-' + sweep_folder_name              # Prepend cfg.name to sweep_folder_name
                cfg.name = f"{cfg.name}-{sweep_file.split('.')[0]}"
                cfg.sweep = tmp_sweep
                main(cfg, copied_args.config, copied_args.sweep_config, tmp_sweep)
            except Exception as e:
                if is_main_process():
                    wandb.finish()
                    torch.cuda.empty_cache()
                synchronize()
                print(f"An error occurred during sweep {sweep_file}.")
                if e == KeyboardInterrupt:
                    print("\nExecution interrupted by user. Exiting gracefully...")
                    exit(0)
            print(f"================== End of Sweep {sweep_file} ==================")
        destroy_process_group()
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