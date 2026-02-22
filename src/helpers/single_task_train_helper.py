
import numpy as np

import rootutils

import torch
import torch.distributed as dist

import warnings
from tqdm.rich import tqdm
# Suppress all warnings (including experimental ones)
warnings.filterwarnings("ignore")

from relbench.base import EntityTask, RecommendationTask, TaskType

rootutils.setup_root('.', indicator='.project-root', pythonpath=True)

from src.utils.io_utils import ( 
    is_main_process, 
    get_world_size,
    get_rank,
)

def train(cfg, model, task, task_name, data_loader, optimizer, loss_fn, device, **kwargs):
    if isinstance(model, tuple):
        src_model, dst_model = model
        src_model.train()
        dst_model.train()
    else:
        model.train()
    if issubclass(task.__class__, EntityTask):
        loss = train_one_epoch_entity(cfg, model, task, data_loader, optimizer, loss_fn, device, **kwargs)
    elif issubclass(task.__class__, RecommendationTask):
        loss = train_one_epoch_recommendation(cfg, model, task, data_loader, optimizer, loss_fn, device, **kwargs)
    else:
        raise ValueError(f"Unknown task type: {task}")
    
    if get_world_size() > 1:
        reduced_loss = torch.tensor(loss).to(device)
        dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
        reduced_loss = reduced_loss / get_world_size()
        reduced_loss = reduced_loss.item()
    else:
        reduced_loss = loss
        
    loss_dict = {
        f'{cfg.train.loss_fn.cls}': reduced_loss
    }
    
    return {task_name: loss_dict}

@torch.no_grad()
def test(cfg, model, task, task_name, data_loader, data_dict, device, is_test=True):    
    if isinstance(model, tuple):
        src_model, dst_model = model
        src_model.eval()
        dst_model.eval()
    else:
        model.eval()
    if issubclass(task.__class__, EntityTask):
        pred_list = inference_entity(model, task, data_loader, device, 
                                     train_table=data_dict['train_table'], 
                                     val_table=data_dict['val_table'], 
                                     test_table=data_dict['test_table'], 
                                     is_test=is_test)
    else:
        raise ValueError(f"Unknown task type: {task}")
    metrics = task.evaluate(pred_list) if is_test else task.evaluate(pred_list, data_dict['val_table'])
    return {task_name: metrics}

def train_one_epoch_entity(cfg, model, task, data_loader, optimizer, loss_fn, device, **kwargs):
    loss_accum = count_accum = 0
    steps = 0
    total_steps = min(len(data_loader), cfg.train.max_steps_per_epoch)
    rank = get_rank()
    entity_table = task.entity_table
    for batch in tqdm(data_loader, total=total_steps, desc=f"Training rank {rank}..."):
        batch_size = batch[task.entity_table].batch_size
        optimizer.zero_grad()
        batch = batch.to(device)
        out = model(batch, task.entity_table)
        out = out.view(-1) if out.size(1) == 1 else out
        
        # slicing until batch_size: https://github.com/pyg-team/pytorch_geometric/discussions/7233
        if task.task_type == TaskType.REGRESSION or task.task_type == TaskType.BINARY_CLASSIFICATION:
            batch[entity_table].y = batch[entity_table].y.float()
        loss = loss_fn(out[:batch_size], batch[entity_table].y)
        
        loss.backward()
        optimizer.step()
        
        loss_accum += loss.detach().item() * batch_size
        count_accum += batch_size
    
        steps += 1
        if steps > cfg.train.max_steps_per_epoch:
            break
    return loss_accum / count_accum

def inference_entity(model, task, data_loader, device, **kwargs):
    pred_list = []
    
    if task.task_type == TaskType.REGRESSION:
        train_table = kwargs['train_table']
        clamp_min, clamp_max = np.percentile(
            train_table.df[task.target_col].to_numpy(), [2, 98]
        )

    # Get the global rank and world size for DDP
    rank = get_rank()
    world_size = get_world_size()

    # Step 1: Compute predictions for the current rank
    for batch in tqdm(data_loader, desc=f"Evaluating rank {rank}..."):
        batch = batch.to(device)
        batch_size = batch[task.entity_table].batch_size

        # Obtain predictions from the model
        pred = model(batch, task.entity_table)

        # Post-process the prediction based on the task type
        if task.task_type == TaskType.REGRESSION:
            assert clamp_min is not None
            assert clamp_max is not None
            pred = torch.clamp(pred, clamp_min, clamp_max)

        if task.task_type in [
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTILABEL_CLASSIFICATION,
        ]:
            pred = torch.sigmoid(pred)

        # Adjust prediction shape if needed
        pred = pred[:batch_size].view(-1) if pred.size(1) == 1 else pred[:batch_size]

        # Collect predictions for this rank
        pred_list.append(pred.detach().cpu())

    # Concatenate predictions into a single tensor
    pred_tensor = torch.cat(pred_list, dim=0)
    
    if world_size > 1:
        # Gather all the padded predictions
        pred_tensor = pred_tensor.to(device)
        gathered_preds = [torch.zeros_like(pred_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_preds, pred_tensor)

        # Stack the gathered predictions and transpose
        stacked_preds = torch.stack(gathered_preds)     # Shape: (num_processes, max_size, pred_dim)
        stacked_preds = stacked_preds.transpose(0, 1)   # Shape: (max_size, num_processes, pred_dim)

        # Flatten the transposed tensor
        flattened_preds = stacked_preds.flatten(0, 1)   # Shape: (max_size * num_processes, pred_dim)

        # Mask out the padding using the total number of predictions
        total_preds = len(kwargs.get("test_table").df) if kwargs.get("is_test") else len(kwargs.get("val_table").df)         # Total number of valid predictions
        final_predictions = flattened_preds[:total_preds].cpu().numpy()     # Keep only the valid predictions
    else:
        final_predictions = pred_tensor.cpu().numpy()
    return final_predictions
