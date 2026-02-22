
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

def train(cfg, model, task, task_name, data_loader, optimizer, loss_fn, col_stats_dict, device, **kwargs):
    model.train()
    loss, categorical_loss, numerical_loss, text_loss, timestamp_loss = train_one_epoch(cfg, model, task, data_loader, optimizer, loss_fn, col_stats_dict, device, **kwargs)
    if get_world_size() > 1:
        reduced_loss = torch.tensor(loss).to(device)
        dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
        reduced_loss = reduced_loss / get_world_size()
        reduced_loss = reduced_loss.item()
        
        reduced_categorical_loss = torch.tensor(categorical_loss).to(device)
        dist.all_reduce(reduced_categorical_loss, op=dist.ReduceOp.SUM)
        reduced_categorical_loss = reduced_categorical_loss / get_world_size()
        reduced_categorical_loss = reduced_categorical_loss.item()
        
        reduced_numerical_loss = torch.tensor(numerical_loss).to(device)
        dist.all_reduce(reduced_numerical_loss, op=dist.ReduceOp.SUM)
        reduced_numerical_loss = reduced_numerical_loss / get_world_size()
        reduced_numerical_loss = reduced_numerical_loss.item()
        
        reduced_text_loss = torch.tensor(text_loss).to(device)
        dist.all_reduce(reduced_text_loss, op=dist.ReduceOp.SUM)
        reduced_text_loss = reduced_text_loss / get_world_size()
        reduced_text_loss = reduced_text_loss.item()
        
        reduced_timestamp_loss = torch.tensor(timestamp_loss).to(device)
        dist.all_reduce(reduced_timestamp_loss, op=dist.ReduceOp.SUM)
        reduced_timestamp_loss = reduced_timestamp_loss / get_world_size()
        reduced_timestamp_loss = reduced_timestamp_loss.item()
    else:
        reduced_loss = loss
        reduced_categorical_loss = categorical_loss
        reduced_numerical_loss = numerical_loss
        reduced_text_loss = text_loss
        reduced_timestamp_loss = timestamp_loss
        
    loss_dict = {
        "total_loss": reduced_loss,
        "categorical_loss": reduced_categorical_loss,
        "numerical_loss": reduced_numerical_loss,
        "text_loss": reduced_text_loss,
        "timestamp_loss": reduced_timestamp_loss
    }
    return {task_name: loss_dict}

def train_one_epoch(cfg, model, task, data_loader, optimizer, loss_fn, col_stats_dict, device, **kwargs):
    loss_accum = count_accum = categorical_accum = numerical_accum = text_accum = timestamp_accum = 0
    steps = 0
    total_steps = min(len(data_loader), cfg.train.max_steps_per_epoch)
    rank = get_rank()
    
    for batch in tqdm(data_loader, total=total_steps, desc=f"Training rank {rank}..."):
        batch_size = batch[task.entity_table].batch_size
        optimizer.zero_grad()
        batch = batch.to(device)
        out = model(batch, task.entity_table)
        
        loss, categorical_loss, numerical_loss, text_loss, timestamp_loss = loss_fn(out, batch, col_stats_dict)
        
        loss.backward()
        optimizer.step()
        
        loss_accum += loss.detach().item() * batch_size
        categorical_accum += categorical_loss.detach().item() * batch_size
        numerical_accum += numerical_loss.detach().item() * batch_size
        text_accum += text_loss.detach().item() * batch_size
        timestamp_accum += timestamp_loss.detach().item() * batch_size
        count_accum += batch_size
    
        steps += 1
        if steps > cfg.train.max_steps_per_epoch:
            break
    return loss_accum / count_accum, categorical_accum / count_accum, numerical_accum / count_accum, text_accum / count_accum, timestamp_accum / count_accum

@torch.no_grad()
def test(cfg, model, task, task_name, data_loader, loss_fn, col_stats_dict, device):
    model.eval()
    loss, categorical_loss, numerical_loss, text_loss, timestamp_loss = test_one_epoch(cfg, model, task, data_loader, loss_fn, col_stats_dict, device)
    if get_world_size() > 1:
        reduced_loss = torch.tensor(loss).to(device)
        dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
        reduced_loss = reduced_loss / get_world_size()
        reduced_loss = reduced_loss.item()

        reduced_categorical_loss = torch.tensor(categorical_loss).to(device)
        dist.all_reduce(reduced_categorical_loss, op=dist.ReduceOp.SUM)
        reduced_categorical_loss = reduced_categorical_loss / get_world_size()
        reduced_categorical_loss = reduced_categorical_loss.item()
        
        reduced_numerical_loss = torch.tensor(numerical_loss).to(device)
        dist.all_reduce(reduced_numerical_loss, op=dist.ReduceOp.SUM)
        reduced_numerical_loss = reduced_numerical_loss / get_world_size()
        reduced_numerical_loss = reduced_numerical_loss.item()
        
        reduced_text_loss = torch.tensor(text_loss).to(device)
        dist.all_reduce(reduced_text_loss, op=dist.ReduceOp.SUM)
        reduced_text_loss = reduced_text_loss / get_world_size()
        reduced_text_loss = reduced_text_loss.item()
        
        reduced_timestamp_loss = torch.tensor(timestamp_loss).to(device)
        dist.all_reduce(reduced_timestamp_loss, op=dist.ReduceOp.SUM)
        reduced_timestamp_loss = reduced_timestamp_loss / get_world_size()
        reduced_timestamp_loss = reduced_timestamp_loss.item()
    else:
        reduced_loss = loss
        reduced_categorical_loss = categorical_loss
        reduced_numerical_loss = numerical_loss
        reduced_text_loss = text_loss
        reduced_timestamp_loss = timestamp_loss
        
    loss_dict = {
        "total_loss": reduced_loss,
        "categorical_loss": reduced_categorical_loss,
        "numerical_loss": reduced_numerical_loss,
        "text_loss": reduced_text_loss,
        "timestamp_loss": reduced_timestamp_loss
    }
    return {task_name: loss_dict}

def test_one_epoch(cfg, model, task, data_loader, loss_fn, col_stats_dict, device):
    loss_accum = count_accum = categorical_accum = numerical_accum = text_accum = timestamp_accum = 0
    rank = get_rank()
    entity_table = task.entity_table
    for batch in tqdm(data_loader, desc=f"Evaluating rank {rank}..."):
        batch_size = batch[task.entity_table].batch_size
        batch = batch.to(device)
        out = model(batch, task.entity_table)
        
        loss, categorical_loss, numerical_loss, text_loss, timestamp_loss = loss_fn(out, batch, col_stats_dict)
        
        loss_accum += loss.detach().item() * batch_size
        categorical_accum += categorical_loss.detach().item() * batch_size
        numerical_accum += numerical_loss.detach().item() * batch_size
        text_accum += text_loss.detach().item() * batch_size
        timestamp_accum += timestamp_loss.detach().item() * batch_size
        count_accum += batch_size
    
    return loss_accum / count_accum, categorical_accum / count_accum, numerical_accum / count_accum, text_accum / count_accum, timestamp_accum / count_accum