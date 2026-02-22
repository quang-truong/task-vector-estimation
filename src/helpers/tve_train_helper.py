
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
    loss = train_one_epoch(cfg, model, task, data_loader, optimizer, loss_fn, col_stats_dict, device, **kwargs)
    if get_world_size() > 1:
        reduced_loss = torch.tensor(loss).to(device)
        dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
        reduced_loss = reduced_loss / get_world_size()
        reduced_loss = reduced_loss.item()
    else:
        reduced_loss = loss
    
    loss_dict = {
        "total_loss": reduced_loss,
    }
    
    return {task_name: loss_dict}

def train_one_epoch(cfg, model, task, data_loader, optimizer, loss_fn, col_stats_dict, device, **kwargs):
    loss_accum = count_accum = 0
    steps = 0
    total_steps = min(len(data_loader), cfg.train.max_steps_per_epoch)
    rank = get_rank()
    
    for batch in tqdm(data_loader, total=total_steps, desc=f"Training rank {rank}..."):
        batch_size = batch[task.entity_table].batch_size
        optimizer.zero_grad()
        batch = batch.to(device)
        out = model(batch, task.entity_table)
        
        loss = loss_fn(out, batch[task.entity_table].y.float())
        
        loss.backward()
        optimizer.step()
        
        loss_accum += loss.detach().item() * batch_size
        count_accum += batch_size
    
        steps += 1
        if steps > cfg.train.max_steps_per_epoch:
            break
    return loss_accum / count_accum

@torch.no_grad()
def test(cfg, model, task, task_name, data_loader, loss_fn, col_stats_dict, device):
    model.eval()
    loss = test_one_epoch(cfg, model, task, data_loader, loss_fn, col_stats_dict, device)
    
    if get_world_size() > 1:
        reduced_loss = torch.tensor(loss).to(device)
        dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
        reduced_loss = reduced_loss / get_world_size()
        reduced_loss = reduced_loss.item()
    else:
        reduced_loss = loss
    
    loss_dict = {
        "total_loss": reduced_loss,
    }
    
    return {task_name: loss_dict}

def test_one_epoch(cfg, model, task, data_loader, loss_fn, col_stats_dict, device):
    loss_accum = count_accum = 0
    rank = get_rank()
    entity_table = task.entity_table
    for batch in tqdm(data_loader, desc=f"Evaluating rank {rank}..."):
        batch_size = batch[task.entity_table].batch_size
        batch = batch.to(device)
        out = model(batch, task.entity_table)
        
        loss = loss_fn(out, batch[task.entity_table].y.float())
        
        loss_accum += loss.detach().item() * batch_size
        count_accum += batch_size
    
    return loss_accum / count_accum