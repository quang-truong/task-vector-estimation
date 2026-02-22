import torch
import torch.distributed as dist

from src.utils.io_utils import is_main_process, get_rank

from torch.optim import Optimizer

class WarmupScheduler:
    def __init__(self, optimizer: Optimizer, warmup_steps: int, base_lr: float):
        """
        Args:
            optimizer (Optimizer): PyTorch optimizer
            warmup_steps (int): Number of steps for warmup
            base_lr (float): Target learning rate after warmup
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.current_step = 1

    def step(self):
        """Update learning rate based on the current step."""
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            lr = self.base_lr * (self.current_step / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        # After warmup, the learning rate remains constant (base_lr)

    def get_lr(self):
        """Get the current learning rate."""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]

def get_lr_scheduler(cfg, optimizer, metadata):
    cls = None if cfg.lr_scheduler is None else cfg.lr_scheduler.cls
    if cls == 'ReduceLROnPlateau':
        mode = 'max' if metadata['higher_is_better'] else 'min'
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode,
                                                            factor=cfg.lr_scheduler.decay_rate,
                                                            patience=cfg.lr_scheduler.patience)
    elif cls == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.lr_scheduler.decay_steps,
                                                    gamma=cfg.lr_scheduler.decay_rate)
    elif cls == 'WarmupLR':
        scheduler = WarmupScheduler(optimizer, cfg.lr_scheduler.warmup_steps, cfg.optimizer.lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = scheduler.base_lr * (scheduler.current_step / scheduler.warmup_steps)
    elif cls is None:
        scheduler = None
    else:
        raise NotImplementedError(f'Scheduler {cfg.lr_scheduler.cls} is not currently supported.')
    return scheduler

def step_scheduler(cfg, scheduler, optimizer, val_perf, metadata):
    break_signal = False
    if scheduler is not None:
        if cfg.lr_scheduler.cls == 'ReduceLROnPlateau':
            val_value = val_perf[metadata['tune_metric']]
            if val_value is not None:
                val_value = val_value.item()
                scheduler.step(val_value)
                if optimizer.param_groups[0]['lr'] < cfg.lr_scheduler.min_lr:
                    print("\n!! The minimum learning rate has been reached.")
                    break_signal = True
            else:
                raise ValueError(f"Validation performance is None for task {metadata['task_name']}")
        else:
            scheduler.step()
    return break_signal