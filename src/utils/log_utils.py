import os
import numpy as np
import wandb
import json
from typing import Dict, List, Any
from collections import defaultdict
from src.utils.io_utils import rank_zero_print, synchronize, is_main_process

class CurvesLogger:
    def __init__(self, task_names, metadata_dict):
        self.loss_curves_dict: Dict[str, Dict[str, Dict[str, List[float]]]] = {task_name: {'train': defaultdict(lambda: []), 'val': defaultdict(lambda: []), 'test': defaultdict(lambda: [])} for task_name in task_names}
        self.perf_curves_dict: Dict[str, Dict[str, Dict[str, List[float]]]] = {task_name: {'val': defaultdict(lambda: []), 'test': defaultdict(lambda: [])} for task_name in task_names}
        self.best_epoch_dict: Dict[str, int] = {task_name: 0 for task_name in task_names}
        self.metadata_dict: Dict[str, Dict[str, Any]] = metadata_dict
    
    def log_loss(self, 
                 train_loss_dict: Dict[str, Dict[str, float]], 
                 val_loss_dict: Dict[str, Dict[str, float]] = None, 
                 test_loss_dict: Dict[str, Dict[str, float]] = None):
        for task_name, train_loss in train_loss_dict.items():
            for loss_name, loss_value in train_loss.items():
                self.loss_curves_dict[task_name]['train'][loss_name].append(loss_value)
        if val_loss_dict is not None:
            for task_name, val_loss in val_loss_dict.items():
                for loss_name, loss_value in val_loss.items():
                    self.loss_curves_dict[task_name]['val'][loss_name].append(loss_value)
        if test_loss_dict is not None:
            for task_name, test_loss in test_loss_dict.items():
                for loss_name, loss_value in test_loss.items():
                    self.loss_curves_dict[task_name]['test'][loss_name].append(loss_value)
    
    def log_perf(self, 
                 val_perf_dict: Dict[str, Dict[str, float]], 
                 test_perf_dict: Dict[str, Dict[str, float]]):
        for task_name, _ in val_perf_dict.items():
            metadata = self.metadata_dict[task_name]
            tune_metric = metadata['tune_metric']
            higher_is_better = metadata['higher_is_better']
            for metric, value in val_perf_dict[task_name].items():
                self.perf_curves_dict[task_name]['val'][metric].append(value)
                # Update the best epoch for the task
                if metric == tune_metric:
                    best_epoch = self.best_epoch_dict[task_name]
                    is_best = (not higher_is_better and value < self.perf_curves_dict[task_name]['val'][metric][best_epoch]) \
                            or (higher_is_better and value > self.perf_curves_dict[task_name]['val'][metric][best_epoch])
                    if is_best:
                        self.best_epoch_dict[task_name] = len(self.perf_curves_dict[task_name]['val'][metric]) - 1
            for metric, value in test_perf_dict[task_name].items():
                self.perf_curves_dict[task_name]['test'][metric].append(value)
    
def print_epoch_metrics(run, epoch, curves_logger, optimizer=None):
    """
    Prints the epoch information including train, validation, and test losses as a tuple,
    as well as performance metrics from validation and test sets.

    Args:
    - epoch (int): Current epoch number.
    - train_loss_dict (dict): Dictionary of training losses.
    - val_loss_dict (dict): Dictionary of validation losses.
    - test_loss_dict (dict): Dictionary of test losses.
    - val_perf_dict (dict): Dictionary of validation metrics.
    - test_perf_dict (dict): Dictionary of test metrics.
    - optimizer: Optimizer object that contains learning rate information.
    """
    if is_main_process():
        out = ""
        loss_curves_dict = curves_logger.loss_curves_dict
        perf_curves_dict = curves_logger.perf_curves_dict
        for task in loss_curves_dict.keys():
            train_loss_dict: Dict[str, List[float]] = loss_curves_dict[task]['train']
            val_loss_dict: Dict[str, List[float]] = loss_curves_dict[task]['val']
            test_loss_dict: Dict[str, List[float]] = loss_curves_dict[task]['test']
            val_perf_dict: Dict[str, List[float]] = perf_curves_dict[task]['val']
            test_perf_dict: Dict[str, List[float]] = perf_curves_dict[task]['test']
            
            out_str = f"{task}: ["
            
            train_loss_str = " | ".join(
                [f'train-{loss_name}: {loss_values[-1]:.3f}' for loss_name, loss_values in train_loss_dict.items()]
            )
            
            val_loss_str = " | ".join(
                [f'val-{loss_name}: {loss_values[-1]:.3f}' for loss_name, loss_values in val_loss_dict.items()]
            ) if len(val_loss_dict) > 0 else ""
                
            test_loss_str = " | ".join(
                [f'test-{loss_name}: {loss_values[-1]:.3f}' for loss_name, loss_values in test_loss_dict.items()]
            ) if len(test_loss_dict) > 0 else ""
            
            val_perf_str = " | ".join(
                [f'val-{metric}: {values[-1]:.3f}' for metric, values in val_perf_dict.items()]
            ) if len(val_perf_dict) > 0 else ""
                
            test_perf_str = " | ".join(
                [f'test-{metric}: {values[-1]:.3f}' for metric, values in test_perf_dict.items()]
            ) if len(test_perf_dict) > 0 else ""
            
            out_str += train_loss_str
            if val_loss_str:
                out_str += ' | ' + val_loss_str
            if test_loss_str:
                out_str += ' | ' + test_loss_str
            if val_perf_str:
                out_str += ' | ' + val_perf_str
            if test_perf_str:
                out_str += ' | ' + test_perf_str
            out += out_str
            out += ']\n'

        rank_zero_print()
        lr_str = f'lr: {optimizer.param_groups[0]["lr"]:.6f}' if optimizer is not None else ""
        print(
            f'epoch: {epoch + 1} | {lr_str} \n{out}'
        )
        
        # Prepare the log dictionary
        log_dict = {
            f"run{run+1}/epoch": epoch + 1,
            f"run{run+1}/lr": optimizer.param_groups[0]["lr"] if optimizer is not None else None
        }
        
        # Add individual task losses to the log
        for task in loss_curves_dict.keys():
            train_loss_dict: Dict[str, List[float]] = loss_curves_dict[task]['train']
            val_loss_dict: Dict[str, List[float]] = loss_curves_dict[task]['val']   
            test_loss_dict: Dict[str, List[float]] = loss_curves_dict[task]['test']
            for loss_name, loss_values in train_loss_dict.items():
                log_dict[f"run{run+1}/{task}/train-{loss_name}"] = loss_values[-1]
            if len(val_loss_dict) > 0:
                for loss_name, loss_values in val_loss_dict.items():
                    log_dict[f"run{run+1}/{task}/val-{loss_name}"] = loss_values[-1]
            if len(test_loss_dict) > 0:
                for loss_name, loss_values in test_loss_dict.items():
                    log_dict[f"run{run+1}/{task}/test-{loss_name}"] = loss_values[-1]

        # Add validation performance metrics to the log
        for task in perf_curves_dict.keys():
            val_perf_dict: Dict[str, List[float]] = perf_curves_dict[task]['val']
            if len(val_perf_dict) > 0:
                for metric, values in val_perf_dict.items():
                    log_dict[f"run{run+1}/{task}/val-{metric}"] = values[-1]        # Log the latest value for each metric
            
            test_perf_dict: Dict[str, List[float]] = perf_curves_dict[task]['test']
            if len(test_perf_dict) > 0:
                for metric, values in test_perf_dict.items():
                    log_dict[f"run{run+1}/{task}/test-{metric}"] = values[-1]       # Log the latest value for each metric


        # Log everything with wandb
        wandb.log(log_dict)
    synchronize()
    
def print_final_results(cfg, run, curves_logger, log_dir):
    """
    Prints the final results of the training process, including the best epoch and performance metrics.
    """
    if is_main_process():
        loss_curves_dict = curves_logger.loss_curves_dict
        perf_curves_dict = curves_logger.perf_curves_dict
        best_epoch_dict = curves_logger.best_epoch_dict
        # General information
        msg = (
            '\n'
            f'========== Result ============\n'
            f'Dataset:          {cfg.dataset}\n'
        )
        
        for task_name in loss_curves_dict.keys():
            best_epoch = best_epoch_dict[task_name]
            msg += f'--------- Best epoch ---------\n'
            msg += f'{task_name}:\n'
            msg += 'Train loss:\n'
            for loss_name, loss_values in loss_curves_dict[task_name]['train'].items():
                msg += f'    {loss_name}: {loss_values[best_epoch]:.3f}\n'

            # Adding validation performance metrics
            msg += 'Val performance:\n'
            for metric, values in perf_curves_dict[task_name]['val'].items():
                msg += f'    {metric:<18}: {values[best_epoch]:<10.4f}\n'

            # Adding test loss and performance metrics
            msg += 'Test performance:\n'
            for metric, values in perf_curves_dict[task_name]['test'].items():
                msg += f'    {metric:<18}: {values[best_epoch]:<10.4f}\n'

            # Best epoch information
            msg += f'Best epoch:       {best_epoch + 1:<10}\n'   # Epoch starts from 1 for user readability
            msg += '----------------------------\n\n'
        
        if len(best_epoch_dict) > 1:
            best_epochs = [best_epoch_dict[task_name] for task_name in best_epoch_dict.keys()]
            for task_name in loss_curves_dict.keys():
                # Print average performance across best epochs
                msg += f'--------- Average performance across best epochs ---------\n'
                msg += f'{task_name}:\n'
                msg += '    Val performance:\n'
                for metric, perf_values in perf_curves_dict[task_name]['val'].items():
                    perf_array = np.array([perf_values[best_epoch] for best_epoch in best_epochs])
                    msg += (
                        f'        {metric}:\n'
                        f'            mean ± std:       {np.mean(perf_array):.4f} ± {np.std(perf_array, ddof=1):.4f}\n'
                        f'            min:              {np.min(perf_array):.4f}\n'
                        f'            max:              {np.max(perf_array):.4f}\n'
                    )
                msg += '    Test performance:\n'
                for metric, perf_values in perf_curves_dict[task_name]['test'].items():
                    perf_array = np.array([perf_values[best_epoch] for best_epoch in best_epochs])
                    msg += (
                        f'        {metric}:\n'
                        f'            mean ± std:       {np.mean(perf_array):.4f} ± {np.std(perf_array, ddof=1):.4f}\n'
                        f'            min:              {np.min(perf_array):.4f}\n'
                        f'            max:              {np.max(perf_array):.4f}\n'
                    )
                msg += '----------------------------\n\n'

        # Print the final message
        rank_zero_print()
        print(msg)
        
        filename = os.path.join(log_dir, f'run{run+1}_results.txt')
        with open(filename, 'w') as handle:
            handle.write(msg)
        rank_zero_print(f"==> Saved results to {filename}\n")
        wandb.save(filename)
        
        # Dump all loss curves to a file
        for task_name, loss_dict in loss_curves_dict.items():
            with open(os.path.join(log_dir, f'run{run+1}_all_{task_name}_loss.json'), 'w') as f:
                json.dump(loss_dict, f)
        # Dump all performance metrics to a file
        for task_name, perf_dict in perf_curves_dict.items():
            with open(os.path.join(log_dir, f'run{run+1}_all_{task_name}_perf.json'), 'w') as f:
                json.dump(perf_dict, f)
        wandb.save(os.path.join(log_dir, f'*.json'))
    synchronize()

def print_runs_stats(cfg, all_perf, task_name, log_dir, num_params: tuple[int, int]):
    log_dict = {
            "num_trainable_params": num_params[0],
            "num_total_params": num_params[1]
        }
    if is_main_process():
        # Initialize dictionary to store final aggregated results
        final_results = {
            'val': {},
            'test': {}
        }
        
        # Log the average, min, max, and std for validation performance metrics
        for metric, perf_values in all_perf['val'].items():
            perf_array = np.array(perf_values)
            final_results['val'][metric] = {
                'min': np.min(perf_array),
                'max': np.max(perf_array),
                'mean': np.mean(perf_array),
                'std': np.std(perf_array, ddof=1)
            }

        # Log the average, min, max, and std for test performance metrics
        for metric, perf_values in all_perf['test'].items():
            perf_array = np.array(perf_values)
            final_results['test'][metric] = {
                'min': np.min(perf_array),
                'max': np.max(perf_array),
                'mean': np.mean(perf_array),
                'std': np.std(perf_array, ddof=1)
            }
        
        # Construct the final results message
        msg = (
            "\n"
            f"========= Final result ==========\n"
            f"Dataset: {cfg.dataset}\n"
            f"SHA: {cfg.sha}\n"
            f"----------- Best epoch ----------\n"
        )

        # Add validation performance metrics to the message
        msg += "Val Performance:\n"
        for metric, stats in final_results['val'].items():
            msg += (
                f'{metric}:\n'
                f'  mean ± std:       {stats["mean"]:.4f} ± {stats["std"]:.4f}\n'
                f'  min:              {stats["min"]:.4f}\n'
                f'  max:              {stats["max"]:.4f}\n'
            )

        # Add test performance metrics to the message
        msg += "Test Performance:\n"
        for metric, stats in final_results['test'].items():
            msg += (
                f'{metric}:\n'
                f'  mean ± std:       {stats["mean"]:.4f} ± {stats["std"]:.4f}\n'
                f'  min:              {stats["min"]:.4f}\n'
                f'  max:              {stats["max"]:.4f}\n'
            )

        msg += "-------------------------------\n\n"

        # Print the final message
        rank_zero_print()
        print(msg)
        
        # Add validation performance metrics to the log
        for metric, stats in final_results['val'].items():
            log_dict[f"{task_name}/mean_best_val_{metric}"] = stats['mean']

        # Add test performance metrics to the log
        for metric, stats in final_results['test'].items():
            log_dict[f"{task_name}/mean_best_test_{metric}"] = stats['mean']

        # Log everything with wandb
        wandb.log(log_dict)
        
        # Create WandB table for the final results
        columns = ["run", "name", "note"]
        row = [log_dir.split('/')[-1], cfg.name, cfg.note]
        for split in ['val', 'test']:
            for metric, stats in final_results[split].items():
                columns.append(f"{task_name}/mean_best_{split}_{metric}")
                row.append(f'{stats["mean"]:.4f} ± {stats["std"]:.4f}')
        table=wandb.Table(data=[row], columns=columns)
        wandb.log({"best_final_results": table})
        
        filename = os.path.join(log_dir, f'average_results.txt')
        with open(filename, 'w') as handle:
            handle.write(msg)
        rank_zero_print(f"==> Saved results to {filename}")
        wandb.save(filename)
    synchronize()
    return log_dict
