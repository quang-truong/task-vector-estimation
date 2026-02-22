'''
    Some of the following code is borrowed from the following repository:
    https://github.com/KiddoZhu/NBFNet-PyG/blob/master/nbfnet/util.py
'''
import argparse
import ast
import datetime
import easydict
import inspect
import jinja2
import os
import numpy as np
import rich
import rich.tree
import shutil
import torch
import time
import yaml
import wandb
import json

from jinja2 import meta
from torch import distributed as dist
from easydict import EasyDict

from src.definitions import LOG_DIR, SRC_DIR, DATA_DIR

def parse_sweep_config(file_path):
    if file_path == "":
        return {}
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            config = yaml.safe_load(file)
        
        # Extract values and format keys as required
        parsed_config = {
            "lr": config["optimizer"]["lr"],
            "warmup_steps": config["lr_scheduler"]["warmup_steps"],
        }

        return parsed_config
    else:
        raise FileNotFoundError(f"File not found: {file_path}")

def parse_cfg_args():
    """
    Parses command-line arguments and dynamic arguments defined in a YAML configuration file.
    Returns:
        tuple: A tuple containing:
            - args (argparse.Namespace): Parsed config file argument.
            - vars (dict): Parsed other undefined arguments in the configuration file from command line.
            These arguments can override the arguments defined in the configuration file if they have the same name.
    Raises:
        ValueError: If there are unrecognized arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="yaml configuration file", required=True)
    parser.add_argument("-s", "--sweep_config", help="sweep configuration file. If a directory is \
                        input, run every sweep files in the directory.", type=str, default="")
    parser.add_argument("--sweep_only_for", help="sweeping for specific files. File index start from 1.", type=str, default="")

    # parse config file and other arguments
    args, unparsed = parser.parse_known_args()
    return args, unparsed

def parse_string_or_list(value):
    """
    Parses a string or a list of strings.
    If the value contains commas, split it into a list of strings.
    Otherwise, return it as a single string.
    """
    if ',' in value:
        return sorted(value.split(','))
    return value
    
def parse_train_args(args, unparsed):
    sweep_cfg_dict = parse_sweep_config(args.sweep_config)
    
    # get dynamic arguments (undeclared arguments) defined in the config file
    num_gpus = len(os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(','))
    vars = detect_variables(args.config)
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", help="random seed", type=int, default=43)
    parser.add_argument("-d", "--debug", help="debug mode", action="store_true")
    parser.add_argument("-f", "--fast_dev_run", help="fast_dev_run mode", action="store_true")
    parser.add_argument("--num_runs", help="number of runs", type=int, default=5)
    parser.add_argument("--name", help="experiment name", type=str, default="")
    parser.add_argument("--note", help="note for the experiment", type=str, default="")
    parser.add_argument("--project", help="project name", type=str, default="rl-dl")
    parser.add_argument("--checkpoint", help="checkpoint path", type=str, default="")
    parser.add_argument("--checkpoint_dst", help="checkpoint path for dst model", type=str, default="")
    parser.add_argument("--gpus", help="list of gpus", type=str, default=str(list(range(num_gpus))))
    parser.add_argument("--num_workers", help="number of workers", type=int, default=0)
    added_arguments = ["seed", "debug", "fast_dev_run", "num_runs", "name", "note", "project", "checkpoint", "checkpoint_dst", "num_workers", "gpus"]
    
    # add undefined dynamic arguments to the parser
    for var in vars:
        if var not in added_arguments and var not in sweep_cfg_dict:
            parser.add_argument("--%s" % var, required=True)
    
    # parse the dynamic arguments
    vars, unparsed = parser.parse_known_args(unparsed)
    
    # parse string or list of strings for multi tasks
    # vars.task = parse_string_or_list(vars.task)
    
    if unparsed:
        raise ValueError("Arguments %s are not recognized" % unparsed)
    vars = {k: literal_eval(v) for k, v in vars._get_kwargs()}
    vars.update(sweep_cfg_dict)
    return args, vars

def handle_exit(signal_num, frame):
    print("Received exit signal, terminating processes...")
    if is_main_process():
        wandb.finish()
        time.sleep(10)
        torch.cuda.empty_cache()
    synchronize()
    destroy_process_group()
    exit(0)

def rank_zero_print(*args, print_details=True, **kwargs):
    """Custom print function that only prints if the current process is rank zero."""
    if get_rank() == 0:
        if print_details:
            # Get current date and time
            current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Get the name of the file where rank_zero_print was called
            frame = inspect.currentframe()
            caller_frame = frame.f_back
            full_file_path = caller_frame.f_code.co_filename
            file_name = os.path.relpath(full_file_path)  # Make file path relative to current directory
            line_number = caller_frame.f_lineno  # Get line number
            function_name = caller_frame.f_code.co_name  # Get function name
            
            # Format the message with datetime, filename, line number, and function name
            prefix = f"\[{current_time}]\[{file_name}:{line_number}]\[{function_name}] -"
        else:
            prefix = ""
        # Print the custom message using rich.print
        rich.print(prefix, *args, **kwargs)
    
def rank_zero_pprint(*args, print_details=True, **kwargs):
    """Custom pprint function that only prints if the current process is rank zero."""
    if get_rank() == 0:
        if print_details:
            # Get current date and time
            current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Get the name of the file where rank_zero_print was called
            frame = inspect.currentframe()
            caller_frame = frame.f_back
            full_file_path = caller_frame.f_code.co_filename
            file_name = os.path.relpath(full_file_path)  # Make file path relative to current directory
            line_number = caller_frame.f_lineno  # Get line number
            function_name = caller_frame.f_code.co_name  # Get function name
            
            # Format the message with datetime, filename, line number, and function name
            prefix = f"\[{current_time}]\[{file_name}:{line_number}]\[{function_name}] -"
        else:
            prefix = ""
        
        # Print the custom message using rich.print
        rich.print(prefix)
        rich.pretty.pprint(args, **kwargs, expand_all=True)

def load_config(cfg_file, context=None):
    """
    Load and parse a configuration file using Jinja2 templating and YAML parsing.

    Args:
        cfg_file (str): Path to the configuration file.
        context (dict, optional): Dictionary containing context variables for Jinja2 templating. Defaults to None.

    Returns:
        EasyDict: Parsed configuration as an EasyDict object.
    """
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    template = jinja2.Template(raw)
    instance = template.render(context)
    cfg = yaml.safe_load(instance)
    cfg = easydict.EasyDict(cfg)
    # Sort tasks
    if isinstance(cfg.dataset.task, list):
        cfg.dataset.task = sorted(cfg.dataset.task)
    return cfg


def literal_eval(string):
    """
    Safely evaluate an expression node or a string containing a Python literal or container display.

    Args:
        string (str): The string to evaluate.

    Returns:
        Any: The result of evaluating the string as a Python literal. If the string cannot be evaluated,
             it returns the original string.
    """
    try:
        return ast.literal_eval(string)
    except (ValueError, SyntaxError):
        return string

def detect_variables(cfg_file):
    """
    Detects undeclared variables in a Jinja2 template file.

    Args:
        cfg_file (str): Path to the Jinja2 template file.

    Returns:
        set: A set of undeclared variable names found in the template.
    """
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    env = jinja2.Environment()
    tree = env.parse(raw)
    vars = meta.find_undeclared_variables(tree)
    return vars


def get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    return 0

def is_main_process():
    return get_rank() == 0


def get_world_size():
    if dist.is_initialized():
        return dist.get_world_size()
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    return 1


def synchronized_fnc(fnc, *args, **kwargs):
    """
    Executes a function in a synchronized manner across multiple processes.
    This function ensures that only the main process performs certain actions 
    (like downloading or writing) while other processes wait. After the main 
    process completes, all processes synchronize, and non-main processes 
    execute the function.
    Args:
        fnc (callable): The function to be executed.
        *args: Variable length argument list to be passed to the function.
        **kwargs: Arbitrary keyword arguments to be passed to the function.
    Returns:
        The result of the function execution.
    """
    # Ensure only the main process performs downloading/writing
    if is_main_process():
        result = fnc(*args, **kwargs)
    
    # Synchronize all processes so others wait for rank 0 to finish
    synchronize()

    # Non-main processes run the function with download=False
    if not is_main_process():
        if "download" in kwargs and kwargs.get("download"):
            kwargs["download"] = False
        result = fnc(*args, **kwargs)
    
    return result

def synchronize():
    if get_world_size() > 1:
        dist.barrier()


def get_device(cfg):
    if cfg.train.gpus:
        device = torch.device(cfg.train.gpus[get_rank()])
    else:
        device = torch.device("cpu")
    return device

def init_process_group(cfg):
    world_size = get_world_size()
    if cfg.train.gpus is not None and len(cfg.train.gpus) != world_size:
        error_msg = "World size is %d but found %d GPUs in the argument"
        if world_size == 1:
            error_msg += ". Did you launch with `python -m torch.distributed.launch`?"
        raise ValueError(error_msg % (world_size, len(cfg.train.gpus)))
    if world_size > 1 and not dist.is_initialized():
        torch.cuda.set_device(get_device(cfg))
        dist.init_process_group("nccl", init_method="env://", world_size=get_world_size(), rank=get_rank())
        
def destroy_process_group():
    if dist.is_initialized():
        dist.destroy_process_group()
        
def check_for_invalid_values(cfg):
    # Check for invalid values
    for key, value in cfg.items():
        if isinstance(value, EasyDict):
            check_for_invalid_values(value)
        else:
            assert value not in ['None', 'null', '', 'True', 'true', 'False', 'false'], f"Invalid value for {key}: {value}"

def sanity_check(cfg):
    check_for_invalid_values(cfg)
    
    if isinstance(cfg.dataset.task, list) and all(isinstance(task, str) for task in cfg.dataset.task):
        with open(os.path.join(DATA_DIR, "dataset_task.json"), "r") as f:
            dataset_task_dict = json.load(f)
        for task in cfg.dataset.task:
            assert task in dataset_task_dict[cfg.dataset.cls], f"Unknown task: {task}"
            assert f"{task}_loss" in cfg.train.loss_fns, f"Loss function for task {task} must be specified."
            assert cfg.train.loss_fns[f"{task}_loss"].get("weight", None) is not None, f"Weight for loss function {task}_loss must be specified."
    
def infer_dataset_task(cfg, cfg_file):
    with open(os.path.join(DATA_DIR, "dataset_task.json"), "r") as f:
        dataset_task_dict = json.load(f)
    if cfg.dataset is not None and cfg.dataset.cls is not None and cfg.dataset.task is not None:
        assert cfg.dataset.cls in dataset_task_dict.keys(), f"Unknown dataset class: {cfg.dataset.cls}"
        if isinstance(cfg.dataset.task, list):
            assert all(task in dataset_task_dict[cfg.dataset.cls] for task in cfg.dataset.task), f"Unknown task: {cfg.dataset.task}"
        else:
            assert cfg.dataset.task in dataset_task_dict[cfg.dataset.cls], f"Unknown task: {cfg.dataset.task}"
        return cfg.dataset.cls, cfg.dataset.task
    else:
        split_ls = cfg_file.split("/")
        dataset_class = next((s for s in split_ls if s.startswith("rel-")), None)
        try:
            dataset_task = split_ls[split_ls.index(dataset_class) + 1]
        except ValueError:
            dataset_task = None
        if dataset_class and dataset_task:
            assert dataset_class in dataset_task_dict.keys(), f"Unknown dataset class: {dataset_class}"
            assert dataset_task in dataset_task_dict[dataset_class], f"Unknown task: {dataset_task}"
        else:
            if int(os.getenv("NUM_CUDA_DEVICES", "1")) > 1:
                print("Dataset class and task must be specified for DDP.")
                exit(0)
            print("Dataset class and task could not be inferred from the config file.")
            dataset_class = input(f"Enter a dataset class from {list(dataset_task_dict.keys())}: ").strip()
            assert dataset_class in dataset_task_dict.keys(), f"Unknown dataset class: {dataset_class}"
            dataset_task = input(f"Enter a task from {dataset_task_dict[dataset_class]}: ").strip()
            assert dataset_task in dataset_task_dict[dataset_class], f"Unknown task: {dataset_task}"
        return dataset_class, dataset_task   
        
def setup_cfg(cfg):
    cfg.debug = cfg.debug or cfg.fast_dev_run
    if cfg.fast_dev_run:
        cfg.train.num_epochs = 2
        cfg.train.max_steps_per_epoch = 5
        cfg.num_runs = 2
    if is_main_process():
        if not cfg.debug:
            if not cfg.name:
                cfg.name = input("Enter a name for this run (Press Enter for model name): ").strip()
            if not cfg.note:
                cfg.note = input("Enter notes for this run (Press Enter for no notes): ").strip()
        if cfg.name == "" or cfg.name is None:
            if isinstance(cfg.dataset.task, str) or isinstance(cfg.dataset.task, list) and all(isinstance(task, str) for task in cfg.dataset.task):
                cfg.name = "-".join([cfg.model.hetero_encoder.cls, cfg.model.conv.cls])
            else:
                raise ValueError("Invalid dataset task.")
        if cfg.note == "":
            cfg.note = None
    synchronize()
    
def setup_wandb(cfg, log_dir):
    if is_main_process():
        wandb.init(
            project = cfg.project,
            config = cfg,
            mode = "disabled" if cfg.debug else "online",
            job_type = "debug" if cfg.debug else "train",
            dir = log_dir,
            name = cfg.name,
            notes = cfg.note
        )
        wandb.run.log_code(root=SRC_DIR)
        wandb.save(os.path.join(log_dir, "config.yaml"))
        if os.path.exists(os.path.join(log_dir, "sweep.yaml")):
            wandb.save(os.path.join(log_dir, "sweep.yaml"))
    synchronize()

def create_log_dir(cfg, cfg_file, sweep_file, sweep_folder_name=None):
    file_name = "log_dir.tmp"
    mode = 'debug' if cfg.debug else 'train'
    if sweep_folder_name:
        folder_name = cfg.name.split("-")[-1]                           # Get cfg_id from cfg.name
        if isinstance(cfg.dataset.task, str):
            log_dir = os.path.join(LOG_DIR, mode, cfg.dataset.cls, cfg.dataset.task, "-".join([cfg.model.hetero_encoder.cls, cfg.model.conv.cls]), sweep_folder_name, folder_name)
    else:
        folder_name = datetime.datetime.now().isoformat(timespec='seconds').replace('T', '_').replace(':','-')
        if isinstance(cfg.dataset.task, str):
            log_dir = os.path.join(LOG_DIR, mode, cfg.dataset.cls, cfg.dataset.task, "-".join([cfg.model.hetero_encoder.cls, cfg.model.conv.cls]), folder_name)

    # synchronize working directory
    if get_rank() == 0:
        with open(file_name, "w") as fout:
            fout.write(log_dir)
        os.makedirs(log_dir)
    synchronize()
    if get_rank() != 0:
        with open(file_name, "r") as fin:
            log_dir = fin.read()
    synchronize()
    if get_rank() == 0:
        os.remove(file_name)
    
    rank_zero_print(f"Logging to directory: {log_dir}")
    # copy the config file to the log directory
    if is_main_process():
        shutil.copy(cfg_file, os.path.join(log_dir, "config.yaml"))
        if sweep_file:
            shutil.copy(sweep_file, os.path.join(log_dir, "sweep.yaml"))
    
    return log_dir

def print_config_tree(
    cfg: dict,
    save_to_file: bool = False,
    output_dir: str = "./",
) -> None:
    """Prints the contents of a dictionary as a tree structure using the Rich library.

    :param cfg: A dictionary representing your configuration.
    :param save_to_file: Whether to export the config tree to a file. Default is False.
    :param output_dir: Directory to save the config tree file, if save_to_file is True.
    """
    
    def add_branch(tree, key, value, key_style="bold cyan", value_style="green"):
        """Recursively add branches for nested dictionaries, and print simple values for leaf nodes."""
        if isinstance(value, dict):  # If the value is a nested dictionary
            branch = tree.add(f"[{key_style}]{key}[/]", style=key_style, guide_style="dim")
            for sub_key, sub_value in value.items():
                add_branch(branch, sub_key, sub_value, key_style, value_style)
        else:  # If it's a leaf node (non-dictionary)
            tree.add(f"[{key_style}]{key}[/]: [{value_style}]{value}[/]")
    
    if is_main_process():
        # Set styles for keys and values
        key_style = "bold cyan"
        value_style = "green"

        # Create the root of the tree
        tree = rich.tree.Tree("CONFIG", style="bold red", guide_style="dim")

        # Generate the config tree from the dictionary
        for field, value in cfg.items():
            add_branch(tree, field, value, key_style, value_style)

        # Print config tree to console only if rank is zero
        rich.print(tree)
        print()

        # Optionally save config tree to a file
        if save_to_file:
            output_path = os.path.join(output_dir, "config_tree.log")
            with open(output_path, "w") as file:
                rich.print(tree, file=file)
            wandb.save(output_path)
    synchronize()
                
def compute_params(model, print_params=False):
    rank_zero_print("Model Parameters")
    num_trainable_params = -1
    num_total_params = -1
    if is_main_process():
        for name, child in model.named_children():
            tmp_trainable_params = 0
            tmp_total_params = 0
            for param in child.parameters():
                if param.requires_grad:
                    tmp_trainable_params += param.numel()
                tmp_total_params += param.numel()
            print(f"=================== {name} ========================")
            print(f"Trainable params: {tmp_trainable_params}")
            print(f"Total params    : {tmp_total_params}")
            
    if is_main_process():
        if print_params:
            print("==================== Model Parameters =======================")
        trainable_params = 0
        total_params = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                if print_params:
                    print(name, param.size())
                trainable_params += param.numel()
            total_params += param.numel()
        print("=================== Params stats ========================")
        print(f"Trainable params: {trainable_params}")
        print(f"Total params    : {total_params}")
        print("=========================================================")
        num_trainable_params = trainable_params
        num_total_params = total_params
    synchronize()
    return num_trainable_params, num_total_params