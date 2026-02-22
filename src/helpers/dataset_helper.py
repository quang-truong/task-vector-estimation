import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
from relbench.base import EntityTask, RecommendationTask

import torch
from torch import Tensor
import torch.utils
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.loader import NeighborLoader
from torch_geometric.typing import NodeType

from typing import Dict, Tuple

from src.definitions import DATA_DIR
from src.helpers.text_embedding_helper import AllMiniLML6v2TextEmbedding, GloveTextEmbedding
from src.helpers.relbench_helper import get_dataset
from src.helpers.task_helper import get_task
from src.helpers.schema_helper import create_schema_graph
from src.helpers.database_helper import (
    get_text_db, 
    get_cat_text_db, 
    get_default_db, 
    make_pkey_fkey_graph, 
    get_node_train_table_input
)
from src.utils.io_utils import rank_zero_print as print
from src.utils.io_utils import rank_zero_pprint as pprint
from src.utils.io_utils import synchronized_fnc, get_world_size, get_rank, is_main_process

def get_text_relation_embedding(db, text_embedder_name, text_embedder, save_dir):
    if is_main_process():
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{text_embedder_name}-relation_embeddings.pt")
        if not os.path.isfile(save_path):
            print("Generating Text Relation Embeddings... Cache Directory:", save_path)
            graph, line_graph = create_schema_graph(db)
            edges = []
            for edge in graph.edge_dict.keys():
                src, _, dst = edge.split('__')
                edges.append(f"{src}, {dst}")
            edge_embeddings = text_embedder(edges)
            torch.save(edge_embeddings, save_path)
            
def save_schema_graph(db, cls, device):
    schema_graph_dir = os.path.join(DATA_DIR, f"{cls}/schema_graph.pt")
    schema_line_graph_dir = os.path.join(DATA_DIR, f"{cls}/schema_line_graph.pt")
    if not os.path.isfile(schema_graph_dir) or not os.path.isfile(schema_line_graph_dir):
        print("Dump Schema Graph... Cache Directory:", schema_graph_dir)
        print("Dump Schema Line Graph... Cache Directory:", schema_line_graph_dir)
        schema_graph, schema_line_graph = create_schema_graph(db)
        torch.save(schema_graph, schema_graph_dir)
        torch.save(schema_line_graph, schema_line_graph_dir)
        schema_graph = schema_graph.to(device)
        schema_line_graph = schema_line_graph.to(device)        
    else:
        print("Loading Schema Graph... Cache Directory:", schema_graph_dir)
        print("Loading Schema Line Graph... Cache Directory:", schema_line_graph_dir)
        schema_graph = torch.load(schema_graph_dir, map_location=device)
        schema_line_graph = torch.load(schema_line_graph_dir, map_location=device)
    return schema_graph, schema_line_graph

def get_data(cfg, device, print_stype=False):
    cls = cfg.dataset.cls
    task_name_ls = [cfg.dataset.task] if isinstance(cfg.dataset.task, str) else cfg.dataset.task
    
    assert cls in ["rel-amazon", "rel-stack", "rel-trial", "rel-f1", "rel-hm", "rel-event", "rel-avito"], f"Unknown dataset: {cls}"
    
    dataset = synchronized_fnc(get_dataset, cls, download=True)

    # Text embedding configuration
    text_embedder_name = cfg.model.hetero_encoder.kwargs.text_embedder
    if text_embedder_name == "minilm":
        text_embedder = AllMiniLML6v2TextEmbedding(device=device)
    elif text_embedder_name == "glove":
        text_embedder = GloveTextEmbedding(device=device)
    else:
        raise ValueError(f"Unknown text embedder: {cfg.model.hetero_encoder.kwargs.text_embedder}")
    text_embedder_cfg = synchronized_fnc(TextEmbedderConfig,
        text_embedder=text_embedder, batch_size=64,
    )
    if cfg.model.hetero_encoder.kwargs.get("use_text_embedding_only", False):
        db, col_to_stype_dict, cache_dir = synchronized_fnc(get_text_db, cls, dataset, text_embedder_name)
    elif cfg.model.hetero_encoder.kwargs.get("use_cat_text_embedding", False):
        db, col_to_stype_dict, cache_dir = synchronized_fnc(get_cat_text_db, cls, dataset, text_embedder_name)
    else:
        db, col_to_stype_dict, cache_dir = synchronized_fnc(get_default_db, cls, dataset, text_embedder_name)
    
    if cfg.model.conv.kwargs.get("text_relation_embeddings", False):
        relation_embedding_save_dir = os.path.join(DATA_DIR, f"{cls}/relation_embeddings")
        synchronized_fnc(get_text_relation_embedding, db, text_embedder_name, text_embedder, relation_embedding_save_dir)
                
    if print_stype:
        print("SType Proposal")
        pprint(col_to_stype_dict)
    
    print("Make PK-FK Graph and Dump Cache... Cache Directory:", cache_dir)
    # Generate the materialized graph
    data, col_stats_dict = synchronized_fnc(make_pkey_fkey_graph,
        db,
        cls,
        col_to_stype_dict=col_to_stype_dict,    # speficied column types
        text_embedder_cfg=text_embedder_cfg,    # our chosen text encoder
        cache_dir=cache_dir,                    # store materialized graph for convenience
        num_workers=8                           # number of workers for materialization
    )
    
    # Save schema graph
    schema_graph, schema_line_graph = synchronized_fnc(save_schema_graph, db, cls, device)
    
    train_table_ls = []
    val_table_ls = []
    test_table_ls = []
    task_ls = []
    for task_name in task_name_ls:
        task: EntityTask | RecommendationTask = synchronized_fnc(get_task, cls, task_name, download=False)       # download=False for self-created tasks

        train_table_ls.append(task.get_table("train"))
        val_table_ls.append(task.get_table("val"))
        test_table_ls.append(task.get_table("test", mask_input_cols=False))
        task_ls.append(task)
    
    print(f"Loaded Dataset: {cfg.dataset.cls}(task={cfg.dataset.task})")
    
    if len(task_ls) == 1:
        return {
            "dataset": dataset,                             # dataset
            "task": task_ls[0],                             # task
            "task_name": task_name_ls[0],                   # task name
            "train_table": train_table_ls[0],               # training table
            "val_table": val_table_ls[0],                   # validation table
            "test_table": test_table_ls[0],                 # testing table
            "data": data,                                   # data as HeteroData (heterogeneous graph)
            "schema_graph": schema_graph,                   # schema graph
            "schema_line_graph": schema_line_graph,         # schema line graph
            "col_stats_dict": col_stats_dict,               # column statistics for HeteroEncoder
        }
    else:
        return {
            "dataset": dataset,                             # dataset
            "task_ls": task_ls,                             # task
            "task_name_ls": task_name_ls,                   # task name
            "train_table_ls": train_table_ls,               # training table
            "val_table_ls": val_table_ls,                   # validation table
            "test_table_ls": test_table_ls,                 # testing table
            "data": data,                                   # data as HeteroData (heterogeneous graph)
            "schema_graph": schema_graph,                   # schema graph
            "schema_line_graph": schema_line_graph,         # schema line graph
            "col_stats_dict": col_stats_dict,               # column statistics for HeteroEncoder
        }

def get_loaders(cfg, data_dict):
    if "task" in data_dict:
        task = data_dict["task"]
        if issubclass(task.__class__, EntityTask):
            loaders = get_node_task_loaders(cfg, data_dict)
        else:
            raise ValueError(f"Unknown task type: {task}")
    print("Loaded Data Loaders.")
    return loaders

def get_node_task_loaders(cfg, data_dict):
    loader_dict = {}
    train_table = data_dict["train_table"]
    val_table = data_dict["val_table"]
    test_table = data_dict["test_table"]
    data = data_dict["data"]
    task = data_dict["task"]
    
    for split, table in [
        ("train", train_table),
        ("val", val_table),
        ("test", test_table),
    ]:
        table_input = get_node_train_table_input(       # NodeTrainTableInput(nodes=(entity type, node idx), time=time, target=target, transform=transform)
            table=table,                                # time is extracted from the table
            task=task,
        )
        shuffle = split == "train"
        # sampler = torch.utils.data.distributed.DistributedSampler(
        #     convert_node_input_table_to_list(table_input), num_replicas=get_world_size(), rank=get_rank(), shuffle=shuffle, seed=cfg.seed
        # )
        sampler = torch.utils.data.distributed.DistributedSampler(
            table_input.nodes[1], num_replicas=get_world_size(), rank=get_rank(), shuffle=shuffle, seed=cfg.seed
        )
        loader_dict[split] = NeighborLoader(
            data,                                       # data is graph
            num_neighbors=cfg.train.num_neighbors,
            time_attr="time",                           # time attribute of `table_input`, not from the original table
            input_nodes=table_input.nodes,
            input_time=table_input.time,
            subgraph_type=cfg.train.subgraph_type,
            transform=table_input.transform,
            batch_size=cfg.train.batch_size,
            temporal_strategy=cfg.train.temporal_strategy,
            sampler=sampler,
            num_workers=cfg.train.num_workers,
            persistent_workers=cfg.train.num_workers > 0,
        )
    return loader_dict
