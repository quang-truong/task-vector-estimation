import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
from relbench.base import EntityTask, RecommendationTask, TaskType
from relbench.base.database import Database, Table
from relbench.modeling.graph import AttachTargetTransform, NodeTrainTableInput
from relbench.modeling.utils import get_stype_proposal, remove_pkey_fkey, to_unix_time

import numpy as np
import pandas as pd

import torch
from torch import Tensor
import torch.utils
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame import stype
from torch_frame.data.stats import StatType
from torch_geometric.typing import NodeType
from torch_geometric.data import HeteroData
from torch_geometric.utils import sort_edge_index

from typing import Any, Dict, Tuple, NamedTuple, Optional, List

from src.definitions import DATA_DIR
from src.helpers.torchframe_helper import Dataset
from src.utils.io_utils import rank_zero_print as print

with open(os.path.join(DATA_DIR, "time_format.json"), "r") as f:
    time_format_dict = json.load(f)

def get_pk_fk_dict(db):
    pk_fk_dict = {}     # Primary and Foreign Key columns for each table
    for table_name, table in db.table_dict.items():
        pk_fk_dict[table_name] = []
        pk_fk_dict[table_name].append(table.pkey_col)
        pk_fk_dict[table_name].extend(table.fkey_col_to_pkey_table.keys())
    return pk_fk_dict

def remove_unnamed_columns(df):
    unnamed_columns = [col for col in df.columns if col.startswith("Unnamed:")]
    df.drop(columns=unnamed_columns, inplace=True)

def drop_nan_columns(df):
    df.dropna(axis=1, how='all', inplace=True)
    
def read_stypes_cache(stypes_cache_path):
    with open(stypes_cache_path, "r") as f:
        print(f"Reading SType Proposal from {stypes_cache_path}")
        col_to_stype_dict = json.load(f)
    for _, col_to_stype in col_to_stype_dict.items():
        for col, stype_str in col_to_stype.items():
            col_to_stype[col] = stype(stype_str)
    return col_to_stype_dict

def get_node_train_table_input(
    table: Table,
    task: EntityTask,
) -> NodeTrainTableInput:
    r"""Get the training table input for node prediction."""

    nodes = torch.from_numpy(table.df[task.entity_col].astype(int).values)
    
    # # Temporary
    # table.df = table.df.head(10000)
    # nodes = torch.from_numpy(table.df[task.entity_col].astype(int).values)

    time: Optional[Tensor] = None
    if table.time_col is not None:
        time = torch.from_numpy(to_unix_time(table.df[table.time_col]))

    target: Optional[Tensor] = None
    transform: Optional[AttachTargetTransform] = None
    if isinstance(task.target_col, str) and task.target_col in table.df:
        target_type = float
        if task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            target_type = int
        if task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
            target = torch.from_numpy(np.stack(table.df[task.target_col].values))
        else:
            target = torch.from_numpy(
                table.df[task.target_col].values.astype(target_type)
            )
        transform = AttachTargetTransform(task.entity_table, target)
    # Multi-target case
    elif isinstance(task.target_col, list) and all(col in table.df.columns for col in task.target_col):
        target = torch.from_numpy(np.stack(table.df[task.target_col].values))
        transform = AttachTargetTransform(task.entity_table, target)
    
    return NodeTrainTableInput(
        nodes=(task.entity_table, nodes),
        time=time,
        target=target,
        transform=transform,
    )

def make_pkey_fkey_graph(
    db: Database,
    dataset_cls: str,
    col_to_stype_dict: Dict[str, Dict[str, stype]],
    text_embedder_cfg: Optional[TextEmbedderConfig] = None,
    cache_dir: Optional[str] = None,
    num_workers: int = 4,
) -> Tuple[HeteroData, Dict[str, Dict[str, Dict[StatType, Any]]]]:
    r"""Given a :class:`Database` object, construct a heterogeneous graph with primary-
    foreign key relationships, together with the column stats of each table.

    Args:
        db: A database object containing a set of tables.
        col_to_stype_dict: Column to stype for
            each table.
        text_embedder_cfg: Text embedder config.
        cache_dir: A directory for storing materialized tensor
            frames. If specified, we will either cache the file or use the
            cached file. If not specified, we will not use cached file and
            re-process everything from scratch without saving the cache.

    Returns:
        HeteroData: The heterogeneous :class:`PyG` object with
            :class:`TensorFrame` feature.
    """
    data = HeteroData()
    col_stats_dict = dict()
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    for table_name, table in db.table_dict.items():
        # Materialize the tables into tensor frames:
        df = table.df
        # Ensure that pkey is consecutive.
        if table.pkey_col is not None:
            assert (df[table.pkey_col].values == np.arange(len(df))).all()

        col_to_stype = col_to_stype_dict[table_name]

        # Remove pkey, fkey columns since they will not be used as input
        # feature.
        remove_pkey_fkey(col_to_stype, table)

        if len(col_to_stype) == 0:  # Add constant feature in case df is empty:
            col_to_stype = {"__const__": stype.numerical}
            # We need to add edges later, so we need to also keep the fkeys
            fkey_dict = {key: df[key] for key in table.fkey_col_to_pkey_table}
            df = pd.DataFrame({"__const__": np.ones(len(table.df)), **fkey_dict})

        path = (
            None if cache_dir is None else os.path.join(cache_dir, f"{table_name}.pt")
        )

        print(f"Materializing '{table_name}' to {path}")
        dataset = Dataset(
            df=df,
            col_to_stype=col_to_stype,
            col_to_text_embedder_cfg=text_embedder_cfg,
            col_to_time_format=time_format_dict[dataset_cls].get(table_name, None) if dataset_cls in time_format_dict else None,
        ).materialize(num_workers=num_workers, path=path)

        data[table_name].tf = dataset.tensor_frame
        col_stats_dict[table_name] = dataset.col_stats

        # Add time attribute:
        if table.time_col is not None:
            data[table_name].time = torch.from_numpy(
                to_unix_time(table.df[table.time_col])
            )

        # Add edges:
        for fkey_name, pkey_table_name in table.fkey_col_to_pkey_table.items():
            pkey_index = df[fkey_name]
            # Filter out dangling foreign keys
            mask = ~pkey_index.isna()
            fkey_index = torch.arange(len(pkey_index))
            # Filter dangling foreign keys:
            pkey_index = torch.from_numpy(pkey_index[mask].astype(int).values)
            fkey_index = fkey_index[torch.from_numpy(mask.values)]
            # Ensure no dangling fkeys
            assert (pkey_index < len(db.table_dict[pkey_table_name])).all()

            # fkey -> pkey edges
            edge_index = torch.stack([fkey_index, pkey_index], dim=0)
            edge_type = (table_name, f"f2p_{fkey_name}", pkey_table_name)
            data[edge_type].edge_index = sort_edge_index(edge_index)

            # pkey -> fkey edges.
            # "rev_" is added so that PyG loader recognizes the reverse edges
            edge_index = torch.stack([pkey_index, fkey_index], dim=0)
            edge_type = (pkey_table_name, f"rev_f2p_{fkey_name}", table_name)
            data[edge_type].edge_index = sort_edge_index(edge_index)

    data.validate()

    return data, col_stats_dict

class AttachMultiTargetTransform:
    r"""Attach the target labels to the heterogeneous mini-batch.

    The batch consists of disjoins subgraphs loaded via temporal sampling. The same
    input node can occur multiple times with different timestamps, and thus different
    subgraphs and labels. Hence labels cannot be stored in the graph object directly,
    and must be attached to the batch after the batch is created.
    """

    def __init__(self, entity: str, target_ls: List[Tensor], task_name_ls: List[str]):
        self.entity = entity
        self.target_ls = target_ls
        self.task_name_ls = task_name_ls

    def __call__(self, batch: HeteroData) -> HeteroData:
        for i, target in enumerate(self.target_ls):
            attr_name = f"y_{self.task_name_ls[i]}"                        # Create the dynamic attribute name, e.g., y_0, y_1
            batch[self.entity][attr_name] = target[batch[self.entity].input_id]
        return batch

def get_default_db(cls, dataset, text_embedder_name):
    cache_dir = os.path.join(
            DATA_DIR, f"{cls}/materialized_cache/{text_embedder_name}"
    )
    stypes_cache_path = os.path.join(DATA_DIR, f"{cls}/stypes.json")
    db_path = os.path.join(DATA_DIR, f"{cls}/dbs/default_db/")
    
    # Load stype cache if it exists, otherwise generate it
    try:
        col_to_stype_dict = read_stypes_cache(stypes_cache_path)
    except FileNotFoundError:
        print("Generating SType Proposal... Please double check the column types or any missing columns.")
        db = dataset.get_db()
        col_to_stype_dict = get_stype_proposal(db)
        new_col_to_stype_dict = {}
        for table, col_to_stype in col_to_stype_dict.items():       # Remove unnamed columns. N/A columns are eliminated by get_stype_proposal by default.
            new_col_to_stype_dict[table] = {col: stype_str for col, stype_str in col_to_stype.items() if not col.startswith("Unnamed:")}
        col_to_stype_dict = new_col_to_stype_dict
        os.makedirs(os.path.dirname(stypes_cache_path), exist_ok=True)
        with open(stypes_cache_path, "w") as f:
            json.dump(col_to_stype_dict, f, indent=2, default=str)
    
    # Load db if it exists, otherwise generate it
    os.makedirs(db_path, exist_ok=True)
    if not any(fname.endswith('.parquet') for fname in os.listdir(db_path)):
        print("Materializing Database... Cache Directory:", db_path)
        db = dataset.get_db()
        for table_name, table in db.table_dict.items():
            df = table.df
            remove_unnamed_columns(df)                         # Remove unnamed columns
            drop_nan_columns(df)                               # Drop columns with all NaN values
            for col in df.columns:                                  # Check if all columns are in stype proposal
                assert col in col_to_stype_dict[table_name].keys(), f"Column {col} not in stype proposal."
        db.save(db_path)
    print("Loading Database... Cache Directory:", db_path)
    db = Database.load(db_path)
    
    return db, col_to_stype_dict, cache_dir

def get_text_db(cls, dataset, text_embedder_name):
    cache_dir = os.path.join(
        DATA_DIR, f"{cls}/materialized_text_emb_only_cache/{text_embedder_name}"
    )
    stypes_cache_path = os.path.join(DATA_DIR, f"{cls}/text_stypes.json")
    db_path = os.path.join(DATA_DIR, f"{cls}/dbs/text_db/")
    
    # Load stype cache if it exists, otherwise generate it
    try:
        col_to_stype_dict = read_stypes_cache(stypes_cache_path)
    except FileNotFoundError:
        db = dataset.get_db()
        try:
            col_to_stype_dict = read_stypes_cache(os.path.join(DATA_DIR, f"{cls}/stypes.json"))
        except:
            raise FileNotFoundError("Please generate the default stype proposal first.")
        pk_fk_dict = get_pk_fk_dict(db)
        for table_name, table in db.table_dict.items():         # Edit the stype proposal so that non-PK-FK and non-time columns are text_embedded
            time_col = table.time_col
            col_to_stype = col_to_stype_dict[table_name]
            for col, _ in col_to_stype.items():
                if col != time_col and col not in pk_fk_dict[table_name]:
                    col_to_stype[col] = stype("text_embedded")
        with open(stypes_cache_path, "w") as f:                 # Save the new stype proposal
            json.dump(col_to_stype_dict, f, indent=2, default=str)

    # Load db if it exists, otherwise generate it
    os.makedirs(db_path, exist_ok=True)
    if not any(fname.endswith('.parquet') for fname in os.listdir(db_path)):
        print("Materializing Database... Cache Directory:", db_path)
        db = dataset.get_db()
        for table_name, table in db.table_dict.items():
            df = table.df
            remove_unnamed_columns(df)                     # Remove unnamed columns
            drop_nan_columns(df)                           # Drop columns with all NaN values
            for col in df.columns:
                assert col in col_to_stype_dict[table_name].keys(), f"Column {col} not in stype proposal."
                val = col_to_stype_dict[table_name].get(col)    # Get the stype of the column. If text_embedded, add column name to the column value
                if val == stype("text_embedded"):
                    df[col] = col + ": " + df[col].astype(str)  # Add column name to the column value for text embedding
        db.save(db_path)
    # Always reload from parquet for consistent table ordering across ranks (see get_default_db).
    print("Loading Database... Cache Directory:", db_path)
    db = Database.load(db_path)

    return db, col_to_stype_dict, cache_dir

def get_cat_text_db(cls, dataset, text_embedder_name):
    cache_dir = os.path.join(
        DATA_DIR, f"{cls}/materialized_cat_text_emb_only_cache/{text_embedder_name}"
    )
    stypes_cache_path = os.path.join(DATA_DIR, f"{cls}/cat_text_stypes.json")
    db_path = os.path.join(DATA_DIR, f"{cls}/dbs/cat_text_db/")
    
    # Load stype cache if it exists, otherwise generate it
    try:
        col_to_stype_dict = read_stypes_cache(stypes_cache_path)
    except FileNotFoundError:
        db = dataset.get_db()
        try:
            col_to_stype_dict = read_stypes_cache(os.path.join(DATA_DIR, f"{cls}/stypes.json"))
        except:
            raise FileNotFoundError("Please generate the default stype proposal first.")
        pk_fk_dict = get_pk_fk_dict(db)
        columns_to_keep_dict = {table_name: [] for table_name in db.table_dict.keys()}
        for table_name, table in db.table_dict.items():
            time_col = table.time_col
            columns_to_keep_dict[table_name] = [col for col in col_to_stype_dict[table_name].keys() if col == time_col or \
                    col in pk_fk_dict[table_name]]              # IDs and timestamps are kept as they are
            columns = col_to_stype_dict[table_name].keys()      # Get all columns
            cat_columns = [col for col in columns if col not in columns_to_keep_dict[table_name]]    # Get remaining columns
            if len(cat_columns) > 0:                            # If there are remaining columns, create a new column "cat_text" that concatenates all remaining columns
                columns_to_keep_dict[table_name].append("cat_text")
                col_to_stype_dict[table_name]["cat_text"] = stype("text_embedded")
        
        new_col_to_stype_dict = {}
        for table, col_to_stype in col_to_stype_dict.items():   # Edit the stype proposal
            new_col_to_stype_dict[table] = {col: stype_str for col, stype_str in col_to_stype.items() if col in columns_to_keep_dict[table]}
        col_to_stype_dict = new_col_to_stype_dict
        with open(stypes_cache_path, "w") as f:                 # Save the new stype proposal
            json.dump(col_to_stype_dict, f, indent=2, default=str)
    
    # Load db if it exists, otherwise generate it
    os.makedirs(db_path, exist_ok=True)
    if not any(fname.endswith('.parquet') for fname in os.listdir(db_path)):
        print("Materializing Database... Cache Directory:", db_path)
        db = dataset.get_db()
        for table_name, table in db.table_dict.items():
            df = table.df
            remove_unnamed_columns(df)                     # Remove unnamed columns
            drop_nan_columns(df)                           # Drop columns with all NaN values
            columns = df.columns
            cat_columns = []
            columns_to_keep = col_to_stype_dict[table_name].keys()  # Columns to be kept are in the stype proposal
            for col in columns:
                if col not in columns_to_keep: # If the column is not in the stype proposal, the column will be concatenated
                    df[col] = col + ": " + df[col].astype(str)      # Add column name to the column value for text embedding
                    cat_columns.append(col)
            df["cat_text"] = df[cat_columns].apply(lambda row: " | ".join(row.astype(str)), axis=1)
            table.df = df[columns_to_keep]
        db.save(db_path)
    # Always reload from parquet for consistent table ordering across ranks (see get_default_db).
    print("Loading Database... Cache Directory:", db_path)
    db = Database.load(db_path)
    return db, col_to_stype_dict, cache_dir