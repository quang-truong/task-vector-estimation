from typing import Any, Dict, List, Type

import torch
from torch import Tensor
from torch.nn import Embedding, ModuleDict
import torch_frame

from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData, Data
from torch_geometric.nn import MLP
from torch_geometric.nn.inits import reset
from torch_geometric.typing import NodeType

from src.helpers.text_embedding_helper import embedding_size_dict

from copy import deepcopy

from src.models.components import HeteroEncoder, HeteroTemporalEncoder, HeteroModel

class TVEMAE(torch.nn.Module):
    '''
    Task Vector Estimation with Masked Autoencoder.
    '''
    def __init__(
        self,
        data: HeteroData,
        schema_graph: Data,
        schema_line_graph: Data,
        out_channels: int,
        col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],      # A dictionary that maps column name into stats
        hetero_encoder_cls: Type[torch.nn.Module] = None,
        hetero_encoder_kwargs: Dict[str, Any] = None,
        temporal_encoder_kwargs: Dict[str, Any] = None,
        conv_cls: Type[torch.nn.Module] = None,
        conv_kwargs: Dict[str, Any] = None,
        # List of node types to add shallow embeddings to input
        shallow_list: List[NodeType] = [],
        mask_rate: float = 0.25,
        **kwargs,
    ):
        super().__init__()
        
        self.mask_rate = mask_rate

        self.encoder = HeteroEncoder(
            node_to_col_names_dict={
                node_type: data[node_type].tf.col_names_dict        # col_names_dict is a dictionary that maps stype to column name
                for node_type in data.node_types                    # node_type is the table name
            },
            node_to_col_stats=col_stats_dict,
            torch_frame_model_cls=hetero_encoder_cls,
            torch_frame_model_kwargs=hetero_encoder_kwargs,
        )
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[
                node_type for node_type in data.node_types if "time" in data[node_type]     # node_type is the table name
            ],
            **temporal_encoder_kwargs
        )
        self.gnn = HeteroModel(
            node_types=data.node_types,
            edge_types=data.edge_types,
            conv_cls=conv_cls,
            text_embedder=hetero_encoder_kwargs["text_embedder"],
            **conv_kwargs,
        )

        self.embedding_dict = ModuleDict(                                   # ModuleDict is a dictionary that maps node type to Embedding
            {
                node: Embedding(data.num_nodes_dict[node], conv_kwargs["channels"])        # mapping node id to embedding
                for node in shallow_list
            }
        )
        
        # Task-specific head
        self.pred_head = torch.nn.Linear(conv_kwargs["channels"], out_channels)
        
        # Decoder
        self.enc_to_dec = torch.nn.ModuleDict(
            {
                node_type: MLP(conv_kwargs["channels"], out_channels=conv_kwargs["channels"], num_layers=1)
                for node_type in data.node_types
            }
        )
        
        clone_conv_kwargs = deepcopy(conv_kwargs)
        clone_conv_kwargs["num_layers"] = 1
        self.decoder_gnn = HeteroModel(
            node_types=data.node_types,
            edge_types=data.edge_types,
            conv_cls=conv_cls,
            text_embedder=hetero_encoder_kwargs["text_embedder"],
            **clone_conv_kwargs,
        )
        self.recon_head = torch.nn.ModuleDict(
            {
                node_type: torch.nn.ModuleDict(
                    {}
                )
                for node_type in data.node_types
            }
        )
        for node_type in data.node_types:
            for col_name in col_stats_dict[node_type].keys():
                if torch_frame.numerical in data[node_type].tf.col_names_dict and col_name in data[node_type].tf.col_names_dict[torch_frame.numerical]:
                    self.recon_head[node_type][col_name] = torch.nn.Linear(conv_kwargs["channels"], 1)
                elif torch_frame.categorical in data[node_type].tf.col_names_dict and col_name in data[node_type].tf.col_names_dict[torch_frame.categorical]:
                    # categorical columns are already encoded as integers starting from 0
                    num_categories = data[node_type].tf.get_col_feat(col_name).max() + 1
                    self.recon_head[node_type][col_name] = torch.nn.Linear(conv_kwargs["channels"], num_categories)
                elif torch_frame.embedding in data[node_type].tf.col_names_dict and col_name in data[node_type].tf.col_names_dict[torch_frame.embedding]:
                    text_embedding_size = embedding_size_dict[hetero_encoder_kwargs["text_embedder"]]
                    self.recon_head[node_type][col_name] = torch.nn.Linear(conv_kwargs["channels"], text_embedding_size)
        
        self.schema_graph = schema_graph
        self.schema_line_graph = schema_line_graph

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.temporal_encoder)
        reset(self.gnn)
        for embedding in self.embedding_dict.values():
            torch.nn.init.normal_(embedding.weight, std=0.1)
        
        reset(self.pred_head)
        reset(self.enc_to_dec)
        reset(self.decoder_gnn)
        reset(self.recon_head)
    
    def mask_input(self, batch):
        batch_clone = deepcopy(batch)
        if self.training:
            for node_type, tf in batch_clone.tf_dict.items():
                for stype in tf.stypes:
                    if stype in [torch_frame.categorical, torch_frame.numerical]:
                        # feat shape: (batch_size, num_cols)
                        feat = batch_clone[node_type].tf.feat_dict[stype]
                        shuffled_cols = []
                        for i in range(feat.size(1)):
                            feat_col = feat[:, i]
                            shuffled_cols.append(feat_col[torch.randperm(feat_col.size(0))])
                        shuffle_feat = torch.stack(shuffled_cols, dim=1)
                        mask_matrix = (torch.rand(feat.size(0), feat.size(1), device=feat.device) < self.mask_rate).int()
                        new_feat = mask_matrix * shuffle_feat + (1 - mask_matrix) * feat
                        batch_clone[node_type].tf.feat_dict[stype] = new_feat
                    elif stype == torch_frame.embedding:
                        feat = batch_clone[node_type].tf.feat_dict[stype].values
                        # Zero out embeddings
                        mask_matrix = (torch.rand(feat.size(0), feat.size(1), device=feat.device) < self.mask_rate).float()
                        feat = feat * (1 - mask_matrix)
                        batch_clone[node_type].tf.feat_dict[stype].values = feat
        return batch_clone

    def forward(
        self,
        batch: HeteroData,
        src_table: NodeType,                        # entity_table is the table name of the root node
        readout_table: NodeType = None,             # readout_entity_table is the table that the output is readout from. default is the root node
    ) -> Tensor:
        readout_table = src_table if readout_table is None else readout_table
             
        seed_time = batch[src_table].seed_time                                       # seed_time is the time of the root node. shape = (batch_size, )
        
        # Mask input
        batch_out = self.mask_input(batch)
        
        # HeteroEncoder: Encode the input data
        x_dict = self.encoder(batch_out.tf_dict)

        # Encode relative time to seed node. Return a dictionary of node type (w. time) mapped to embeddings
        rel_time_dict = self.temporal_encoder(
            seed_time, batch_out.time_dict, batch_out.batch_dict
        )

        # Add time embeddings to the encoded input. x_dict and rel_time_dict have the same keys
        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time                            # Add time embeddings to the encoded input

        # Add shallow embeddings to the input. Shallow embeddings are used to encode the node id
        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(batch_out[node_type].n_id)    # Add shallow embeddings to the encoded input
        
        x_dict = self.gnn(                                                              # Apply GNN
            x_dict,
            batch_out.edge_index_dict,
            # batch.num_sampled_nodes_dict,
            # batch.num_sampled_edges_dict,
        )
        
        # Task-specific output
        out = self.pred_head(x_dict[readout_table])
        
        # Decoder
        for node_type, enc in self.enc_to_dec.items():
            x_dict[node_type] = enc(x_dict[node_type])
        
        # Reconstruction
        recon_x_dict = self.decoder_gnn(
            x_dict,
            batch_out.edge_index_dict,
        )
        recon_out = {node_type: {} for node_type in recon_x_dict.keys()}
        for node_type, recon_x in recon_x_dict.items():
            for col_name, head in self.recon_head[node_type].items():
                recon_out[node_type][col_name] = head(recon_x)

        return out, recon_out