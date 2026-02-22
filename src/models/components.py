from typing import Any, Dict, List, Type

import torch
from torch import Tensor
from torch.nn import Embedding, ModuleDict
import torch_frame

from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData, Data
from torch_geometric.nn import MLP, PositionalEncoding, BatchNorm, LayerNorm, GCNConv
from torch_geometric.nn.models import JumpingKnowledge
from torch_geometric.nn.inits import reset
from torch_geometric.typing import NodeType, EdgeType

from src.helpers.model_helper import get_norm_layer, get_act_fnc
from src.utils.io_utils import rank_zero_print as print
from src.helpers.text_embedding_helper import embedding_size_dict

from copy import deepcopy

class SimpleLinearBlock(torch.nn.Module):
    """
    A simple linear block with normalization and activation functions.
    
    Parameters
    ----------
    in_channels: int
        The number of input features.
    out_channels: int
        The number of output features.
    norm: str
        The type of normalization layer. Supported types: bn, ln, id
    act_fnc: str
        The name of the activation function. Supported types: relu, elu, sigmoid, tanh, leaky_relu, id
    dropout_rate: float
        The dropout rate. Default: 0.0
    bias: bool
        If set to False, the layer will not learn an additive bias. Default: True
    """
    def __init__(self, in_channels: int, out_channels: int, norm: str, act_fnc: str, dropout_rate: float=0.0, bias: bool = True):
        super(SimpleLinearBlock, self).__init__()
        self.linear = torch.nn.Linear(in_channels, out_channels, bias=bias)
        self.norm = get_norm_layer(norm)(out_channels)
        self.act_fnc = get_act_fnc(act_fnc)()
        self.dropout = torch.nn.Dropout(dropout_rate)

        self.reset_parameters()
        
    def reset_parameters(self):
        reset(self.linear)
        reset(self.norm)
        reset(self.act_fnc)

    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x)
        x = self.act_fnc(x)
        x = self.dropout(x)
        return x

class HeteroEncoder(torch.nn.Module):
    r"""HeteroEncoder based on PyTorch Frame.

    Args:
        channels (int): The output channels for each node type.
        node_to_col_names_dict (Dict[NodeType, Dict[torch_frame.stype, List[str]]]):
            A dictionary mapping from node type to column names dictionary
            compatible to PyTorch Frame.
        torch_frame_model_cls: Model class for PyTorch Frame. The class object
            takes :class:`TensorFrame` object as input and outputs
            :obj:`channels`-dimensional embeddings. Default to
            :class:`torch_frame.nn.ResNet`.
        torch_frame_model_kwargs (Dict[str, Any]): Keyword arguments for
            :class:`torch_frame_model_cls` class. Default keyword argument is
            set specific for :class:`torch_frame.nn.ResNet`. Expect it to
            be changed for different :class:`torch_frame_model_cls`.
        default_stype_encoder_cls_kwargs (Dict[torch_frame.stype, Any]):
            A dictionary mapping from :obj:`torch_frame.stype` object into a
            tuple specifying :class:`torch_frame.nn.StypeEncoder` class and its
            keyword arguments :obj:`kwargs`.
    """

    def __init__(
        self,
        node_to_col_names_dict: Dict[NodeType, Dict[torch_frame.stype, List[str]]],
        node_to_col_stats: Dict[NodeType, Dict[str, Dict[StatType, Any]]],
        torch_frame_model_cls: Type[torch.nn.Module],
        torch_frame_model_kwargs: Dict[str, Any],
        default_stype_encoder_cls_kwargs: Dict[torch_frame.stype, Any] = {
            torch_frame.categorical: (torch_frame.nn.EmbeddingEncoder, {}),
            torch_frame.numerical: (torch_frame.nn.LinearEncoder, {}),
            torch_frame.multicategorical: (
                torch_frame.nn.MultiCategoricalEmbeddingEncoder,
                {},
            ),
            torch_frame.embedding: (torch_frame.nn.LinearEmbeddingEncoder, {}),
            torch_frame.timestamp: (torch_frame.nn.TimestampEncoder, {}),
        },
    ):
        super().__init__()

        self.encoders = torch.nn.ModuleDict()
        
        self.use_text_embedding_only = torch_frame_model_kwargs.get("use_text_embedding_only", False)  \
            or torch_frame_model_kwargs.get("use_cat_text_embedding", False)
        if self.use_text_embedding_only:        # If use_text_embedding_only is True, then use only text embedding directly
            default_stype_encoder_cls_kwargs = {
                torch_frame.embedding: (torch_frame.nn.LinearEmbeddingEncoder, {}),
                torch_frame.timestamp: (torch_frame.nn.TimestampEncoder, {}),
            }

        print("Columns and their semantic types:")
        for node_type in node_to_col_names_dict.keys():
            print(f"Table: {node_type}")
            for stype in node_to_col_names_dict[node_type].keys():
                print(f"\tColumns: {node_to_col_names_dict[node_type][stype]}, Stype: {stype}")
            stype_encoder_dict = {
                stype: default_stype_encoder_cls_kwargs[stype][0](
                    **default_stype_encoder_cls_kwargs[stype][1]
                )
                for stype in node_to_col_names_dict[node_type].keys()
            }
            torch_frame_model = torch_frame_model_cls(
                **torch_frame_model_kwargs,
                col_stats=node_to_col_stats[node_type],
                col_names_dict=node_to_col_names_dict[node_type],
                stype_encoder_dict=stype_encoder_dict,
            )
            self.encoders[node_type] = torch_frame_model

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.encoders)

    def forward(
        self,
        tf_dict: Dict[NodeType, torch_frame.TensorFrame],
    ) -> Dict[NodeType, Tensor]:
        x_dict = {}
        for node_type, tf in tf_dict.items():
            x_dict[node_type] = self.encoders[node_type](tf)
        return x_dict

class HeteroTemporalEncoder(torch.nn.Module):
    def __init__(self, 
                 node_types: List[NodeType], 
                 in_channels: int, 
                 **kwargs):
        super().__init__()

        self.encoder_dict = torch.nn.ModuleDict(
            {node_type: PositionalEncoding(in_channels) for node_type in node_types}
        )
        self.lin_dict = torch.nn.ModuleDict(
            {node_type: SimpleLinearBlock(in_channels, **kwargs) for node_type in node_types}
        )

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.encoder_dict)
        reset(self.lin_dict)

    def forward(
        self,
        seed_time: Tensor,
        time_dict: Dict[NodeType, Tensor],
        batch_dict: Dict[NodeType, Tensor],
    ) -> Dict[NodeType, Tensor]:
        out_dict: Dict[NodeType, Tensor] = {}

        for node_type, time in time_dict.items():
            rel_time = seed_time[batch_dict[node_type]] - time
            rel_time = rel_time / (60 * 60 * 24)  # Convert seconds to days.

            x = self.encoder_dict[node_type](rel_time)
            x = self.lin_dict[node_type](x)
            out_dict[node_type] = x

        return out_dict
    
class HeteroModel(torch.nn.Module):
    def __init__(
        self,
        node_types: List[NodeType],
        edge_types: List[EdgeType],
        channels: int,
        num_layers: int,
        aggr: str,
        norm: str,
        act_fnc: str,
        dropout_rate: float,
        bias: bool = True,
        conv_cls: Type[torch.nn.Module] = None,
        jump_mode: str = "cat",
        **kwargs,
    ):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = conv_cls(
                node_types=node_types,
                edge_types=edge_types,
                channels=channels,
                aggr=aggr,
                act_fnc=act_fnc,
                bias=bias,
                **kwargs,
            )
            self.convs.append(conv)

        self.norms = torch.nn.ModuleList()
        self.act_fncs = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()
        for _ in range(num_layers):
            norm_dict = torch.nn.ModuleDict()
            act_fnc_dict = torch.nn.ModuleDict()
            dropout_dict = torch.nn.ModuleDict()
            for node_type in node_types:
                norm_dict[node_type] = LayerNorm(channels, mode="node") if norm == "ln" else BatchNorm(channels)
                act_fnc_dict[node_type] = get_act_fnc(act_fnc)()
                dropout_dict[node_type] = torch.nn.Dropout(dropout_rate)
            self.norms.append(norm_dict)
            self.act_fncs.append(act_fnc_dict)
            self.dropouts.append(dropout_dict)
        if jump_mode:
            self.jumps = torch.nn.ModuleDict()
            self.lins = torch.nn.ModuleDict()
            for node_type in node_types:
                self.jumps[node_type] = JumpingKnowledge(jump_mode, channels)
                if jump_mode == "cat":
                    self.lins[node_type] = SimpleLinearBlock(channels * num_layers, channels, norm, act_fnc, dropout_rate, bias)
                else:
                    self.lins[node_type] = SimpleLinearBlock(channels, channels, norm, act_fnc, dropout_rate, bias)
        else:
            self.jumps = None
        self.reset_parameters()
            

    def reset_parameters(self):
        reset(self.convs)
        reset(self.norms)
        reset(self.act_fncs)
        if self.jumps:
            reset(self.jumps)
            reset(self.lins)

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[NodeType, Tensor],
    ) -> Dict[NodeType, Tensor]:
        if self.jumps:
            xs_dict = {key: [] for key, _ in x_dict.items()}
        for i, (conv, norm_dict, act_fnc_dict, dropout_dict) in enumerate(zip(self.convs, self.norms, self.act_fncs, self.dropouts)):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: dropout_dict[key](act_fnc_dict[key](norm_dict[key](x))) for key, x in x_dict.items()}
            if self.jumps:
                xs_dict = {key: xs_dict[key] + [x] for key, x in x_dict.items()}
        if self.jumps:
            x_dict = {key: self.lins[key](self.jumps[key](xs)) for key, xs in xs_dict.items()}
        return x_dict
