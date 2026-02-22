from __future__ import annotations

import math
from typing import Any

from torch import Tensor
from torch.nn import (
    BatchNorm1d,
    Dropout,
    LayerNorm,
    Identity,
    Linear,
    Module,
    Sequential,
)

import torch_frame
from torch_frame import TensorFrame, stype
from torch_frame.data.stats import StatType
from torch_frame.nn.encoder.stype_encoder import (
    EmbeddingEncoder,
    LinearEncoder,
    StypeEncoder,
)
from torch_frame.nn.encoder.stypewise_encoder import StypeWiseFeatureEncoder
from torch_geometric.nn.inits import reset
from src.helpers.model_helper import get_norm_layer, get_act_fnc


class FCResidualBlock(Module):
    r"""Fully connected residual block.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        normalization (str, optional): The type of normalization to use.
            :obj:`layer_norm`, :obj:`batch_norm`, or :obj:`None`.
            (default: :obj:`layer_norm`)
        dropout_prob (float): The dropout probability (default: `0.0`, i.e.,
            no dropout).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: str,
        act_fnc: str,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.lin1 = Linear(in_channels, out_channels)
        self.lin2 = Linear(out_channels, out_channels)
        self.act_fnc1 = get_act_fnc(act_fnc)()
        self.act_fnc2 = get_act_fnc(act_fnc)()
        self.dropout = Dropout(dropout_rate)

        self.norm1: BatchNorm1d | LayerNorm | Identity = get_norm_layer(norm)(out_channels)
        self.norm2: BatchNorm1d | LayerNorm | Identity = get_norm_layer(norm)(out_channels)

        self.shortcut: Linear | Identity = Linear(in_channels, out_channels) if in_channels != out_channels else Identity()
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        reset(self.lin1)
        reset(self.lin2)
        reset(self.act_fnc1)
        reset(self.act_fnc2)
        reset(self.norm1)
        reset(self.norm2)
        reset(self.shortcut)

    def forward(self, x: Tensor) -> Tensor:
        out = self.lin1(x)
        out = self.norm1(out)
        out = self.act_fnc1(out)
        out = self.dropout(out)

        out = self.lin2(out)
        out = self.norm2(out)
        out = self.act_fnc2(out)
        out = self.dropout(out)

        x = self.shortcut(x)

        out = out + x

        return out


class ResNet(Module):
    r"""The ResNet model introduced in the
    `"Revisiting Deep Learning Models for Tabular Data"
    <https://arxiv.org/abs/2106.11959>`_ paper.

    .. note::

        For an example of using ResNet, see `examples/revisiting.py
        <https://github.com/pyg-team/pytorch-frame/blob/master/examples/
        revisiting.py>`_.

    Args:
        in_channels (int): The number of channels in the backbone layers.
        out_channels (int): The number of output channels in the decoder.
        num_layers (int): The number of layers in the backbone.
        col_stats(dict[str,Dict[:class:`torch_frame.data.stats.StatType`,Any]]):
             A dictionary that maps column name into stats.
             Available as :obj:`dataset.col_stats`.
        col_names_dict (dict[:class:`torch_frame.stype`, List[str]]): A
            dictionary that maps stype to a list of column names. The column
            names are sorted based on the ordering that appear in
            :obj:`tensor_frame.feat_dict`. Available as
            :obj:`tensor_frame.col_names_dict`.
        stype_encoder_dict
            (dict[:class:`torch_frame.stype`,
            :class:`torch_frame.nn.encoder.StypeEncoder`], optional):
            A dictionary mapping stypes into their stype encoders.
            (default: :obj:`None`, will call :obj:`EmbeddingEncoder()`
            for categorical feature and :obj:`LinearEncoder()` for
            numerical feature)
        normalization (str, optional): The type of normalization to use.
            :obj:`batch_norm`, :obj:`layer_norm`, or :obj:`None`.
            (default: :obj:`layer_norm`)
        dropout_prob (float): The dropout probability (default: `0.2`).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: str,
        act_fnc: str,
        dropout_rate: float = 0.2,
        num_layers: int = 4,
        col_stats: dict[str, dict[StatType, Any]] = None,
        col_names_dict: dict[torch_frame.stype, list[str]] = None,
        stype_encoder_dict: dict[torch_frame.stype, StypeEncoder] | None = None,
        **kwargs
    ) -> None:
        super().__init__()

        if stype_encoder_dict is None:
            stype_encoder_dict = {
                stype.categorical: EmbeddingEncoder(),
                stype.numerical: LinearEncoder(),
            }

        self.encoder = StypeWiseFeatureEncoder(
            out_channels=in_channels,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
        )

        num_cols = sum(
            [len(col_names) for col_names in col_names_dict.values()])
        self.backbone = Sequential(*[
            FCResidualBlock(
                in_channels * num_cols if i == 0 else in_channels,
                in_channels,
                norm=norm,
                act_fnc=act_fnc,
                dropout_rate=dropout_rate,
            ) for i in range(num_layers)
        ])

        self.decoder = Sequential(
            get_norm_layer(norm)(in_channels),
            get_act_fnc(act_fnc)(),
            Linear(in_channels, out_channels),
            Dropout(dropout_rate),
        )
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        reset(self.encoder)
        reset(self.backbone)
        reset(self.decoder)

    def forward(self, tf: TensorFrame) -> Tensor:
        r"""Transforming :class:`TensorFrame` object into output prediction.

        Args:
            tf (TensorFrame): Input :class:`TensorFrame` object.

        Returns:
            torch.Tensor: Output of shape [batch_size, out_channels].
        """
        x, _ = self.encoder(tf)

        # Flattening the encoder output
        x = x.view(x.size(0), math.prod(x.shape[1:]))

        x = self.backbone(x)
        out = self.decoder(x)
        return out
