import torch.nn.functional as F
from torch.nn import BatchNorm1d, LayerNorm, Identity, ReLU, ELU, GLU, Sigmoid, Tanh, LeakyReLU, PReLU, SiLU


norm_layers = {
    "bn": BatchNorm1d,
    "ln": LayerNorm,
    "id": Identity
}

act_fncs = {
    "relu": (F.relu, ReLU),
    "elu": (F.elu, ELU),
    "sigmoid": (F.sigmoid, Sigmoid),
    "tanh": (F.tanh, Tanh),
    "leaky_relu": (F.leaky_relu, LeakyReLU),
    "prelu": (F.prelu, PReLU),
    "glu": (F.glu, GLU),
    "swish": (lambda x: x * F.sigmoid(x), SiLU),
    "id": (lambda x: x, Identity)
}

    
def get_norm_layer(norm_type: str):
    """
    Returns the normalization layer.
    
    Parameters
    ----------
    norm_type: str
        The type of normalization layer. Supported types: bn, ln, id
    """
    return norm_layers[norm_type]

def get_act_fnc(act_fnc: str, return_module: bool = True):
    """
    Returns the activation function.
    
    Parameters
    ----------
    act_fnc: str
        The name of the activation function. Supported types: relu, elu, sigmoid, tanh, leaky_relu, id
    return_module: bool
        If True, returns the module of the activation function. If False, returns the function.
    """
    fnc, module = act_fncs[act_fnc]
    if return_module:
        return module
    return fnc

from src.models.conv_layers.hetero_graphsage_conv import HeteroGraphSAGEConv
from src.models.hetero_encoders.resnet import ResNet

torch_frame_models = {
        "resnet": ResNet,
    }

conv_layers = {
        "graphsage": HeteroGraphSAGEConv,
    }
    
def get_hetero_encoder(cfg):
    return torch_frame_models[cfg.model.hetero_encoder.cls]

def get_conv_layer(cfg):
    return conv_layers[cfg.model.conv.cls]
