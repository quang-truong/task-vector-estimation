'''
    Code from https://github.com/xinzhuma/patchnet/blob/master/lib/helpers/save_helper.py
'''

import os
import torch

from src.utils.io_utils import rank_zero_print as print

def average_state_dicts(state_dicts):
    # Initialize an empty state_dict with the same structure
    avg_state_dict = {}

    # Iterate over all keys in the state_dict
    for key in state_dicts[0]:
        # Stack the tensors from all state_dicts along a new dimension
        stacked_tensors = torch.stack([state_dict[key] for state_dict in state_dicts], dim=0)
        
        # Compute the mean along the new dimension
        avg_state_dict[key] = torch.mean(stacked_tensors, dim=0)

    return avg_state_dict

def save_backbone_checkpoint(model, optimizer, epoch, filename):
    state = {
        "model.encoder": model.encoder.state_dict(),
        "model.temporal_encoder": model.temporal_encoder.state_dict(),
        "model.gnn": model.gnn.state_dict(),
        "model.embedding_dict": {key: embedding.state_dict() for key, embedding in model.embedding_dict.items()},
        "epoch": epoch,
    }
    torch.save(state, filename)
    print(f"==> Saved to {filename}") 
    
def load_backbone_checkpoint(model, optimizer, filename, device='cpu'):
    if os.path.isfile(filename):
        print(f"==> Loading from checkpoint {filename}")
        state = torch.load(filename, map_location=device)
        epoch = state['epoch']
        print(f"==> Loading weights from the best epoch (Epoch: {epoch + 1})")
        if model is not None:
            model.encoder.load_state_dict(state['model.encoder'])
            model.temporal_encoder.load_state_dict(state['model.temporal_encoder'])
            model.gnn.load_state_dict(state['model.gnn'])
            for key, embedding in model.embedding_dict.items():
                embedding.load_state_dict(state['model.embedding_dict'][key])
        print("==> Done")
    else:
        raise FileNotFoundError

def save_checkpoint(model, optimizer, epoch, filename):
    state = {
                "model": model.state_dict(),
                "epoch": epoch,
            }
    torch.save(state, filename)
    print(f"==> Saved to {filename}")


def load_checkpoint(model, optimizer, filename, device='cpu'):
    if os.path.isfile(filename):
        print(f"==> Loading from checkpoint {filename}")
        state = torch.load(filename, map_location=device)
        epoch = state['epoch']
        print(f"==> Loading weights from the best epoch (Epoch: {epoch + 1})")
        if model is not None and state['model'] is not None:
            model.load_state_dict(state['model'])
        print("==> Done")
    else:
        raise FileNotFoundError