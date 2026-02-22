import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss, L1Loss

import torch.nn.functional as F

from torch_frame.data.stats import StatType
# from torch_frame.nn.encoder.stype_encoder import TimestampTensorMapper
# from torch_frame.nn.encoder.stype_encoder import PositionalEncoding, CyclicEncoding

def get_loss(cls, device, **kwargs) -> torch.nn.Module:
    """
    Returns the loss function.
    
    Parameters
    ----------
    name: str
        The name of the loss function. Supported types: cross_entropy, bce, mse, mae
    """
    if cls == 'cross_entropy':
        return CrossEntropyLoss()
    elif cls == 'bce':
        pos_weight = kwargs.get("pos_weight", None)
        if pos_weight is not None:
            pos_weight = torch.tensor(pos_weight).to(device)
        return BCEWithLogitsLoss(pos_weight=pos_weight)
    elif cls == 'mse':
        return MSELoss()
    elif cls == 'mae':
        return L1Loss()
    elif cls == 'bpr':
        return BPRLoss()
    elif cls == 'masked':
        return MaskedLoss(device=device)
    elif cls == 'scl':
        return ScaledCosineLoss(weight=kwargs['weight'], device=device)
    elif cls == 'info_nce':
        return InfoNCELoss(device=device)
    elif cls == 'scl_info_nce':
        return ScaledCosineInfoNCELoss(weight=kwargs['weight'], device=device)
    elif cls == 'scl_masked':
        return ScaledCosineMaskedLoss(weight=kwargs['weight'], device=device)
    else:
        raise ValueError(f'Unknown loss function: {cls}.')
    
class BPRLoss(torch.nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()

    def forward(self, pos_score, neg_score):
        return torch.nn.functional.softplus(-(pos_score - neg_score)).mean()
    
class MaskedLoss(torch.nn.Module):
    def __init__(self, device):
        super(MaskedLoss, self).__init__()
        self.device = device

        # # Init positional/cyclic encoding
        # self.positional_encoding = PositionalEncoding(128).to(device)
        # self.cyclic_encoding = CyclicEncoding(128).to(device)

    def forward(self, x_dict, y_dict, col_stats_dict):
        """
            x_dict: Output from the model
            y_dict: Original input
        """
        all_numerical_losses = []
        all_categorical_losses = []
        all_text_losses = []
        all_timestamp_losses = []
        categorical_loss = torch.tensor(0.0, device=self.device)
        numerical_loss = torch.tensor(0.0, device=self.device)
        text_loss = torch.tensor(0.0, device=self.device)
        timestamp_loss = torch.tensor(0.0, device=self.device)
        total_loss = torch.tensor(0.0, device=self.device)
        for node_type in x_dict.keys():
            for col_name in x_dict[node_type].keys():
                x = x_dict[node_type][col_name]
                y = y_dict[node_type].tf.get_col_feat(col_name)
                
                if StatType.COUNT in col_stats_dict[node_type][col_name]:               # Categorical
                    # Exclude y = -1 (NaN for categorical features)
                    rows_not_minus_1 = (y != -1).squeeze()
                    x = x[rows_not_minus_1]
                    y = y[rows_not_minus_1]
                    
                    # y includes target labels starting from 0
                    loss = F.cross_entropy(x, y.flatten().long())
                    all_categorical_losses.append(loss)
                elif StatType.MEAN in col_stats_dict[node_type][col_name]:              # Numerical
                    # Remove rows with NaN
                    rows_without_nan = ~torch.any(torch.isnan(y), dim=-1)
                    x = x[rows_without_nan]
                    y = y[rows_without_nan]
                    
                    col_mean = col_stats_dict[node_type][col_name][StatType.MEAN]
                    col_std = col_stats_dict[node_type][col_name][StatType.STD]
                    if col_std != 0.0:                                                  # To avoid the case of constant column
                        y = (y - col_mean) / col_std
                        loss = F.mse_loss(x, y)
                        all_numerical_losses.append(loss)
                elif StatType.EMB_DIM in col_stats_dict[node_type][col_name]:           # Text embedded
                    # y.values retrieve the text embeddings from MultiEmbeddingTensor
                    loss = F.mse_loss(x, y.values)
                    all_text_losses.append(loss)
        if len(all_categorical_losses) > 0:
            categorical_loss = torch.stack(all_categorical_losses).mean()
            total_loss += categorical_loss
        if len(all_numerical_losses) > 0:
            numerical_loss = torch.stack(all_numerical_losses).mean()
            total_loss += numerical_loss
        if len(all_text_losses) > 0:
            text_loss = torch.stack(all_text_losses).mean()
            total_loss += text_loss
        if len(all_timestamp_losses) > 0:
            timestamp_loss = torch.stack(all_timestamp_losses).mean()
            total_loss += timestamp_loss
        return total_loss, categorical_loss, numerical_loss, text_loss, timestamp_loss
    
class InfoNCELoss(torch.nn.Module):
    def __init__(self, temperature: float = 0.1, device: torch.device = torch.device('cpu')):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, x_dict, y_dict):
        '''
            x_dict: Dictionary of Augmented feature (num_nodes, channels)
            y_dict: Dictionary of Original feature (num_nodes, channels)
        '''
        losses = []
        for node_type in x_dict.keys():
            x = F.normalize(x_dict[node_type], p=2, dim=1)
            y = F.normalize(y_dict[node_type], p=2, dim=1)
            
            sim_matrix = torch.matmul(x, y.T) / self.temperature
            labels = torch.arange(sim_matrix.size(0)).to(self.device)
            loss = (F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.T, labels)) / 2
            losses.append(loss)
        loss = torch.stack(losses).mean()
        return loss

class ScaledCosineLoss(torch.nn.Module):
    def __init__(self, weight: float, alpha: float = 2, device: torch.device = torch.device('cpu')):
        super(ScaledCosineLoss, self).__init__()
        self.alpha = alpha
        self.weights = torch.tensor([weight, 1-weight]).to(device)

    def forward(self, x, y):
        '''
            x: (batch_size, num_tasks + 1)
            y: (batch_size, num_tasks + 1)
        '''
        label = y[:, -1]
        assert torch.all((label == 0) | (label == 1)), "Label must only contain 0 or 1"
        x = F.normalize(x, p=2, dim=1)
        y = F.normalize(y, p=2, dim=1)
        
        loss = (1 - (x * y).sum(dim=-1)).pow_(self.alpha)
        # MSELoss
        # loss = (x - y[:, :-1]).pow_(2).sum(dim=-1)
        weights = torch.where(label == 1, self.weights[0], self.weights[1])         # Higher weight for non-null entries
        loss = (loss * weights).sum() / weights.sum()
        return loss
    
class ScaledCosineInfoNCELoss(torch.nn.Module):
    def __init__(self, weight: float, alpha: float = 2, beta: float = 0.1, temperature: float = 0.1, device: torch.device = torch.device('cpu')):
        super(ScaledCosineInfoNCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.weights = torch.tensor([weight, 1-weight]).to(device)
        self.device = device
        
    def forward(self, pred_out, y, aug_out, out):
        '''
            pred_out: (batch_size, num_tasks + 1)
            y: (batch_size, num_tasks + 1)
            aug_out: Dictionary of Augmented feature (num_nodes, channels)
            out: Dictionary of Original feature (num_nodes, channels)
        '''
        # Scale cosine loss
        label = y[:, -1]
        assert torch.all((label == 0) | (label == 1)), "Label must only contain 0 or 1"
        pred_out = F.normalize(pred_out, p=2, dim=1)
        y = F.normalize(y, p=2, dim=1)

        loss = (1 - (pred_out * y).sum(dim=-1)).pow_(self.alpha)
        weights = torch.where(label == 1, self.weights[0], self.weights[1])         # Higher weight for non-null entries
        sce_loss = (loss * weights).sum() / weights.sum()
        
        # InfoNCE loss
        losses = []
        for node_type in aug_out.keys():
            x = F.normalize(aug_out[node_type], p=2, dim=1)
            y = F.normalize(out[node_type], p=2, dim=1)
            
            sim_matrix = torch.matmul(x, y.T) / self.temperature
            labels = torch.arange(sim_matrix.size(0)).to(self.device)
            loss = (F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.T, labels)) / 2
            losses.append(loss)
        contrastive_loss = torch.stack(losses).mean()
        return sce_loss + self.beta * contrastive_loss, sce_loss, contrastive_loss
    
class ScaledCosineMaskedLoss(torch.nn.Module):
    def __init__(self, weight: float, alpha: float = 2, beta: float = 0.1, device: torch.device = torch.device('cpu')):
        super(ScaledCosineMaskedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.weights = torch.tensor([weight, 1-weight]).to(device)
        self.device = device
        
    def forward(self, pred_out, y, x_dict, y_dict, col_stats_dict):
        # Scale cosine loss
        label = y[:, -1]
        assert torch.all((label == 0) | (label == 1)), "Label must only contain 0 or 1"
        pred_out = F.normalize(pred_out, p=2, dim=1)
        y = F.normalize(y, p=2, dim=1)

        loss = (1 - (pred_out * y).sum(dim=-1)).pow_(self.alpha)
        weights = torch.where(label == 1, self.weights[0], self.weights[1])         # Higher weight for non-null entries
        sce_loss = (loss * weights).sum() / weights.sum()
        
        # MAE loss
        all_numerical_losses = []
        all_categorical_losses = []
        all_text_losses = []
        categorical_loss = torch.tensor(0.0, device=self.device)
        numerical_loss = torch.tensor(0.0, device=self.device)
        text_loss = torch.tensor(0.0, device=self.device)
        total_loss = torch.tensor(0.0, device=self.device)
        for node_type in x_dict.keys():
            for col_name in x_dict[node_type].keys():
                x = x_dict[node_type][col_name]
                y = y_dict[node_type].tf.get_col_feat(col_name)
                
                if StatType.COUNT in col_stats_dict[node_type][col_name]:               # Categorical
                    # Exclude y = -1 (NaN for categorical features)
                    rows_not_minus_1 = (y != -1).squeeze()
                    x = x[rows_not_minus_1]
                    y = y[rows_not_minus_1]
                    
                    # y includes target labels starting from 0
                    loss = F.cross_entropy(x, y.flatten().long())
                    all_categorical_losses.append(loss)
                elif StatType.MEAN in col_stats_dict[node_type][col_name]:              # Numerical
                    # Remove rows with NaN
                    rows_without_nan = ~torch.any(torch.isnan(y), dim=-1)
                    x = x[rows_without_nan]
                    y = y[rows_without_nan]
                    
                    col_mean = col_stats_dict[node_type][col_name][StatType.MEAN]
                    col_std = col_stats_dict[node_type][col_name][StatType.STD]
                    if col_std != 0.0:                                                  # To avoid the case of constant column
                        y = (y - col_mean) / col_std
                        loss = F.mse_loss(x, y)
                        all_numerical_losses.append(loss)
                elif StatType.EMB_DIM in col_stats_dict[node_type][col_name]:           # Text embedded
                    # y.values retrieve the text embeddings from MultiEmbeddingTensor
                    loss = F.mse_loss(x, y.values)
                    all_text_losses.append(loss)
        if len(all_categorical_losses) > 0:
            categorical_loss = torch.stack(all_categorical_losses).mean()
            total_loss += categorical_loss
        if len(all_numerical_losses) > 0:
            numerical_loss = torch.stack(all_numerical_losses).mean()
            total_loss += numerical_loss
        if len(all_text_losses) > 0:
            text_loss = torch.stack(all_text_losses).mean()
            total_loss += text_loss
        return sce_loss + self.beta * total_loss, sce_loss, total_loss
    