from typing import List, Optional
import os

import torch
from torch import Tensor
from sentence_transformers import SentenceTransformer

from src.definitions import WEIGHT_DIR

embedding_size_dict = {
    'glove': 300,
    'minilm': 384,
}

class GloveTextEmbedding:
    def __init__(self, device: Optional[torch.device
                                       ] = None):
        self.model = SentenceTransformer(
            "sentence-transformers/average_word_embeddings_glove.6B.300d",
            cache_folder=os.path.join(WEIGHT_DIR, "sentence-transformers"),
            device=device,
        )

    def __call__(self, sentences: List[str]) -> Tensor:
        return torch.from_numpy(self.model.encode(sentences))       # DO NOT USE self.model.encode(sentences, convert_to_tensor=True)! It will raise an error for DDP.
    
class AllMiniLML6v2TextEmbedding:
    def __init__(self, device: Optional[torch.device
                                       ] = None):
        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            cache_folder=os.path.join(WEIGHT_DIR, "sentence-transformers"),
            device=device,
        )

    def __call__(self, sentences: List[str]) -> Tensor:
        return torch.from_numpy(self.model.encode(sentences))       # DO NOT USE self.model.encode(sentences, convert_to_tensor=True)! It will raise an error for DDP.