import torch
import torch.nn as nn
from typing import Optional, Callable

from ...utils import collaps_mask
                

class TextEncoder(nn.Module):

    def __init__(self, pooler: nn.Module, 
        p_dropout: float,
        out_features: int,
        in_features: Optional[int] = 768, 
        head: bool = True,
        activation: nn.Module = nn.ReLU(),
        att: Optional[nn.Module] = None, 
        bias: bool = True
        ):
        super(TextEncoder, self).__init__()
        # to make sure the model has at least one param
        self.dummy_param = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(p=p_dropout)
        self.att = att
        self.pooler = pooler
        if head:
            assert in_features is not None, 'in_features is required if head is True'
            self.head = nn.Sequential(
                nn.Linear(in_features, out_features, bias=bias),
                activation,
                nn.Linear(out_features, out_features, bias=bias)
            )
        self.out_dim = out_features
        
    def forward(self, inpt: tuple):
        '''
        Args:
            inpt (tuple): (
                x: sequential (ie not reduced) news embeddings, shape (B, N, S, D)
                m: attention masks, shape (B, N, S, 1)
                )
        '''
        x, m = inpt
        # TODO: moving to device should not be done here
        device = next(self.parameters()).device
        x = x.to(device)
        m = m.to(device)
        b, n, s, d = x.shape
        x = x.reshape((b * n, s, d))
        m = m.reshape((b * n, s, 1))
        x = self.dropout(x)
        if self.att is not None:
            x = self.att(x, m)
        x = self.pooler(x, m)
        if hasattr(self, 'head'):
            x = self.head(x)
        x = x.reshape((b, n, self.out_dim))
        m = m.reshape((b, n, s, 1))
        m = collaps_mask(m, dim=2)
        return (x, m)      

  
class CategoryEncoder(nn.Module):

    def __init__(
            self, 
            n_categories: int,
            embedding_dim: int,
            head: bool = True,
            activation: Optional[Callable] = torch.relu
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings= n_categories + 1,
            embedding_dim=embedding_dim
        )
        if head:
            self.linear = nn.Linear(
                in_features=embedding_dim,
                out_features=embedding_dim
            )
        if activation is not None:
            self.activation = activation

    def forward(self, x: torch.Tensor):
        x = self.embedding(x)
        if hasattr(self, 'linear'):
            x = self.linear(x)
        if hasattr(self, 'activation'):
            x = self.activation(x)
        return x
