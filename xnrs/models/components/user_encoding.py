import torch
import torch.nn as nn
from typing import Optional


class UserEncoder(nn.Module):

    def __init__(self, 
        pooler: nn.Module, 
        p_dropout: float,
        emb_dim: Optional[int] = None,
        # out_dim: Optional[int] = None,
        att: Optional[nn.Module] = None,
        head: bool = False,
        activation: nn.Module = nn.ReLU(),
        bias: bool = True
        ):
        super(UserEncoder, self).__init__()
        # to make sure the model has at least one param to 
        self.dummy_param = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(p=p_dropout)
        self.att = att
        self.pooler = pooler
        if head:
            assert emb_dim is not None
            # assert out_dim is not None
            # self.head = nn.Linear(emb_dim, emb_dim, bias=False)
            self.head = nn.Sequential(
                nn.Linear(emb_dim, emb_dim, bias=bias),
                activation,
                nn.Linear(emb_dim, emb_dim, bias=bias)
            )
        
    def forward(
        self, 
        inpt: tuple, # (x,m)
        add_features: Optional[dict] = None,
        return_weights: bool = False    
    ):
        '''
        D: embedding dim
        Args:
            inpt (tuple): (
                x: news embeddings, shape (B, N, D)
                m: attention masks, shape (B, N, 1)
            )
        '''
        x, m = inpt
        # inputs should already be on the right device
        device = next(self.parameters()).device
        x = x.to(device)
        m = m.to(device)
        x = self.dropout(x)
        if self.att is not None:
            x = self.att(x, m)
        if return_weights:  
            x, a = self.pooler(x, m, return_weights=True)
        else:
            x = self.pooler(x, m)
        if hasattr(self, 'head'):
            x = self.head(x)
        if return_weights:
            return x, a
        else:
            return x
