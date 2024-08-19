from typing import Callable
import torch
import torch.nn as nn


class DotScoring(nn.Module):

    def __init__(self, normalize: bool = False):
        super(DotScoring, self).__init__()
        self.normalize = normalize

    def forward(self, u: torch.Tensor, c: torch.Tensor):
        '''
        Args:
            u: user representation, shape (B, 1, D)
            c: candidates representation, shape (B, N, D)
        Returns:
            scores, shape (B, N, 1)
        '''
        if self.normalize:
            u = u / u.norm(p=2, dim=2, keepdim=True)
            c = c / c.norm(p=2, dim=2, keepdim=True)
        return torch.bmm(c, u.transpose(-1, -2))


class CAUMScoring(DotScoring):

    def forward(self, u: torch.Tensor, c: torch.Tensor):
        '''
        Args:
            u: candidate aware user representation, shape (B, N, D)
            c: candidates representation, shape (B, N, D)
        Returns:
            scores, shape (B, N, 1)
        '''
        s_mat = super().forward(u, c)
        s = torch.diagonal(s_mat, dim1=-1, dim2=-2)
        return s.unsqueeze(-1)
    

class BilinScoring(nn.Module):

    def __init__(self, emb_dim: int, normalize: bool = False, bias: bool = True):
        super(BilinScoring, self).__init__()
        self.bilin = nn.Bilinear(
            in1_features=emb_dim,
            in2_features=emb_dim,
            out_features=1,
            bias=bias
        )
        self.normalize = normalize

    def forward(self, u: torch.Tensor, c: torch.Tensor):
        '''
        Args:
            u: user representation, shape (B, 1, D)
            c: candidates representation, shape (B, N, D)
        Returns:
            scores, shape (B, N, 1)
        '''
        if self.normalize:
            u = u / u.norm(p=2, dim=2, keepdim=True)
            c = c / c.norm(p=2, dim=2, keepdim=True)
        n_cand = c.shape[1]
        u = torch.cat([u] * n_cand, dim=1)
        return self.bilin(u, c)


class FCScoring(nn.Module):

    def __init__(
        self, 
        emb_dim: int, 
        hidden_dim: int, 
        activation: Callable = torch.tanh, 
        bias: bool = True
    ):
        super(FCScoring, self).__init__()
        self.fc1 = nn.Linear(
            in_features= 2 * emb_dim, 
            out_features=hidden_dim,
            bias=bias
        )
        self.fc2 = nn.Linear(
            in_features=hidden_dim,
            out_features=1,
            bias=bias
        )
        self.activation = activation

    def forward(self, u: torch.Tensor, c: torch.Tensor):
        '''
        D corresponds to the embedding dim
        Args:
            u: user representation, shape (B, 1, D)
            c: candidate representation, shape (B, N, D)
        '''
        # TODO: dropout?
        N = c.shape[1]
        u = u.repeat((1, N, 1))
        x = torch.cat([u, c], dim=2)
        return self.fc2(self.activation(self.fc1(x)))
