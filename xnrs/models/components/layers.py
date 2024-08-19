import torch
import torch.nn as nn
from typing import List
from typing import Optional
import math


class MaskedMax(nn.Module):

    def __init__(self, dim: int):
        super(MaskedMax, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, m: torch.Tensor):
        x = x * m
        return torch.max(x, dim=self.dim, keepdim=True).values


class MaskedMean(nn.Module):
    # TODO: rename to WeightedMean

    def __init__(self):
        super(MaskedMean, self).__init__()

    # TODO: just add **kwargs here to enable return_weights?
    def forward(self, x: torch.Tensor, m: torch.Tensor):
        '''
        Args:
            x: input tensor, shape (B, N, D)
            m: optional input mask, shape (B, N, 1)
        Returns:
            masked average over dim N, shape (B, 1, D)
        '''        
        # mean ignoring padding
        u = torch.sum(x * m, dim=1, keepdim=True)\
             / (torch.sum(m, dim=1, keepdim=True) + 1e-8)
        return u


class AdditiveAttention(torch.nn.Module):

    def __init__(self, in_features, hidden_features):
        super(AdditiveAttention, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, 1)

    def forward(
        self, 
        x: torch.Tensor, 
        m: torch.Tensor = None, 
        return_weights: bool = False
    ):
        '''
        Args:
            x: input tensor, shape (B, N, D)
            m: optional input mask, shape (B, N, 1)
        Returns:
            weighted average over dim N, shape (B, 1, D)
        '''
        a = self.fc2(torch.tanh(self.fc1(x)))
        a = torch.exp(a)
        if m is not None:
            a = a * m
        a = a / (torch.sum(a, dim=1, keepdim=True) + 1e-8)
        x = torch.bmm(a.transpose(-1, -2), x)
        if return_weights:
            return x, a
        else:
            return x


class PersonalizedAttention(torch.nn.Module):

    def __init__(self, in_features, hidden_features, query_features):
        super(PersonalizedAttention, self).__init__()
        self.x_fc = nn.Linear(in_features, hidden_features)
        self.q_fc = nn.Linear(query_features, hidden_features)

    def forward(self, q: torch.Tensor, x: torch.Tensor, m: torch.Tensor = None):
        '''
        Args:
            q: personalized user query, shape (B, 1, D)
            x: input tensor, shape (B, N, D)
            m: optional input mask, shape (B, N, 1)
        Returns:
            weighted average over dim N, shape (B, 1, D)
        '''
        xa = torch.tanh(self.x_fc(x))
        q = self.q_fc(q)
        b, n, hd = xa.shape
        q = q.repeat((1, n, 1))
        a = torch.bmm(
            q.reshape(b * n, hd).unsqueeze(-2),
            xa.reshape(b * n, hd).unsqueeze(-1)
        )
        a = a.reshape(b, n, 1)
        a = torch.exp(a)
        if m is not None:
            a = a * m
        a = a / (torch.sum(a, dim=1, keepdim=True) + 1e-8)
        x = torch.bmm(a.transpose(-1, -2), x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, dropout = 0.1, scaled = True):
        super().__init__()

        self.scaled = scaled
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.h = n_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, x: torch.Tensor, m: torch.Tensor):
        # TODO: check tensor shapes 

        k = x; v = x; q = x

        B = q.size(0)
        S = q.size(1)
        
        k = self.k_linear(k)
        q = self.q_linear(q)
        v = self.v_linear(v)
        
        # reshape into (B * h * S * d_mode)
        k = k.view(B, S, self.h, self.d_k).transpose(1, 2)
        q = q.view(B, S, self.h, self.d_k).transpose(1, 2)
        v = v.view(B, S, self.h, self.d_k).transpose(1, 2)

        # attention scores, shape (B, h, S, S)
        att = torch.matmul(q, k.transpose(-2, -1))
        if self.scaled:
            att = att / math.sqrt(self.d_k)
    
        if m is not None:
            m = m.unsqueeze(1)  # to shape (B, 1, S, 1)
            att = att.masked_fill(m == 0, -1e9)
        
        # row-wise normalization
        att = torch.softmax(att, dim=-1)
        att = self.dropout(att)  # TODO: check dropout after normalizing... ?

        # applying attention 
        output = torch.matmul(att, v)
        # reshape int (B, S, d_model), final layer
        output = output.transpose(1,2).contiguous().view(B, S, self.d_model)
        output = self.out(output)
    
        return output


class DenseAttention(nn.Module):

    def __init__(self, input_dim: int, hidden_dim1: int, hidden_dim2: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim1)
        self.tanh1 = nn.Tanh()
        self.linear2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.tanh2 = nn.Tanh()
        self.linear3 = nn.Linear(hidden_dim2, 1)

    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        x = self.tanh1(x)
        x = self.linear2(x)
        x = self.tanh2(x)
        x = self.linear3(x)
        return x