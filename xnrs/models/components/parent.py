import torch
import torch.nn as nn
from typing import Tuple, Optional

from ...utils import collaps_mask


class ParentRec(nn.Module):
    # TODO: rename to SingleTextParentRec
    
    def __init__(self, 
        news_encoder: nn.Module, 
        user_encoder: nn.Module, 
        rec_model: nn.Module, 
        text_feature: str = 'title_emb'
    ):
        super(ParentRec, self).__init__()
        self.news_encoder = news_encoder
        self.user_encoder = user_encoder
        self.rec_model = rec_model
        self.text_feature = text_feature

    def _forward(
        self, 
        history: Tuple[torch.Tensor], 
        candidates: Tuple[torch.Tensor],
        add_user_feats: Optional[Tuple[torch.Tensor]] = None,
        return_embeddings: bool = False
    ):
        # TODO: make this more general
        h, hm = self.news_encoder(history) # history news
        c, _ = self.news_encoder(candidates) # candidate news
        u = self.user_encoder((h, hm), add_user_feats) # user embedding through history
        r = self.rec_model(u, c)
        if return_embeddings:
            return r, u, c
        else:
            return r

    def forward(self, batch: dict, return_embeddings: bool = False):
        
        return self._forward(
            history=batch['user_features']['history'][self.text_feature],
            candidates=batch['candidate_features'][self.text_feature],
            add_user_feats=batch['user_features']['other'],
            return_embeddings=return_embeddings
        )
        
    def get_user_embeddings(self, batch: dict) -> torch.Tensor:
        """
        give user history and get user embedding
        Used for CL

        B: batch size
        H: history size
        L: sequence length
        D: embedding dimension
        """
        # TODO: add impression
        # TODO: do the combination of impression and history
        history = batch['user_features']['history'][self.text_feature]

        if isinstance(history, list) and len(history) == 2:
            history = tuple(history)

        x, m = history  # x: [B, H, L, D], m: [B, H, L, 1]
        device = next(self.parameters()).device
        x = x.to(device)
        m = m.to(device)

        B, H, L, D = x.shape
        # print(f"[DEBUG] Input shape: x={x.shape}, m={m.shape}")

        # pool to get[B, H, D]
        news_emb, news_mask = self.news_encoder((x, m))  # mask [B, H, 1]

        # encode user to get [B, D]
        user_emb = self.user_encoder((news_emb, news_mask))  # pooling + head
        # print(f"[DEBUG] User embedding shape: {user_emb.shape}")

        return user_emb.squeeze(1)