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