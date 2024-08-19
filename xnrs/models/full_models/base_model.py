import torch.nn as nn

from ..components import ParentRec, TextEncoder, layers, UserEncoder


class BaseRec(ParentRec):

    def __init__(self, cfg, rec_model: nn.Module):
        title_pooler = layers.AdditiveAttention(
            in_features=cfg.d_backbone,
            hidden_features=256
        )
        title_encoder = TextEncoder(
            att=None,
            pooler=title_pooler,
            p_dropout=cfg.p_dropout,
            in_features=cfg.d_backbone,
            out_features=cfg.title_emb_dim,
            bias=cfg.bias
        )
        hist_pooler = layers.AdditiveAttention(
            in_features=cfg.title_emb_dim,
            hidden_features=256
        )
        user_encoder = UserEncoder(
            pooler=hist_pooler,
            att=None,
            head=False,
            p_dropout=cfg.p_dropout,
            emb_dim=cfg.title_emb_dim
        )
        super(BaseRec, self).__init__(
            news_encoder=title_encoder,
            user_encoder=user_encoder,
            rec_model=rec_model
        )