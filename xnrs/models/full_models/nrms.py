import torch
import torch.nn as nn

from ..components import layers, news_encoding, parent

from ..components import user_encoding


class NRMS(parent.ParentRec):

    def __init__(self, cfg, rec_model: nn.Module):
        title_att = layers.MultiHeadAttention(
            n_heads=cfg.n_heads,
            d_model=cfg.d_backbone
        )
        title_pooler = layers.AdditiveAttention(
            in_features=cfg.d_backbone,
            hidden_features=256
        )
        title_encoder = news_encoding.TextEncoder(
            att=title_att,
            pooler=title_pooler,
            p_dropout=cfg.p_dropout,
            in_features=cfg.d_backbone,
            out_features=cfg.title_emb_dim
        )
        # ablation
        user_att = layers.MultiHeadAttention(
            n_heads=cfg.n_heads,
            d_model=cfg.title_emb_dim
        )
        user_pooler = layers.AdditiveAttention(
            in_features=cfg.title_emb_dim,
            hidden_features=256
        )
        user_encoder = user_encoding.UserEncoder(
            att=user_att,
            pooler=user_pooler,
            emb_dim=cfg.title_emb_dim,
            p_dropout=cfg.p_dropout,
            head=False
        )
        super(NRMS, self).__init__(
            news_encoder=title_encoder,
            user_encoder=user_encoder,
            rec_model=rec_model
        )

class NRMS_LF(parent.ParentRec):
    '''NRSMS model with mean pooling user encoder ("late fusion")'''

    def __init__(self, cfg, rec_model: nn.Module):
        title_att = layers.MultiHeadAttention(
            n_heads=cfg.n_heads,
            d_model=cfg.d_backbone
        )
        title_pooler = layers.AdditiveAttention(
            in_features=cfg.d_backbone,
            hidden_features=256
        )
        title_encoder = news_encoding.TextEncoder(
            att=title_att,
            pooler=title_pooler,
            p_dropout=cfg.p_dropout,
            in_features=cfg.d_backbone,
            out_features=cfg.title_emb_dim
        )
        user_pooler = layers.MaskedMean()
        user_encoder = user_encoding.UserEncoder(
            att=None,
            pooler=user_pooler,
            emb_dim=cfg.title_emb_dim,
            p_dropout=cfg.p_dropout,
            head=False
        )
        super(NRMS_LF, self).__init__(
            news_encoder=title_encoder,
            user_encoder=user_encoder,
            rec_model=rec_model
        )