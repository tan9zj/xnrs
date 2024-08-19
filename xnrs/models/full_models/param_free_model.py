import torch.nn as nn

from ..components import ParentRec, TextEncoder, layers, UserEncoder


class ParamFreeRec(ParentRec):

    def __init__(self, cfg, rec_model: nn.Module):
        assert cfg.title_emb_dim == cfg.d_backbone
        title_pooler = layers.MaskedMean()
        title_encoder = TextEncoder(
            att=None,
            head=False,
            pooler=title_pooler,
            p_dropout=cfg.p_dropout,
            out_features=cfg.d_backbone
        )
        hist_pooler = layers.MaskedMean()
        user_encoder = UserEncoder(
            att=None,
            head=False,
            pooler=hist_pooler,
            p_dropout=cfg.p_dropout,
            emb_dim=cfg.title_emb_dim
        )
        super(ParamFreeRec, self).__init__(
            news_encoder=title_encoder,
            user_encoder=user_encoder,
            rec_model=rec_model
        )