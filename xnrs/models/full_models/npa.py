import torch
import torch.nn as nn

from ..components import layers
from ...utils import collaps_mask


class NPA(nn.Module):

    def __init__(self, cfg, rec_model):
        super(NPA, self).__init__()
        self.user_embedder = nn.Embedding(
            num_embeddings=cfg.n_users + 1,  # 0 for padding
            embedding_dim=cfg.user_emb_dim
        )
        self.title_pooler = layers.PersonalizedAttention(
            in_features=cfg.d_backbone,
            hidden_features=128,
            query_features=cfg.user_emb_dim
        )
        self.dropout = nn.Dropout(p=cfg.p_dropout)
        self.news_head = nn.Sequential(
            nn.Linear(cfg.d_backbone, cfg.title_emb_dim),
            nn.ReLU(),
            nn.Linear(cfg.title_emb_dim, cfg.title_emb_dim)  # deactivate bias?
        )
        self.user_encoder = layers.PersonalizedAttention(
            in_features=cfg.title_emb_dim,
            hidden_features=128,
            query_features=cfg.user_emb_dim
        )
        self.rec_model = rec_model

    def _forward(self, 
        hist_title_features: tuple,
        cand_title_features: tuple, 
        uid: torch.Tensor):
        '''
        Args:
            hist_title_features (tuple): ( 
                h: history news embeddings, shape (B, N, S, D)
                hm: history attention masks, shape (B, N, S)
            )
            cand_title_features (tuple): (
                c: candidate news embeddings, shape (B, N, S, D)
                cm: candidate attention masks, shape (B, N, S, D)
            )
            uid: user ids of every instance, shape (B, 1)
        '''
        h, hm = hist_title_features
        c, cm = cand_title_features

        # TODO: moving to device should not be done here
        device = next(self.parameters()).device

        h, hm = h.to(device), hm.to(device)
        c, cm = c.to(device), cm.to(device)
        uid = uid.to(device)
        
        user_emb = self.user_embedder(uid)  # (B, du)

        # user encoding
        b, nh, s, d = h.shape
        h = h.reshape((b * nh, s, d))
        hm_title = hm.reshape((b * nh, s, 1))
        h = self.dropout(h)
        hist_user_emb = user_emb.repeat_interleave(nh, dim=0)  # (b * nh, du)
        h = self.title_pooler(q=hist_user_emb, x=h, m=hm_title)  # (b * nh, d)
        h = self.news_head(h)  # (b * nh, 1, d_emb)
        emb_dim = h.shape[2]
        h = h.reshape((b, nh, emb_dim))  # (b, nh, 1, d_emb)
        hm = collaps_mask(hm, dim=2)  # (b, nh)
        u = self.user_encoder(q=user_emb, x=h, m=hm)  # (b, 1, e_emb)

        # candidates encoding
        b, nc, s, d = c.shape
        c = c.reshape((b * nc, s, d))
        cm = cm.reshape((b * nc, s, 1))
        c = self.dropout(c)
        cand_user_emb = user_emb.repeat_interleave(nc, dim=0)
        c = self.title_pooler(q=cand_user_emb, x=c, m=cm)
        c = self.news_head(c)
        emb_dim = c.shape[2]
        c = c.reshape((b, nc, emb_dim))

        # scores
        r = self.rec_model(u, c)

        return r

    def forward(self, batch: dict):
        return self._forward(
            hist_title_features=batch['user_features']['history']['title_emb'],
            cand_title_features=batch['candidate_features']['title_emb'],
            uid=batch['user_features']['other']['user_index']
        )