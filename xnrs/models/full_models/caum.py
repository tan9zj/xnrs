# Adapted from https://github.com/taoqi98/CAUM/blob/main/Code/Models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from ..components.layers import AdditiveAttention, MultiHeadAttention, DenseAttention
from ..components.news_encoding import TextEncoder, CategoryEncoder


class CAUM(nn.Module):

    def __init__(self, cfg, rec_model: nn.Module):
        super().__init__()
        self.cfg = cfg
        self.news_encoder = CAUMNewsEncoder(cfg)
        self.user_encoder = CAUMUserEncoder(cfg)
        self.rec_model = rec_model
        
    def forward(self, batch: dict, return_embeddings: bool = False):
        h, hm = self.news_encoder(batch['user_features']['history'])
        c, cm = self.news_encoder(batch['candidate_features'])
        u = self.user_encoder((h, hm), (c, cm))
        r = self.rec_model(u, c)
        if return_embeddings:
            return r, u, c
        else:
            return r
    

class CAUMUserEncoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # initialize
        self.dropout1 = nn.Dropout(p=cfg.p_dropout)
        self.dropout2 = nn.Dropout(p=cfg.p_dropout)
        self.dropout3 = nn.Dropout(p=cfg.p_dropout)

        self.linear1 = nn.Linear(in_features=cfg.total_emb_dim * 4, out_features=cfg.total_emb_dim)
        self.linear2 = nn.Linear(in_features=cfg.total_emb_dim * 2, out_features=cfg.total_emb_dim)
        self.linear3 = nn.Linear(
            in_features=cfg.total_emb_dim + cfg.total_emb_dim, out_features=cfg.total_emb_dim
        )

        self.dense_att = DenseAttention(
            input_dim=cfg.total_emb_dim * 2,
            hidden_dim1=cfg.total_emb_dim,
            hidden_dim2=cfg.total_emb_dim // 2,
        )
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=cfg.total_emb_dim, num_heads=cfg.n_heads
        )

    def forward(
        self, 
        history_features: Tuple[torch.Tensor], 
        cand_features: Tuple[torch.Tensor]
    ):
        h, hm = history_features
        c, cm = cand_features

        c = self.dropout1(c)
        h = self.dropout2(h)

        # training with multiple candidates per history
        n_c = c.shape[1]  # number of candidates
        n_h = h.shape[1]  # number of users / histories per batch
        c_repeat = c.unsqueeze(dim=2).repeat(1, 1, n_h, 1)
        h_repeat = h.unsqueeze(dim=1).repeat(1, n_c, 1, 1)

        # dims become b x n_c x n_h x d
        # candi-cnn
        h_left = torch.cat(
            [h_repeat[:, :, -1:, :], h_repeat[:, :, :-1, :]], dim=2
        )
        h_right = torch.cat(
            [h_repeat[:, :, 1:, :], h_repeat[:, :, :1, :]], dim=2
        )
        h_cnn = torch.cat(
            [h_left, h_repeat, h_right, c_repeat], dim=-1
        )

        h_cnn = self.linear1(h_cnn)

        # candi-selfatt
        h_selfatt = torch.cat([c_repeat, h_repeat], dim=-1)
        h_selfatt = self.linear2(h_selfatt)
        b, n_c, n_h, d = h_selfatt.shape
        h_selfatt = h_selfatt.view((b * n_c, n_h, d))
        h_selfatt, _ = self.multihead_attention(h_selfatt, h_selfatt, h_selfatt)
        h_selfatt = h_selfatt.view((b, n_c, n_h, d))

        h_all = torch.cat([h_cnn, h_selfatt], dim=-1)
        h_all = self.dropout3(h_all)
        h_all = self.linear3(h_all)

        # candi-att
        hc = torch.cat([h_all, c_repeat], dim=-1)
        a = self.dense_att(hc)
        # a = a.squeeze(dim=-1)
        a = a.transpose(-1, -2)
        a = F.softmax(a, dim=-1)

        h_all = h_all.view((b * n_c, n_h, d))
        a = a.view(b * n_c, 1, n_h)
        u = torch.bmm(a, h_all).squeeze(1)
        u = u.view(b, n_c, d)
        
        return u
    

class CAUMNewsEncoder(nn.Module):
    '''without entity embedding module'''

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        title_pooler = AdditiveAttention(
            in_features=cfg.d_backbone,
            hidden_features=cfg.title_emb_dim
        )
        title_att = MultiHeadAttention(
            n_heads=cfg.n_heads,
            d_model=cfg.d_backbone
        )
        self.title_encoder = TextEncoder(
            pooler=title_pooler,
            att=title_att,
            p_dropout=cfg.p_dropout,
            out_features=cfg.title_emb_dim,
            in_features=cfg.d_backbone,
            head=True,
            bias=cfg.bias
        )
        self.cat_embedder = CategoryEncoder(
            n_categories=cfg.n_categories,
            embedding_dim=cfg.cat_emb_dim
        )
        if 'subcategory_index' in cfg.catg_features:
            self.subcat_embedder = CategoryEncoder(
                n_categories=cfg.n_subcategories,
                embedding_dim=cfg.cat_emb_dim
            )

    def _forward(
        self,
        title_emb: Tuple[torch.tensor],
        cat_idxs: torch.tensor,
        subcat_idxs: Optional[torch.tensor]
    ):
        title_emb, m = self.title_encoder(title_emb)
        cat_emb = self.cat_embedder(cat_idxs)
        emb = torch.cat([title_emb, cat_emb], dim=2)
        if subcat_idxs is not None:
            assert hasattr(self, 'subcat_embedder')
            subcat_emb = self.subcat_embedder(subcat_idxs)
            emb = torch.cat([emb, subcat_emb], dim=2)
        return emb, m
    
    def forward(self, news_features: dict):
        if 'subcategory_index' in news_features.keys():
            subcat_idxs = news_features['subcategory_index']
        else:
            subcat_idxs = None
        return self._forward(
            title_emb=news_features['title_emb'],
            cat_idxs=news_features['category_index'],
            subcat_idxs=subcat_idxs
        )

if __name__ == '__main__':

    import yaml
    from dotmap import DotMap
    from ..components.scoring import CAUMRec

    cfg_dict = yaml.full_load(open('config/adressa_CAUM.yml'))
    cfg = DotMap(cfg_dict)

    rec = CAUMRec()
    model = CAUM(cfg, rec)

    b, s, d = 3, 50, 768
    nh, nc = 25, 5
    batch = {
        'user_features': {'history': {
            'title_emb': (torch.randn((b, nh, s, d)), torch.ones((b, nh, s, 1))),
            'category_index': torch.randint(0, cfg.n_categories + 1, (b, nh))
            # 'subcategory_index': torch.randint(0, cfg.n_subcategories + 1, (b, nh))
        }},
        'candidate_features': {
            'title_emb': (torch.randn((b, nc, s, d)), torch.ones((b, nc, s, 1))),
            'category_index': torch.randint(0, cfg.n_categories + 1, (b, nc))
            # 'subcategory_index': torch.randint(0, cfg.n_subcatgories + 1, (b, nc))
        }
    }

    model(batch)