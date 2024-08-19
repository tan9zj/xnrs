import torch
import torch.nn as nn
from typing import Optional, Tuple
import traceback

from ..components import layers, news_encoding, user_encoding


class LSTUR(nn.Module):

    def __init__(self, cfg, rec_model):
        super().__init__()
        self.news_encoder = LSTURNewsEncoder(cfg)
        self.user_encoder = LSTURUserEncoder(cfg)
        self.rec_model = rec_model
        self.cfg = cfg

    def _forward(self,
        user_ids: torch.tensor,
        hist_title_features: Tuple[torch.tensor],
        cand_title_features: Tuple[torch.tensor],
        hist_cat_idxs: torch.tensor,
        cand_cat_idxs: torch.tensor,
        hist_subcat_idxs: Optional[torch.tensor] = None,
        cand_subcat_idxs: Optional[torch.tensor] = None,
        return_embeddings: bool = False
    ):
        h, hm = self.news_encoder(
            title_features=hist_title_features,
            cat_idxs=hist_cat_idxs,
            subcat_idxs=hist_subcat_idxs
        )
        c, _ = self.news_encoder(
            title_features=cand_title_features,
            cat_idxs=cand_cat_idxs,
            subcat_idxs=cand_subcat_idxs
        )

        u = self.user_encoder((h, hm), user_ids)
        r = self.rec_model(u, c)

        if return_embeddings:
            return r, u, c
        else:
            return r
        
    def forward(self, batch:dict, return_embeddings: bool = False):
        if 'subcategory_index' in self.cfg.catg_features:
            hist_subcat_idxs=batch['user_features']['history']['subcategory_index']
            cand_subcat_idxs=batch['candidate_features']['subcategory_index']
        else:
            hist_subcat_idxs=None
            cand_subcat_idxs=None
        return self._forward(
            user_ids=batch['user_features']['other']['user_index'],
            hist_title_features=batch['user_features']['history']['title_emb'],
            cand_title_features=batch['candidate_features']['title_emb'],
            hist_cat_idxs=batch['user_features']['history']['category_index'],
            cand_cat_idxs=batch['candidate_features']['category_index'],
            hist_subcat_idxs=hist_subcat_idxs,
            cand_subcat_idxs=cand_subcat_idxs,
            return_embeddings=return_embeddings
        )


class LSTURUserEncoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg 

        long_term_emb_dim = cfg.total_emb_dim
        if cfg.long_short_term_method == 'con':
            long_term_emb_dim //= 2

        if self.cfg.long_term_method == 'embedding':
            self.long_term_encoder = nn.Embedding(
                num_embeddings=cfg.n_users + 1,
                embedding_dim=long_term_emb_dim,
                padding_idx=0,
            )
        elif self.cfg.long_term_method == 'mean':
            hist_pooler = layers.MaskedMean()
            self.long_term_encoder = user_encoding.UserEncoder(
                pooler=hist_pooler,
                att=None,
                head=True,
                emb_dim=self.cfg.total_emb_dim,
                out_dim=long_term_emb_dim,
                p_dropout=cfg.p_dropout,
                bias=cfg.bias
            )
        else:
            raise ValueError(f'long_term_method must be in [mean, embedding], got {self.cfg.long_term_method}')
        self.dropout = nn.Dropout(p=cfg.p_user_dropout)
        self.gru = nn.GRU(
            cfg.total_emb_dim, long_term_emb_dim,
            batch_first=True
        )

    def forward(
        self,
        history_features: Tuple[torch.tensor],
        user_ids: torch.tensor
    ):

        if self.cfg.long_term_method == 'mean':
            u_lt = self.long_term_encoder(history_features)
            u_lt = u_lt.squeeze(1)
        elif self.cfg.long_term_method == 'embedding':
            u_lt = self.long_term_encoder(user_ids).squeeze(1)
        u_lt = self.dropout(u_lt)

        h, hm = history_features
        # TODO: What is the order of reading histories?
        h_st = h[:, :self.cfg.st_hist_len]
        hm_st = hm[:, :self.cfg.st_hist_len]
        l = hm_st.sum(dim=1).cpu().squeeze(1)
        u_st = nn.utils.rnn.pack_padded_sequence(
            h_st, lengths=l, 
            batch_first=True, enforce_sorted=False
        )

        if self.cfg.long_short_term_method == "ini":
            # first dim of gru initial hidden state is the number of gru layers
            _, u = self.gru(u_st, u_lt.unsqueeze(0))
            return u.squeeze(dim=0).unsqueeze(1)
        elif self.cfg.long_short_term_method == "con":
            _, u_st = self.gru(u_st)
            u = torch.cat((u_st.squeeze(dim=0), u_lt), dim=1)
            return u.unsqueeze(1)
        elif self.cfg.long_short_term_method == "lt_only":
            u = u_lt.unsqueeze(1)
            return u
        else:
            raise ValueError(f'invalid value for long_short_term_method, got {self.cfg.long_short_term_method}')
        

class LSTURNewsEncoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        title_pooler = layers.AdditiveAttention(
            in_features=cfg.d_backbone,
            hidden_features=cfg.title_emb_dim
        )
        self.title_encoder = news_encoding.TextEncoder(
            pooler=title_pooler,
            p_dropout=cfg.p_dropout,
            out_features=cfg.title_emb_dim,
            in_features=cfg.d_backbone,
            head=True,
            bias=cfg.bias
        )
        self.cat_embedder = nn.Embedding(
            num_embeddings=cfg.n_categories + 1,
            embedding_dim=cfg.cat_emb_dim
        )

        if 'subcategory_index' in cfg.catg_features:
            self.subcat_embedder = nn.Embedding(
                num_embeddings=cfg.n_subcategories + 1,
                embedding_dim=cfg.cat_emb_dim
            )

    def forward(
        self,
        title_features: Tuple[torch.tensor],
        cat_idxs: torch.tensor,
        subcat_idxs: Optional[torch.tensor]
    ):
        title_emb, m = self.title_encoder(title_features)
        cat_emb = self.cat_embedder(cat_idxs)
        emb = torch.cat([title_emb, cat_emb], dim=2)
        if subcat_idxs is not None:
            assert hasattr(self, 'subcat_embedder')
            subcat_emb = self.subcat_embedder(subcat_idxs)
            emb = torch.cat([emb, subcat_emb], dim=2)
        return emb, m
    

if __name__ == '__main__':

    import yaml
    from dotmap import DotMap
    from ..components.scoring import DotRec

    cfg_dict = yaml.full_load(open('config/mind_LSTUR.yml'))
    cfg = DotMap(cfg_dict)

    rec = DotRec()
    model = LSTUR(cfg, rec)
