import torch
import torch.nn as nn

from ..components import layers, TextEncoder


class NAML(nn.Module):

    def __init__(self, cfg, rec_model):
        super(NAML, self).__init__()
    
        title_pooler = layers.AdditiveAttention(
            in_features=cfg.d_backbone,
            hidden_features=256
        )
        self.title_encoder = TextEncoder(
            att=None,
            pooler=title_pooler,
            p_dropout=cfg.p_dropout,
            in_features=cfg.d_backbone,
            out_features=cfg.title_emb_dim
        )
        body_pooler = layers.AdditiveAttention(
            in_features=cfg.d_backbone,
            hidden_features=256
        )
        self.body_encoder = TextEncoder(
            att=None,
            pooler=body_pooler,
            p_dropout=cfg.p_dropout,
            in_features=cfg.d_backbone,
            out_features=cfg.title_emb_dim
        )
        self.cat_embedder = nn.Embedding(
            num_embeddings=cfg.n_categories + 1,  # 0 for padding
            embedding_dim=cfg.cat_emb_dim
        )
        self.cat_fc = nn.Linear(
            in_features=cfg.cat_emb_dim,
            out_features=cfg.total_emb_dim
        )
        self.subcat_embedder = nn.Embedding(
            num_embeddings=cfg.n_subcategories + 1,  # 0 for padding
            embedding_dim=cfg.sub_emb_dim
        )
        self.subcat_fc = nn.Linear(
            in_features=cfg.sub_emb_dim,
            out_features=cfg.total_emb_dim
        )
        self.feature_pooler = layers.AdditiveAttention(
            in_features=cfg.total_emb_dim,
            hidden_features=256
        )
        self.user_encoder = layers.AdditiveAttention(
            in_features=cfg.title_emb_dim,
            hidden_features=256
        )
        self.rec_model = rec_model
        self.emb_dim = cfg.total_emb_dim

    def _forward(self,
        hist_title_features: tuple, hist_abstract_features: tuple, hist_ctg, hist_subctg,
        cand_title_features: tuple, cand_abstract_features: tuple, cand_ctg, cand_subctg,
        ):
        # TODO: moving to device should not be done here
        device = next(self.parameters()).device
        # print(" Max category index (hist):", hist_ctg.max().item())
        # print(" Max subcategory index (hist):", hist_subctg.max().item())
        # print(" Max category index (cand):", cand_ctg.max().item())
        # print(" Max subcategory index (cand):", cand_subctg.max().item())


        b, nh, s, d = hist_title_features[0].shape
        nc = cand_title_features[0].shape[1]

        hist_title_emb, hist_mask = self.title_encoder(hist_title_features)
        cand_title_emb, _ = self.title_encoder(cand_title_features)

        hist_abstract_emb, _ = self.body_encoder(hist_abstract_features)
        cand_abstract_emb, _ = self.body_encoder(cand_abstract_features)

        hist_ctg_emb = self.cat_fc(self.cat_embedder(hist_ctg.to(device)))
        cand_ctg_emb = self.cat_fc(self.cat_embedder(cand_ctg.to(device)))

        hist_subctg_emb = self.subcat_fc(self.subcat_embedder(hist_subctg.to(device)))
        cand_subctg_emb = self.subcat_fc(self.subcat_embedder(cand_subctg.to(device)))

        # debug
        # print("hist_title_emb", hist_title_emb.shape)
        # print("hist_abstract_emb", hist_abstract_emb.shape)
        # print("hist_ctg_emb", hist_ctg_emb.shape)
        # print("hist_subctg_emb", hist_subctg_emb.shape)

        hist = torch.cat([hist_title_emb, hist_abstract_emb, hist_ctg_emb, hist_subctg_emb], dim=2)
        cand = torch.cat([cand_title_emb, cand_abstract_emb, cand_ctg_emb, cand_subctg_emb], dim=2)
        
        # hist = torch.stack([hist_title_emb, hist_abstract_emb, hist_ctg_emb, hist_subctg_emb], dim=2)
        # cand = torch.stack([cand_title_emb, cand_abstract_emb, cand_ctg_emb, cand_subctg_emb], dim=2)

        hist = hist.reshape((b * nh, 4, self.emb_dim))
        cand = cand.reshape((b * nc, 4, self.emb_dim))

        hist = self.feature_pooler(hist)
        cand = self.feature_pooler(cand)

        hist = hist.reshape((b, nh, self.emb_dim))
        cand = cand.reshape((b, nc, self.emb_dim))
        
        urep = self.user_encoder(hist, hist_mask)
        score = self.rec_model(urep, cand)
        
        return score
    def get_user_embeddings(self, batch: dict):
        """
        user embedding (urep)，shape [B, D]
        """
        device = next(self.parameters()).device

        # history 
        hist_title_feats = batch['user_features']['history']['title_emb']
        hist_abstract_feats = batch['user_features']['history']['abstract_emb']
        hist_ctg = batch['user_features']['history']['category_index']
        hist_subctg = batch['user_features']['history']['subcategory_index']

        # encode title & abstract
        hist_title_emb, hist_mask = self.title_encoder(hist_title_feats)
        hist_abstract_emb, _ = self.body_encoder(hist_abstract_feats)

        # category embedding 
        hist_ctg_emb = self.cat_fc(self.cat_embedder(hist_ctg.to(device)))
        hist_subctg_emb = self.subcat_fc(self.subcat_embedder(hist_subctg.to(device)))

        # cat + feature_pooler
        # concat / reshape / pooler
        b, nh, _ = hist_title_emb.shape
        emb_dim = self.emb_dim
        hist = torch.cat([
            hist_title_emb,
            hist_abstract_emb,
            hist_ctg_emb,
            hist_subctg_emb
        ], dim=2).view(b * nh, 4, emb_dim)
        hist = self.feature_pooler(hist).view(b, nh, emb_dim)

        # user encoding
        urep = self.user_encoder(hist, hist_mask)
        return urep

    def forward(self, batch: dict):
        return self._forward(
            hist_title_features=batch['user_features']['history']['title_emb'],
            hist_abstract_features=batch['user_features']['history']['abstract_emb'],
            hist_ctg=batch['user_features']['history']['category_index'],
            hist_subctg=batch['user_features']['history']['subcategory_index'],
            cand_title_features=batch['candidate_features']['title_emb'],
            cand_abstract_features=batch['candidate_features']['abstract_emb'],
            cand_ctg=batch['candidate_features']['category_index'],
            cand_subctg=batch['candidate_features']['subcategory_index'],
        )
    

class SmallNAML(nn.Module):

    def __init__(self, cfg, rec_model):
        super(SmallNAML, self).__init__()
    
        title_pooler = layers.AdditiveAttention(
            in_features=cfg.d_backbone,
            hidden_features=256
        )
        self.title_encoder = TextEncoder(
            att=None,
            pooler=title_pooler,
            p_dropout=cfg.p_dropout,
            in_features=cfg.d_backbone,
            out_features=cfg.title_emb_dim
        )

        self.cat_embedder = nn.Embedding(
            num_embeddings=cfg.n_categories + 1,  # 0 for padding
            embedding_dim=cfg.cat_emb_dim
        )
        self.cat_fc = nn.Linear(
            in_features=cfg.cat_emb_dim,
            out_features=cfg.total_emb_dim
        )

        self.feature_pooler = layers.AdditiveAttention(
            in_features=cfg.total_emb_dim,
            hidden_features=256
        )
        self.user_encoder = layers.AdditiveAttention(
            in_features=cfg.title_emb_dim,
            hidden_features=256
        )
        self.rec_model = rec_model
        self.emb_dim = cfg.total_emb_dim

    def _forward(self,
        hist_title_features: tuple, hist_ctg,
        cand_title_features: tuple, cand_ctg,
        ):
        # TODO: moving to device should not be done here
        device = next(self.parameters()).device

        b, nh, s, d = hist_title_features[0].shape
        nc = cand_title_features[0].shape[1]

        hist_title_emb, hist_mask = self.title_encoder(hist_title_features)
        cand_title_emb, _ = self.title_encoder(cand_title_features)

        hist_ctg_emb = self.cat_fc(self.cat_embedder(hist_ctg.to(device)))
        cand_ctg_emb = self.cat_fc(self.cat_embedder(cand_ctg.to(device)))

        hist = torch.stack([hist_title_emb, hist_ctg_emb], dim=2)
        cand = torch.stack([cand_title_emb, cand_ctg_emb], dim=2)

        hist = hist.reshape((b * nh, 2, self.emb_dim))
        cand = cand.reshape((b * nc, 2, self.emb_dim))

        hist = self.feature_pooler(hist)
        cand = self.feature_pooler(cand)

        hist = hist.reshape((b, nh, self.emb_dim))
        cand = cand.reshape((b, nc, self.emb_dim))
        
        urep = self.user_encoder(hist, hist_mask)
        score = self.rec_model(urep, cand)
        
        return score

    def forward(self, batch: dict):
        return self._forward(
            hist_title_features=batch['user_features']['history']['title_emb'],
            hist_ctg=batch['user_features']['history']['category_index'],
            cand_title_features=batch['candidate_features']['title_emb'],
            cand_ctg=batch['candidate_features']['category_index'],
        )