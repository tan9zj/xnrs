import torch.nn as nn
import torch
from ..components import ParentRec, TextEncoder, layers, UserEncoder


class StandardRec(ParentRec):

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
            head=True,
            p_dropout=cfg.p_dropout,
            emb_dim=cfg.title_emb_dim,
            bias=cfg.bias
        )
        super(StandardRec, self).__init__(
            news_encoder=title_encoder,
            user_encoder=user_encoder,
            rec_model=rec_model
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
    
    def get_news_embeddings(self, batch: dict, mode: str = 'candidate') -> torch.Tensor:
        """
        Get news embeddings from the news encoder.

        Args:
            batch (dict): input batch with keys 'candidate_features' or 'user_features'
            mode (str): 'candidate' for current news; 'history' for user clicked news

        Returns:
            torch.Tensor: News embeddings [B, D] or [B, N, D]
        """
        if mode == 'candidate':
            news_input = batch['candidate_features'][self.text_feature]
        elif mode == 'history':
            news_input = batch['user_features']['history'][self.text_feature]
        else:
            raise ValueError("mode must be 'candidate' or 'history'")

        if isinstance(news_input, list) and len(news_input) == 2:
            news_input = tuple(news_input)

        x, m = news_input
        device = next(self.parameters()).device
        x = x.to(device)
        m = m.to(device)

        news_emb, _ = self.news_encoder((x, m))
        return news_emb
