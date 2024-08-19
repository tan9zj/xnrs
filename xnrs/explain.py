import torch
from typing import Optional
from os import PathLike, makedirs
from os.path import join, exists
from pandas import DataFrame
import random
from tqdm import tqdm
from typing import Callable, List, Tuple, Optional
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel

from .models.utils import load_model_from_ckpt
from .data.mind import make_mind_data
from .data.utils import precompute_embeddings
from .utils import add_batch_dim_


class Explainer:

    def __init__(self, 
        device: str = 'cpu', 
        activation: Callable = torch.relu
    ):
        self.cfg = None
        self.model = None
        self.backbone = None
        self.tokenizer = None
        self.data = None
        self.device_str = torch.device(device)
        self.activation = activation

    def load_data(self):
        raise NotImplementedError()

    def load_checkpoint(
        self, 
        path: PathLike,
        device: Optional[str] = None,
        init_backbone: bool = False
    ):
        model, cfg = load_model_from_ckpt(path)
        self.model = model
        self.cfg = cfg
        if device is not None:
            cfg.device = device
        self.device = cfg.device
        self.model.to(torch.device(self.device))
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.backbone)
        if init_backbone:
            self.backbone = AutoModel.from_pretrained(self.cfg.backbone)
        self.cfg.add_features = ['title']
        self.cfg.load_article_ids = True

    def show_history(self, session_idx: int):
        history = self.data[session_idx]['user_features']['history']['title']
        return DataFrame({'title': history})

    def show_candidates(self, session_idx: int):
        cand = self.data[session_idx]['candidate_features']['title']
        return DataFrame({'candidate': cand})

    def score_session(self, session_idx: int, return_embeddings: bool = False):
        b = self.data[session_idx]
        add_batch_dim_(b)
        return self.score_batch(b, return_embeddings)

    def score_batch(self, 
        batch: dict, 
        return_embeddings: bool = False
        ):
        b = batch
        s, u, c = self.model(b, return_embeddings=True)
        s = self.activation(s)
        s = s.detach().cpu().squeeze()
        u = u.detach().cpu()
        c = c.detach().cpu()
        if return_embeddings:
            return b, s, u, c
        else:
            return b, s

    def explain_score_in_session(
        self,
        session_idx: int, 
        candidate_idx: int, 
        n_steps: int = 100
    ):
        b = self.data[session_idx]
        add_batch_dim_(b)
        return self.explain_score_in_batch(
            batch=b, 
            candidate_idx=candidate_idx, 
            n_steps=n_steps
            )

    def explain_score_from_str(
        self,
        candidate: str,
        history: List[str],
        n_steps: int = 100      
    ):
        print('embedding history news')
        hist_emb_backbone = precompute_embeddings(
            texts=history, 
            tokenizer=self.tokenizer,
            model=self.backbone, 
            seq_len=50, 
            relative_to_reference=True, 
            device=self.device_str
        )
        print('embedding candidate news')
        cand_emb_backbone = precompute_embeddings(
            texts=[candidate],
            tokenizer=self.tokenizer,
            model=self.backbone, 
            seq_len=50, 
            relative_to_reference=True, 
            device=self.device_str
        )
        hist_emb_backbone_stack = torch.cat([torch.FloatTensor(h[0]) for h in hist_emb_backbone]).unsqueeze(0)
        hist_mask_backbone_stack = torch.cat([torch.FloatTensor(h[1]) for h in hist_emb_backbone]).unsqueeze(0)
        cand_emb_backbone_stack = torch.FloatTensor(cand_emb_backbone[0][0]).unsqueeze(0)
        cand_mask_backbone_stack = torch.FloatTensor(cand_emb_backbone[0][1]).unsqueeze(0).unsqueeze(0)
    
        batch = {
            'user_features': {
                'history': {
                    'title_emb': (hist_emb_backbone_stack, hist_mask_backbone_stack),
                    'title': history
                }, 
                'other': {}
            }, 
            'candidate_features': {
                'title_emb': (cand_emb_backbone_stack, cand_mask_backbone_stack),
                'title': candidate
                }, 
            'targets': None
        }
        print('computing attributions')
        return self.explain_score_in_batch(batch=batch, candidate_idx=0, n_steps=n_steps)


    def explain_score_in_batch(self,
        batch: dict,
        candidate_idx: int = 0,
        n_steps: int = 100,
    ):
        b = batch
        device = torch.device(self.device_str)
        cidx = candidate_idx
        (hist_emb, hist_att) = b['user_features']['history'][self.model.text_feature]
        (hist_emb, hist_att) = (hist_emb.to(device).requires_grad_(), hist_att.to(device).requires_grad_())
        # only scoring one candidate
        cand_emb = b['candidate_features'][self.model.text_feature][0][:, cidx:cidx+1, :, :].to(device).requires_grad_()
        cand_att = b['candidate_features'][self.model.text_feature][1][:, cidx:cidx+1, :, :].to(device).requires_grad_()
        c, _ = self.model.news_encoder((cand_emb, cand_att))
        da = 1 / n_steps
        grads = []
        for a in tqdm(torch.arange(da, 1 + da, da)):
            ga = a * hist_emb
            ha, ham = self.model.news_encoder((ga, hist_att))
            ua = self.model.user_encoder.forward(inpt=(ha, ham))
            sa = self.activation(self.model.rec_model(ua, c))
            grada = torch.autograd.grad(sa, ga)
            grads.append(grada[0])
        grads = torch.cat(grads)
        int_grads = torch.sum(grads * da, dim=0)
        attr = int_grads * hist_emb.detach()
        attr = torch.sum(attr, dim=(0, 3))
        s_true = sa.item()  # last sa (for a = 1) is the true score
        s_attr = torch.sum(attr).item()
        t = b['targets'][0][cidx].int().item() if b['targets'] is not None else None
        cand = b['candidate_features']['title'][cidx]
        titles = b['user_features']['history']['title']
        attr_dict = {
            'title': titles,
            'tokens': [self.tokenizer.tokenize(t) for t in titles],
            'news_attribution': list(torch.sum(attr, dim=1).numpy())[:len(titles)],  # cut off padding
            'token_attributions': list(attr.numpy())[:len(titles)]
        }
        return attr_dict, cand, s_attr, s_true, t

    def sample_random_session(self, min_hist_len: int = 1):
        L = 0
        while L < min_hist_len:
            s_idx = random.randint(0, len(self.data)-1)
            b = self.data[s_idx]
            L = len(b['user_features']['history']['title'])
        return s_idx

    
class MindExplainer(Explainer):

    def load_data(
        self, 
        news_path: Optional[PathLike] = None, 
        user_path: Optional[PathLike] = None
    ):
        assert self.cfg is not None, 'load checkpoint first'
        if news_path is not None:
            self.cfg.test_news_data_path = news_path
        if user_path is not None:
            self.cfg.test_user_data_path = user_path
        self.cfg.train_news_data_path = None
        self.cfg.train_user_data_path = None
        _, test_ds = make_mind_data(self.cfg)
        self.data = test_ds
        

if __name__ == '__main__':

    import pandas as pd

    explainer = MindExplainer()
    explainer.load_checkpoint(path = '../../experiments/exp_base_bilin/checkpoints/ckpt_2')
    explainer.load_data(
        news_path= '../../data/MIND/MINDlarge_dev/news.pkl',
        user_path= '../../data/MIND/MINDlarge_dev/behaviors.csv'
    )

    # clicks = []
    # ids = []
    # rec_scores = []
    # for i in tqdm(range(len(explainer.data))):
    #     # clicks
    #     cands = explainer.data[i]['candidate_features']['article_id']
    #     n_pos = int(explainer.data[i]['targets'].sum())
    #     clicks += cands[:n_pos]
    #     # scores
    #     results = explainer.score_candidates(i)
    #     ids += results['id'].to_list()
    #     rec_scores += results['score'].to_list()

    # counts = pd.Series(clicks).value_counts().to_frame().rename(columns={0: 'count'})
    # scores = DataFrame({'id': ids, 'score': rec_scores})
    # mean_scores = scores.groupby('id').mean()
    # all = counts.join(mean_scores, how='outer').fillna(0)
    # all.to_pickle('../../experiments/popularity_mind/counts_and_scores.pkl')

    # device = torch.device('cuda:0')
    # explainer.model.to(device)

    # us = []
    # for i in tqdm(range(len(explainer.data))):
    # # for i in tqdm(range(10)):
    #     b = explainer.data[i]
    #     h, hm = explainer.model.news_encoder((
    #         b['user_features']['history']['title_emb'][0].unsqueeze(0).to(device),
    #         b['user_features']['history']['title_emb'][1].unsqueeze(0).to(device)
    #     ))
    #     u = explainer.model.user_encoder((h, hm), b['user_features']['other'])
    #     us.append(u.detach().cpu())

    # user_embeddings = DataFrame({'user_emb': us})
    # user_embeddings.to_pickle('../../experiments/popularity_mind/user_embeddings.pkl')

    ids = []
    scores = []
    targets = []
    emb = []
    hist_len = []
    for i in tqdm(range(len(explainer.data))):
    # for i in tqdm(range(100)):
        # scores
        results, u = explainer.score_candidates(i, return_user_emb=True)
        ids.append(results['id'].to_list())
        scores.append(results['score'].to_list())
        targets.append(results['target'].to_list())
        emb.append(u.detach().cpu().squeeze())
        hist_len.append(len(explainer.show_history(i)))

    users = DataFrame({
        'user_emb': emb,
        'articles': ids,
        'scores': scores,
        'targets': targets,
        'hist_len': hist_len
    })

    users.to_pickle('../../experiments/popularity_mind/users_w_hist_len.pkl')