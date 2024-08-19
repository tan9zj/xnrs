import torch
from torch.utils.data import Dataset
from typing import Optional
import random

from ..utils import stack_scalars


class NewsRecDataset(Dataset):

    def __init__(
        self, uds, news_feat,
        mode: str, 
        n_negatives: Optional[int], 
        l_seq: int = 50, 
        l_hist: int = 25,
        text_features: list = ['title_emb'], 
        catg_features: list = [], 
        user_features: list = [], 
        add_features: list = [],
        load_article_ids: bool = False,
        loss_weights: bool = False,
        loss_weights_exponent: Optional[float] = None,
        device: torch.device = torch.device('cpu')
    ):
        super().__init__()
        self.uds = uds
        self.news_feat = news_feat
        
        self.mode = mode
        self.n_neg = n_negatives
        self.l_seq = l_seq
        self.l_hist = l_hist

        self.text_features = text_features
        self.catg_features = catg_features
        self.user_features = user_features
        self.add_features = add_features
        self.load_article_ids = load_article_ids
        self.loss_weights = loss_weights
        self.loss_weights_exp = loss_weights_exponent

        self.device = device

    def __len__(self):
        return len(self.uds)

    def __getitem__(self, idx, return_news: bool = False):
        
        session = self.uds[idx]
        history = session['history']

        if self.mode == 'train':
            # one pos and k neg for training
            positives = [random.choice(session['positives'])]
            negatives = random.choices(session['negatives'], k=self.n_neg)
        elif self.mode == 'eval':
            # all pos and neg for eval
            positives = session['positives']
            negatives = session['negatives']
        
        hist_features = [self.news_feat[n] for n in history]
        pos_features = [self.news_feat[n] for n in positives]
        neg_features = [self.news_feat[n] for n in negatives]

        return_dict = {
            'user_features': {
                'history': {}, 'other': {}
            }, 
            'candidate_features': {}, 
            'targets': {}
        }
        
        for feat in self.text_features:
        
            hist = [n[feat] for n in hist_features[-self.l_hist:]]
            hist_emb = torch.cat([torch.FloatTensor(e[0]) for e in hist])
            hist_mask = torch.cat([torch.FloatTensor(e[1]) for e in hist]).unsqueeze(-1)
            # padding history to constant number of news
            d_backbone = hist_emb.shape[-1]
            hist_emb_pad = torch.zeros((self.l_hist - len(hist), self.l_seq, d_backbone))
            hist_emb = torch.cat([hist_emb, hist_emb_pad])
            hist_mask_pad = torch.zeros((self.l_hist - len(hist), self.l_seq, 1))
            hist_mask = torch.cat([hist_mask, hist_mask_pad])

            # weights
            neg_corr = self.n_neg or 1
            if self.loss_weights:
                pos_counts = torch.tensor([n['clicks'] for n in pos_features])
                pos_weights = (1 / pos_counts) ** self.loss_weights_exp
                neg_weight = torch.mean(pos_weights) * neg_corr
                neg_weights = torch.full((len(negatives),), neg_weight)
                weights = torch.cat([pos_weights, neg_weights])
                return_dict['weights'] = weights.unsqueeze(-1)

            pos = [n[feat] for n in pos_features]
            pos_emb = torch.cat([torch.FloatTensor(e[0]) for e in pos])
            pos_mask = torch.cat([torch.FloatTensor(e[1]) for e in pos]).unsqueeze(-1)

            neg = [n[feat] for n in neg_features]
            neg_emb = torch.cat([torch.FloatTensor(e[0]) for e in neg])
            neg_mask = torch.cat([torch.FloatTensor(e[1]) for e in neg]).unsqueeze(-1)

            cand_emb = torch.cat([pos_emb, neg_emb])
            cand_mask = torch.cat([pos_mask, neg_mask])

            return_dict['user_features']['history'][feat] = (hist_emb, hist_mask)
            return_dict['candidate_features'][feat] = (cand_emb, cand_mask)

        for feat in self.catg_features:
            
            # ctg features need to be 1d numpy arrays
            # TODO: change this to simple ints as for user indices?
            hist_feat = [n[feat] for n in hist_features]
            hist_feat = stack_scalars(hist_feat, N=self.l_hist, pad_label=0)

            pos_feat = [n[feat] for n in pos_features]
            neg_feat = [n[feat] for n in neg_features]
            cand_feat = pos_feat + neg_feat

            return_dict['user_features']['history'][feat] = torch.IntTensor(hist_feat)
            return_dict['candidate_features'][feat] = torch.IntTensor(cand_feat)

        for feat in self.add_features:  # not loaded to self.device
            hist_feat = [n[feat] for n in hist_features][-self.l_hist:]
            pos_feat = [n[feat] for n in pos_features]
            neg_feat = [n[feat] for n in neg_features]
            cand_feat = pos_feat + neg_feat
            return_dict['user_features']['history'][feat] = hist_feat
            return_dict['candidate_features'][feat] = cand_feat
        
        if self.load_article_ids:
            return_dict['user_features']['history']['article_id'] = history[-self.l_hist:]
            return_dict['candidate_features']['article_id'] = positives + negatives

        # user features must all be categorical, i.e. int (converted from str)

        user_dict = {
            feat: torch.IntTensor([int(session[feat])]) \
                for feat in self.user_features
        }
        if user_dict:
            return_dict['user_features']['other'] = user_dict
            
        if self.mode == 'train':
            targets = torch.FloatTensor([1] + [0] * self.n_neg).unsqueeze(1)
        elif self.mode == 'eval':
            targets = torch.FloatTensor([1] * len(positives) + [0] * len(negatives)).unsqueeze(1)

        return_dict['targets'] = targets

        if return_news:  # for additional features
            return return_dict, hist_features, pos_features, neg_features
        else:
            return return_dict