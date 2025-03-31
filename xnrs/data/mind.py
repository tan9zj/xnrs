import torch
from os import PathLike
from typing import Optional, List
import pandas as pd
from pandas import DataFrame
import datasets

from .utils import collect_features, precompute_embeddings
from .dataset import NewsRecDataset

#TODO: change tensor dimensions, especially masks

def make_mind_data(cfg):

    features = collect_features(cfg)

    if cfg.train_news_data_path is not None:
        train_news = MindHandler.load_news_data_as_dict(
            path=cfg.train_news_data_path,
            columns=features
        )
    else:
        train_news = None
    if cfg.train_user_data_path is not None:
        train_behaviors = MindHandler.load_behaviors_as_hf_dataset(
            cfg.train_user_data_path, 
            split='train',
            num_proc=1
        )
    else:
        train_behaviors = None
    if cfg.test_news_data_path is not None:
        test_news = MindHandler.load_news_data_as_dict(
            path=cfg.test_news_data_path,
            columns=features
        )
    else:
        test_news = None
    if cfg.test_user_data_path is not None:
        test_behaviors = MindHandler.load_behaviors_as_hf_dataset(
            cfg.test_user_data_path, 
            split='train',  # weird api -> can only be train...
            num_proc=1
        )
    else:
        test_behaviors = None
    if train_news is not None and train_behaviors is not None:
        train_ds = NewsRecDataset(
            uds=train_behaviors, news_feat=train_news, 
            n_negatives=cfg.n_negatives,
            mode='train',
            l_seq=cfg.seq_len,
            l_hist=cfg.hist_len,
            text_features=cfg.text_features,
            catg_features=cfg.catg_features,
            user_features=cfg.user_features,
            add_features=cfg.add_features,
            load_article_ids=cfg.load_article_ids,
            device=torch.device(cfg.device)
        )
    else:
        train_ds = None
    if test_news is not None and test_behaviors is not None:
        test_ds = NewsRecDataset(
            uds=test_behaviors, news_feat=test_news, 
            n_negatives=None,
            mode='eval',
            l_seq=cfg.seq_len,
            l_hist=cfg.hist_len,
            text_features=cfg.text_features,
            catg_features=cfg.catg_features,
            user_features=cfg.user_features,
            add_features = cfg.add_features,
            load_article_ids=cfg.load_article_ids,
            device=torch.device(cfg.device)
        )
    else:
        test_ds = None

    return train_ds, test_ds


class MindHandler:

    @staticmethod
    def read_behaviors_as_df(
        path: PathLike, 
        sep: str = '\t', 
        drop_cold_start: bool = True,    
        **kwargs
    ):

        def impression_converter(impr: str):
            return [
                (n[:-2], int(n[-1])) for n in impr.split()
            ]

        def filter_impressions(impressions: list, label: int):
            return [
                i[0] for i in impressions if i[1] == label
            ]

        user_data = pd.read_csv(
            path,
            sep=sep,
            header=None,
            index_col=False,
            names=['index', 'user', 'time', 'history', 'impression'],
            usecols=['user', 'time', 'history', 'impression'],
            converters={
                'history': lambda h: h.split(),
                'impression' : impression_converter
            },
            **kwargs
        )
        if drop_cold_start:
            user_data['cold_start'] = user_data.history.map(lambda h: not bool(h))
            user_data = user_data[user_data.cold_start == False]
            user_data = user_data.drop('cold_start', axis=1)
        user_data['positives'] = user_data['impression'].apply(filter_impressions, label=1)
        user_data['negatives'] = user_data['impression'].apply(filter_impressions, label=0)
        return user_data

    @staticmethod
    def read_news_as_df(path: PathLike, usecols=['id', 'category', 'subcategory', 'title', 'abstract'], **kwargs):
        news_data = pd.read_csv(
            path,
            sep='\t',
            header=None,
            index_col=0,
            names=['id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'],
            usecols=usecols,
            **kwargs
        )
        return news_data

    @staticmethod
    def precompute_embeddings(news_data: DataFrame, src_feat, dst_feat, tokenizer, model, 
        device: torch.device = torch.device('cuda:0'), reduction: str = 'none', relative_to_reference: bool = True,
        seq_len: int = 50):
        assert src_feat in news_data.columns
        news_data[src_feat].fillna('', inplace=True)
        assert not news_data[src_feat].isna().any()
        feat_list = list(news_data[src_feat])
        emb = precompute_embeddings(
            texts=feat_list, tokenizer=tokenizer, model=model, 
            device=device, reduction=reduction, seq_len=seq_len,
            relative_to_reference=relative_to_reference
        )
        news_data[dst_feat] = emb
        return news_data

    @staticmethod
    def pre_tokenize(news_data: DataFrame, src_feat, tokenizer):
        texts = news_data[src_feat].to_list()
        tokenized = tokenizer(texts, truncation=True, padding=True)
        news_data[src_feat + '_tokens'] = tokenized['input_ids']
        news_data[src_feat + '_mask'] = tokenized['attention_mask']
        return news_data

    @staticmethod
    def load_news_data_as_dict(path, columns=['title_emb', 'clicks']):
        df = pd.read_pickle(path)
        return df[columns].to_dict('index')

# # change how to read zip data
#     @staticmethod
#     def load_news_data_as_dict(path, columns=['title_emb', 'clicks']):
#         import gzip, pickle
#         with gzip.open(path, 'rb') as f:
#             df = pickle.load(f)
#         return df[columns].to_dict('index')

    @staticmethod
    def read_behaviours_tsv(
        src_path: PathLike,
        columns: List = ['user', 'time', 'history', 'impression']
    ):
        assert src_path.endswith('.tsv')
        df = pd.read_csv(src_path, sep='\t', names=columns)
        return df

    @staticmethod
    def load_behaviors_as_hf_dataset(path, split, remove_columns=['time', 'user', 'impression'], num_proc: int = 1):
        ds = datasets.load_dataset('csv', data_files=path, split=split)
        # removing empty histories
        ds = ds.filter(lambda row: row['history'] is not None, num_proc=num_proc)
        # splitting hsitory and impressions
        ds = ds.map(lambda inst: {'history': inst['history'].split()}, num_proc=num_proc)
        ds = ds.map(lambda row: {
            'positives': [id[:-2] for id in row['impression'].split() if id[-1]=='1'],
            'negatives': [id[:-2] for id in row['impression'].split() if id[-1]=='0']
        }, num_proc=num_proc)
        ds = ds.remove_columns(remove_columns)
        return ds
        
    @staticmethod
    def compute_clicks(news: DataFrame, behaviors: DataFrame):
        clicks = behaviors.positives.to_list()
        all_clicks = []
        for l in clicks:
            all_clicks += l
        counts = pd.Series(all_clicks).value_counts()
        news['clicks'] = news.index.map(
            lambda id: counts[id] if id in counts.index else 0
        )    
        return news