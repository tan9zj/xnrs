import pandas as pd
import datasets
from os import PathLike, makedirs
from os.path import join
from pathlib import Path
from typing import Optional
import json
from tqdm import tqdm
import random
import pickle
from transformers import AutoModel, AutoTokenizer

from .dataset import NewsRecDataset
from .utils import collect_features
from .utils import precompute_embeddings


class AdressaHandler:

    @staticmethod
    def load_behaviors_as_hf_dataset(path):
        ds = datasets.load_dataset('csv', data_files=path, split='train')
        ds = ds.map(lambda r: {
            'history': eval(r['history']),
            'positives': eval(r['positives']),
            'negatives': eval(r['negatives'])
        })
        return ds
    
    @staticmethod
    def load_news_as_dict(path, columns: list = ['title_emb']):
        news = pd.read_pickle(path)
        news = news[columns].to_dict('index')
        return news    

    @staticmethod
    def init_datasets(cfg):

        features = collect_features(cfg)

        news = AdressaHandler.load_news_as_dict(
            path=cfg.news_data_path, 
            columns=features
        )

        if cfg.train_user_data_path is not None:
            train_behaviors = AdressaHandler.load_behaviors_as_hf_dataset(
                path=cfg.train_user_data_path
            )
            train_ds = NewsRecDataset(
                uds=train_behaviors,
                news_feat=news,
                mode='train',
                n_negatives=cfg.n_negatives,
                text_features=cfg.text_features,
                catg_features=cfg.catg_features,
                user_features=cfg.user_features,
                add_features=cfg.add_features
            )
        else:
            train_ds = None

        if cfg.test_user_data_path is not None:
            test_behaviors = AdressaHandler.load_behaviors_as_hf_dataset(
                path=cfg.test_user_data_path
            )
            test_ds = NewsRecDataset(
                uds=test_behaviors,
                news_feat=news,
                mode='eval',
                n_negatives=None,
                text_features=cfg.text_features,
                catg_features=cfg.catg_features,
                user_features=cfg.user_features,
                add_features=cfg.add_features
            )
        else:
            test_ds = None

        return train_ds, test_ds
    
    @staticmethod
    def extract_data_for_day(dir: PathLike, day: str, dst_dir: Optional[PathLike] = None):
        users = {}
        news = {}
        with open(join(dir, day), 'r') as f:
            for line in f:
                event = json.loads(line.strip('\n'))
                if 'id' in event and 'title' in event:
                    nid = event['id']
                    t = event['title']
                    if 'category1' in event.keys():
                        c = event['category1']
                    else:
                        c = None
                    if nid not in news:
                        news[nid] = {'title': t, 'category': c}
                    uid = event['userId']
                    # time = event['time']
                    if uid not in users:
                        users[uid] = []
                    # users[uid].append((nid, time))
                    users[uid].append(nid)
        if dst_dir is not None:
            with open(join(dst_dir, f'clicks_{day}.pkl'), 'wb') as f:
                pickle.dump(users, f)
            with open(join(dst_dir, f'news_{day}.pkl'), 'wb') as f:
                pickle.dump(news, f)
        return users, news
    
    @staticmethod
    def extract_days(dir: PathLike, days: list[str], dst_dir: PathLike):
        path = Path(dst_dir)
        if not path.exists():
            path.mkdir(parents=True)
        for d in tqdm(days):
            AdressaHandler.extract_data_for_day(dir, day=d, dst_dir=dst_dir)
    
    @staticmethod
    def load_extracted_data(day: str, root_dir: PathLike, file_prefix: str):
        data = pickle.load(open(join(root_dir, f'{file_prefix}{day}.pkl'), 'rb'))
        return data

    @staticmethod
    def load_click_data(days: list[str], root_dir: PathLike, file_prefix: str = 'clicks_'):
        histories = {}
        for d in tqdm(days):
            clicks = AdressaHandler.load_extracted_data(d, root_dir, file_prefix)
            for u, c in clicks.items():
                if u in histories.keys():
                    histories[u] += c
                else:
                    histories[u] = c
        return histories
    
    @staticmethod
    def load_news(days: list[str], root_dir: PathLike, file_prefix: str = 'news_', return_df: bool = False):
        all_news = {}
        for d in tqdm(days):
            n = AdressaHandler.load_extracted_data(d, root_dir, file_prefix)
            all_news.update(n)
        if return_df:
            all_news = pd.DataFrame.from_dict(all_news, orient='index')
        return all_news
    
    @staticmethod
    def make_dataset(
        history_clicks: dict, 
        candidate_clicks: dict, 
        candidate_news: dict,
        k_negatives: int = 20,
        dst_path: Optional[PathLike] = None
        ):
        users = []
        histories = []
        positives = []
        negatives = []
        candidates = set(candidate_news.keys())
        for u, clicks in tqdm(candidate_clicks.items()):
            if u in history_clicks:
                history = history_clicks[u]
                histories.append(history)
                all_skips = list(candidates - set(clicks) - set(history))
                sample_skips = random.sample(all_skips, k_negatives)
                positives.append(clicks)
                negatives.append(sample_skips)
                users.append(u)
        df = pd.DataFrame({
            'user': users,
            'history': histories,
            'positives': positives,
            'negatives': negatives
        }).set_index('user')
        if dst_path is not None:
            df.to_csv(dst_path)
        return df
    
    @staticmethod
    def make_daily_datasets(all_days: list[str], N_days: int, src_dir: PathLike, dst_dir: PathLike):
        path = Path(dst_dir)
        if not path.exists():
            path.mkdir(parents=True)
        for d in range(-1, -N_days, -1):
        
            history_days = all_days[:d]
            candidate_day = all_days[d]
            assert candidate_day not in history_days
            print(f'\nday:{candidate_day}\n_______\n')

            print('processing history')
            history_clicks = AdressaHandler.load_click_data(days=history_days, root_dir=src_dir)
            print('loading candidate clicks')
            candidate_clicks = AdressaHandler.load_click_data(days=[candidate_day], root_dir=src_dir)
            print('loading candidate news')
            candidate_news = AdressaHandler.load_news(days=[candidate_day], root_dir=src_dir)

            print('processing dataset')
            AdressaHandler.make_dataset(
                history_clicks=history_clicks,
                candidate_clicks=candidate_clicks,
                candidate_news=candidate_news,
                dst_path=join(dst_dir, f'histories_and_clicks_for_{candidate_day}.csv')
            )

    @staticmethod
    def combine_daily_datasets(
        days: list[str], 
        src_dir: PathLike, 
        file_prefix: str = 'histories_and_clicks_for_',
        dst_path: Optional[PathLike] = None
        ):
        daily_dfs = []
        for d in tqdm(days):
            daily_dfs.append(
                pd.read_csv(join(src_dir, file_prefix + str(d) + '.csv'))
            )
        df = pd.concat(daily_dfs, ignore_index=True)
        if dst_path is not None:
            path = Path(dst_path)
            if not path.parent.exists():
                path.parent.mkdir(parents=True)
            df.to_csv(dst_path)
        return df
    
    @staticmethod
    def embed_news(
        days: list[str], 
        src_dir: PathLike, 
        model_name: str = 'ltg/norbert3-base', 
        dst_path: Optional[PathLike] = None,
        seq_len: int = 50,
        reduction: str = 'none',
        relative_to_reference: bool = True
        ):
        news = AdressaHandler.load_news(days, src_dir, return_df=True)
        titles = list(news.title)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        emb = precompute_embeddings(
            texts=titles, 
            tokenizer=tokenizer, 
            model=model,
            seq_len=seq_len,
            reduction=reduction,
            relative_to_reference=relative_to_reference
            )
        news['title_emb'] = emb
        if dst_path is not None:
            path = Path(dst_path)
            if not path.parent.exists():
                path.parent.mkdir(parents=True)
            news.to_pickle(dst_path)
        return news
