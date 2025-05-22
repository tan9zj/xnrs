import pandas as pd
from transformers import AutoModel, AutoTokenizer
from torch import device
import json
import gzip
import pickle
import os

from .mind import MindHandler
from .utils import index_category


# configuration

TRANSFORM_BEHAVIORS = False
MAIN_CATEGORIES = True
# PROCESS_NEWS = True
PROCESS_TRAIN_NEWS = False
PROCESS_TEST_NEWS = False

MODEL = 'sentence-transformers/all-mpnet-base-v2'
# MODEL = 'microsoft/mpnet-base'
# MODEL = 'roberta-base'
# MODEL = 'uclanlp/newsbert'

ABBR = 'smpnet'
SEQ_LEN = 50
REFERENCE = True

DEVICE = 'cuda:0'

# SRC_TRAIN_NEWS_PATH = '/var/scratch/zta207/data/MINDlarge_train/news.tsv'
# DST_TRAIN_NEWS_PATH = f'/var/scratch/zta207/data/MINDlarge_train/news_full_{ABBR}.pkl'
# SRC_TRAIN_USER_PATH = '/var/scratch/zta207/data/MINDlarge_train/behaviors.tsv'
# DST_TRAIN_USER_PATH = '/var/scratch/zta207/data/MINDlarge_train/behaviors.csv'
SRC_TRAIN_NEWS_PATH = '/var/scratch/zta207/data/MINDsmall_train/news.tsv'
DST_TRAIN_NEWS_PATH = f'/var/scratch/zta207/data/MINDsmall_train/news_full_{ABBR}.pkl'
SRC_TRAIN_USER_PATH = '/var/scratch/zta207/data/MINDsmall_train/behaviors.tsv'
# DST_TRAIN_USER_PATH3 = '/var/scratch/zta207/data/MINDsmall_train/behaviors3.csv' # main category from history
DST_TRAIN_USER_PATH4 = '/var/scratch/zta207/data/MINDsmall_train/behaviors4.csv' # main category and theme from history + clicks impression

# SRC_TEST_NEWS_PATH = '/var/scratch/zta207/data/MINDlarge_dev/news.tsv'
# DST_TEST_NEWS_PATH = f'/var/scratch/zta207/data/MINDlarge_dev/news_full_{ABBR}.pkl'
# SRC_TEST_USER_PATH = '/var/scratch/zta207/data/MINDlarge_dev/behaviors.tsv'
# DST_TEST_USER_PATH = '/var/scratch/zta207/data/MINDlarge_dev/behaviors.csv'

SRC_TEST_NEWS_PATH = '/var/scratch/zta207/data/MINDsmall_dev/news.tsv'
DST_TEST_NEWS_PATH = f'/var/scratch/zta207/data/MINDsmall_dev/news_full_{ABBR}.pkl'
SRC_TEST_USER_PATH = '/var/scratch/zta207/data/MINDsmall_dev/behaviors.tsv'
DST_TEST_USER_PATH = '/var/scratch/zta207/data/MINDsmall_dev/behaviors.csv'
# DST_TEST_USER_PATH3 = '/var/scratch/zta207/data/MINDsmall_dev/behaviors3.csv'
DST_TEST_USER_PATH4 = '/var/scratch/zta207/data/MINDsmall_dev/behaviors4.csv' # main category from history + clicks impression

CONFIG_PATH = f'/var/scratch/zta207/data/{ABBR}_config.json'

CATEGORY_INDEX_PATH = f'/var/scratch/zta207/data/category_index.pkl'
SUBCATEGORY_INDEX_PATH = f'/var/scratch/zta207/data/sub_category_index.pkl'
USER_INDEX_PATH = f'/var/scratch/zta207/data/user_index.pkl'

CATEGORY_THEME_MAP = {
    "news": "news",
    "weather": "news", 

    "foodanddrink": "lifestyle",
    "health": "lifestyle",
    "lifestyle": "lifestyle",
    "travel": "lifestyle",

    "video": "entertainment",
    "entertainment": "entertainment",
    "kids": "entertainment",
    "music": "entertainment",
    "tv": "entertainment",
    "movies": "entertainment",
    "autos": "entertainment",

    "northamerica": "world",
    "middleeast": "world",

    "finance": "finance",
    "sports": "sports"
}

# transforming beahiours

if TRANSFORM_BEHAVIORS:
    print('preparing user data')

    train_user_df = MindHandler.read_behaviours_tsv(src_path=SRC_TRAIN_USER_PATH)

    if not os.path.exists(USER_INDEX_PATH):
        train_user_df, user_idx = index_category(train_user_df, 'user', return_category_idx=True)
        with open(USER_INDEX_PATH, 'wb') as f:
            pickle.dump(user_idx, f)
    else:
        with open(USER_INDEX_PATH, 'rb') as f:
            user_idx = pickle.load(f)
        train_user_df = index_category(train_user_df, column='user', category_idx=user_idx)

    train_user_df.to_csv(DST_TRAIN_USER_PATH)

    # indexing users
    test_user_df = MindHandler.read_behaviours_tsv(src_path=SRC_TEST_USER_PATH)
    test_user_df = index_category(test_user_df, column='user', category_idx=user_idx)
    test_user_df.to_csv(DST_TEST_USER_PATH)

if MAIN_CATEGORIES:
    print('computing main categories and clicks for train users...')
    train_news = MindHandler.read_news_as_df(SRC_TRAIN_NEWS_PATH)
    train_news.reset_index(inplace=True)
    news_cat_map = dict(zip(train_news['id'], train_news['category']))

    user_main_category = {}
    user_main_theme = {}

    # train_user_df = MindHandler.read_behaviours_tsv(src_path=SRC_TRAIN_USER_PATH)
    train_user_df = pd.read_csv('/var/scratch/zta207/data/MINDsmall_train/behaviors.csv')
    # add clicks and nonclicks
    clicks_list = []
    nonclicks_list = []
    
    for idx, row in train_user_df.iterrows():
        user_id = row['user']
        history_str = row['history']
        impression_str = row['impression']
        
        # add clicks and nonclicks
        clicks = []
        nonclicks = []
        if not pd.isna(impression_str):
            for item in impression_str.split(' '):
                if item.endswith('-1'):
                    clicks.append(item.rsplit('-', 1)[0])
                elif item.endswith('-0'):
                    nonclicks.append(item.rsplit('-', 1)[0])
        clicks_list.append(' '.join(clicks))
        nonclicks_list.append(' '.join(nonclicks))

        # add main category from history
        history_list = [] if pd.isna(history_str) else history_str.split(' ')
        combined_news = history_list + clicks
        # cats = [news_cat_map.get(news_id) for news_id in history_list if news_id in news_cat_map]
        cats = [news_cat_map.get(news_id) for news_id in combined_news if news_id in news_cat_map]
        themes = [CATEGORY_THEME_MAP.get(cat) for cat in cats if cat in CATEGORY_THEME_MAP]
        # print(f"[DEBUG] user_id={user_id}")
        # print(f"[DEBUG] history_list={history_list}")
        # print(f"[DEBUG] clicks={clicks}")
        # print(f"[DEBUG] combined_news={combined_news}")
        # print(f"[DEBUG] cats={cats}")
        # print(f"[DEBUG] mapped themes (raw)={[cat.lower() if isinstance(cat, str) else cat for cat in cats]}")
        # print(f"[DEBUG] themes={themes}")
        
        main_theme = max(set(themes), key=themes.count)
        user_main_theme[user_id] = main_theme
        
        main_cat = max(set(cats), key=cats.count)
        user_main_category[user_id] = main_cat

 
        
    train_user_df['main_category'] = train_user_df['user'].map(user_main_category)
    train_user_df['main_theme'] = train_user_df['user'].map(user_main_theme)

    train_user_df['clicks'] = clicks_list
    train_user_df['nonclicks'] = nonclicks_list
    
    print('saving train user behavior')
    train_user_df.to_csv(DST_TRAIN_USER_PATH4, index=False)


    print('computing main categories and clicks for test users...')
    test_news = MindHandler.read_news_as_df(SRC_TEST_NEWS_PATH)
    test_news.reset_index(inplace=True)
    news_cat_map_test = dict(zip(test_news['id'], test_news['category']))

    user_main_category_test = {}
    user_main_theme_test = {}
    test_user_df = pd.read_csv('/var/scratch/zta207/data/MINDsmall_dev/behaviors.csv')
    clicks_list_test = []
    nonclicks_list_test = []

    for idx, row in test_user_df.iterrows():
        user_id = row['user']
        history_str = row['history']
        impression_str = row['impression']
        
        clicks = []
        nonclicks = []
        if not pd.isna(impression_str):
            for item in impression_str.split(' '):
                if item.endswith('-1'):
                    clicks.append(item.rsplit('-', 1)[0])
                elif item.endswith('-0'):
                    nonclicks.append(item.rsplit('-', 1)[0])
        clicks_list_test.append(' '.join(clicks))
        nonclicks_list_test.append(' '.join(nonclicks))
        
        history_list = [] if pd.isna(history_str) else history_str.split(' ')
        combined_news = history_list + clicks
        # cats = [news_cat_map_test.get(news_id) for news_id in history_list if news_id in news_cat_map_test]
        cats = [news_cat_map_test.get(news_id) for news_id in combined_news if news_id in news_cat_map_test]
        themes = [CATEGORY_THEME_MAP.get(c) for c in cats if c in CATEGORY_THEME_MAP]
        
        main_cat = max(set(cats), key=cats.count)
        user_main_category_test[user_id] = main_cat

        main_theme = max(set(themes), key=themes.count)
        user_main_theme_test[user_id] = main_theme

    test_user_df['main_category'] = test_user_df['user'].map(user_main_category_test)
    test_user_df['main_theme'] = test_user_df['user'].map(user_main_theme_test)

    test_user_df['clicks'] = clicks_list_test
    test_user_df['nonclicks'] = nonclicks_list_test
    print('saving test user behavior')
    test_user_df.to_csv(DST_TEST_USER_PATH4, index=False)


    
# pre-processing news

print('init transformer')
tokenizer = AutoTokenizer.from_pretrained(MODEL)
transformer = AutoModel.from_pretrained(MODEL)

if PROCESS_TRAIN_NEWS:

    with open(CONFIG_PATH, 'w+') as f:
        json.dump({
            'model': MODEL,
            'abbreviation': ABBR,
            'seq_len': SEQ_LEN,
            'reference': REFERENCE
        }, f)

    print('indexing train news categories')
    print('computing embeddings with', MODEL)

    train_news = MindHandler.read_news_as_df(SRC_TRAIN_NEWS_PATH)

    # train_news, category_index = index_category(
    #     data=train_news,
    #     column='category',
    #     return_category_idx=True
    # )
    #
    # train_news, sub_category_index = index_category(
    #     data=train_news,
    #     column='subcategory',
    #     return_category_idx=True
    # )
    #
    # with open(CATEGORY_INDEX_PATH, 'wb') as f:
    #     pickle.dump(category_index, f)
    # with open(SUBCATEGORY_INDEX_PATH, 'wb') as f:
    #     pickle.dump(sub_category_index, f)

    # read the saved index
    with open(CATEGORY_INDEX_PATH, 'rb') as f:
        category_index = pickle.load(f)
    with open(SUBCATEGORY_INDEX_PATH, 'rb') as f:
        sub_category_index = pickle.load(f)

    train_news = index_category(
        data=train_news,
        column='category',
        category_idx=category_index
    )

    train_news = index_category(
        data=train_news,
        column='subcategory',
        category_idx=sub_category_index
    )

    print('computing train news title embeddings')

    train_news = MindHandler.precompute_embeddings(
        news_data=train_news,
        src_feat='title',
        dst_feat='title_emb',
        tokenizer=tokenizer,
        model=transformer,
        reduction='none',
        seq_len=SEQ_LEN,
        device=device(DEVICE),
        relative_to_reference=REFERENCE
    )

    print('computing train news abstract embeddings')

    train_news = MindHandler.precompute_embeddings(
        news_data=train_news,
        src_feat='abstract',
        dst_feat='abstract_emb',
        tokenizer=tokenizer,
        model=transformer,
        reduction='none',
        seq_len=SEQ_LEN,
        device=device(DEVICE),
        relative_to_reference=REFERENCE
    )

    train_news.to_pickle(DST_TRAIN_NEWS_PATH)

if PROCESS_TEST_NEWS:
    print('indexing test news categories')

    test_news = MindHandler.read_news_as_df(SRC_TEST_NEWS_PATH)

    with open(CATEGORY_INDEX_PATH, 'rb') as f:
        category_index = pickle.load(f)
    with open(SUBCATEGORY_INDEX_PATH, 'rb') as f:
        sub_category_index = pickle.load(f)

    test_news = index_category(
        data=test_news,
        column='category',
        category_idx=category_index
    )

    test_news = index_category(
        data=test_news,
        column='subcategory',
        category_idx=sub_category_index
    )

    print('computing test news title embeddings')

    # NOTE: for the accuracy of our attributions, it is crucial to to set the argument relative_to_reference=True (cf. Section 4.3 of the paper)

    test_news = MindHandler.precompute_embeddings(
        news_data=test_news,
        src_feat='title',
        dst_feat='title_emb',
        tokenizer=tokenizer,
        model=transformer,
        reduction='none',
        seq_len=SEQ_LEN,
        device=device(DEVICE),
        relative_to_reference=REFERENCE
    )

    print('computing test news abstract embeddings')

    test_news = MindHandler.precompute_embeddings(
        news_data=test_news,
        src_feat='abstract',
        dst_feat='abstract_emb',
        tokenizer=tokenizer,
        model=transformer,
        reduction='none',
        seq_len=SEQ_LEN,
        device=device(DEVICE),
        relative_to_reference=REFERENCE
    )

    test_news.to_pickle(DST_TEST_NEWS_PATH)
    print("Completed!")
