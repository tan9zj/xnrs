from transformers import AutoModel, AutoTokenizer
from torch import device
import json
import gzip
import pickle

from .mind import MindHandler
from .utils import index_category


# configuration

TRANSFORM_BEHAVIORS = False
# PROCESS_NEWS = True
PROCESS_TRAIN_NEWS = False
PROCESS_TEST_NEWS = True

MODEL = 'sentence-transformers/all-mpnet-base-v2'
# MODEL = 'microsoft/mpnet-base'
# MODEL = 'roberta-base'
# MODEL = 'uclanlp/newsbert'

ABBR = 'smpnet'
SEQ_LEN = 50
REFERENCE = True

DEVICE = 'cuda:0'

SRC_TRAIN_NEWS_PATH = '/var/scratch/zta207/data/MINDlarge_train/news.tsv'
DST_TRAIN_NEWS_PATH = f'/var/scratch/zta207/data/MINDlarge_train/news_full_{ABBR}.pkl'
SRC_TRAIN_USER_PATH = '/var/scratch/zta207/data/MINDlarge_train/behaviors.tsv'
DST_TRAIN_USER_PATH = '/var/scratch/zta207/data/MINDlarge_train/behaviors.csv'

SRC_TEST_NEWS_PATH = '/var/scratch/zta207/data/MINDlarge_dev/news.tsv'
DST_TEST_NEWS_PATH = f'/var/scratch/zta207/data/MINDlarge_dev/news_full_{ABBR}.pkl'
SRC_TEST_USER_PATH = '/var/scratch/zta207/data/MINDlarge_dev/behaviors.tsv'
DST_TEST_USER_PATH = '/var/scratch/zta207/data/MINDlarge_dev/behaviors.csv'

CONFIG_PATH = f'/var/scratch/zta207/data/{ABBR}_config.json'

CATEGORY_INDEX_PATH = f'/var/scratch/zta207/data/category_index.pkl'
SUBCATEGORY_INDEX_PATH = f'/var/scratch/zta207/data/sub_category_index.pkl'



# transforming beahiours

if TRANSFORM_BEHAVIORS:

    print('preparing user data')

    train_user_df = MindHandler.read_behaviours_tsv(src_path=SRC_TRAIN_USER_PATH)
    train_user_df, user_idx = index_category(train_user_df, 'user', return_category_idx=True)
    train_user_df.to_csv(DST_TRAIN_USER_PATH)

    # indexing users
    test_user_df = MindHandler.read_behaviours_tsv(src_path=SRC_TEST_USER_PATH)
    test_user_df = index_category(test_user_df, column='user', category_idx=user_idx)
    test_user_df.to_csv(DST_TEST_USER_PATH)

print('init transformer')
tokenizer = AutoTokenizer.from_pretrained(MODEL)
transformer = AutoModel.from_pretrained(MODEL)

# pre-processing news

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
