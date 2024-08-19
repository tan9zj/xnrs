import os
from os import PathLike
from typing import Optional
import yaml
from dotmap import DotMap

from train import train
from xnrs.explain import MindExplainer


def test_training(config_path: PathLike = 'config/mind_standard.yml'):
    print('testing config: ', config_path)
    train(cfg_path=config_path, debug=True)
    print('seems fine')
    print('-------')


def test_init_mind_explainer(
    model_path: PathLike = 'checkpoints/xnrs_mind_checkpoint',
    news_path: PathLike = '../data/mind/MINDlarge_dev/news_full_sbert.pkl',
    user_path: PathLike = '../data/mind/MINDlarge_dev/behaviors.csv',
    device: Optional[str] = None
):
    print('testing explainer initialization')
    assert os.path.exists(model_path)
    assert os.path.exists(news_path)
    assert os.path.exists(user_path)
    print('init explainer')
    explainer = MindExplainer()
    print('loading model checkpoint')
    explainer.load_checkpoint(path=model_path, device=device)
    print('loading data')
    explainer.load_data(
        news_path=news_path,
        user_path=user_path
    )
    print('seems fine')
    print('-------')


def test_imports():
    print('testing imports')
    import xnrs
    from xnrs import MindExplainer
    from xnrs import BCELogitsRankingTrainer, BCERankingTrainer, MSERankingTrainer
    from xnrs.data import AdressaHandler, NewsRecDataset, make_mind_data
    from xnrs.models import make_model
    print('seems fine')
    print('-------')


def test_adressa_dataset(cfg_path: PathLike = 'config/adressa_standard.yml'):
    print('testing adressa dataset')
    from xnrs.data import AdressaHandler
    print('loading config')
    cfg = yaml.full_load(open(cfg_path, 'r'))
    cfg = DotMap(cfg)
    print('init datasets')
    train_ds, test_ds = AdressaHandler.init_datasets(cfg)
    print('init successful, train and test set sizes:')
    print(len(train_ds), len(test_ds))
    print('-------')


def test_mind_dataset(cfg_path: PathLike = 'config/mind_standard.yml'):
    print('testing mind dataset')
    from xnrs.data import make_mind_data
    print('loading config')
    cfg = yaml.full_load(open(cfg_path, 'r'))
    cfg = DotMap(cfg)
    print('init datasets')
    train_ds, test_ds = make_mind_data(cfg)
    print('init successful, train and test set sizes:')
    print(len(train_ds), len(test_ds))
    print('-------')


if __name__ == '__main__':

    test_imports()
    test_adressa_dataset()
    test_mind_dataset()
    test_training(config_path='config/adressa_standard.yml')
    test_training(config_path='config/mind_standard.yml')
    test_init_mind_explainer()