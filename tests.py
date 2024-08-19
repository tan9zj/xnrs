import os
from os import PathLike
from typing import Optional
import yaml
from dotmap import DotMap

from train import train
from xnrs.explain import MindExplainer


def test_training():
    print('testing mind training')
    print('-------')
    train(cfg_path='config/mind_standard.yml', debug=True)
    print('seems fine\n')
    print('testing adressa training')
    print('-------')
    train(cfg_path='config/mind_standard.yml', debug=True)
    print('seems fine\n')


def test_init_mind_explainer(
    model_path: PathLike = '../../experiments/mind_standard_test_0/checkpoints/ckpt_hack',
    news_path: PathLike = '/mount/arbeitsdaten/tcl/data/mind/MINDlarge_dev/news_full_sbert.pkl',
    user_path: PathLike = '/mount/arbeitsdaten/tcl/data/mind/MINDlarge_dev/behaviors.csv',
    device: Optional[str] = None
):
    print('\ntesting explainer initialization')
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


def test_imports():
    print('\ntesting imports')
    import xnrs
    from xnrs import MindExplainer
    from xnrs import BCELogitsRankingTrainer, BCERankingTrainer, MSERankingTrainer
    from xnrs.data import AdressaHandler, NewsRecDataset, make_mind_data
    from xnrs.models import make_model
    print('seems fine')


def test_adressa_dataset(cfg_path: PathLike = 'config/adressa_standard.yml'):
    print('testing adressa dataset')
    from xnrs.data import AdressaHandler
    print('loading config')
    cfg = yaml.full_load(open(cfg_path, 'r'))
    cfg = DotMap(cfg)
    print('init datasets')
    train_ds, test_ds = AdressaHandler.init_datasets(cfg)
    print('init successful, dataset sizes:')
    print(len(train_ds), len(test_ds))
    return 


def test_mind_dataset(cfg_path: PathLike = 'config/mind_standard.yml'):
    print('testing mind dataset')
    from xnrs.data import make_mind_data
    print('loading config')
    cfg = yaml.full_load(open(cfg_path, 'r'))
    cfg = DotMap(cfg)
    print('init datasets')
    train_ds, test_ds = make_mind_data(cfg)
    print('init successful, dataset sizes:')
    print(len(train_ds), len(test_ds))
    return 


if __name__ == '__main__':

    # test_imports()
    # test_init_mind_explainer()
    # test_training()
    # test_adressa_dataset()
    test_mind_dataset()