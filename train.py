from os import PathLike
import wandb as wb
import yaml
import os
from os.path import join, exists
import torch
import random
from os import PathLike
import argparse
from dotmap import DotMap

from xnrs.models import make_model
from xnrs.data import make_mind_data
from xnrs.data import AdressaHandler
from xnrs.training import BCELogitsRankingTrainer, MSERankingTrainer


def train(cfg_path: PathLike, debug: bool = False):

    print('loading config')
    cfg = yaml.full_load(open(cfg_path, 'r'))

    if debug:
        cfg['debug'] = True
        cfg['name'] = 'debug_run'
    else:
        cfg['debug'] = False
    dir = join(cfg['dir'], cfg['name'])
    if not exists(dir):
        os.makedirs(dir)

    print('init logging')
    if cfg['wandb']:
        wb.init(
            config=cfg,
            dir=dir,
            project=cfg['project'],
            tags=cfg['tags'],
            notes=cfg['notes'],
            name=cfg['name'],
            mode=cfg['mode']
        )
        cfg = wb.config
    else:
        cfg = DotMap(cfg)

    torch.manual_seed(cfg.random_seed)
    random.seed(cfg.random_seed)

    print('init model')
    model = make_model(cfg)
    if cfg['wandb']:
        wb.watch(model)

    print('init dataset (the first time this will take a while)')
    if cfg.dataset == 'mind':
        train_ds, test_ds = make_mind_data(cfg)
    elif cfg.dataset == 'adressa':
        train_ds, test_ds = AdressaHandler.init_datasets(cfg)
    else:
        raise ValueError('unknown dataset')

    print('init trainer')
    # trainer = BCELogitsRankingTrainer(cfg, model, train_ds, test_ds)
    trainer = MSERankingTrainer(cfg, model, train_ds, test_ds)

    print('starting training')
    trainer.train()
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c',
        help='path to config file',
        default='config/mind_standard.yml'
        # default='config/mind_small.yml'
    )
    args = parser.parse_args()
    train(cfg_path=args.config)