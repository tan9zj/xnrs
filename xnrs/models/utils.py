import torch
from dotmap import DotMap
import os
from os import PathLike
from os.path import exists, join
import wget
import zipfile
import sys


from .make_model import make_model


def load_model_from_ckpt(
    path: PathLike
):
    ckpt = torch.load(path, map_location=torch.device('cpu'))
    cfg = DotMap(ckpt['config'])
    model = make_model(cfg)
    model.load_state_dict(ckpt['state_dict'])
    return model, cfg


def progress(current, total, width):
  progress_message = f"downloading model: {int(current / total * 100)}%"
  sys.stdout.write("\r" + progress_message)
  sys.stdout.flush()


def get_checkpoint(
    name: str, 
    dst_dir: PathLike = 'checkpoints',
    base_url: str = 'https://www2.ims.uni-stuttgart.de/data/xnrs/'
):
    assert name in ['xnrs_adressa', 'xnrs_mind'], \
        'available models are: xnrs_mind and xnrs_adressa'
    if not exists(dst_dir):
        os.makedirs(dst_dir)
    path = join(dst_dir, name + '_checkpoint')
    if not exists(path):
        zip_path = path + '.zip'
        if not exists(zip_path):
            url = base_url + name + '_checkpoint.zip'

            wget.download(url, zip_path, bar=progress)
            print()
        print('unzipping')
        with zipfile.ZipFile(zip_path, 'r') as f:
            f.extractall(dst_dir)
    print('ready')