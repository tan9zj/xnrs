from typing import Callable
import torch
from tqdm import tqdm
from typing import Optional
import pandas as pd


# data

def tokenize_history(history:list, tokenizer:Callable, tokenize_features:list, drop_features:list=[]):
    tokenized_history = []
    for article in history:
        tokenized_article = {}
        for feat, value in article.items():
            if feat in tokenize_features:
                # TODO: this should not be specific to huggingface tokenizers.. 
                # wrap hugging face tokenizers?
                tokenized_article[feat] = torch.tensor(tokenizer(value)['input_ids'])
            elif feat not in drop_features:
                tokenized_article[feat] = value
        tokenized_history.append(tokenized_article)
    return tokenized_history


def filter_len(df, col: str, n_min: Optional[int] = 0, n_max: Optional[int] = None):
        assert col in df.columns, f'{col} is not a valid column'
        assert n_min is not None or n_max is not None, 'at least one bound must be given'
        if n_min is not None:
            mask = df[col].map(lambda h: len(h) >= n_min)
            df = df[mask]
        if n_max is not None:
            mask = df[col].map(lambda h: len(h) < n_max)
            df = df[mask]
        df.reset_index(drop=True, inplace=True)
        return df


# training

def stack_embeddings(emb: list, S: int, N: Optional[int] = None):
    assert emb, 'The emb list is empty'
    D = emb[0].shape[1]
    if N is not None and len(emb) > N:
        emb = emb[-N:]
    padded_emb = []
    masks = []
    for e in emb:
        if len(e) > S:
            e = e[:S]
        s = len(e)
        e = torch.cat([e, torch.zeros(S - s, D)])
        m = torch.tensor([1] * s + [0] * (S - s))
        padded_emb.append(e)
        masks.append(m)
    if N is not None and len(padded_emb) < N:
        n = len(padded_emb)
        padded_emb += [torch.zeros(S, D)] * (N - n)
        masks += [torch.zeros(S)] * (N - n)
    padded_emb = torch.stack(padded_emb)
    masks = torch.stack(masks).unsqueeze(-1)
    return padded_emb, masks


def stack_scalars(labels: list, N: Optional[int] = None, pad_label: int = 0):
    n = len(labels)
    if N is not None:
        if n > N:
            labels = labels[-N:]
        if n < N:
            labels += [pad_label] * (N - n)
    return labels 


def collaps_mask(m: torch.Tensor, dim: int):
    return torch.clamp(torch.sum(m, dim=dim), 0, 1)


def add_batch_dim_(input: dict):
    for v in input.values():
        if isinstance(v, dict):
            add_batch_dim_(v)
        elif isinstance(v, torch.Tensor):
            v.unsqueeze_(0)
        elif isinstance(v, tuple):
            for t in v:
                t.unsqueeze_(0)

def batch_to_device(batch: dict, device: torch.device):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
        elif isinstance(v, dict):
            batch_to_device(v, device)


# logs

def process_case(history, positives, negatives, scores, targets):
    pos_titles = [a['title'] for a in positives]
    neg_titles = [a['title'] for a in negatives]
    cand_titles = pos_titles + neg_titles
    targets = targets.numpy().flatten().tolist()
    scores = scores.numpy().flatten().tolist()
    cand_df = pd.DataFrame({
        'candidates': cand_titles,
        'targets': targets,
        'scores': scores
    })
    cand_df = cand_df.sort_values(by='scores', ascending=False, ignore_index=True)
    hist_titles = [a['title'] for a in history]
    cand_df['history'] = pd.Series(hist_titles)
    return cand_df


# loss

def ranking_loss(p: torch.Tensor, n: torch.Tensor, reduction: str = 'mean'):
    """negative sampling loss
    Args:
        p: positives, shape (B, 1)
        n: negatives, shape (B, K)
    """
    p = torch.exp(p)
    n = torch.sum(torch.exp(n), dim=1, keepdim=True)
    loss = - torch.log(p / (p + n))
    if reduction == 'mean':
        return torch.mean(loss)
    elif reduction == 'none':
        return loss
    else:
        raise ValueError(f'got {reduction} for arg "reduction", but only support "mean" or "none".')
    
# polar plot
import numpy as np

def to_polar(vectors, reference):
    norms = np.linalg.norm(vectors, axis=1)
    unit_vectors = vectors / (norms[:, None] + 1e-8)
    ref = reference / (np.linalg.norm(reference) + 1e-8)
    angles = np.arccos(np.clip(unit_vectors @ ref, -1.0, 1.0))
    return np.stack([angles, norms], axis=1)


import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def plot_polar(data1, data2, labels=('User', 'News'), out_path='polar_plot.png'):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 4))
    
    datasets = []
    if data1 is not None:
        datasets.append((data1, labels[0]))
    if data2 is not None:
        datasets.append((data2, labels[1]))
    
    for data, label in datasets:
        angles, lengths = data[:, 0], data[:, 1]
        print("News polar stats:")
        print(" - Angle min/max:", np.min(angles), np.max(angles))
        print(" - Length min/max:", np.min(lengths), np.max(lengths))
        print(" - Length percentiles:", np.percentile(lengths, [25, 50, 75, 90, 95, 99]))

        x = lengths * np.cos(angles)
        y = lengths * np.sin(angles)
        kde = gaussian_kde(np.vstack([x, y]))
        # print("max length (data1):", np.max(data1[:, 1]) if data1 is not None else "None")
        # print("max length (data2):", np.max(data2[:, 1]) if data2 is not None else "None")

        r = np.linspace(0, 1.0, 700)
        theta = np.linspace(0, np.pi, 100)
        R, T = np.meshgrid(r, theta)
        X = R * np.cos(T)
        Y = R * np.sin(T)
        Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(R.shape)
        ax.contour(T, R, Z)
        max_idx = np.unravel_index(Z.argmax(), Z.shape)
        ax.text(T[max_idx], R[max_idx], label)

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ax.set_thetamin(0)
    ax.set_thetamax(180)

    ax.set_title("Embedding Polar Distribution")
    fig.savefig(out_path)
    plt.close(fig)

