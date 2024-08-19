import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import PathLike
from typing import Optional


def plot_history_attributions(
    attributions: dict,
    candidate_title: str, 
    color_range = .02
):
    attr_df = pd.DataFrame(attributions)
    attr_df = attr_df.sort_values(by='news_attribution', ascending=False)
    sorted_attr = np.stack(attr_df.token_attributions)
    all_tokens = list(attr_df.tokens)
    max_len = max([len(t) for t in all_tokens])
    f, ax = plt.subplots(figsize=(max_len, len(all_tokens)/3))
    im = ax.imshow(sorted_attr[:, 0:max_len+2], aspect=.5, vmin=-color_range, vmax=color_range, cmap='bwr')
    for i in range(len(all_tokens)):
        for j in range(min(len(all_tokens[i]), max_len)):
            ax.text(j+1, i, all_tokens[i][j], ha='center', va='center')
    plt.colorbar(im, ax=ax)
    ax.set_xticks([])
    ax.set_yticks(np.arange(len(all_tokens)))
    ax.set_yticklabels([
        f'{np.round(l, 3)}' for l in list(attr_df.news_attribution)
    ])
    attr_tot = attr_df.news_attribution.sum()
    ax.set_title(f'Recommendation Score: {np.round(attr_tot, 3):.3f}  -  Candidate: {candidate_title}')
    return f 


def history_attributions_to_latex(
    attributions: dict,
    min_attr: int = 15
):
    df = pd.DataFrame(attributions).sort_values('news_attribution', ascending=False).reset_index(drop=True)
    max_attr = np.concatenate(list(df.token_attributions)).max()
    latex = ''
    for i in range(len(df)):
        tokens = df.tokens.loc[i]
        attr = df.token_attributions.loc[i]
        score = df.news_attribution.loc[i]
        latex += f'{score:.3f} & '
        for t, v in zip(tokens, attr[1:-1]):
            if t.startswith('##'):
                t = t[2:]
            else:
                latex += ' '
            v = v / max_attr * 100
            if v >= min_attr:
                latex += '\\'
                latex += f'adjustbox{{bgcolor=red!{v:.1f}}}{{\strut {t}}}'
            else:
                latex += t
        latex += ' \\\\\n'
    return latex