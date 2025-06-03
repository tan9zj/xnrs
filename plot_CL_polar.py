import pandas as pd
import torch
import yaml
import numpy as np
from tqdm import tqdm
import os

from xnrs.models.components.news_encoding import TextEncoder
from xnrs.models.components import layers
from xnrs.utils import to_polar, plot_polar 
from xnrs.data.mind import MindHandler

cfg_path = 'config/mind_small_CL.yml'  
with open(cfg_path, 'r') as f:
    cfg = yaml.full_load(f)
    
# # load data
# news_data_path = cfg['train_news_data_path']
# df = pd.read_pickle(news_data_path)
# # print(f"Loaded columns: {df.columns.tolist()}")
# # print(df.index.name)
# print(f"Loaded {len(df)} news items from {news_data_path}")


# # load parameters
# device = torch.device(cfg['device'])
# text_feature = cfg['text_features'][0]  # e.g., 'title_emb'
# in_dim = cfg['d_backbone']              # e.g., 768
# out_dim = cfg['title_emb_dim']          # e.g., 256
# p_dropout = cfg['p_dropout']
# bias = cfg['bias']

# # text encoder
# title_pooler = layers.AdditiveAttention(
#     in_features=in_dim,
#     hidden_features=256
# )
# encoder = TextEncoder(
#     att=None,
#     pooler=title_pooler,
#     p_dropout=p_dropout,
#     in_features=in_dim,
#     out_features=out_dim,
#     bias=bias
# )
# encoder.eval()
# encoder.to(device)
# print(f"Initialized TextEncoder with in_dim={in_dim}, out_dim={out_dim}")

# all_news_embeds = []
# for idx, row in tqdm(df.iterrows(), total=len(df)):
#     news_id = idx
#     # print(f"[DEBUG] type(row[text_feature]) = {type(row[text_feature])}")
#     # print(f"[DEBUG] row[text_feature] = {row[text_feature]}")
#     # print(f"size: {row[text_feature][0].shape}")
#     # break
#     title_emb = np.array(row[text_feature][0])  # (1, S, D)
#     while title_emb.ndim > 2:
#         title_emb = title_emb.squeeze(0) # (S, D)

#     S, D = title_emb.shape
#     title_emb_tensor = torch.tensor(title_emb).unsqueeze(0).unsqueeze(0)  # (1,1,S,D)
#     mask = torch.ones(title_emb_tensor.shape[:3] + (1,))  # (1,1,S,1)

#     title_emb_tensor = title_emb_tensor.to(device).float()
#     mask = mask.to(device).float()

#     with torch.no_grad():
#         news_emb, _ = encoder((title_emb_tensor, mask))  # (1,1,D_out)
#         news_emb = news_emb.squeeze().cpu().numpy()  # (D_out,)
    
#     all_news_embeds.append({
#         'news_id': news_id,
#         **{f'dim_{i}': news_emb[i] for i in range(len(news_emb))}
#     })

# # # save to csv
out_dir = os.path.join(cfg['dir'], cfg['name'])
# os.makedirs(out_dir, exist_ok=True)
# out_csv_path = os.path.join(out_dir, 'news_final_emb.csv')
# pd.DataFrame(all_news_embeds).to_csv(out_csv_path, index=False)
# print(f"Saved final news embeddings to {out_csv_path}")


# polar plot
print("Loading embeddings...")
news_emb_df = pd.read_csv('/var/scratch/zta207/experiments/mind_small_news/news_final_emb.csv')
before = pd.read_csv('/var/scratch/zta207/experiments/mind_small_CL_theme_hwc/before_cl_user_emb.csv')
after = pd.read_csv('/var/scratch/zta207/experiments/mind_small_CL_theme_hwc/after_cl_user_emb.csv')

news_embeds = news_emb_df[[col for col in news_emb_df.columns if col.startswith('dim_')]].values
before_embeds = before[[col for col in before.columns if col.startswith('dim_')]].values
after_embeds = after[[col for col in after.columns if col.startswith('dim_')]].values

# mean user embedding 
mean_user = before_embeds.mean(axis=0)
# polar coordinates
news_polar = to_polar(news_embeds, mean_user)
before_polar = to_polar(before_embeds, mean_user)

mean_user2 = after_embeds.mean(axis=0)
after_polar = to_polar(after_embeds, mean_user2)
news_polar2 = to_polar(news_embeds, mean_user2)

# news vs. before plot
polar_out_path1 = os.path.join(out_dir, 'polar_news_vs_before_theme.png')
plot_polar(
    data1=before_polar, 
    data2=news_polar,
    out_path=polar_out_path1
)
print(f"Polar plot (news vs. before) saved to {polar_out_path1}")

# news vs. after plot
polar_out_path2 = os.path.join(out_dir, 'polar_news_vs_after_theme.png')
plot_polar(
    data1=after_polar, 
    data2=news_polar2,
    out_path=polar_out_path2
)
print(f"Polar plot (news vs. after) saved to {polar_out_path2}")