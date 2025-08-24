

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# 文件路径
base_path = "/var/scratch/zta207/experiments/mind_small_CL_theme_tsne/"
before_path = base_path + "before_cl_user_emb.csv"
after_path = base_path + "after_cl_user_emb.csv"

# 读取数据，去除 user_id

before_df = pd.read_csv(before_path).drop(columns=['user_id'])
after_df = pd.read_csv(after_path).drop(columns=['user_id'])

# 随机采样 10,000 个用户（保证两者采样一致）
sample_indices = before_df.sample(n=8000, random_state=42).index

before_sampled = before_df.loc[sample_indices].reset_index(drop=True)
after_sampled = after_df.loc[sample_indices].reset_index(drop=True)


# ✅ PCA 降维到 50D（推荐）
pca = PCA(n_components=50, random_state=42)
before_pca = pca.fit_transform(before_sampled)
after_pca = pca.transform(after_sampled)

# ✅ t-SNE 降维到 2D
tsne = TSNE(n_components=2, perplexity=100, n_iter=1500, random_state=42)

before_tsne = tsne.fit_transform(before_pca)
after_tsne = tsne.fit_transform(after_pca)

# === 🖼️ 绘制 BEFORE 图 ===
plt.figure(figsize=(8, 6))
plt.scatter(before_tsne[:, 0], before_tsne[:, 1], color='#0072B2', alpha=0.6, s=30)
plt.title("t-SNE of User Embeddings (Before CL)", fontsize=16)
plt.xlabel("t-SNE dim 1")
plt.ylabel("t-SNE dim 2")
plt.grid(True)
plt.tight_layout()
plt.savefig(base_path + "tsne_before_cl.pdf", dpi=300, bbox_inches='tight')
plt.close()

# === 🖼️ 绘制 AFTER 图 ===
plt.figure(figsize=(8, 6))
plt.scatter(after_tsne[:, 0], after_tsne[:, 1], color='#E69F00', alpha=0.6, s=30)
plt.title("t-SNE of User Embeddings (After CL)", fontsize=16)
plt.xlabel("t-SNE dim 1")
plt.ylabel("t-SNE dim 2")
plt.grid(True)
plt.tight_layout()
plt.savefig(base_path + "tsne_after_cl.pdf", dpi=300, bbox_inches='tight')
plt.close()

print("✅ 两张 t-SNE 图像已分别保存：")
print(" -", base_path + "tsne_before_cl_8k.pdf")
print(" -", base_path + "tsne_after_cl_8k.pdf")
