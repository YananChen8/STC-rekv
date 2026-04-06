# 保存为比如 model/plot_siglip_similarity.py 方便复用
import os
import matplotlib
matplotlib.use("Agg")  # 服务器上无显示设备时用
import matplotlib.pyplot as plt
import seaborn as sns

from model.cache import STC_CACHE

def plot_siglip_similarity(out_dir: str = "siglip_similarity_plots"):
    cache = STC_CACHE()
    sim_by_layer = getattr(cache, "siglip_similarity_by_layer", None)
    if not sim_by_layer:
        print("No similarity data found in STC_CACHE().siglip_similarity_by_layer")
        return

    os.makedirs(out_dir, exist_ok=True)

    for layer_idx in sorted(sim_by_layer.keys()):
        sim = sim_by_layer[layer_idx].numpy()  # [F, T]
        plt.figure(figsize=(10, 6))
        sns.heatmap(sim, vmin=-1, vmax=1, cmap="coolwarm")
        plt.title(f"SigLIP cosine similarity | layer={layer_idx}")
        plt.xlabel("token index")
        plt.ylabel("frame index")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"layer_{layer_idx:02d}.png"))
        plt.close()