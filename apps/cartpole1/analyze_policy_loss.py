import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ==== 入力設定 ====
# TensorBoard出力ファイル（libtorch側EventWriterのTSV）
tsv_path = r"out\build\x64-Debug\runs\cart1\scalars.tsv"

if not os.path.exists(tsv_path):
    raise FileNotFoundError(f"ログファイルが見つかりません: {tsv_path}")

# ==== ロード ====
df = pd.read_csv(tsv_path, sep="\t")
df.sort_values("step", inplace=True)

# ==== 補助関数 ====
def smooth(y, window=5):
    """単純移動平均スムージング"""
    if len(y) < window:
        return y
    return np.convolve(y, np.ones(window)/window, mode="valid")

def plot_metric(ax, df, tag, color, ylabel, smooth_window=5, logscale=False):
    """共通描画関数"""
    sub = df[df["name"] == tag]
    if len(sub) == 0:
        ax.text(0.5, 0.5, f"[{tag}] データなし", ha="center", va="center", transform=ax.transAxes)
        return
    x = sub["step"].to_numpy()
    y = sub["value"].to_numpy()
    y_smooth = smooth(y, smooth_window)
    ax.plot(x, y, color=color, alpha=0.4, label="Raw")
    ax.plot(x[smooth_window-1:], y_smooth, color=color, linewidth=2, label=f"Smoothed ({smooth_window})")
    ax.set_ylabel(ylabel)
    if logscale:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")

# ==== グラフ設定 ====
plt.style.use("seaborn-v0_8-darkgrid")
fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)

# ==== 各項目描画 ====
plot_metric(axes[0], df, "train/return_mean", "green", "Mean Reward")
plot_metric(axes[1], df, "loss/policy", "orange", "Policy Loss")
plot_metric(axes[2], df, "loss/value", "blue", "Value Loss")
plot_metric(axes[3], df, "loss/entropy", "purple", "Entropy Loss")
plot_metric(axes[4], df, "grad/norm", "red", "Grad Norm")

# ==== 軸ラベル & 出力 ====
axes[-1].set_xlabel("Step (update index)")
plt.suptitle("PPO Training Metrics Overview", fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()
