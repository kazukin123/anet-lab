import os
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

# === CMake実行ディレクトリからの相対パスを自動検出 ===
build_dir = os.path.join("out", "build", "x64-Debug")
input_tsv = os.path.join(build_dir, "runs", "cart1", "scalars.tsv")
output_dir = os.path.join("runs_converted", "cart1_tb")

# === ロード ===
if not os.path.exists(input_tsv):
    raise FileNotFoundError(f"ログファイルが見つかりません: {input_tsv}")

os.makedirs(output_dir, exist_ok=True)
df = pd.read_csv(input_tsv, sep="\t")

writer = SummaryWriter(log_dir=output_dir)
for name, g in df.groupby("name"):
    for _, row in g.iterrows():
        writer.add_scalar(name, row["value"], row["step"])
writer.close()

print(f"✅ TensorBoard形式に変換しました: {output_dir}")
print("👉 実行: tensorboard --logdir runs_converted/cart1_tb")
