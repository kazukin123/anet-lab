# self-attention を ATen `scaled_dot_product_attention` 直接呼び出しに置き換える

`CustomTransformerEncoderLayer` の self-attention が学習 forward の GPU ホットスポットであることを NSight Systems（CUDA_LAUNCH_BLOCKING=ON の真の GPU 帰属）で確認した：`forward.self_attn` が layer の 65%、`TransformerEncoderModule::Forward` が forward の 63%。原因は `torch::nn::MultiheadAttention` が (1) 非 fused（bmm→softmax→dropout→bmm で `[bsz*nhead, S, S]` 行列を materialize）、(2) C++ 版が self-attention 判定に `torch::equal(query,key)` を使い、Python の `query is key` 短絡が無いため同一テンソルでも eq+all+`.item()` D2H が 1 層 2 回走る、(3) `need_weights=true` 既定で捨てる注意重みまで計算する、の 3 点。LibTorch 2.11.0+cu130 に存在する ATen op `at::scaled_dot_product_attention`（Flash/mem-efficient/math を自動選択、autograd 対応）を直接呼ぶ `anet::nn::SdpaSelfAttention` に置き換える。

## Considered Options

- **`torch::nn::MultiheadAttention.forward` を使い続ける（現状）**: 上記 3 問題が残り forward GPU の最大要因のまま。棄却。
- **MHA を自前の `qkv_proj` / `out_proj` Linear に置換（Approach B）**: パラメータ名が変わり旧チェックポイント非互換・重み移行が必要、`_reset_parameters` 相当の初期化も再実装。棄却。
- **MHA を submodule として残し forward だけ SDPA に差し替え（Approach A、採用）**: `self_attn.in_proj_weight` / `in_proj_bias` / `out_proj.*` を流用するためパラメータ名・初期化・checkpoint がすべて自動で不変。SDPA 既定 scale=1/√head_dim が nn::MHA の scaling と一致し、head 分割順も一致するため数学的に等価。`[B,S,E]` のまま処理でき `[S,B,E]` 転置往復も除去。

## Consequences

- ATen 高レベル op `at::scaled_dot_product_attention` への依存が生じる。LibTorch 更新時はシグネチャ追従の確認が必要（0003 と同種の依存）。
- backend（flash/mem-efficient/math）差により従来実装と bitwise 非再現（数学的には同一）。厳密比較や切り戻しのため `net.block.[*].tf.use_sdpa = false`（デフォルト true）で従来 `MultiheadAttention.forward` 経路に戻せる。
- Flash backend は half 精度（fp16/bf16）が条件のため、最大効果は AMP パス。fp32 は mem-efficient/math フォールバックで効果は中程度。
- `torch::equal` 8 回/forward と注意重み平均化が消え、softmax + attention 行列 materialize が flash で不要になる。
- 「LibTorch 高レベル API を迂回し ATen op を直接呼ぶ」方針（ADR 0003 と同系）に加え、「nn モジュールを保持器として残し forward だけ差し替える」互換維持パターンを確立する。
- 実装手順は `docs/memo/012_sdpa_attention_10prd.md`、実装ノートは `docs/memo/012_sdpa_attention_20impl.md`。
