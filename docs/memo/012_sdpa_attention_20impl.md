# SDPA Attention 実装計画

## Summary

- `docs/memo/012_sdpa_attention_10prd.md` に沿って、`CustomTransformerEncoderLayer` の self-attention forward を `at::scaled_dot_product_attention` 経路へ差し替える。
- `torch::nn::MultiheadAttention` は `self_attn` submodule として残し、パラメータ名・初期化・checkpoint 互換を維持する。
- `CONTEXT.md` は glossary なので更新しない。ADR は既存の `0003-fused-adamw-via-aten.md` と PRD で判断根拠が足りているため追加しない。

## Public / Internal Interfaces

- `core/anet-core/src/nn_impl.hpp` に内部 helper として `anet::nn::SdpaSelfAttention(const torch::nn::MultiheadAttention& mha, const torch::Tensor& x)` を宣言する。
- `TransformerConfig` に `bool use_sdpa = true;` を追加し、`TransformerEncoderModuleFactory` で `tf.use_sdpa` を読む。
- `GetCurrentConfigData()` に `use_sdpa` を出力する。
- config key は `net.block.[*].tf.use_sdpa`、既定値は `true`。`false` で従来 `mha_->forward(...)` 経路に戻す。

## Implementation Changes

- `core/anet-core/src/nn_modules.cpp` に `anet::nn::SdpaSelfAttention` を実装する。
  - 入出力は `[B, S, E]`。
  - `mha_->in_proj_weight` / `in_proj_bias` で QKV を一括射影し、`[B, S, H, head_dim] -> [B, H, S, head_dim]` に変換する。
  - `at::scaled_dot_product_attention(q, k, v, {}, dropout_p, false)` を呼び、scale は既定値に任せる。
  - 出力を `[B, S, E]` に戻し、`mha_->out_proj` を適用する。
  - `_qkv_same_embed_dim=false`、`bias_k/bias_v` 定義あり、入力 rank や `E != embed_dim` などは暗黙 fallback せず `ANET_CHECK_MSG` で fail-fast する。
- `CustomTransformerEncoderLayer` に `use_sdpa_` を追加し、Pre-LN / Post-LN 両方の `self_attn` 区間で分岐する。
  - SDPA 経路では `[SeqLen, Batch, E]` への転置往復を削除する。
  - 旧経路は A/B 比較と切り戻し用に維持する。
  - `ANET_PROFILE_SCOPE` 名は `self_attn` を維持する。
- `apps/runner/config/nn.txt` と `apps/runner/config/DropMerge.txt` の Transformer ブロックへ `tf.use_sdpa = true` と短い用途コメントを追記する。
  - 既存の未コミット変更がある `DropMerge.txt` は戻さず、`TransEnc_64/128/256` 近傍への追記だけ行う。

## Test Plan

- `core/anet-core/src/nn_test.cpp` に SDPA テストを追加する。
  - `SdpaSelfAttention` と従来 `mha.forward` の forward 出力を fp32 CPU で `allclose` 比較する。
  - 手動 `q @ k^T / sqrt(head_dim) -> softmax -> v -> out_proj` 参照値と `SdpaSelfAttention` 出力を比較し、head 分割順と scale を固定する。
  - `use_sdpa=true/false` の `TransformerEncoder` を factory 経由で作り、同一パラメータ・同一入力で Pre-LN / Post-LN の forward と backward grad を比較する。
  - `torch::save/load` 後も `self_attn.*` パラメータ名が変わらず、load 前後の forward が一致することを確認する。
  - `add_bias_kv=true` など未対応 MHA option で `SdpaSelfAttention` が `CHECK_THROWS` になることを確認する。
  - CUDA 利用可能時は CUDA fp32 の forward/backward 比較も実行する。half/AMP の flash backend 実測は単体テストではなく NSight 確認項目にする。
- 検証コマンド:
  - `cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test'`
  - `core\anet-core\bin\Debug\anet-core-test.exe`
  - `git diff --check`

## Assumptions

- LibTorch 2.11.0+cu130 の `at::scaled_dot_product_attention` シグネチャは PRD の確認済み前提を採用する。
- `nn_impl.hpp` は内部ヘッダなので、テスト可能性のために `SdpaSelfAttention` を宣言してよい。
- 実測比較は実装後にユーザー側で DropMerge `TransEnc_128` を NSight / steps/sec で確認する。
