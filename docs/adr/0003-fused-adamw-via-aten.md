# optimizer step を ATen `_fused_adamw_` 直接呼び出しの FusedAdamW に置き換える

DQN Learner の `optimizer_->step()` が学習ループのボトルネックであることを Tracy 実測（optimizer_step 区間が支配的）と CUDA_LAUNCH_BLOCKING 実測（カーネル起動律速）で確認した。小型 MLP では param テンソル ~10 個 × per-param 約 12 演算 ≈ 毎 step 100 回超の起動が発生する。PyTorch の `fused=True` 相当は LibTorch の C++ API（`torch::optim`）に存在しないが、その実体である ATen op `at::_fused_adamw_` は LibTorch 2.11.0 に CPU/CUDA 両ディスパッチで存在するため、`torch::optim::AdamW` 派生の `anet::FusedAdamW` がこれを直接呼ぶ構成を採用する（起動数 ~120/step → ~2/step）。なお zero_grad は 2.11 で `set_to_none = true` がデフォルトになっており既にカーネル起動ゼロのため、対象外と判明した。

## Considered Options

- **foreach 手組み（`_foreach_mul_` 等で multi-tensor AdamW を自作）**: 起動 ~10〜15/step とコード量大。CPU では multi-tensor 効果が無い。fused が使えない条件（sparse grad / complex / 異 device 混在）は本プロジェクトに存在せず、保険の価値が無いため棄却。
- **`torch::optim::Optimizer` 派生または完全独立クラス**: state 形式を自由化できるが、自作 GradScaler（`param_groups()` + 仮想 `step()` 依存）とチェックポイント（`WriteTorchObject` の仮想 save/load 依存）が壊れ、呼び出し側の改修と旧チェックポイント非互換が発生するため棄却。
- **`torch::optim::AdamW` 派生 + `step()` override（採用）**: options / `AdamWParamState` / serialize / GradScaler 互換がすべて自動で成立。fused が要求するデバイス上 fp32 step テンソルは、`AdamWParamState` の int64 step と並行管理する（両者は決定的に一致、同期不要。load 後はキャッシュ再構築）。
- **grad_scale / found_inf を fused カーネルへ直渡し（PyTorch GradScaler 完全互換）**: grad clip 使用時は step 前に unscale 済み勾配が必要となり利得が消える。本プロジェクトは grad clip 前提のため見送り。

## Consequences

- ATen 内部 API（アンダースコア付き op）への依存が生じる。LibTorch 更新時はシグネチャ追従の確認が必要。
- fused は演算順序差により従来実装と bitwise 非再現（数学的には同一）。学習曲線の厳密比較が必要な場合は `learner.use_fused_optimizer = false`（デフォルト true）で従来 `torch::optim::AdamW` に切替できる。
- 同時に `GradScaler::Unscale_` を `at::_amp_foreach_non_finite_check_and_unscale_` の 1 カーネルに置換し、grad_norm / clip も foreach ヘルパへ統一する。これにより従来 no-op だった inf 検出が実際に動き出し、FP16 overflow 時の step スキップ + scale backoff が本来の AMP の動作として発生するようになる。
- 適用は DQN Learner のみ。muzero / image_cls への展開は将来のコンストラクタ 1 行変更として残す。
- 実装手順は `docs/memo/011_fused_adamw_10prd.md`。
