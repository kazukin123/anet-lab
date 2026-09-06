# FusedAdamW 導入実装計画

## Summary

- `docs/memo/011_fused_adamw_10prd.md` と `docs/adr/0003-fused-adamw-via-aten.md` に沿って、DQN Learner の AdamW step、grad norm、grad clip、AMP unscale を fused/foreach 経路へ置き換える。
- `CONTEXT.md` は用語集なので更新しない。既存 ADR で判断根拠は足りているため ADR 追加もしない。
- 編集前に `git diff` を確認し、既存の未コミット変更は戻さず、今回の変更だけを重ねる。

## Public Interfaces

- `LearnerConfig` に `bool use_fused_optimizer = true;` を追加する。
- `DefaultDQNAgentConfig` で `learner.use_fused_optimizer` を読む。
- `apps/runner/config/agent.txt` の DefaultDQN baseline に `DefaultDQNAgent.baseline.learner.use_fused_optimizer = true` と用途コメントを追加する。
- Rainbow は今回の対象外。`RainbowAgentConfig` では TBO と同様に `learner.use_fused_optimizer = false;` を明示し、暗黙に fused 化しない。
- `muzero_based_agent.cpp` と `image_cls_agent.cpp` は変更しない。

## Implementation Changes

- `core/anet-core/include/anet/nn_util.hpp` に `FusedAdamW : public torch::optim::AdamW`、foreach helper、`GradScaler` 改修を追加する。
- `FusedAdamW::step()` は closure 対応、`ANET_PROFILE_FUNC()`、sparse grad fail-fast、lazy state 初期化、`(device, dtype)` グループ化、CPU int64 step と device fp32 step tensor の同期前進、`at::_fused_adamw_` 呼び出しを実装する。
- `FusedAdamW::load()` は親 `AdamW::load()` 後に step tensor cache を clear し、次回 step で再構築する。
- `CollectDefinedGrads`、`ForeachGradNorm`、`ForeachClipGradNorm_` を追加し、既存の `+ 1e-6` / `clamp_max(1.0)` の clip セマンティクスを維持する。
- `GradScaler::Unscale_()` は `_amp_foreach_non_finite_check_and_unscale_` に置換し、`Step(optimizer)` overload で inf 検出時の step skip と scale backoff 連携を有効にする。既存 `Step(optimizer, bool)` は互換用に残す。
- `Learner::SetupOptimizer()` は `config_.use_fused_optimizer` で `anet::FusedAdamW` と従来 `torch::optim::AdamW` を分岐し、verbose log に `fused=` を追加する。
- `Learner::Optimize()` は FP32 / AMP+BF16 / AMP+FP16 の grad norm と clip を foreach helper 経由へ統一し、AMP+FP16 は新 `grad_scaler_.Step(*optimizer_)` を使う。`zero_grad` は本機能では変更しない。

## Test Plan

- `nn_util.hpp` の変更に関するテストは PRD 通り `core/anet-core/src/nn_test.cpp` に追加する。
  - `FusedAdamW` vs `torch::optim::AdamW`: CPU 必須、CUDA は利用可能時のみ、weight_decay `{0, 1e-2}`、10 step、param と optimizer state を `allclose` 比較する。
  - checkpoint: `FusedAdamW` save/load 後の継続更新一致、旧 `torch::optim::AdamW` checkpoint を `FusedAdamW` へ load した後の更新一致を検証する。
  - `GradScaler`: inf grad で `Unscale_()` 後 `Step()` が param 更新を skip し、`Update()` 後 scale が backoff することを検証する。
  - foreach helper: norm が手計算と一致し、clip 後 grad が既存手動式と一致することを検証する。
- DQN 設定・Learner 経路に関するテストは必要に応じて既存の `core/anet-core/src/dqn_based_agent_test.cpp` に追加する。
  - `DefaultDQNAgentConfig` は既定 true と明示 false を読む。
  - `RainbowAgentConfig` は `RainbowAgent.learner.use_fused_optimizer=true` を渡しても false のままになる。
- 検証コマンド:
  - `cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test'`
  - `core\anet-core\bin\Debug\anet-core-test.exe`
  - `git diff --check`

## Assumptions

- LibTorch 2.11.0+cu130 の ATen op シグネチャは PRD の確認済み前提を採用する。ビルドで差異が出た場合は、挙動を変えず呼び出しだけ追従する。
- unsupported condition は暗黙 fallback せず fail-fast にする。
- 実測比較はユーザー実施とし、実装側の完了条件は単体/回帰テストと Debug build 成功までにする。
