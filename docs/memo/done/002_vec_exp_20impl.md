# Vectorized Exploration 実装計画

## Summary
`DefaultDQNAgent` の `train_policy_` だけに空間探索を適用する。`eval_policy_` と `target_policy_` は生成時に必ず `use_spatial_exploration=false` とし、`RainbowAgent` は対象外のままにする。実装後は複数エージェントで差分レビューし、指摘を反映してから検証する。

## Key Changes
- `ActionPolicyConfig` に `use_spatial_exploration=false` と `spatial_scale_type="log"` を追加し、`DefaultDQNAgentConfig` では `train_policy.*` のみ新設定を読む。
- `DefaultDQNAgent::CreateActionPolicy` を `CreateActionPolicy(policy_config, enable_spatial, num_envs, device)` に変更する。
- policy 生成は次の固定ルールにする:
  - `train_policy_`: `enable_spatial = config_.train_policy.use_spatial_exploration`
  - `eval_policy_`: `enable_spatial = false`
  - `target_policy_`: `enable_spatial = false`
- `use_optimistic_target` で `target_policy = train_policy` した後、さらに明示設定を読んだ後でも、最終的に `target_policy.use_spatial_exploration=false` に正規化する。
- `eval_policy_` も最終的に `eval_policy.use_spatial_exploration=false` に正規化し、設定ファイルに指定があっても false 扱いにする。
- `spatial_scale_type` が `"log"` / `"linear"` 以外の場合は、既存マクロ名に合わせて `ANET_SYSTEM_ERROR` で例外落ちさせる。

## Implementation Details
- 空間テンソル生成は `ActionPolicy` 初期化時に行う。`num_envs == 1` でも Shape は `[1]` にする。
- `log` 指定時に `start_val <= 0` または `end_val <= 0` は `1e-4f` にクランプし、WARN を1回出す。
- 対象パラメータは `eps_*`、`uqe_eps_*`、`uqe_tau_*`。
- 空間探索有効時は `current_epsilon_` / `current_uqe_tau_` を `NaN` にし、`OnLearn` の時間減衰更新を早期 return する。
- epsilon-greedy はスカラー経路に `NaN` を流さず、空間探索時専用の `[N]` epsilon tensor 経路で `rand({N}).lt(eps_tensor)` を使う。
- UQE / ThompsonSampling は `spatial_tau_tensor_` を `[N, 1]` に整形して既存の vectorized UQE 経路へ渡す。
- `SelectAction` 時に実バッチ `N` と空間テンソル長が一致しない場合は明示的にエラーにする。

## Review Flow
- 実装エージェント: `ActionPolicyConfig`、`DefaultDQNAgent`、`ActionPolicy`、関連テストを修正する。
- レビューエージェントA: `ActionPolicy` の tensor/scalar 分岐、device、batch shape、NaN フォールスルーを確認する。
- レビューエージェントB: `train/eval/target` の propagation、`use_optimistic_target`、`RainbowAgent` 非対象維持、テスト不足を確認する。
- 親エージェントが両レビューを統合し、必要な修正を入れてからビルド/テストを実行する。

## Test Plan
- 空間テンソル生成: `linear`、`log`、log clamp、`num_envs==1`、不正 `spatial_scale_type` の例外。
- `DefaultDQNAgentConfig`: `train_policy.use_spatial_exploration=true` でも `eval_policy` / `target_policy` は false。
- `use_optimistic_target=true`: target が train 設定をコピーしても spatial は false。
- `EpsilonGreedy` / `UQE` / `ThompsonSampling`: 空間探索有効時に `GetScalar("epsilon")` / `GetScalar("uqe_tau")` が `NaN` のまま、`OnLearn` 後も変化しない。
- 検証は `cmake --build --preset x64-Debug --target anet-core` と、可能なら `anet-core-test` の DQN 周辺テストで行う。

## Assumptions
- `eval_policy.use_spatial_exploration` と `target_policy.use_spatial_exploration` は、設定されていてもエラーにせず false に正規化する。
- ユーザ指定の不正 `spatial_scale_type` は補正せず、即座に `ANET_SYSTEM_ERROR` で失敗させる。
