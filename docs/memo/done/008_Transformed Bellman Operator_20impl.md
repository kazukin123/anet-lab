# Transformed Bellman Operator (TBO) 導入 実装計画

## 概要

`docs/memo/008_Transformed Bellman Operator_10prd.md` に沿って、DefaultDQN の学習ターゲットへ Transformed Bellman Operator を追加する。
TBO は DefaultDQN 専用の設定として公開し、Rainbow / MuZero には公開しない。
Q メトリクスは学習 scalar 中心に実空間版を追加し、action-info aux への TBO 設定伝播は行わない。

## 実装変更

- `core/anet-core/include/anet/agent.hpp` の `LearnerConfig` に `use_tbo = false` と `tbo_epsilon = 1e-2f` を追加する。
- `core/anet-core/include/anet/default_dqn_agent.hpp` の `DefaultDQNAgentConfig` で `learner.use_tbo` と `learner.tbo_epsilon` を読む。
- `tbo_epsilon <= 0`、NaN、非有限値は起動時に `ANET_SYSTEM_ERROR` で落とし、エラーにはキー、指定値、期待条件を含める。
- `RainbowAgentConfig` / `RainbowAgent` では `learner.use_tbo = false` を明示し、config ファイルだけでなくコード経由でも TBO を有効化できないようにする。
- `core/anet-core/src/dqn_based_agent.hpp` の `anet::rl::dqn::Learner` に protected 共通処理として `TransformH` / `TransformHInv` を追加する。
- `TDLearner::UpdateFromSamples` では、TBO 有効時に `max_next_q` を `TransformHInv` で実空間へ戻してから Bellman 加算し、得られた target を `TransformH` で h 空間へ戻す。
- `QuantileLearnerBase::CalcTargetQuantiles` では、TBO 有効時に `next_dist` を `TransformHInv` で実空間へ戻してから n-step return と加算し、各分位点 target を `TransformH` で h 空間へ戻す。
- Double DQN、n-step、UQE の行動選択ロジックは変更しない。TBO はターゲット値の変換だけに適用する。
- `DefaultDQNAgent` 初期化時に、`learner.use_tbo = true` かつ `reward_scaler.use_dynamic_scaling` または `reward_scaler.use_auto_post_scale` が有効なら warn を出す。
- `BatchUpdateResult` に実空間 Q tensor を追加し、`q_max_real_mean`、`q_max_real_max`、`q_max_real_std`、`q_sa_real_mean` を `GetScalar` で遅延 CPU 転送する。
- `apps/runner/config/agent.txt` に `DefaultDQNAgent.baseline.learner.use_tbo = false` と `DefaultDQNAgent.baseline.learner.tbo_epsilon = 0.01` を追加する。
- `apps/runner/config/metrics_scalar.txt` に実空間 Q scalar の tag を追加する。
- `apps/runner/config/DropMerge.txt` に検証用の `R.learner.use_tbo = true` オーバーライドを追加する。

## テスト

- `dqn_based_agent_test.cpp` に protected 変換関数を公開する test-only subclass を用意し、代表値で `h(h^-1(x))`、`h^-1(h(x))`、単調性を検証する。
- 代表値は `0`、`+-1`、`+-10`、`+-1e3` を含め、`epsilon` は `1e-2` と境界確認用の別値を使う。
- `DefaultDQNAgentConfig` の `learner.use_tbo` / `learner.tbo_epsilon` 読み込みを検証する。
- `tbo_epsilon` の `0` 以下、NaN、非有限値で `ANET_SYSTEM_ERROR` になることを検証する。
- Rainbow では `learner.use_tbo` が強制 OFF になることを検証する。
- `LogCaptureGuard` を使い、TBO と動的 reward scaler / auto post scale の併用時だけ warn が出ることを検証する。
- `BatchUpdateResult` の実空間 Q scalar が TBO 有効時に `TransformHInv` 後の値を返すことを検証する。
- 検証コマンドは `VsDevCmd.bat` 経由で `anet-core-test` をビルドし、リポジトリルートから `core\anet-core\bin\Debug\anet-core-test.exe` を実行する。

## 前提

- `CONTEXT.md` は glossary 専用とし、実装手順や ADR 的な判断は入れない。
- TBO の公開範囲判断は `docs/adr/0001-default-dqn-tbo-scope.md` に記録する。
- Q の実空間メトリクス追加は学習 scalar 中心とし、action-info aux の `q_values_real` 追加は今回行わない。
- `tbo_epsilon` は `use_tbo = false` でも設定値として検証する。
- Munchausen RL 本体、MuZero の値変換、Rainbow への TBO 公開、`td_clip_value` などの自動再チューニングは対象外とする。
