# 325 Agent個別実装 MuZero系列

[anet-lab](../../../README.md) › [wiki の目録](../index.md) › [機能カテゴリ](../index.md#機能カテゴリ)

## 概要・責務

- MuZeroAgentは登録済みAgentの試作実装である。（出典: [Agentと学習](../../design/110_agents_and_learning.jp.md)、§2.4）。

## 成立の経緯

- 2026-07-20: batch Envの仕様とAgent生成の接続を更新した（出典: [PRD034](../../memo/done/034_imagecls_batch_input_10prd.md)、題・目的）。
- 2026-08-30: Network Head構築のweight norm設定を接続した（出典: [PRD065](../../memo/done/065_nn_spectral_norm_10prd.md)、題・目的）。
- 2026-09-08: Actor 設定は名前付きカタログ参照とし、Agent インタフェースから RunMode を外す（出典: [ADR 0038](../../adr/0038-actor-config-catalog-without-runmode.md)、本文）。
- 2026-09-19: MuZero試作AgentのActor生成をカタログ参照へ移行した（出典: [PRD061](../../memo/done/061_eval_slot_policy_override_10prd.md)、題・目的）。

## 選ばなかった形

記述なし。

## 変えるときの条件

一般の条件は[リファクタリング観点](../../quality/refactor.md)の「ガードレール」、テストの充足は[定点統計](../../../reports/stats/trend.md)の「テスト充足」を見る。

記述なし。

## 関係

- [共通ReplayBufferとの境界](../../design/150_replay_buffer.jp.md)、§2.5。
- 設計: [Agentと学習](../../design/110_agents_and_learning.jp.md)、§2.4。
- ユーザーガイド: [Run実行ガイド](../../design/020_user_guide_run.jp.md)。
- 含まれるアプリケーション: なし。

## ファイル

点検が作り直す。手で直さない。

- [muzero_proto_agent.hpp](../../../core/anet-core/include/anet/muzero_proto_agent.hpp)
- [muzero_proto_agent.cpp](../../../core/anet-core/src/muzero_proto_agent.cpp)

## PRD の一覧

点検が作り直す。手で直さない。

- [PRD034](../../memo/done/034_imagecls_batch_input_10prd.md) — ImageCls 専用 batch 入力コンポーネント / BatchEnv adapter
- [PRD061](../../memo/done/061_eval_slot_policy_override_10prd.md) — PRD 061: Actor 設定カタログと eval スロットごとの方策 / network 指定
- [PRD065](../../memo/done/065_nn_spectral_norm_10prd.md) — NN Spectral Normalization（weight_norm.mode）PRD

## 台帳の指摘

台帳のツールが作り直す。手で直さない。

<!-- registry:begin -->
なし

[台帳の一覧](../../../reports/registry/findings.md)
<!-- registry:end -->
