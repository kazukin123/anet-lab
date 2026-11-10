# 325 Agent個別実装 MuZero系列

[anet-lab](../../../README.md) › [wiki の目録](../index.md) › [機能カテゴリ](../index.md#機能カテゴリ)

## 概要・責務

- MuZeroAgentは登録済みAgentの試作実装である。（出典: [Agentと学習](../../design/110_agents_and_learning.jp.md)、§2.4）。

## 成立の経緯

- 2026-07-20: batch Envの仕様とAgent生成の接続を更新した（出典: [PRD034](../../memo/done/034_imagecls_batch_input_10prd.md)、題・目的）。
- 2026-08-30: Network Head構築のweight norm設定を接続した（出典: [PRD065](../../memo/done/065_nn_spectral_norm_10prd.md)、題・目的）。
- 2026-09-08: Actor 設定は名前付きカタログ参照とし、Agent インタフェースから RunMode を外す（出典: [ADR 0038](../../adr/0038-actor-config-catalog-without-runmode.md)、本文）。
- 2026-09-19: MuZero試作AgentのActor生成をカタログ参照へ移行した（出典: [PRD061](../../memo/done/061_eval_slot_policy_override_10prd.md)、題・目的）。
- 2026-10-26: ProfiledValueの導入対象を本線の探索・学習へ絞り、試作MuZero固有の温度は対象外とした（出典: [PRD101](../../memo/done/101_exploration_schedule_profiled_value_10prd.md)、§3の非ゴール）。

## 選ばなかった形

- MuZero固有の温度を今回ProfiledValueへ移す形 — 範囲外: 試作Agentの探索設定で本線のDQNと器が違い、今は本線へ戻す予定が無い。再開条件: MuZeroを本線へ戻すとき（出典: [PRD101](../../memo/done/101_exploration_schedule_profiled_value_10prd.md)、§3の非ゴール）。

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
- [PRD101](../../memo/done/101_exploration_schedule_profiled_value_10prd.md) — 探索と学習のパラメータを ProfiledValue で時間変化させる PRD

## 課題

issueの道具（`wiki_support.py maps`）が作り直す。手で直さない。

<!-- issues:begin -->
| # | ID | 状態 | WSJF | 待ち | 対象 | 工程 | 領域 | 分類 | 種類 | PRD | AIの判定 | 課題 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | [issue-20261103-2d66d2](../../management/issues/issue-20261103-2d66d2.md) | 未着手 | 2.2 | — | 成果物 | PRD | 324_dqn, 321_agent_core, 325_muzero, 315_nn | 学習基盤 | 調査 | [PRD910](../../memo/910_network_lock_audit_10prd.md) | 残件有り | 999: train/eval mode 整理後の排他精査 PRD（草案） |
| 2 | [issue-20261103-5addea](../../management/issues/issue-20261103-5addea.md) | 未着手 | 2.2 | — | 成果物 | PRD | 311_dev_tools, 318_rl_core, 315_nn, 312_util, 317_metrics, 321_agent_core, 319_gui, 316_config, 322_env_individual, 324_dqn, 110_runner_app, 325_muzero, 313_profiling | 開発 | リファクタリング | [PRD079](../../memo/079_cpp_internal_linkage_convention_10prd.md) | 残件有り | PRD 079: .cpp の内部リンケージと namespace インデント規約 |
| 3 | [issue-20261103-785e68](../../management/issues/issue-20261103-785e68.md) | 未着手 | 2.0 | — | 成果物 | PRD | 315_nn, 318_rl_core, 321_agent_core, 324_dqn, 319_gui, 320_env_core, 314_tensor, 312_util, 110_runner_app, 322_env_individual, 325_muzero | 開発 | リファクタリング | [PRD945](../../memo/945_outside_core_extension_layout_10prd.md) | 残件有り | anet-core の外で実装する拡張点とコード配置の規約 |
| 4 | [issue-20261103-aa5758](../../management/issues/issue-20261103-aa5758.md) | 未着手 | 2.0（引継ぎ） | — | 成果物 | PRD | 321_agent_core, 318_rl_core, 324_dqn, 314_tensor, 325_muzero | 学習基盤 | リファクタリング | [PRD944](../../memo/944_observation_pipeline_10prd.md) | 残件有り | 観測パイプライン（観測加工の定義と spec の一元化）と強制行動の `MakeAction` 入力化 |
| 5 | [issue-20261103-f993bc](../../management/issues/issue-20261103-f993bc.md) | 未着手 | 0.5 | — | 成果物 | PRD | 321_agent_core, 324_dqn, 325_muzero, 315_nn | 学習基盤 | リファクタリング | [PRD936](../../memo/936_nn_forward_amp_consolidation_10prd.md) | 残件有り | NN 実行設定への AMP の集約（全 Agent 対象） |
| 6 | [issue-20261103-2dd820](../../management/issues/issue-20261103-2dd820.md) | 未着手 | 1.0 | [issue-20261103-785e68](../../management/issues/issue-20261103-785e68.md) | 成果物 | PRD | 321_agent_core, 315_nn, 324_dqn, 312_util, 318_rl_core, 325_muzero, 110_runner_app, 311_dev_tools, 322_env_individual | 開発 | リファクタリング | [PRD957](../../memo/957_source_layout_10prd.md) | 残件有り | ソース配置の見直し（トップ階層を `src/` に、Env と同じ具象別のライブラリへ Agent を分ける） |

着手できる 5 / 待ち 1 / 保留 0 / 次版 0 / 解決 0
<!-- issues:end -->
