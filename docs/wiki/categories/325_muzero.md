# 325 Agent個別実装 MuZero系列

[anet-lab](../../../README.md) › [wiki の目録](../index.md) › [機能カテゴリ](../index.md#機能カテゴリ)

## 概要・責務

- MuZeroAgentは登録済みAgentの試作実装である。（出典: [Agentと学習](../../design/110_agents_and_learning.jp.md)、§2.4）。
- MuZeroAgentも、Runnerが系統の公開初期化関数を呼んで登録し、同じ関数を何度呼んでも登録は1回である（出典: [Agentと学習](../../design/110_agents_and_learning.jp.md)、§2.4）。

## 成立の経緯

- 2026-07-20: batch Envの仕様とAgent生成の接続を更新した（出典: [RFD0034](../../memo/done/0034_imagecls_batch_input_10rfd.md)、題・目的）。
- 2026-08-30: Network Head構築のweight norm設定を接続した（出典: [RFD0065](../../memo/done/0065_nn_spectral_norm_10rfd.md)、題・目的）。
- 2026-09-08: Actor 設定は名前付きカタログ参照とし、Agent インタフェースから RunMode を外す（出典: [ADR 0038](../../adr/0038-actor-config-catalog-without-runmode.md)、本文）。
- 2026-09-19: MuZero試作AgentのActor生成をカタログ参照へ移行した（出典: [RFD0061](../../memo/done/0061_eval_slot_policy_override_10rfd.md)、題・目的）。
- 2026-10-26: ProfiledValueの導入対象を本線の探索・学習へ絞り、試作MuZero固有の温度は対象外とした（出典: [RFD0101](../../memo/done/0101_exploration_schedule_profiled_value_10rfd.md)、§3の非ゴール）。
- 2026-11-11: MuZeroのReplayのエピソード終端での価値ターゲットの打ち切りとunroll窓のmaskに、短いエピソードを手で作る5場面のテストを足した（出典: [RFD0122](../../memo/0122_small_refactor_and_test_gap_fixes_10rfd.md)、§1.1・§4 D4）。
- 2026-11-12: MuZeroのActorのカタログ項目を読み口の範囲で読み、読まれない設定キーがあれば構築時に止めるようにした（出典: [RFD0117](../../memo/0117_warn_policy_and_unread_config_10rfd.md)、§4 D3）。
- 2026-11-17: MuZeroのActorは、nn_traceを求める呼び出しを受けても何も入れず、最初の1回だけWARNを出すようにした（出典: [RFD0170](../../memo/0170_nn_trace_capture_unification_10rfd.md)、§4 D12・§6.4）。
- 2026-11-19: MuZero系を基盤の外の系統のライブラリへ出し、登録をライブラリの初期化関数（`InitMuZero`）に移した（出典: [RFD0182](../../memo/0182_source_layout_10rfd.md)、§4.1 D4・D7・D8）。
- 2026-11-20: `InitMuZero`も、何度呼んでも登録を1回だけ行うようにした（出典: [RFD0192](../../memo/0192_concurrency_and_boundary_defect_fixes_10rfd.md)、§5 D11・§6.5）。
- 2026-11-21: MuZeroの試作Agentの設定の誤りの文面（隠れ状態の次元、温度の予定のstart・end）が、設定キーを完全キーで示すようにした（出典: [RFD0194](../../memo/0194_config_key_full_names_in_messages_10rfd.md)、§8.3）。
- 2026-11-23: 文書の呼び名をRFDへ改めるのに合わせて、MuZeroのテストのファイル名の番号の形も改めた（出典: [RFD0195](../../memo/0195_document_term_rfd_and_4digit_numbers_10rfd.md)、§5.2・§9.6）。

## 選ばなかった形

- MuZero固有の温度を今回ProfiledValueへ移す形 — 範囲外: 試作Agentの探索設定で本線のDQNと器が違い、今は本線へ戻す予定が無い。再開条件: MuZeroを本線へ戻すとき（出典: [RFD0101](../../memo/done/0101_exploration_schedule_profiled_value_10rfd.md)、§3の非ゴール）。
- MuZeroのReplayのテストで見つかった食い違い（打ち切りの式など）を、テストを足すRFD0122の中で直す形 — 範囲外: 振る舞いを変えると、今の振る舞いをテストで固めるゴールに反し、直し方の判断が要る。見つかったら起票の候補にする（出典: [RFD0122](../../memo/0122_small_refactor_and_test_gap_fixes_10rfd.md)、§3.2・§4 D15）。
- Agent全体を1本のライブラリにする形 — 不採用: MuZeroやImageClsがDQNの部品を読むことを、機械が止めない（出典: [RFD0182](../../memo/0182_source_layout_10rfd.md)、§4.3）。

## 変えるときの条件

一般の条件は[リファクタリング観点](../../quality/refactor.md)の「ガードレール」、テストの充足は[定点統計](../../../reports/stats/trend.md)の「テスト充足」を見る。

- MuZeroのReplayの価値ターゲットの打ち切りとunroll窓のmaskは、5場面のテストが今の値のまま断定している。truncatedで終わるときに補う値が、打ち切った次の状態ではなく最後のstep自身の`root_value`であることも、誤りの疑いをテストのコメントに書いたうえで断定している（出典: [RFD0122](../../memo/0122_small_refactor_and_test_gap_fixes_10rfd.md)、§4 D4・D15）。
- MuZeroのライブラリは基盤の公開ヘッダだけを読み、テストは兄弟のライブラリに依存しない。ターゲットの依存とincludeは、検索と道具の検査で確かめる（出典: [RFD0182](../../memo/0182_source_layout_10rfd.md)、§5.4・§6 のS3-1〜S3-5）。

## 関係

- [共通ReplayBufferとの境界](../../design/150_replay_buffer.jp.md)、§2.5。
- 設計: [Agentと学習](../../design/110_agents_and_learning.jp.md)、§2.4。
- ユーザーガイド: [Run実行ガイド](../../design/020_user_guide_run.jp.md)。
- 含まれるアプリケーション: なし。

## ファイル

点検が作り直す。手で直さない。

- [CMakeLists.txt](../../../src/algo/muzero/CMakeLists.txt)
- [muzero.hpp](../../../src/algo/muzero/include/anet/algo/muzero.hpp)
- [muzero.cpp](../../../src/algo/muzero/src/muzero.cpp)
- [muzero_actor_test.cpp](../../../src/algo/muzero/src/muzero_actor_test.cpp)
- [muzero_based_agent.cpp](../../../src/algo/muzero/src/muzero_based_agent.cpp) / [muzero_based_agent.hpp](../../../src/algo/muzero/src/muzero_based_agent.hpp)
- [muzero_config_baseline_test.cpp](../../../src/algo/muzero/src/muzero_config_baseline_test.cpp)
- [muzero_proto_agent.cpp](../../../src/algo/muzero/src/muzero_proto_agent.cpp) / [muzero_proto_agent.hpp](../../../src/algo/muzero/src/muzero_proto_agent.hpp)
- [rfd0101_muzero_test.cpp](../../../src/algo/muzero/src/rfd0101_muzero_test.cpp)

## RFD の一覧

点検が作り直す。手で直さない。

- [RFD0034](../../memo/done/0034_imagecls_batch_input_10rfd.md) — ImageCls 専用 batch 入力コンポーネント / BatchEnv adapter
- [RFD0061](../../memo/done/0061_eval_slot_policy_override_10rfd.md) — PRD 061: Actor 設定カタログと eval スロットごとの方策 / network 指定
- [RFD0065](../../memo/done/0065_nn_spectral_norm_10rfd.md) — NN Spectral Normalization（weight_norm.mode）PRD
- [RFD0084](../../memo/done/0084_noisynet_10rfd.md) — NoisyNet (Noisy Networks for Exploration) PRD
- [RFD0101](../../memo/done/0101_exploration_schedule_profiled_value_10rfd.md) — 探索と学習のパラメータを ProfiledValue で時間変化させる PRD
- [RFD0117](../../memo/0117_warn_policy_and_unread_config_10rfd.md) — RFD0117: WARN の意味の線引きと、読まれない設定の検出 RFD
- [RFD0122](../../memo/0122_small_refactor_and_test_gap_fixes_10rfd.md) — RFD0122: 小さなリファクタリングとテストの空白をまとめて片づける
- [RFD0170](../../memo/0170_nn_trace_capture_unification_10rfd.md) — RFD0170: forward の名前付き出力を capture に一本化し、nn_trace を要求した step だけ作る RFD
- [RFD0182](../../memo/0182_source_layout_10rfd.md) — ソース配置の見直し（置き場の規則、Agent の系統ごとのライブラリ、DQN のファイル分割、テストの参照範囲）RFD
- [RFD0192](../../memo/0192_concurrency_and_boundary_defect_fixes_10rfd.md) — RFD0192: 監査で見つかった不具合 5 件をまとめて直す（Notifier の同時実行、動画の画像の大きさ、強制行動の device、重複登録、QValuePanel の平滑化）
- [RFD0194](../../memo/0194_config_key_full_names_in_messages_10rfd.md) — RFD0194: エラー・WARN・INFO が示す設定キーを完全キーにする（device の二重の fatal、指標名、用語の不整合 6 件を含む）
- [RFD0195](../../memo/0195_document_term_rfd_and_4digit_numbers_10rfd.md) — RFD0195: 文書の呼び名を RFD に改め、番号を 0 埋めの 4 桁にし、今使っている場所をすべて移す PRD

## 課題

issueの道具（`wiki_support.py maps`）が作り直す。手で直さない。

<!-- issues:begin -->
| ID | 状態 | WSJF | 待ち | 対象 | 工程 | 領域 | 分類 | 種類 | RFD | AIの判定 | 課題 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [issue-20261103-aa5758](../../management/issues/issue-20261103-aa5758.md) | 調査中 | 2.0（引継ぎ） | — | コード | RFD | 321_agent_core, 318_rl_core, 324_dqn, 314_tensor, 325_muzero | 学習基盤 | リファクタリング | [RFD0161](../../memo/0161_observation_pipeline_10rfd.md) | 残件有り | 観測パイプライン（観測加工の定義と spec の一元化）と強制行動の `MakeAction` 入力化 |
| [issue-20261103-2d66d2](../../management/issues/issue-20261103-2d66d2.md) | 未着手 | 2.2 | — | コード | RFD | 324_dqn, 321_agent_core, 325_muzero, 315_nn | 学習基盤 | 調査 | [RFD0134](../../memo/0134_network_lock_audit_10rfd.md) | 残件有り | 999: train/eval mode 整理後の排他精査 RFD（草案） |
| [issue-20261103-5addea](../../management/issues/issue-20261103-5addea.md) | 未着手 | 2.2 | — | コード | RFD | 311_dev_tools, 318_rl_core, 315_nn, 312_util, 317_metrics, 321_agent_core, 319_gui, 316_config, 322_env_individual, 324_dqn, 110_runner_app, 325_muzero, 313_profiling | 開発 | リファクタリング | [RFD0079](../../memo/0079_cpp_internal_linkage_convention_10rfd.md) | 残件有り | RFD0079: .cpp の内部リンケージと namespace インデント規約 |
| [issue-20261103-785e68](../../management/issues/issue-20261103-785e68.md) | 未着手 | 2.0 | — | コード | RFD | 315_nn, 318_rl_core, 321_agent_core, 324_dqn, 319_gui, 320_env_core, 314_tensor, 312_util, 110_runner_app, 322_env_individual, 325_muzero | 開発 | リファクタリング | [RFD0169](../../memo/0169_outside_core_extension_layout_10rfd.md) | 残件無し | anet-core の外で実装する拡張点とコード配置の規約 |
| [issue-20261110-d21e5f](../../management/issues/issue-20261110-d21e5f.md) | 未着手 | 1.2 | — | コード | RFD | 318_rl_core, 325_muzero | 学習基盤 | 不具合 | — | 残件有り | MuZeroのReplayで、truncatedで終わるときに補う価値が、打ち切った次の状態ではなく最後のstep自身のroot_valueになっている |
| [issue-20261103-f993bc](../../management/issues/issue-20261103-f993bc.md) | 未着手 | 0.5 | — | コード | RFD | 321_agent_core, 324_dqn, 325_muzero, 315_nn | 学習基盤 | リファクタリング | [RFD0164](../../memo/0164_nn_forward_amp_consolidation_10rfd.md) | 残件有り | NN 実行設定への AMP の集約（全 Agent 対象） |

着手できる 6 / 待ち 0 / 保留 0 / 次版 0 / 解決 0
<!-- issues:end -->
