# 313 プロファイリング

[anet-lab](../../../README.md) › [wiki の目録](../index.md) › [機能カテゴリ](../index.md#機能カテゴリ)

## 概要・責務

- ProfileRangeとmacroで関数・処理phaseをCPU/GPU profilerへ安定名で記録する。（出典: [可観測性](../../design/140_observability.jp.md)、§2・§3）。

## 成立の経緯

記述なし。

## 選ばなかった形

記述なし。

## 変えるときの条件

一般の条件は[リファクタリング観点](../../quality/refactor.md)の「ガードレール」、テストの充足は[定点統計](../../../reports/stats/trend.md)の「テスト充足」を見る。

記述なし。

## 関係

- 設計: [可観測性](../../design/140_observability.jp.md)、§2・§3。
- ユーザーガイド: [Run実行ガイド](../../design/020_user_guide_run.jp.md)。
- 含まれるアプリケーション: なし。

## ファイル

点検が作り直す。手で直さない。

- [profile.hpp](../../../core/anet-core/include/anet/profile.hpp)
- [profile.cpp](../../../core/anet-core/src/profile.cpp) / [profile_test.cpp](../../../core/anet-core/src/profile_test.cpp)

## PRD の一覧

点検が作り直す。手で直さない。
なし

## 課題

issueの道具（`wiki_support.py maps`）が作り直す。手で直さない。

<!-- issues:begin -->
| ID | 状態 | WSJF | 待ち | 対象 | 工程 | 領域 | 分類 | 種類 | PRD | AIの判定 | 課題 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [issue-20261103-5addea](../../management/issues/issue-20261103-5addea.md) | 未着手 | 2.2 | — | 成果物 | PRD | 311_dev_tools, 318_rl_core, 315_nn, 312_util, 317_metrics, 321_agent_core, 319_gui, 316_config, 322_env_individual, 324_dqn, 110_runner_app, 325_muzero, 313_profiling | 開発 | リファクタリング | [PRD079](../../memo/079_cpp_internal_linkage_convention_10prd.md) | 残件有り | PRD 079: .cpp の内部リンケージと namespace インデント規約 |
| [issue-20261111-d1d12b](../../management/issues/issue-20261111-d1d12b.md) | 未着手 | 1.0 | — | 成果物 | 監査 | 321_agent_core, 315_nn, 318_rl_core, 317_metrics, 316_config, 312_util, 313_profiling, 314_tensor, 322_env_individual, 110_runner_app | アルゴリズム | リファクタリング | — | 未確認 | 今は起こさないリファクタリング候補 16 群を暫定で 1 件にまとめた（手を付けるときに群ごとに分けて起票し直す） |

着手できる 2 / 待ち 0 / 保留 0 / 次版 0 / 解決 0
<!-- issues:end -->
