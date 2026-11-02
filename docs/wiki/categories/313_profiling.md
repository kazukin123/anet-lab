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

## 台帳の指摘

台帳のツールが作り直す。手で直さない。

<!-- registry:begin -->
なし

[台帳の一覧](../../../reports/registry/findings.md)
<!-- registry:end -->
