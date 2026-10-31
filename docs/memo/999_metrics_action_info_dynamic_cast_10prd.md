# MetricsLogObserverBaseのaction_info取得から既存dynamic_castを整理する

> 2026-10-20 起票（暫定。グリル前）。PRD097工程末の候補②をユーザーが採用した。
> 具体的な契約変更・実装方式は未決。

## 背景と問題

[observers.cpp](../../core/anet-core/src/observers.cpp)の`MetricsLogObserverBase::GetMetricsData`、`EventField::ACTION_INFO`分岐（記録時917行）に、`BatchActionInfo`から`anet::Module`へのproduction `dynamic_cast`が残っている。[AGENTS.md](../../AGENTS.md)のコーディング規約は、productionの`dynamic_cast`を禁止し、仮想関数・明示的interface・既存APIの拡張等による表現を求めている。

[PRD097実装記録](097_env_auto_reset_after_notification_21impl.md)の独立レビューで見つかった既存コードで、今回の変更hunk外。PRD097の受入結果に影響する新規不具合としては扱っていない。

## 現在の振る舞い

- `action_info`がないときは、TrainEventでだけ使えるというエラーで停止する。
- `Module`へ変換できる場合は、その`GetScalar(key)`から値を読む。
- 変換できない場合は、指定step軸のstepとNaNを返す。

## 取り組むこと

- `BatchActionInfo`・各派生結果とmetricsの責任境界を調べ、productionの実行時型判別を使わずscalar取得の可否を表現する。
- 公開メトリクス宣言と通知から観測できる値・step・NaN/nullopt・不正イベントのfail-fastを確認し、既存の意味を維持して移行する。契約変更が必要なら現用の全利用側を同時に移行し、互換aliasや旧型専用の分岐は追加しない。
- 規約を満たす最小の変更に絞り、無関係なObserverの整理へ広げない。

## 完了の目安と未決

- 対象のproduction `dynamic_cast`がなくなり、scalarを返せる/返せないaction_info、未知キー、値未成立、対象外イベントを公開のmetrics/通知経路で検証できる。
- 関連回帰とDebugビルドを確認し、必要な計測範囲・日本語段階コメントを維持する。
- scalar取得をどのinterfaceで表すかは未決。着手時に既存の設計・関連ADRと照合して決める。

## 出典

- [実装記録](097_env_auto_reset_after_notification_21impl.md)の独立レビュー・工程末の選択。
- レビュー原文: `.scratch/prd097/review-20261020-1634-d7-impl.md`、`.scratch/prd097/review-20261020-2158-final-impl.md`。
