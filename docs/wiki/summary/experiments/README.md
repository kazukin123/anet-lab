# ハイパーパラメータ探索記録

## 元の材料

[ハイパーパラメータ探索記録](../../../experiments/README.md) blob:980f99c1d53fa41626ce2817d9acd387a940b3bd

## 要点

- 実験は Agent アルゴリズム、環境、campaign の順に分ける。
- 一つの campaign は比較の前提を共有する Run または一つの探索テーマに対応する。
- 実効設定は Run の config/config_data.txt、到達 step・停止理由・メトリクスは各 Run の記録で確認する。
- 過去の観測と当時の判断を保持し、解釈の更新や無効化は状態と理由を追記する。
- 新たに集計した数値は先に実験記録へ書き、wiki から原本を引く。

## 主要な概念

- [評価と比較の条件](../../concepts/measurement/evaluation.md)
- [開発プロセス・開発ツール・テスト基盤（311）](../../concepts/framework/311_dev_tools.md)
