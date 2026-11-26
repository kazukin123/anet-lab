# 可塑性メトリクスは実 forward の branch capture と学習経路外の部分 forward の2系統で測る

## 元の材料

[ADR 0031](../../../adr/0031-plasticity-metrics-out-of-band-partial-forward.md) blob:feeaccd7819292f854dba9e7029b2212f2538f62

## 要点

- 学習バッチの特徴は更新に用いた実際の forward の途中結果から取り出す。
- 固定 probe の特徴は必要な部分だけを eval で実行し、学習用乱数を消費しない。
- 測定は購読要求があるときに行い、無条件に全分岐をキャプチャしない。
- 後の設計では要求キーを持つ共通の ForwardCapture に途中結果を集める。
- 共通 capture への改名と移設は RFD0161、Forwarder と処理の並びは ADR 0084 が定めた。

## 内容

### 測定する表現

srank と dormant unit 率を学習が実際に見た特徴、条件を固定した probe の双方で測る。前者は dropout や batch 統計を含み、後者は replay ratio の違う Run でも同じ測定条件を取れる。

### 選んだ取り出し方

全分岐を強制実行すると IQN の分位など本来不要な入力や計算まで生じる。probe は指定分岐の依存部分だけを通し、学習時の乱数に干渉しない。

### 実装時の補足

online の実バッチ、target の実バッチ、probe は購読行から個別の測定周期を得る。購読がなければ要求も追加の推論も作らない。実バッチの取得に共通 capture を使うようになってもこの分担は変わらない。

## 主要な概念

- [可塑性指標と成績](../../concepts/phenomena/plasticity.md)
- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
- [計測の負荷と比較条件](../../concepts/measurement/diagnostic_cost.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
