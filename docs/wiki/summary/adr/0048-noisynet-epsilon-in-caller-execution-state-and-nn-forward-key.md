# NoisyNet の ε は module に持たせず、呼び出し側所有の NN 実行状態と NN 実行設定キーで駆動する

## 元の材料

[ADR 0048](../../../adr/0048-noisynet-epsilon-in-caller-execution-state-and-nn-forward-key.md) blob:3f923fbc4cd5ababff1914ba383b12745823687f

## 要点

- NoisyNet の ε は同じ重みを使う複数の Actor や Learner で共有しない。
- ノイズの保持先は module の buffer ではなく、呼び出し側が持つ用途別の実行状態にする。
- 同じ状態・層・時計での共有規則を定め、別用途の推論や snapshot と干渉させない。
- 後に設定の毎回の解釈を廃止し、事前準備された実行オブジェクトを使う形へ改めた。
- target 計算に使う online と target も network ごとに独立の状態を持つ。

## 内容

### 状態の持ち主

重みが共有されても Actor、Learner の現在値計算、target 構築がノイズの更新時点を共有するとは限らない。module の buffer に置くと別用途の forward や snapshot が影響を受ける。

### 設定と寿命

重み・保存の対象と、その推論で利用する ε の状態を分ける。後の事前準備方式でも、同じ用途で繰り返し使う状態と、別 Network 用の独立状態の区別を保つ。

## 主要な概念

- [NoisyNetの観測](../../concepts/methods/noisynet.md)
- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
