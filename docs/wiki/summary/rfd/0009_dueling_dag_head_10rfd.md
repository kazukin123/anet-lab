# DefaultDQNAgent Dueling: V/A ストリームを body DAG で分岐できるようにする

## 元の材料

[DefaultDQNAgent Dueling: V/A ストリームを body DAG で分岐できるようにする](../../../memo/done/0009_dueling_dag_head_10rfd.md) blob:9f9426d907aeef1ff9d13f7dd65af784fd4b63d4

## 要点

- Dueling DQN の状態価値 V と行動優位 A に独立した隠れ表現を持たせる。
- 既存の Head は共有特徴から二つの Linear へ直接射影しており、ストリームごとに隠れ層を設定できなかった。
- V と A の分岐は body DAG の出力として設定し、Head でそれぞれの特徴を読む。
- 通常の Dueling と分位を返す QuantileDueling の両方を対象にする。
- 分岐を指定していない既存設定は従来の共有特徴で動かす。

## 内容

### Context（なぜやるか）

古典的 Dueling DQN（Wang et al. 2016）は、共有バックボーンの後で V 用と A 用に独立した隠れ層（FC+活性）へ枝分かれし、それぞれ最終射影する。

### 現状の要点

factory コンストラクタの引数は変えない（`action_dim`/`num_quantiles`/`init`）。construction site（DefaultDQN・Rainbow 双方）に波及しない。変更は `CreateHead` 内部と head クラスに限定。

### 設計（MuZero 流・後方互換）

head に「2モード」を持たせ、config で V/A ストリームを定義したら分岐、しなければ従来通りにする。

## 主要な概念

- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
- [316 設定管理](../../concepts/framework/316_config.md)
