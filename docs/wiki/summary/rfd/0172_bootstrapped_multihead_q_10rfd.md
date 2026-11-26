# 共有 Body と複数の Q Head による探索案

## 元の材料

[RFD0172　共有 Body と複数の Q Head による探索案](../../../memo/0172_bootstrapped_multihead_q_10rfd.md) blob:78ee4405b191186431285e68661a0771e2d7387e

## 要点

- 共有 Body と複数の価値関数で探索する暫定案であり、既定や実装の順序は再グリル前の候補である。
- Kung-Fu Master では同じ設定でも seed により到達する方策が異なり、単一の価値関数の探索に偏りが生じた。
- Bootstrapped DQN のように特徴抽出を共有し、同じ Run 内で複数の Q Head を別々の価値関数として学習させる案である。
- 各 lane がエピソード単位で担当する Head を選び、一貫した行動系列を作る。
- IQN を含む DQN 系で既定の Head 数を1にし、従来条件との比較を可能にする。
- ボスへ挑む頻度と、挑戦中の命中から撃破に至る割合を区別して測る。

## 内容

### 目的と問題

1 つのネットワークの中に、Body（特徴抽出）を共有し、Q の Head を K 個持たせる（Bootstrapped DQN、Osband ら 2016）。Head はそれぞれ別の価値関数として学び、Actor の各 lane はエピソードの初めに Head を 1 つ選んで、そのエピソードの間はその Head で行動する。Head ごとに違う谷（KFM ならボス行きか稼ぎか）に落ちうるので、1 本の Run の中で、独立の seed を K 本回すのに近い多様性を、Body の計算を共有した安い形で得る。

### 背景

KFM の実時間優先（既定）で、ボス行きに入ったのは 5 本中 1 本。seed 3 と seed 2 の設定の差は `run.seed` だけだった。

### 起草時点の現状

NoisyNetは、ノイズを lane ごとに、episode の始まりで引き直して保持する仕組み（CONTEXT の「保持方式」の `episode`、「共有範囲」の `sample`）を持つ。Head の選択も「lane ごとに episode の間保持する値」で、同じ形をしている。

### 提案の骨子

Head: 価値と優位の流れと射影を K 組持つ（既定 K = 8）。Body（`main_feature` と IQN の τ の埋め込みの合成）は共有する。target network も同じ形で全 Head を持つ。

### User Stories

研究者として、1 本の Run の中で、別々の戦略に落ちる価値関数を複数育て、どの Head がどの戦略を取ったかを見たい。
研究者として、Head の数を 1 にしたときに今と同じ学習系列になることを確かめたい。

### 候補を採る場合の成功の測り方と受入の目安

`num_q_heads = 1` で、本機能の前後の同 seed・決定的モードの短い Run の metrics checksum が一致する。
単体テスト: Head ごとの損失が、その Head の Q と target だけで計算され、Body の勾配が全 Head の平均になる。

## 主要な概念

- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
- [kung_fu_master（KFM）](../../concepts/subjects/kung_fu_master.md)
