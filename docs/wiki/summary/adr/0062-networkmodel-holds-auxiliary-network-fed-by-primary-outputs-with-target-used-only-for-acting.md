# NetworkModel は primary の Network に加えて補助 Network を持ち、補助は primary の公開出力を入力に取る。補助の target は bootstrap に使わず、target を元にする Actor の行動にだけ使う

## 元の材料

[ADR 0062](../../../adr/0062-networkmodel-holds-auxiliary-network-fed-by-primary-outputs-with-target-used-only-for-acting.md) blob:45d6cd8a38bae498c8b9276e0525567a616fe3aa

## 要点

- NetworkModel は primary と、primary の公開出力を入力に取る補助 Network を持てる。
- skip の補助 target は target ベースの Actor の行動に使い、学習の bootstrap には使わない。
- 特徴への勾配の扱いを選べ、後に split として loss ごとに Adam のモーメントを分ける方法を追加した。
- skip 有効時は primary・補助の online/target を同期、複製、保存の対象にする。
- 価値分解の内的 Head には行動入力で二度推論する問題がないため、後の ADR 0074 は primary の合成 Head を選んだ。

## 内容

### 補助の役割

skip の行動を入力に取る構成では、すべてを primary に入れると行動決定後に primary の計算まで繰り返す。補助を分ければ2回目は補助だけを実行でき、特徴への勾配の扱いを境界で選べる。

### 同期と保存

補助があれば primary と同じ時点で target 同期と Actor の複製を行う。補助を使う checkpoint は各 Network の online と target を保持する。後続の価値分解では、この分割を必要としない内的 Head を primary 内に置いた。

## 主要な概念

- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
- [行動の時間尺度](../../concepts/methods/temporal_abstraction.md)
