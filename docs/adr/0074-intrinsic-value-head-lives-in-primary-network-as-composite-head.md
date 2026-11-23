# 価値の分解の内的の Head は、補助 Network ではなく、primary の Network の合成 Head に置く

価値の分解（[PRD 121](../memo/0121_extrinsic_intrinsic_value_decomposition_10rfd.md)）の内的の Head の置き場は、primary の Network の Head を 2 組の出力を出す合成 Head にするか、skip と同じ補助 Network にするかの 2 つで、**合成 Head にする**ことに決める。forward が 1 回のままで、target の同期・Actor の写し・checkpoint が今の Network の仕組みにそのまま乗るためである。[ADR 0062](0062-networkmodel-holds-auxiliary-network-fed-by-primary-outputs-with-target-used-only-for-acting.md) が skip でこの形を退けた理由のうち「行動を入力に取る構成で forward が 2 回になる」は内的の Head には当たらず、skip の判断は変えない。

## Considered Options

- **補助 Network（skip と同じ器）**: 共有の Body への勾配を継ぎ目で切れ、無効で取った重みから続けられる。代わりに、forward が Actor と Learner で 1 回ずつ増え、skip 専用の箇所を内的用にも書くことになり、ADR 0062 の「補助の target は bootstrap に使わない」を改める必要がある。勾配の clip と Adam の状態を分けることは、合成 Head のままでも、損失ごとに backward を取るモード（skip の損失ごとの最適化と同じ作り）で満たせる。先送り。そのモードでも共有が学習を乱し、Network ごとに分ける必要があると分かったら作る

## Consequences

- 合成 Head は、外的と内的の出力を別の鍵で出す。合わせた Q は、重みを知っている DQN 側が作る。
- 勾配の clip は、既定では外的と内的の勾配を合わせて 1 回になる。
- 価値の分解の有無が違う checkpoint は、Network の形が違うので読めない。
