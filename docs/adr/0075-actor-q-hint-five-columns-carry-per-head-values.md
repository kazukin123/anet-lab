# Actor Q ヒントを常時 5 列にし、価値の分解の Head ごとの値を運ぶ。列の数はモードで変えない

価値の分解（[PRD 121](../memo/0121_extrinsic_intrinsic_value_decomposition_10rfd.md)）は、PER の優先度を Head ごとの TD 誤差の重み付きの和にするので、Actor の値から初期優先度を見積もる構成（`actor_approx`）では Head ごとの Q_sa と V_s が要り、Actor Q ヒントを `[q_sa, state_value, munchausen_term, intrinsic_q_sa, intrinsic_state_value]` の**常時 5 列**にすることに決める。価値の分解が有効のとき前の 3 列は外的の Head の値で、無効のとき前の 3 列は今と同じ 1 本の Q の値、後ろの 2 列は 0 にする。[ADR 0036](0036-actor-q-hint-three-columns-munchausen.md) の「列の数をモードで変えず、使わない列を 0 にする」判断を保ち、列の数だけを 3 から 5 に改める。

## Considered Options

- **価値の分解が有効のときだけ 5 列にする**: 無効の構成のヒントは 3 列のままになる。代わりに、検証と詰め方の契約がモードに依る。ADR 0036 が Munchausen で退けた形と同じである。却下
- **ヒントは 3 列のままにし、初期優先度だけ合わせた Q の誤差で見積もる**: ヒントの形は変わらない。代わりに、初期優先度と、学習のあとの優先度（Head ごとの誤差の重み付きの和）で決まりが分かれ、内的の Head だけが外している遷移の初期優先度が低く出る。却下

## Consequences

- `kActorQHintColumnCount` は 5 になり、列の数が 5 でないヒントは ERROR にする。互換の読み替えは作らない。
- 価値の分解が無効で `actor_approx` を使う構成では、0 の 2 列が増える。初期優先度の数値は 3 列のときと同じである。
