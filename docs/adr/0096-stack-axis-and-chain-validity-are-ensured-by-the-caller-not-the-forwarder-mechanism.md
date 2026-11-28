# stack 軸の確保と、並びが成り立つことの確かめは、Forwarder の仕組みではなく使う側が受け持つ

並びは、「`FrameStack` は先頭に 1 つ」「`Normalize` は 1 つまで」「`FrameStack` が無ければ 1 枚ぶんを足す」という決まりを持ち、stack 軸を必ず持たせることを、仕組みの側で受け持っていた（[RFD0200](../memo/0200_forwarder_public_interface_rework_10rfd.md)。[ADR 0086](0086-perception-stack-target-keys-always-carry-stack-axis.md) の一部を改める）。**並びは決まりを持たずに、設定に書いたとおりに組み、stack の対象のキーに長さ 1 の軸を足すことと、行動選択・Sample・可視化の出力の spec が同じであることの確かめは、使う側（DefaultDQN）が行う**ことを決定する。stack 軸を必ず持つことは RL の側の決まりで、設定で自由に並べられる仕組みが縛ることではないためである。

## Considered Options

- **並びの決まりを残す**: 設定で自由に並べられるのに、並べ方を縛ることになる。却下。
- **stack 軸の確保を、`FrameStack` の部品か、専用の部品で行う**: RL の側の決まりを、Forwarder の仕組みに持たせることになる。却下。
