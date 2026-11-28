# 重ねる枚数とキーの設定は Agent 全体に置き、Sample の束の spec は ReplayBuffer が答える

重ねる枚数とキーは並びの中の `FrameStack` の部品の設定が持ち、ReplayBuffer へは Agent が並びに聞いて渡していたので、重ね方だけがデータと逆の向きに流れていた（[RFD0200](../memo/0200_forwarder_public_interface_rework_10rfd.md)）。**重ねる設定を Agent 全体の 1 か所（`frame_stack.*`）に置いて、Agent が ReplayBuffer と `FrameStack` の部品の両方へ配り（部品は、並びの中の重ねる位置だけを表す）、Sample で返す束の spec は ReplayBuffer が答えて、Sample 用の Forwarder はその spec を受けて作る**ことを決定する。重ね方は、行動選択と ReplayBuffer の両方が使う情報で、1 つの部品の性質ではなく、束を作るのは ReplayBuffer だからである。

## Considered Options

- **並びが、束の spec か、枚数とキーを答える**: 束を作るのは ReplayBuffer なのに、Forwarder の側が主になる。却下。
- **ReplayBuffer と部品の 2 か所に書き、既定の設定で結ぶ**: 1 つの事実に、正本が 2 つになる。却下。
- **ReplayBuffer の設定の 1 か所に書く**: 両方が使う設定が、片方の下に入る。却下。
- **重ねるのを、並びの外へ出す**: 重ねる位置を、設定の並びで書けなくなる。却下。
