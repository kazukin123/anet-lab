# forwarder.hpp の公開の口の作り直し（作る側と作られるものを分け、spec を依頼ごとに渡す） RFD

> 2026-11-24 起票。グリルを行った（2026-11-23〜24。hpp の草案を直しながら、会話の中で決めた。草案は §10.3）。起草のあと、別モデルの検討（2026-11-24）を受けて直した（§10.2）。
> 2026-11-24 改訂: 起票のあとに出た材料を §10.4 に足した（RFD0142 を本 RFD の口に照らした机上の確かめと、直近の系列を並びの外のバッファに置く案）。確定事項と実装契約は変えていない。開発者は、今の形のまま先に実装すると述べた。
> 2026-11-25 改訂: 設定の誤りを表の順で報告するため、外部の部品にも適用する設定キーの読み方と、並びで使える Forward の種類の検査を §6.3 に明記した。受入で複数の誤りがある場合の順を確かめ、移行対象へ実行バッチの設定を加えた。
> 2026-11-25 改訂: `FrameStack` の出力に軸を付ける条件と、DefaultDQN が最終出力に軸を補う条件を明確にした。開始の印の spec（§6.7）と Actor の出力検査（§6.5）を明記し、通常の Actor 経路での受入を §7 に加えた。既存の Montezuma 起動は出荷設定の既定を使うと §6.9 に記した。
> 起点: [0161_observation_pipeline_10rfd.md](0161_observation_pipeline_10rfd.md) の実装のあとの、開発者のレビュー（2026-11-23）。forwarder.hpp について、依頼の型の名前が中身と合っていないこと、型の関係が込み入っていること、RL に依らない層に RL の概念が入っていることが挙がった（§10.1）。
> 基準コード: 2026-11-24 の main（`0411d5eb`）。実装の入口で、§1 と §4 の行番号をシンボル名で引き直す。
> 関連: [ADR 0094](../adr/0094-forwarders-are-built-per-request-and-builders-are-not-forwarders.md)、[ADR 0095](../adr/0095-frame-stack-setting-is-agent-wide-and-replay-buffer-answers-sample-spec.md)、[ADR 0096](../adr/0096-stack-axis-and-chain-validity-are-ensured-by-the-caller-not-the-forwarder-mechanism.md)。一部を改める ADR は [ADR 0084](../adr/0084-forwarder-is-common-interface-and-chain-serves-requests-named-by-input-and-product.md) と [ADR 0086](../adr/0086-perception-stack-target-keys-always-carry-stack-axis.md)（§5.5 の D29）。

## 1. 問題

RFD0161 は、観測の加工の定義を 1 つの仕組み（Forwarder の並び）にまとめた。動きは目的どおりだが、公開の口（[forwarder.hpp](../../src/core/base/include/anet/forwarder.hpp)）に、次の 5 つが残った。

### 1.1 型の関係が込み入っている

公開の型は 6 つある（`Forwarder`・`SpecForwarder`・`ForwardRequest`・`ForwardExecutor`・`ForwarderProvider`・`ForwarderChain`。37〜94 行）。

- 実行するものが 2 種類ある。`Forwarder::Forward` は `TensorDict` だけを受け、`ForwardExecutor::Forward` は `episode_start` も受ける。使う側は、使い道によって口を使い分ける。
- 部品の口が 2 種類ある。部品は `SpecForwarder` を実装し、依頼によって動きが違う部品だけ、足して `ForwarderProvider` を実装する。
- 並び（`ForwarderChain`）は、自分が `SpecForwarder` であり、依頼ごとの Forwarder と実行器を作る側でもある（73〜83 行）。
- `ForwardRequest` は、依頼そのものではなく、依頼の種類の列挙である（49〜53 行）。

### 1.2 RL に依らない層に、RL の概念が入っている

forwarder.hpp は、全部が `anet` 名前空間（RL に依らない層）にある。そこに、次の RL の概念が入っている。

- `ForwardExecutor::Forward` の引数 `episode_start`（59 行）。
- 依頼の種類の列挙の値（`FrameWithStatsUpdate`・`PerceptionFromStack`・`PerceptionFromFrame`）。
- 並びの決まり。`FrameStack` は先頭に 1 つ、`Normalize` は 1 つまでで、`FrameStack` が無ければ 1 枚ぶんを足す（[forwarder.cpp](../../src/core/base/src/forwarder.cpp) 612〜627 行）。
- 重ねる枚数とキーを答える口（`GetStackCount`・`GetStackKeys`。84〜86 行）。
- 図の名前（`perception`・`Frame`・`Perception`。forwarder.cpp 717〜754 行）。

### 1.3 部品の種類が固定で、外から足せない

部品の種類は、forwarder.cpp の中の 3 つ（`FrameStack`・`Normalize`・`DropKeys`）に固定で、ほかの名前はエラーになる（576 行）。NN の module は、type の名前で登録する形である。

### 1.4 spec を、構築のときの 1 つしか渡せない

`ForwarderChain` は、構築のときに spec を 1 つだけ受ける（75〜76 行）。渡しているのは、Env の観測 1 枚ぶんの spec である。ところが、依頼ごとに、実際に渡すものが違う。

| 使い道 | 実際に渡すもの |
|---|---|
| 行動選択 | Env の観測 1 枚と、episode の開始の印 |
| 経験の受入れ | Env の観測 1 枚（obs と next_obs） |
| Sample | ReplayBuffer が重ねて返した束 |
| 可視化 | Env の観測と同じ形の 1 枚 |

`ForwardSpec` は、依頼の指定なしで出力の spec を答える（80 行）。どの依頼の spec なのかが、口から読めない。

### 1.5 重ね方が、データと逆の向きに渡る

重ねる枚数とキーは、並びの中の `FrameStack` の部品の設定（`perception.block.[stack].count`・`keys.*`）が持つ。ReplayBuffer へは、Agent が並びに聞いて（[default_dqn_agent.cpp](../../src/algo/dqn/src/default_dqn_agent.cpp) 655 行）、Learner 経由で渡す（[dqn_learner.cpp](../../src/algo/dqn/src/dqn_learner.cpp) 942〜944 行）。データは Env → ReplayBuffer → Sample 用の Forwarder と流れるのに、重ね方だけが、並び → Agent → Learner → ReplayBuffer と流れる。行動選択と ReplayBuffer の両方が使う設定の正本が、並びの中の 1 つの部品にある。

## 2. ゴールと非ゴール

**ゴール**

- **G1**: 公開の口を、「依頼を出して Forwarder を受け取り、`Forward` を呼ぶ」の 1 つの使い方にする。実行するものは `Forwarder` だけで、作る側は Forwarder を兼ねない。
- **G2**: forwarder.hpp の `anet` の層から RL の概念を外す。依頼の種類は RL の側が定義し、並びの決まりは持たず、stack 軸の確保は使う側が受け持つ。部品の種類は登録制にする。
- **G3**: spec が、データと同じ向きに流れる。依頼ごとに、実際に渡すものの spec を渡し、束の spec は ReplayBuffer が答える。重ねる設定は、Agent 全体の 1 か所に書く。
- **G4**: 出荷設定（新しい設定キーへ移したもの）で、学習の挙動が変わらない（同じ seed・決定的なバックエンドの Run で、scalar と trace が一致する）。
- あわせて、本 RFD の課題（[issue-20261124-6b50cb](../management/issues/issue-20261124-6b50cb.md)。公開の口の作り直し）が片付く。

**非ゴール**

行の「本当の制約か」は、クリーンブレークで今回一緒に移せない理由があるかを書く。

| # | 入れないもの（消える案） | 理由と、本当の制約か |
|---|---|---|
| NG1 | 部品の中身を変えること（正規化の式と統計の持ち方、重ねる履歴の持ち方、キーを落とす動き） | 口の作り直しに要らない。動きを変えると、G4 の等価性で確かめられなくなる。制約ではなく、範囲の線である。ただし、どのキーに統計を取るかの決まり方（D18・D28）と、`FrameStack` が枚数とキーを受け取る先（D15・D17）は変わる |
| NG2 | 統計つきの正規化の読み書きの競合（[issue-20261122-258ef8](../management/issues/issue-20261122-258ef8.md)）と、統計の checkpoint（[issue-20261113-b96b2c](../management/issues/issue-20261113-b96b2c.md)）を直すこと | 別の関心。出荷設定の正規化は素通しで、等価性の確かめに乗らない。制約ではなく分担で、それぞれの課題が持つ |
| NG3 | RND の観測加工を、同じ仕組みに載せること（[issue-20261122-ba42c9](../management/issues/issue-20261122-ba42c9.md)） | 別の関心。本 RFD で、依頼の種類を RL の側が定義でき、部品を登録できるようになるので、載せる準備にはなる。制約ではなく分担で、課題が持つ |
| NG4 | ReplayBuffer の重ね直しの方式を変えること。1 枚のとき stack 軸を付けずに返す動きを変えること | Rainbow が、同じ ReplayBuffer を 1 枚で使い、Network を stack 軸の無い spec で作っている（P31）。足すのは、束の spec を答える口だけである。1 枚のときの軸は、`FrameStack` の部品か、呼び出し元が足す（§6.4、D11）。移せない制約ではなく、Rainbow へ波及させないための線である |
| NG5 | Network を `SpecForwarder` にすること。nn.hpp の口を変えること | 出力の spec を導くには、Head まで仮の forward を通すことになる（ADR 0084 の先送りのまま）。口の作り直しとは別の関心である。制約ではなく、Network の出力の spec を使う側が出たら、別の RFD で足す |
| NG6 | Rainbow・MuZero・ImageCls の観測の加工を、この仕組みに載せること | 載せても消える重複が無い（RFD0161 の NG7 と同じ）。制約ではなく、範囲の線である |

## 3. 範囲

| # | 入れる仕組み | 効くゴール |
|---|---|---|
| 1 | 実行するものを `Forwarder` の 1 つにし、episode の開始の印を入力に含める。`ForwardExecutor` を無くす | G1・G2 |
| 2 | 作る側（`ForwarderBuilder` と部品）と、作られるもの（`SpecForwarder`）を分ける。`ForwarderProvider` を無くす | G1 |
| 3 | 依頼の型 `ForwarderRequest` と、種類の名前 `ForwardKind`。種類は RL の側（`anet::rl`）が定義する | G1・G2 |
| 4 | 部品の種類の登録制。部品の工場が、扱う種類を申告する | G2 |
| 5 | spec を、依頼ごとに渡す。作った Forwarder が、入力と出力の spec を答える | G3 |
| 6 | ReplayBuffer が、Sample で返す束の spec を答える口 | G3 |
| 7 | 重ねる設定を、Agent 全体の 1 か所（`frame_stack.*`）に置く | G3 |
| 8 | 並びの決まりを外す。stack 軸の確保と、残った開始の印の取り除きを、呼び出し元（DefaultDQN）が受け持つ | G2 |
| 9 | spec の宣言を設定で変える部品 `SetSpec` | G2（正規化の対象を spec の宣言だけで決めるための手段。D18・D19） |
| 10 | 図を、作った Forwarder が描く | G1・G2 |
| 11 | 3 つの Perception の出力の spec を、作った側が比べる | G3（成り立たない設定を、構築で止める） |

入れないものは、§2 の非ゴールのとおり。

## 4. 前提

種類は、事実・用語・既存部品の働き・過去の判断・守る線。P1〜P13 は、ゴールの合意のときに示した共通の前提（2026-11-24。説明が要る番号は挙がらなかった）。P14 以降は、hpp の草案の検討で問いに添えた前提と、起草で足した前提である。

| # | 種類 | 内容 | 出典 | 確かめ |
|---|---|---|---|---|
| P1 | 事実 | 今の公開の口は 6 つの型である | §1.1 | 済み |
| P2 | 事実 | `ForwardExecutor::Forward` が `episode_start` を引数に取り、依頼の種類は `anet` の層の列挙である | §1.2 | 済み |
| P3 | 事実 | spec は構築のときの 1 つだけを受け、`ForwardSpec` は依頼なしで答える | §1.4 | 済み |
| P4 | 事実 | 並びは決まりを持ち、部品の種類は 3 つに固定である | §1.2、§1.3 | 済み |
| P5 | 既存部品の働き | stack の部品は、依頼の種類で動きが違う。経験の受入れでは何もしない。Sample では、枚数が 2 以上なら何もせず、1 枚のときだけ長さ 1 の軸を足す。可視化では、同じ 1 枚を枚数ぶん並べる | [forwarder.cpp](../../src/core/base/src/forwarder.cpp) 135〜177 行 | 済み |
| P6 | 事実 | 重ね方は、Agent が並びに聞いて、Learner 経由で ReplayBuffer へ渡している | §1.5 | 済み |
| P7 | 既存部品の働き | ReplayBuffer は、1 枚のとき stack 軸を付けずに返す。設定の `stack_keys` が空なら、全キーを重ねる | [replay_buffer_impl.cpp](../../src/core/base/src/replay_buffer_impl.cpp) 1295 行、[replay_buffer.hpp](../../src/core/base/include/anet/replay_buffer.hpp) 79〜80 行 | 済み |
| P8 | 事実 | 重ねる設定は `perception.block.[stack].count` と `keys.mode`・`keys.listed.names`。出荷設定の並びは `stack > norm` の 1 通りで、重ねる設定を書いているのは 4 ファイルである | [agent.txt](../../apps/runner/config/agent.txt) 194〜197 行、§6.9 | 済み |
| P9 | 既存部品の働き | 図は `ForwarderChain::MakeGraphViz` が作り、名前は `perception`・`Frame`・`Perception` に固定である | [forwarder.cpp](../../src/core/base/src/forwarder.cpp) 715〜754 行 | 済み |
| P10 | 事実 | RFD0161 は実装済みで、その課題 2 件は解決である。実装のあとのレビュー対応のコミット（`b1d3e576`）は、forwarder.cpp とテストを直したが、forwarder.hpp は変えていない | [0161_observation_pipeline_10rfd.md](0161_observation_pipeline_10rfd.md)、`git show --stat b1d3e576`（2026-11-24 に確認） | 済み |
| P11 | 過去の判断 | ADR 0084 は「履歴を使う依頼は、行動選択用の実行器が受ける」と決め、ADR 0086 は「重ねる部品を設定に書かなくても、1 枚の重ねが入る」と決めた。ADR 0086 と用語集は、「stack の対象のキーは、1 枚でも必ず stack 軸を持つ」を、Perception の決まりとして書いている | [ADR 0084](../adr/0084-forwarder-is-common-interface-and-chain-serves-requests-named-by-input-and-product.md)、[ADR 0086](../adr/0086-perception-stack-target-keys-always-carry-stack-axis.md) | 済み |
| P12 | 守る線 | 観測加工を変えるときは、4 環境で、変更の前後とも同じ seed と決定的なバックエンドを指定し、学習 Run の scalar と trace を照合する | [314_tensor.md](../wiki/categories/314_tensor.md) の「変えるときの条件」、[0161_observation_pipeline_10rfd.md](0161_observation_pipeline_10rfd.md) の D34・D35・§9 | 済み |
| P13 | 事実 | 関係する未解決の課題が 3 件ある（統計の読み書きの競合、統計の checkpoint、RND の観測加工） | §2 の NG2・NG3 | 済み |
| P14 | 既存部品の働き | 今ある依頼の型は、「作るもの＋Request」で名付けている（`ActorRequest` を `Agent::CreateActor` が受ける）。動詞を付けた Request の型は、C++ のコードに無い（`src` を `(struct|class) (Create|Build|Make|Get)\w+Request` で検索して、Java の 1 件だけ） | [rl.hpp](../../src/core/base/include/anet/rl.hpp) 760・843 行 | 対象外 |
| P15 | 既存部品の働き | `MakeGraphViz` という名前の口は 2 つ（`ForwarderChain`・`nn::Network`）で、どちらも必ず図を返す。テスト以外の呼び出し 12 か所は、空を確かめずに使う。未対応なら空を返す約束があるのは、別の口 `graphviz::GraphVizProvider::CreateGraph` で、呼ぶ側（`GraphVizObserver`）が確かめる | [nn.hpp](../../src/core/base/include/anet/nn.hpp) 233 行、[graphviz.hpp](../../src/core/base/include/anet/graphviz.hpp) 184 行、[observers.cpp](../../src/core/base/src/observers.cpp) 1225〜1226 行 | 対象外 |
| P16 | 既存部品の働き | NN は、全体の図を `Network::MakeGraphViz`、部品の値を `NetworkHead::GetGraphVizInfo`（既定は空）で答える | [nn.hpp](../../src/core/base/include/anet/nn.hpp) 233 行、[nn_head.hpp](../../src/core/base/include/anet/nn_head.hpp) 158 行 | 対象外 |
| P17 | 過去の判断 | RFD0161 の前は、重ねる設定は Agent 全体の設定（`stucker.*`）で、Agent が、行動選択・可視化・Network の入力の spec・ReplayBuffer へ配っていた。RFD0161 の D25 が、並びの部品の設定へ移した。理由は、部品の置く位置を設定で書けるようにし、書き方を NN とそろえるためである | [0161_observation_pipeline_10rfd.md](0161_observation_pipeline_10rfd.md) §6.4 の D25 | 対象外 |
| P18 | 事実 | Dopamine の DQN は、Agent が枚数（`stack_size`）を引数で受け、自分の状態の形と、ReplayBuffer の構築の両方に使う | [dqn_agent.py（JAX 版）](https://github.com/google/dopamine/blob/master/dopamine/jax/agents/dqn/dqn_agent.py)（2026-11-24 に確認） | 対象外 |
| P19 | 既存部品の働き | 今の `FrameStack` の部品は、`keys.mode` が `all` のとき、自分の位置に届いた spec の全キーを重ねる。`listed` の名前が spec に無ければ、エラーにする | [forwarder.cpp](../../src/core/base/src/forwarder.cpp) 430〜455 行 | 対象外 |
| P20 | 事実 | 設定の基準の検証（固定した 17 構成の解決結果の比較）の期待値 17 ファイルが、今の重ねる設定のキーを含む | [README.md](../../src/core/base/testdata/rfd0072/README.md)、§6.9 の割り振り | 対象外 |
| P21 | 既存部品の働き | NN の部品の登録は、type の名前と工場を、登録先に入れる形である | [nn_impl.hpp](../../src/core/base/src/nn_impl.hpp) 257〜279 行 | 対象外 |
| P22 | 既存部品の働き | 並びに選ばれた block の設定は閉じた範囲で、block の中の読まれないキーは、構築でエラーになる（NN の block と同じ）。選ばれていない block と、並びの外の読まれないキー（RFD0161 が移した `stucker.*` など）は、起動時の WARN の一覧に出る | [forwarder.cpp](../../src/core/base/src/forwarder.cpp) 646 行、[config.cpp](../../src/core/base/src/config.cpp) 382 行、[forwarder_test.cpp](../../src/core/base/src/forwarder_test.cpp) 291〜311 行 | 対象外 |
| P23 | 既存部品の働き | Actor は、観測と `episode_start` を実行器へ渡して、Perception を受ける | [dqn_actor.cpp](../../src/algo/dqn/src/dqn_actor.cpp) 1503 行 | 対象外 |
| P24 | 事実 | 出荷設定は、`DropKeys` を使っていない。並びは `stack > norm` だけである（`apps/runner/config`・`apps/*.bat`・`data` を `perception.structure` と `DropKeys` で検索して、agent.txt の 1 行だけ）。正規化は、どれも素通し（`mode` が `none`）である。このため、等価性の Run（G4）では、並べ方の自由・統計を取るキーの決まり・`SetSpec` の動きは見えず、単体テストだけが確かめる | [agent.txt](../../apps/runner/config/agent.txt) 194・322 行 | 対象外 |
| P25 | 既存部品の働き | 今の統計の入れ物は、Perception の流れで `Normalize` の位置に届く spec の、`Unscaled` の float のキーから作る。正規化の部品は、入れ物に無いキーを、黙ってそのまま通す | [forwarder.cpp](../../src/core/base/src/forwarder.cpp) 539〜548 行、[scaler_impl.cpp](../../src/core/base/src/scaler_impl.cpp) 622〜638 行 | 対象外 |
| P26 | 既存部品の働き | `Normalize` は、正規化したキーの値の尺度を `Scaled` にする。このため、流れによって正規化するかどうかが違っても、出力の spec は同じになりうる | [forwarder.cpp](../../src/core/base/src/forwarder.cpp) 321〜332 行 | 対象外 |
| P27 | 既存部品の働き | 重ねる履歴の部品（`DictFrameStacker`）は、開始の印が CPU にある前提で、device の同期を避けている。履歴を置く device は、構築で受ける | [stacker.cpp](../../src/core/base/src/stacker.cpp) 33〜41 行 | 対象外 |
| P28 | 既存部品の働き | Network の側は、行の開始の印を、呼び出しごとの情報（`NetworkForwardCallInfo`）の型つきの値で受ける。`anet` の層のうち、nn.hpp には `episode_start` の名前がある | [nn.hpp](../../src/core/base/include/anet/nn.hpp) 148〜164 行 | 対象外 |
| P29 | 既存部品の働き | 評価用の Actor は、Run の途中でも作られる | [trainer.cpp](../../src/core/base/src/trainer.cpp) 1258〜1270 行 | 対象外 |
| P30 | 既存部品の働き | 観測の spec の検査（`StateSpec::AssertSanity`）は、整数の dtype と離散のキーの `Unscaled` をエラーにし、`Vector` に 1 次元の形を求める。`labels` の数の検査（`TensorSpec::HasValidLabels`）は、stack 軸を形の先頭に数える | [rl.cpp](../../src/core/base/src/rl.cpp) 158〜186 行、[common.hpp](../../src/core/base/include/anet/common.hpp) 134〜146 行 | 対象外 |
| P31 | 事実 | `ReplayBuffer` の口は rl.hpp にあり、継承するクラスは 5 つある（本体の `DefaultReplayBuffer`・`PrefetchingReplayBuffer` と、テスト用の差し替え 3 つ）。作る関数（`CreateReplayBuffer`）を呼ぶのは dqn の Learner だけで、Rainbow も同じ Learner を使う。MuZero は、自前の ReplayBuffer を使う | [rl.hpp](../../src/core/base/include/anet/rl.hpp) 998 行、[rainbow_agent.cpp](../../src/algo/dqn/src/rainbow_agent.cpp) 66〜80 行、`src` を `public (anet::)?(rl::)?ReplayBuffer` で検索（2026-11-24。6 行。うち 1 行は `muzero_rb.hpp` のコメントで、継承するクラスは 5 つ） | 対象外 |
| P33 | 事実 | Actor は、行動に添える情報の中で、行の開始の印を `episode_start` の名前で運んでいる | [dqn_actor.cpp](../../src/algo/dqn/src/dqn_actor.cpp) 1599 行 | 対象外 |
| P34 | 既存部品の働き | 設定の値が、設定キーの型として読めないとき（整数や数として読めない）は、設定の読み取りがエラーにする。文字列の一覧は、空白で区切って読むので、型に反する値が無い | [config.cpp](../../src/core/base/src/config.cpp) 61〜75 行・429〜438 行・529〜536 行 | 対象外 |
| P35 | 事実 | 設定の基準（17 ファイル）は、キーの並び（`map_order`）と、キーと値（`values`）を持つ。重ねる設定の旧いキーは、どのファイルでも、並びと値に同じ数だけあり（2〜4 個）、並びに重複は無い。`resolution/` の期待値には、旧いキーは無い。RFD0161 は、固定コミットを保ち、移したキーと値だけを解決結果から採って、基準を直した | [README.md](../../src/core/base/testdata/rfd0072/README.md) の「golden の更新方針」、17 ファイルを読んで数えた（2026-11-24） | 対象外 |
| P36 | 守る線 | 拡張点を基盤の外で書けるようにする変更では、interface・部品・登録口を公開し、拡張点の表と設計文書を同時に更新する。設定の特定の値に、別の意味を持たせない | [AGENTS.md](../../AGENTS.md) の「基本方針」と「設定値の扱い」、[010_framework_overview.jp.md](../design/010_framework_overview.jp.md) の「4.3 拡張点」 | 対象外 |
| P32 | 用語 | Frame は、Observation に、Agent が足した情報を合わせたものである。今は足す情報が無いので、Frame の spec は Env の観測の spec と同じである | [CONTEXT.md](../../CONTEXT.md) の「Frame」 | 対象外 |

## 5. 確定事項

「依存」は、理由がよりどころにする前提（P）と、ほかの決定（D）を挙げる。発言の日付と原文は §10.1 にある。退けた案は §5.6 にある。

### 5.1 型と口

| # | 決定 | 理由 | 依存 |
|---|---|---|---|
| D1 | 実行するものは `Forwarder` の 1 つにする。行ごとの履歴を使う変換も、依頼で作った Forwarder が受ける。episode の開始の印は、入力の `TensorDict` に鍵として含め、鍵の名前は依頼を作る側が決める（DefaultDQN は `episode_start` にする）。`ForwardExecutor` は無くす。開始の印は、CPU に置いた真偽の値のまま渡し、device へ移さない。鍵の名前が Frame のキーと重なれば、Agent が構築でエラーにする | 実行の口が 2 つあると、使う側が使い道で口を使い分ける。開始の印を口の引数から外せば、forwarder.hpp の口に RL の概念が残らない。重ねる履歴の部品は、開始の印が CPU にある前提で、device の同期を避けている（device へ移せない制約ではなく、同期を避けるための選択である）。Network の側は、開始の印を呼び出しごとの型つきの値で受けていて、形がそろわなくなるが、`Forwarder` の口は入力と capture だけにする。鍵の名前は、Actor が行動に添える情報で使っている名前にそろえる | P1、P2、P23、P27、P28、P33 |
| D2 | 定義を持つ側（並びと部品）は、Forwarder を兼ねない。どの使い方も、依頼を出して Forwarder を受け取る | 「Forwarder でもあり、Forwarder を作るものでもある」は、責務が 2 つになる | P1 |
| D3 | 並びのクラスの名前は `ForwarderBuilder`、作る口は `BuildForwarder` とする | 設定から組み立てて作るものなので Builder と呼ぶ。作る口は、クラスの名前に合わせる | D2 |
| D4 | 依頼で作ったものは `SpecForwarder` で、入力の spec と出力の spec を答える。依頼なしで spec を答える口（今の `ForwardSpec`）は無くす | `SpecForwarder` のインスタンスがあることは、どういう入力からどういう出力を作るかの様式が決まっていることと同じである。依頼が決まらないと、どの spec かが決まらない | P3、D2 |
| D5 | 依頼の型は `ForwarderRequest` で、Forward の種類の名前（`ForwardKind`。中身は文字列）と、付帯の情報を持つ。`anet` の層は、種類を解釈しない。種類は RL の側（`anet::rl`）が、何を渡して何が欲しいかで名付けて定義する。今の列挙 `ForwardRequest` は無くす | この型は、`Forward` への依頼ではなく、Forwarder を作ってもらうときの依頼である。今ある依頼の型と同じ付け方（作るもの＋Request）にする。種類の名前は、作った Forwarder が行う Forward の種類を表すので `ForwardKind` とする | P2、P14 |
| D6 | 部品の種類は登録制にし、設定の `type` の名前で選ぶ。部品の工場が、扱う `ForwardKind` を申告する。種類を問わない部品は、問わないと申告する。同じ `type` の二重登録と、登録に無い `type` の設定は、エラーにする。標準の部品は、基盤の初期化（NN の標準の module を登録するのと同じ所）で登録する。部品の interface と登録口を公開ヘッダに置くので、設計文書の拡張点の表に行を足す（§6.9） | 公開の拡張点は、表と設計文書を同時に更新する決まりである。NN と同じ形にする。設定で選ばれる具象のクラスが RL に依ることは、問題にしない。使える種類を工場に聞けば、設定に書いた種類の名前の打ち間違いを、構築で止められる | P4、P21、P36 |
| D7 | ヘッダは 1 つ（`anet/forwarder.hpp`）のまま、中を「使う側が触るもの」と「部品を作る側だけが触るもの」の節に分ける。使う側と、部品を作る側で、ヘッダを分ける案は採らない | 機能を使うときも作るときも、その機能のヘッダ 1 つを読めば足りるようにする。分けられない制約ではなく、フレームワークの方針である | D2 |
| D8 | `ForwarderBuilder` は、並びの決まりを持たない。設定に書いた並びを、そのまま組む。「`FrameStack` は先頭に 1 つ」「`Normalize` は 1 つまで」「`FrameStack` が無ければ足す」を外す | 設定で自由に並べられる仕組みに、並べ方の決まりは要らない。設定に書いたとおりに動くのが、正しい動きである。成り立たない並びは、部品の検査か、呼び出し元の比較（D21）で止まる | P4 |
| D9 | spec を部品に通すのは、依頼が来たとき（`BuildForwarder`）だけにする。構築では、設定から部品を作り、spec に依らない設定の誤りだけを検査する | spec が依頼ごとになる（D10）ので、構築の時点では、部品に届く spec が決まらない | D10 |
| D10 | spec は、構築ではなく、依頼ごとに `BuildForwarder` へ渡す。渡すのは、作った Forwarder の `Forward` に実際に入れるものの spec である。部品は、届いた spec に合わせて作り、扱えない spec はエラーにする。行動選択の依頼では、開始の印の鍵の spec も含める | 実際に入れるものの形と合った spec を渡すのが、筋である。依頼ごとに、渡すものが違う | P3、§1.4 |
| D11 | 「stack の対象のキーは、1 枚でも必ず stack 軸を持つ」は、Forwarder の仕組みではなく、使う側（DefaultDQN）の前提として確保する。呼び出し元は、並び全体の最終出力の spec を見て、重ねる対象のキーのうち、出力に残っていて stack 軸の無いものに、長さ 1 の軸を足す。対象でないキーには足さない。`FrameStack` の部品が軸を付ける動きは、変えない（今と同じく、自分の位置で軸を付ける。§6.4。枚数とキーを受け取る先は、D15 で変わる）。観測を Actor の device へ移すのも、呼び出し元が行う | stack 軸を必ず持つことは、RL の側の決まりで、Forwarder の仕組みが縛ることではない。確保する手段を `FrameStack` の部品に限る必要も無い。対象でないキーに足すと、Network の入力の形が変わる。`FrameStack` の部品が自分の位置で軸を付ければ、3 つの Perception の依頼で、そこに届いた対象キーはその出力で stack 軸を持つ（部品の動きを変えられない制約ではなく、この理由で選んだ） | P5、P7、P11、D8 |
| D12 | 並びに、開始の印を読む部品（`FrameStack`）が無いときは、出力に残った開始の印を、呼び出し元（DefaultDQN）が取り除く。Agent が、作った Forwarder にかぶせる処理として持つ（§6.5） | 枚数 1 で `FrameStack` を書かない並びを動かすため。D11 と同じ考え方で、使う側の前提は使う側が満たす | D1、D8、D11 |

### 5.2 spec の流れと重ね方

| # | 決定 | 理由 | 依存 |
|---|---|---|---|
| D13 | `FrameStack` は、並びの部品のままにする。並べ方は自由で、`norm > stack` も `stack > norm` も書ける。Sample の依頼では、届いた束のうち、既に重なっているキーは、そのまま通す | 重ねる位置を設定で書けることは、残す。重ねる前に落とす・宣言を変える部品を置ける（`drop > stack`、`set > stack > norm`）。並びの順で動きが違ってよく、そう書いた設定なら、それが正しい動きである。統計つきの正規化を `norm > stack` に置くと、行動選択の履歴に、その時点の統計で正規化した Frame がたまり、1 つの Perception の中で古い統計と新しい統計が混ざる。これも、そう書いた設定の動きである。重ねるのを並びの外へ出せば、Forward の種類は 2 つで済み、開始の印も並びに入らないが、位置を書けなくなる（§5.6）。移せない制約ではなく、並べ方の自由を選んだ | D8、P5 |
| D14 | Sample で渡す束の spec は、ReplayBuffer が答える。ReplayBuffer を持つ側（Learner）が、その spec を Sample 用の `BuildForwarder` に渡す。並びに枚数とキーを聞く口（`GetStackCount`・`GetStackKeys`）は無くす。ReplayBuffer が受ける 1 枚の spec は、Frame の spec（§6.5 の F）のままにする。経験の受入れ用の Forwarder の出力の spec が、F とキー・形・dtype で同じことを、Agent が構築で確かめる | 束を作るのは ReplayBuffer なので、ReplayBuffer が主で、Forwarder は従である。行動選択の側が Env の spec を受けるのと、同じ関係になる。spec は、データと同じ向きに流れる。経験の受入れは、値もキーも変えない。その出力の spec（並びを 1 度通った宣言）から Sample の流れを始めると、宣言を変える部品が、Sample の流れにだけ 2 度効く。たとえば `norm > set` で、`SetSpec` が `Unscaled` のキーを `Scaled` にすると、行動選択では正規化され、Sample では正規化されない。F のままにするのは、変えられない制約ではなく、これを避けるための選択である | P6、P26、D10 |
| D15 | 重ねる枚数とキーの設定は、Agent 全体の 1 か所に書く。Agent が、ReplayBuffer へは構築の設定で（Learner 経由。今と同じ渡し方）、`FrameStack` の部品へは依頼の付帯の情報で配る。`FrameStack` の部品は、枚数とキーの設定を持たず、並びの中の「重ねる位置」だけを表す。RFD0161 の D25 のうち、重ねる設定を部品の設定に置いた部分を戻す | 行動選択と ReplayBuffer の両方が使う情報なので、Agent 全体の設定が正本になる。枚数は、1 つの部品の性質ではない。部品は、今も行の数と開始の印を依頼から受け取るので、依頼から受け取るものが増えるだけである | P6、P17、P18 |
| D16 | 設定キーは、Agent の直下の `frame_stack.count`・`frame_stack.keys.mode`・`frame_stack.keys.listed.names` とする。値の名前・型・値域・既定は、今の部品の設定を引き継ぐ（§6.6）。今のキー（`perception.block.[stack].count`・`keys.mode`・`keys.listed.names`）は無くす | `frame_stack` は、部品の `type` の名前（`FrameStack`）と同じ語である。`perception` の下は、並びを組む側が読む場所なので、Agent が読むキーを置かない。`stack` だけでは、並びの block の名前や、NN の `StackMerge` と紛れる。値の側は、置き場を移すだけで、意味を変えない（変えると、G4 の等価性で確かめられなくなる。変えられない制約ではなく、範囲の線である） | D15、P8 |
| D17 | 重ねるキーの名前は、Agent が決めて配る。`keys.mode` が `all` なら、Frame の全キー。`listed` なら、一覧に書いた名前で、Frame に無い名前があれば、Agent が構築でエラーにする。ReplayBuffer と依頼には、同じ名前の一覧を渡す。`FrameStack` の部品は、渡された名前のうち、自分の位置に届いたキーを重ねる。並びの途中の部品が作ったキーは、重ねる対象に選べない | 設定の持ち主が Agent になる（D15）ので、名前に直すのも Agent が行う。無い名前を構築で止める今の動きを保つ。ReplayBuffer が重ねるのは Frame なので、選べるのは Frame のキーである。前の部品が落としたキーは、届かないので重ねない。出荷設定はどれも `FrameStack` が先頭なので、動きは変わらない | D15、P19、P24、P32 |

### 5.3 部品

| # | 決定 | 理由 | 依存 |
|---|---|---|---|
| D18 | 正規化の部品が統計を取るかは、キーの spec の宣言（値の尺度が `Unscaled` の float）だけで決める。そのキーが Perception に出るか（あとの部品が落とすか）では決めない。宣言を見るのは、経験の受入れの流れで部品に届いた spec である（D28） | 部品に届くことと、統計が要ることは別である。Perception の流れで届いたキーで決めると、落とすキーの置き方で対象がずれる | D9、D10 |
| D19 | spec の宣言を設定で変える部品 `SetSpec` を、標準の部品に足す。変えられる属性は、値の尺度（`scale`）、範囲の宣言（`min_values`・`max_values`）、`labels` で、データには触らない。当てる `ForwardKind` を設定で絞れる（既定は全部）。絞った設定は、書いたとおりに動き、絞ったこと自体はエラーにしない。Perception を作る種類の一部にだけ当てた結果、出力の spec が食い違えば、D21 の比較で止まる。食い違わなければ、行動選択と Sample で正規化する対象がずれたまま動く（§6.6）。`set` にした属性の一覧が空の設定は、エラーにする（空の一覧に、宣言を消す意味を持たせない）。設定キーとエラーは §6.6・§6.7 | 正規化の対象を spec の宣言だけで決める（D18）ので、宣言を設定で変える手段を先に用意する。あとで小さな実装を足すことにならないようにする。絞る口も、同じ考え方で先に入れる。別モデルの検討は、値が黙って食い違う設定が書けること（P26）を挙げて、絞る口を外す案を出した。開発者に聞いて、残すと決まった（2026-11-24。§5.6、§10.1）。絞った設定の動きは、設定に書いたとおりに動くのが正しい、という D8 と同じ扱いにする。絞る口は、外せない制約ではなく、選んで残す。設定の特定の値に別の意味を持たせない決まりにより、空の一覧を「消す」と読まない | D18、D6、D8、P26、P36 |
| D20 | 標準の `FrameStack`・`Normalize`・`DropKeys` は、4 つの `ForwardKind`（§6.4）を全部扱う。`SetSpec` は、種類を問わない。並びの部品のどれかが、依頼の `ForwardKind` を扱わないときは、`BuildForwarder` がエラーにする | 知らない依頼を、黙って素通しにしない | D5、D6 |

### 5.4 図と、出力の確かめ

| # | 決定 | 理由 | 依存 |
|---|---|---|---|
| D21 | 行動選択・Sample・可視化の 3 つの Forwarder の出力の spec（D11 の軸を足し、D12 の印を除いたあと）が同じであることを、Agent の構築のうちに確かめる。可視化用の出力の spec を Network の入力に使い、Sample 用は Learner が作った時点で、行動選択用は、Agent が行の数 1 の依頼で作って、これと比べる。Actor を作るときは、比べ直さない。キーの集合か、キーごとの spec の属性のどれかが違えば、エラーにする。確かめるのは、形と宣言の食い違いである。値の食い違いは、往復のテスト（§7 の 6）が確かめる | 3 つとも同じ Network に入る。並びの決まりを外した（D8）ので、成り立たない設定（枚数が 2 以上なのに `FrameStack` の部品が無い、など）を、最初の学習や行動選択まで潜ませない。行動選択用の出力の spec は、行の数にも seed にも依らないので、Actor を作るときまで待たない。評価用の Actor は Run の途中でも作られるので、待つと、Run の途中で初めて止まる | D8、D11、D12、D13、P26、P29 |
| D22 | 図（dot）は、作った Forwarder が描く。`ForwarderBuilder` は描かない | Builder は工場であり、作る製品の情報を工場に書かせると、予定の情報になる。出来た製品の情報は、製品に聞く | D2、P9 |
| D23 | 図の口は `SpecForwarder` に足し、図の口のための型は足さない。全体の図を作る口（`MakeGraphViz`）は、対応しないものが空を返す（既定）。呼ぶ側は、空かどうかを確かめてから使う。`ForwarderBuilder` が作ったものは、必ず図を返す。図のノードに書く自分の値を答える口も `SpecForwarder` に足し、既定は空で、部品が作る Forwarder が上書きする。`Network::MakeGraphViz` は必ず図を返すので、同じ名前で約束が違う。違いは、口の説明に書く（§6.2） | 型の数を増やさない。未対応なら空を返す約束は、今ある図の口（`CreateGraph`）と同じである | P15、P16、D22 |
| D24 | 図の題は、依頼の `ForwardKind` の名前にする。両端のノードは `input`・`output`、部品のノードは block の名前と `type` にする。詳しい図には、seed、部品が答えた値、ノードごとの spec を書く。Agent が記録する名前（`perception.structure`・`perception.detail`）は変えず、可視化用の Forwarder の図を記録する | `anet` の層に RL の語を置かない。名前を渡す口を足さずに済む。記録する名前を変えなければ、Run の `dot/` の下のファイル名は変わらない（変えられない制約ではなく、Run の出力の名前を保つための選択である） | P9、D22 |

### 5.5 確かめ方・文書・進め方

| # | 決定 | 理由 | 依存 |
|---|---|---|---|
| D25 | 等価性は、RFD0161 と同じ比べ方で確かめる（§7 の「等価性の比べ方」）。LunarLander・GridMaze・DropMerge・Atari で、変更の前後の exe を、同じ設定・同じ seed・決定的なバックエンドで 1 本ずつ流し、全部の scalar と trace が一致すること | 観測加工を変えるときの条件による | P12 |
| D26 | 新しい RFD として起こす。RFD0161 は改訂しない。実装は、段階に分けずに 1 つとして進める | RFD0161 の仕事（観測加工の定義の一元化）は済んで、課題も解決している。口の作り直しは目的が別なので、番号を分けたほうが経緯を追える（RFD0161 を改訂できない制約ではなく、経緯の追いやすさで選んだ）。口の入れ替えは、途中の形が意味を持たない | P10 |
| D27 | 旧い口（`ForwardExecutor`・`ForwarderProvider`・`ForwarderChain`・列挙の `ForwardRequest`・`ForwardSpec`・`GetStackCount`・`GetStackKeys`）と、旧い設定キー（D16）に、別名も互換の読みも残さない。旧い設定キーを、並びに選ばれた `FrameStack` の block に書くと、block の中の読まれないキーとして、構築でエラーになる。今ある仕組みのままで、旧いキーのための検出も、旧いキーの名前を書くテストも足さない（block の中の読まれないキーの検査は、今あるテストが確かめている） | クリーンブレーク方針による。block の設定は閉じた範囲なので、旧いキーが黙って読まれないことは無い（たとえば、旧いキーで枚数 1 と書いた設定が、基準のプロファイルの 4 枚で動き出すことは無い） | P22 |
| D28 | 可視化用の依頼（`PerceptionFromFrame`）は残す（可視化の動きを変えないための線で、移せない制約ではない）。統計つきの正規化の統計の入れ物は、経験の受入れ用の依頼（`FrameWithStatsUpdate`）の `BuildForwarder` のうちに、届いた spec の、`Unscaled` の float のキーから作る（stack 軸があれば、それを除いた 1 枚ぶんの形）。`Forward` の中では作らない。経験の受入れ用の依頼をもう一度受けても、入れ物は作り直さず、届いたキーか 1 枚ぶんの形が入れ物と違えば、エラーにする。Perception を作る依頼で、`Unscaled` の float のキーが届いたのに、入れ物に無い（経験の受入れ用の依頼がまだ来ていない場合を含む）か、1 枚ぶんの形が違えば、エラーにする。このため、統計を取るキーは今と変わる。今は、Perception の流れで `Normalize` の位置に届くキーで、新しくは、経験の受入れの流れで届くキーである（`DropKeys` が `Normalize` より前にある並びでは、落とすキーにも統計を取る。取らせたくなければ、`SetSpec` で宣言を変える） | 可視化の動き（同じ 1 枚を枚数ぶん並べる）は変えない（NG1）。統計は、どの依頼で作った Forwarder にも共通の値で、部品が持つ。正規化の部品は、入れ物に無いキーを黙ってそのまま通すので、決まりが無いと、宣言は `Scaled` なのに値は正規化されていない、が起きる。入れ物を `Forward` の中で作ると、経験の受入れと行動選択が並んで走るときに、作る時点が重なりうる | D9、D10、D18、P5、P25 |
| D29 | ADR 0084 と ADR 0086 は書き換えず、先頭に注記を足して、新しい ADR を 3 本起こす。ADR 0094 は D1・D2・D5、ADR 0095 は D14・D15、ADR 0096 は D8・D11 を記録する。ADR 0084 のうち「履歴を使う依頼は、行動選択用の実行器が受ける」と「並びが、依頼ごとの Forwarder を返す Forwarder である」を改め、共通の口 `Forwarder` と、依頼を「何を渡して何が欲しいか」で名付けることは残す。ADR 0086 のうち「重ねる部品を書かなくても、1 枚の重ねが入る」を改め、「stack の対象のキーは、1 枚でも必ず stack 軸を持つ」は残す | 元に戻しにくく、文脈が無いと意図が読めず、案を比べて選んだ判断である。食い違う ADR は、どの判断と食い違うかを書く | P11 |
| D30 | `GetScalar`・`GetTensor`（メトリクスの鍵 `obs_norm.*` の読み口）は、`ForwarderBuilder` が部品へ順に問い合わせて、最初に答えた値を返す。今と同じである | 口の作り直しで、読み口を変えない（NG1。移せない制約ではなく、範囲の線である）。並びに `Normalize` を 2 つ書くと、後ろの部品の値は読めないが、設定に書いたとおりの動きである（D8） | D8 |

### 5.6 退けた案

記号は、検討のときに振ったものである。§2 のゴール（G1〜G4）と、§7 の検索の文字列（S1〜S21）とは、別の記号である。

| 案 | 中身 | 退けた理由 |
|---|---|---|
| episode の開始の印を、`Forward` の付帯の引数にする。または、型つきの引数のまま、名前だけ中立にする（Network の実行器が、呼び出しごとの型つきの値で受けているのと同じ形） | 口は 2 つのまま | 口が 1 つにならない（D1） |
| 並びが Forwarder を兼ねる（P） | 今の形 | 責務が 2 つになる（D2） |
| 依頼を、`anet` の層の列挙のままにする（L1・L2） | 種類を足すたびに、`anet` の層を変える | RL の概念が `anet` の層に残る（D5） |
| 依頼の型の名前を `ForwardRequest` のままにする（N1）、`BuildForwarderRequest` にする（N2） | N2 は、使う側の口の名前＋Request | N1 は「Forward の依頼」と読める。N2 は、部品の側の口（`CreateForwarder`）の名前と合わず、今ある型の付け方とも違う（D5） |
| 使える種類を、登録の関数で登録する | 種類の一覧を、別に登録する | 部品の工場が申告すれば足りる（D6） |
| 並びの決まりを残す（V1〜V3） | 「`FrameStack` は先頭」など | 設定で自由に並べられるのに、決まりで縛る理由が無い（D8） |
| 部品が、求める入力の spec を申告する（W1） | spec は 1 枚ぶんを渡し、Sample 用の Forwarder が束の spec を答える | 実際に入れるものの spec を渡す形（D10）のほうが筋である。申告の仕組みが増える |
| stack 軸の確保を、`FrameStack` の部品を足して行う（E1）、並びの最後に専用の部品を足す（E3） | Forwarder の仕組みが受け持つ | RL の側の決まりを、Forwarder の仕組みに持たせない（D11） |
| 束の spec を、依頼の種類で聞く（R1）、専用の口で聞く（R2）、部品の申告で聞く（R3） | どれも、並びが束の spec を答える | 束を作るのは ReplayBuffer である（D14） |
| 重ねるのを、並びの外へ出す（R4） | 呼び出し元が重ねてから、並びに通す。Forward の種類は 2 つ（統計の更新と、重ねた束から Perception を作る）で済み、開始の印は並びに入らず、stack 軸は重ねた時点で付き、3 つの出力の比較も要らなくなる。`FrameStack` を並びに書けなくなる | 重ねる位置を設定で書けることを残す（D13）。別モデルの検討も、比べる材料としてこの案を挙げた |
| ReplayBuffer が受ける 1 枚の spec を、経験の受入れ用の Forwarder の出力の spec にする | 起草のときに置いた形 | 宣言を変える部品が、Sample の流れにだけ 2 度効く（D14） |
| `FrameStack` の部品は Sample では常に素通しにし、1 枚のときの軸は、並びのあとで呼び出し元が足す | 起草のときに置いた形 | 枚数 1 のとき、`FrameStack` の後ろの部品が、依頼の種類によって違う形を受ける。今の部品の動きも変わる（D11） |
| 行動選択用の出力の spec を、Actor を作るたびに比べる | 起草のときに置いた形 | Run の途中で初めて止まりうる（D21） |
| 重ねる設定を、ReplayBuffer と部品の 2 か所に書く（S1） | 既定の設定で、値の参照で結ぶ | 1 つの事実に、正本が 2 つになる（D15） |
| 重ねる設定を、ReplayBuffer の 1 か所に書く（S2） | `learner` の下に置く | 両方が使う設定が、片方の下に入る（D15） |
| 設定キーを `perception` の下に置く（K2）、`stack` にする（K3） | — | K2 は、ReplayBuffer も使う設定が Perception の名前の下に入り、読む側が 2 つになる。K3 は、block の名前や `StackMerge` と紛れる（D16） |
| 正規化の対象を、並びの決まりで決める（H2）、Perception を作る依頼のときに届いたキーで決める（H3） | — | 届くことと、要ることを同じに扱うことになる（D18） |
| `SetSpec` を、値の尺度だけにする、形の意味の属性（`type`・`num_classes`・stack 軸の有無）も変えられるようにする | — | 範囲の宣言と `labels` までを入れる。形の意味の属性は、データと合わなくなるので変えさせない（D19） |
| `SetSpec` の、当てる Forward の種類を絞る口を外す（U2）。残して、Perception を作る 3 つの種類の一部にだけ当てる一覧を、構築でエラーにする（U3） | U2 は、いつも全部の種類に当たる。別モデルの検討が出した案である。落とすキーの統計を取らせない使い道は、全部の種類で `Scaled` にすれば書ける。U3 は、行動選択と Sample で当て方が違う設定を止める | 開発者が、絞る口を残すと選んだ（U1。D19）。U2 は、手段を先に用意しておく考え方（D19）から外れる。U3 は、`SetSpec` が Perception を作る種類の名前を知ることになり、種類を問わない部品（D20）でなくなる |
| 図を、`ForwarderBuilder` が描く（G1・G3） | G1 は依頼と spec を受けて描く。G3 は並びだけを描く | 工場に、製品の情報を書かせることになる（D22） |
| 図の口のために、型を足して分ける（J2・J3）、既定を「自分 1 つをノードにした図」にする（J1） | — | 型を増やさず、未対応は空を返す形にする（D23） |
| RFD0161 を改訂する、暫定で起こして別の枠でグリルする | — | D26。グリルに当たる往復は、起票の会話で済んでいる |

### 5.7 用語

[CONTEXT.md](../../CONTEXT.md) の「観測と可視化」の項目を、次のとおり直した。

| 語 | 直し |
|---|---|
| Forwarder | 「用途ごとの状態は持たない」を外した。依頼に合わせて作った Forwarder は、その使い道の状態（行ごとの履歴）を持てる（D1） |
| Forwarder の並び（`ForwarderBuilder`） | 「Forwarder をつないだもの」から、「部品を並べた定義で、依頼に合った Forwarder を作るもの。自分は Forwarder ではない」へ改めた（D2・D3） |
| Forwarder の部品（`ForwarderModule`） | 新しく足した。並びの 1 つぶんの定義である。既存の語「Forwarder」は実行するものを指すようになったので、定義の側を指す語が要る |
| Forwarder の依頼（`ForwarderRequest`） | 「Forward の依頼（ForwardRequest）」を改めた。Forwarder を作ってもらうときに渡すもの（D5） |
| Forward の種類（`ForwardKind`） | 新しく足した。今の「Forward の依頼」の中身（種類）を、こちらへ移した。種類は 4 つになる（§6.4）。既存の語「Forward の依頼」は、依頼そのものを指すように変わったので、種類を指す語が要る |

「束」（ReplayBuffer が重ねて返したもの）と「重ね方」（枚数とキー）は、本 RFD の中の言い方で、用語集には足さない。

新しい設定キーの語のうち、用語集の別の項目に当たるものは、次のとおりである。

| 語 | 用語集の項目 | 関係 |
|---|---|---|
| `labels`（`SetSpec` の設定キー） | 「targets」「class_names」が、避ける語に `labels` を挙げる | 別の概念である。ここでの `labels` は、`TensorSpec` の今の欄の名前（キーの要素やチャネルに付ける名前）を、そのまま使う |
| `scale`（`SetSpec` の設定キー） | 「値の尺度」（ValueScale） | 同じ意味である。「Q空間」が避ける語に挙げる value scale とは別である |
| `set`（`SetSpec` の `mode` の値と、その名前空間） | 「タグセット」 | 別の概念である。ここでの `set` は、「書いた値にする」の意味で、`keep`（変えない）と対にする |
| `frame`・`stack`（`frame_stack`） | 「Frame」「stack 軸」 | 同じ意味である（Frame を重ねる） |
| `Context`（`ForwarderModuleContext`） | 「Env name」が、避ける語に context を挙げる | 別の概念である。ここでは、部品を作るときに渡す情報を指し、NN の部品の同じ役の型（`ModuleContext`）と同じ付け方にする |
| `Module`（`ForwarderModule`） | 「Module Config」などがいう Module | 同じ意味である（`ModuleBase` を継ぐ、設定から作る部品） |

`PerceptionFromSequence` の `Sequence` は、行ごとに続く Frame の列を指す。`TensorSpec` の種類の `Sequence`（§6.6 の `labels` の検査に出る）とは別で、用語集にこの語の項目は無い。

開始の印の鍵の名前 `episode_start` は、用語集の「NN 呼び出し情報」と「履歴開始」の説明に出る episode_start と同じ意味（行の episode の開始を表す、入力の側の値）である。「履歴開始」が避ける語に挙げる「episode_startフラグ」（保存した値との混同）には当たらない。

## 6. 実装契約

### 6.1 クラス一覧

> 従えない事情があれば、実装する側が理由を実装計画に書いて変えてよい（質問しない）。ただし、一覧に無い機能グループを新しく立てるときは、作る前に開発者に確かめる。

| 名前 | 区分 | 責務 | 置き場（機能グループ） |
|---|---|---|---|
| `Forwarder` | 変更なし | `TensorDict` を受けて返す、実行の口 | `forwarder`（314_tensor。`src/core/base`） |
| `SpecForwarder` | 変更 | 入力の spec と出力の spec を答える。図を作る口と、図のノードに書く値を答える口を持つ。`ForwardSpec` を消す | `forwarder` |
| `ForwarderRequest` | 新規 | Forwarder を作ってもらうときの依頼。Forward の種類の名前と、付帯の情報を持つ | `forwarder` |
| `ForwarderBuilder` | 新規（今の `ForwarderChain` に当たる） | 設定に書いた部品の並びを組み、依頼に合った Forwarder を作る。メトリクスの読み口を部品へ回す | `forwarder` |
| `ForwarderModule` | 新規 | 並びの部品。変換の定義を持ち、必要に応じて並びで使える Forward の種類との組み合わせを検査する。依頼に合った Forwarder を作る | `forwarder` |
| `ForwarderModuleContext` | 新規 | 部品を作るときに渡す情報（seed と、この並びで使える Forward の種類） | `forwarder` |
| `ForwarderModuleFactory` | 新規 | 設定の `type` から部品を作る。扱う Forward の種類と受け付ける設定キーを申告する | `forwarder` |
| `ForwardExecutor` | 削除 | — | `forwarder` |
| `ForwarderProvider` | 削除 | — | `forwarder` |
| `ForwarderChain` | 削除 | `ForwarderBuilder` に替わる | `forwarder` |
| 列挙の `ForwardRequest` | 削除 | `ForwarderRequest` と `ForwardKind` に替わる | `forwarder` |
| 標準の部品（`FrameStack`・`Normalize`・`DropKeys`） | 変更 | `ForwarderModule` と工場の形に直す。`FrameStack` は、枚数とキーを依頼から受ける | `forwarder`（公開しない） |
| 標準の部品 `SetSpec` | 新規 | spec の宣言を、設定に従って変える | `forwarder`（公開しない） |
| `rl::ReplayBuffer` | 変更 | Sample が返す obs の spec を答える口 `GetSampleObsSpecs` を足す | `rl`（318_rl_core。`src/core/base`） |
| `DefaultReplayBuffer`・`PrefetchingReplayBuffer` | 変更 | `GetSampleObsSpecs` を実装する（後者は、包んでいる ReplayBuffer の答えを返す） | `replay_buffer`（318_rl_core。`src/core/base`） |
| `DefaultDQNAgent` | 変更 | 重ねる設定を持ち、依頼ごとに spec を渡して Forwarder を作る。stack 軸を足し、出力の spec を比べ、図を記録する | `default_dqn_agent`（324_dqn。`src/algo/dqn`） |
| `dqn::Actor` | 変更 | 実行器の代わりに、行動選択用の Forwarder を持つ。観測を device へ移し、開始の印を入力に足す | `dqn_actor`（321_agent_core。`src/algo/dqn`） |
| `dqn::Learner` | 変更 | ReplayBuffer を作ったあとに、束の spec を渡して Sample 用の Forwarder を作る | `dqn_learner`（321_agent_core。`src/algo/dqn`） |
| DefaultDQN の設定の構造体 | 変更 | 重ねる設定（`frame_stack`）を足す。Learner へ渡す重ね方の型（`ReplayFrameStack`）は、今のまま使う（D15 の、今と同じ渡し方による。変えられない制約ではなく、変える理由が無い） | `dqn_config`（321_agent_core。`src/algo/dqn`） |

新しい機能グループは立てない。`anet::rl` の依頼を作る関数と種類の名前は、`anet/forwarder.hpp` の `anet::rl` の節に置く（D7）。

### 6.2 公開の口（使う側が触るもの）

宣言の草案は §10.3 にある。名前と意味は、本節が正本である。

| 名前 | 意味 |
|---|---|
| `Forwarder::Forward(input, capture)` | `TensorDict` を受けて `TensorDict` を返す。今のまま |
| `SpecForwarder::GetInputSpecs()` | `Forward` に渡す `TensorDict` の spec。`ForwarderBuilder` が作ったものでは、`BuildForwarder` に渡した spec と同じである |
| `SpecForwarder::GetOutputSpecs()` | `Forward` が返す `TensorDict` の spec |
| `SpecForwarder::MakeGraphViz(detail)` | 自分の中身（入力の spec → 中の部品 → 出力の spec）を図にする。対応しないものは、空を返す（既定）。`ForwarderBuilder` が作ったものは、必ず図を返す。`Network::MakeGraphViz` は必ず図を返すが、こちらは空を返すことがある、と口の説明に書く |
| `SpecForwarder::GetGraphVizDetails()` | 詳しい図のノードに書く、実際に使った値（設定の値や、依頼から受け取った値）。既定は空で、部品が作る Forwarder が上書きする |
| `ForwardKind` | Forward の種類の名前。意味は、依頼を受ける部品が決める |
| `ForwarderRequest(forward_kind, options)` | 依頼。`GetForwardKind()` と `GetOptions()` で読む。付帯の情報（`options`）は設定データで運び、キーの名前と意味は、依頼を作る関数と、読む部品のあいだの約束である |
| `ForwarderBuilder(config_data, config_prefix, seed)` | 設定の `structure` の順に部品を作る。spec は受け取らない。選ばれた全 block について、設定を §6.7 の行順に検査する |
| `ForwarderBuilder::BuildForwarder(request, input_specs)` | 依頼に合った Forwarder を作る。空は返さない。依頼と spec を部品へ順に渡し、返ったものを並べて 1 つにする。先頭の部品には `input_specs`、次の部品には、前の部品が返したものの出力の spec が届く |
| `ForwarderBuilder::GetScalar`・`GetTensor` | 部品へ順に問い合わせて、最初に答えた値を返す（D30） |

### 6.3 公開の口（部品を作る側だけが触るもの）

| 名前 | 意味 |
|---|---|
| `ForwarderModule::ValidateForwardKinds(forward_kinds)` | 並びで使える Forward の種類と部品の設定の組み合わせを検査する。全 block の工場が部品を作り終えたあとに呼ぶ。組み合わせの検査が要らない部品は、何もしない既定のまま使う |
| `ForwarderModule::CreateForwarder(request, input_specs)` | 依頼に合った Forwarder を作る。空は返さない。`input_specs` は、この部品に実際に届くものの spec である。作ったものの `GetInputSpecs()` は、`input_specs` と同じにする。設定と spec の食い違いと、扱えない spec は、ここで検査して止める。その依頼では何もしないときは、素通しの Forwarder を返す |
| `MakePassThroughForwarder(specs)` | 入力をそのまま返す Forwarder を作る。入力と出力の spec は、渡した spec と同じである |
| `ForwarderModuleContext` | `seed`（block の名前から作った seed）と、`forward_kinds`（この並びで使える Forward の種類。並びの部品の全部が扱うもの。どの部品も種類を問わなければ、値なし） |
| `ForwarderModuleFactory::GetForwardKinds()` | この種類の部品が扱う Forward の種類の一覧。種類を問わない部品は、値なしを返す。`ForwarderBuilder` は、部品を作る前に、並びの工場の全部からこれを集める |
| `ForwarderModuleFactory::ReadConfigKeys(config_data)` | 受け付ける設定キーを文字列として読む。設定を持つ工場は、この口で受け付けるキーをすべて読む。設定を持たない工場は、何もしない既定のまま使う。Builder は未読キーを検査し、全 block が通ったあとに部品を作る |
| `ForwarderModuleFactory::CreateModule(config_data, context)` | 設定から部品を作る。部品の設定値の型・値域など、並びや spec に依らない誤りは、ここで検査して止める |
| `RegisterForwarderModuleFactory(type_name, factory)` | 部品の種類を、設定の `type` の名前で登録する |

部品が守ること:

- どの依頼で作った Forwarder にも共通の値（正規化の統計など）は、部品が持つ。用意するのは `CreateForwarder` のうちで、`Forward` の中では作らない（D28）。
- 自分が扱わないキーは、変えずに出力へ残す（開始の印の鍵のように、あとの部品か呼び出し元が読むキーがある）。
- 渡された Tensor のある device の上で動く。行動選択用の依頼の開始の印は、CPU に置いた真偽の値で届く。履歴を置く device は、重ねる対象のキーの Tensor から決める（D1）。
- 乱数を使うときは、自分の seed（`ForwarderModuleContext` の `seed`）と、依頼に載った seed を合わせて使う。標準の部品は、乱数を使わない。

### 6.4 Forward の種類と、標準の部品の動き

種類は 4 つで、`anet::rl` が定義する。名前は、何を渡して何が欲しいかで付ける。依頼は、種類ごとの関数（`Make<種類の名前>Request`）で作る。

| Forward の種類 | 渡すもの | 欲しいもの | 依頼に載せる値 |
|---|---|---|---|
| `FrameWithStatsUpdate` | 観測 1 枚 | 統計を更新しつつ、保存する Frame。キーと形は変えない | なし |
| `PerceptionFromStack` | 保存する側が重ねて返した束 | Perception | 枚数、重ねるキーの名前の一覧 |
| `PerceptionFromFrame` | 観測と同じ形の 1 枚 | 履歴なしの Perception | 枚数、重ねるキーの名前の一覧 |
| `PerceptionFromSequence` | 観測 1 枚に、開始の印の鍵を足したもの | 行ごとの履歴から作った Perception | 行の数、seed、開始の印の鍵の名前、枚数、重ねるキーの名前の一覧 |

`PerceptionFromSequence` は、新しい種類である（今は、種類の値にせず、実行器を作る口で受けている）。開始の印は、「新しい列が始まる行」を表す真偽の値で、行の数ぶんある。DefaultDQN は、鍵の名前を `episode_start` にする（D1）。

| 部品 | `FrameWithStatsUpdate` | `PerceptionFromStack` | `PerceptionFromFrame` | `PerceptionFromSequence` |
|---|---|---|---|---|
| `FrameStack` | 素通し | 重ねるキーのうち届いたものについて、既に stack 軸を持つキーは、そのまま通す。軸の無いキーには、枚数が 1 のとき、長さ 1 の軸を足す（ReplayBuffer は、1 枚のとき軸を付けずに返す） | 重ねるキーのうち届いたものについて、同じ 1 枚を枚数ぶん並べる | 重ねるキーのうち届いたものを、行ごとの履歴で重ねる。開始の印が真の行は、履歴をその行の今の 1 枚で埋め直す。開始の印の鍵を、出力から取り除く |
| `Normalize` | 統計を更新し、値は変えずに返す | 統計を読んで当てる。統計は更新しない | 同左 | 同左 |
| `DropKeys` | 素通し（Frame にはキーを残す） | 書いたキーを落とす | 同左 | 同左 |
| `SetSpec` | 当てる種類なら、書いたキーの spec の宣言を変える。データには触らない | 同左 | 同左 | 同左 |

- `FrameStack` の動きのうち、今と違うのは、枚数とキーを、設定ではなく依頼から受けること（D15）と、渡された名前のうち自分の位置に届かなかったキーを、エラーにせずに重ねないこと（D17。今は、`FrameStack` を先頭にしか置けないので、届かないのは、並びの入力の spec に無い名前だけで、部品がエラーにしている。その名前は、新しくは Agent が構築で止める）である。3 つの Perception の依頼では、`FrameStack` の出力では、そこに届いた対象キーが stack 軸を持つ。`FrameWithStatsUpdate` では 1 枚の Frame をそのまま受ける。
- `DropKeys` の動きは、今と同じである（NG1）。`Normalize` は、当て方と式は今と同じで、統計を取るキーの決まり方だけが変わる（D28）。
- `FrameStack` を 2 つ書いた並びは、2 つ目が、既に stack 軸を持つキーを重ねようとして、部品の検査でエラーになる（§6.7 の 9）。並びの決まりではなく、部品の検査である。

### 6.5 呼び出し元（DefaultDQN）の受け持ち

spec は、データと同じ向きに流れる。

```text
Frame ──F──► FrameWithStatsUpdate ──（値もキーも変えない）──► ReplayBuffer ──S──► PerceptionFromStack ──P──► Network
 ├──F＋開始の印──► PerceptionFromSequence ──P──► Network
 └──F──► PerceptionFromFrame ──P──► Network
```

- **F**: Frame の spec。Agent が作る（今は足すキーが無いので、Env の観測の spec と同じである）。ReplayBuffer の生成にも、この F を渡す（今と同じ）。
- **S**: 重ねた束の spec。ReplayBuffer が答える（`GetSampleObsSpecs`）。F と、Agent から受け取った重ね方から決まる。重ねて返すキーには stack 軸が付く。1 枚のときは、軸の無い spec を答える（P7）。
- **P**: Perception の spec。3 つとも同じ Network に入る。

| 誰が | いつ | すること |
|---|---|---|
| `DefaultDQNAgent` | 構築 | 重ねる設定（`frame_stack.*`）を読み、重ねるキーの名前を決める（D17）。`ForwarderBuilder` を作る。経験の受入れ用（`FrameWithStatsUpdate`）の Forwarder を、F を渡して、最初に作る（統計の入れ物は、このとき用意される。D28）。その出力の spec が、F とキー・形・dtype で同じことを確かめる（D14）。可視化用（`PerceptionFromFrame`）の Forwarder を、F を渡して作る。可視化用の出力の spec（D11 の軸を足したあと）を、Network の入力に使う。行動選択用（`PerceptionFromSequence`）の Forwarder を、行の数 1 の依頼で作って、出力の spec を比べる（D21）。Learner に、`ForwarderBuilder` と重ね方と F を渡す。`ForwarderBuilder` が作った可視化用の Forwarder（軸を足す処理をかぶせる前のもの）の図を記録する。図が空なら、エラーにする |
| `dqn::Learner` | 構築（ReplayBuffer を作ったあと） | S を渡して、Sample 用（`PerceptionFromStack`）の Forwarder を作る。出力の spec（軸を足したあと）を、Network の入力の Perception の spec と比べる（D21）。`ForwarderBuilder` を渡されなかったとき（Rainbow）は、今と同じく、Sample の観測をそのまま使い、比べない |
| `DefaultDQNAgent` | Actor を作るとき | F に開始の印の鍵の spec を足したものを渡して、行動選択用（`PerceptionFromSequence`）の Forwarder を作る。出力に開始の印の鍵が残っていれば、取り除く（D12） |
| `dqn::Actor` | 行動選択 | 観測を自分の device へ移し、開始の印（CPU に置いたまま）を鍵として足して、`Forward` を呼ぶ。返された観測に開始の印があれば、行動選択の前にエラーにする |
| `DefaultDQNAgent` | 経験の受入れ | obs と next_obs を、経験の受入れ用の Forwarder に 1 回ずつ通す（今と同じ） |
| `dqn::Learner` | Sample | obs・next_obs・skip の学習の窓の終点の観測に、Sample 用の Forwarder を当てる（今と同じ） |

stack 軸を足す処理（D11）と、開始の印を取り除く処理（D12）は、作った Forwarder にかぶせて、1 つの Forwarder として持つ。これらの処理と、出力の spec の比較は、DefaultDQN の中（`src/algo/dqn`）に置き、公開の口にはしない（使う側は DefaultDQN だけで、公開の口を最小に保つ）。

### 6.6 設定キー

**重ねる設定（Agent の直下）**

| 設定キー | 型と値域 | 既定 | 意味 |
|---|---|---|---|
| `frame_stack.count` | 1 以上の整数 | コードの既定は 1。基準のプロファイル（`DefaultDQNAgent.@baseline`）は 4 | 重ねる枚数 |
| `frame_stack.keys.mode` | `all`・`listed` | `all` | 重ねるキーの選び方。`all` は Frame の全キー、`listed` は一覧に書いたキー（D17） |
| `frame_stack.keys.listed.names` | キー名の一覧 | 空 | `listed` のときに重ねるキー。`all` のときは、読むが使わない（文字列の一覧なので、型に反する値は無い） |

枚数と、キーの選び方は、互いに独立である。名前の一覧は、選び方が `listed` のときだけ使う。

```text
DefaultDQNAgent.@baseline : frame_stack.count ?= 4
DefaultDQNAgent.@baseline : frame_stack.keys.mode ?= all
DefaultDQNAgent.@baseline : perception.structure ?= stack > norm
DefaultDQNAgent.@baseline : perception.block.[stack].type ?= FrameStack
```

**旧いキーとの対応**

| 旧いキー | 新しいキー |
|---|---|
| `perception.block.[<名前>].count`（`type` が `FrameStack` の block） | `frame_stack.count` |
| `perception.block.[<名前>].keys.mode` | `frame_stack.keys.mode` |
| `perception.block.[<名前>].keys.listed.names` | `frame_stack.keys.listed.names` |

`type` が `FrameStack` の block は、`type` だけを持つ。旧いキーを、並びに選ばれた `FrameStack` の block に書くと、構築でエラーになる（D27）。並び（`perception.structure`・`perception.block.[<名前>].type`）と、`Normalize`・`DropKeys` の設定キーは変えない（設定キーの全体を見直すことは、本 RFD の範囲に入れない。移せない制約ではなく、範囲の線である）。

**`SetSpec` の設定キー**（`perception.block.[<名前>]` の下）

| 設定キー | 型と値域 | 既定 | 意味 |
|---|---|---|---|
| `type` | `SetSpec` | — | — |
| `keys` | キー名の一覧。1 つ以上 | 必須 | 宣言を変えるキー |
| `scale` | `keep`・`Scaled`・`Unscaled` | `keep` | 値の尺度。`keep` は変えない |
| `min_values.mode` | `keep`・`set` | `keep` | 下限の宣言を変えるか |
| `min_values.set.values` | 数の一覧 | 空 | `set` のときの下限 |
| `max_values.mode` | `keep`・`set` | `keep` | 上限の宣言を変えるか |
| `max_values.set.values` | 数の一覧 | 空 | `set` のときの上限 |
| `labels.mode` | `keep`・`set` | `keep` | `labels` を変えるか |
| `labels.set.names` | 名前の一覧 | 空 | `set` のときの `labels` |
| `forward_kinds.mode` | `all`・`listed` | `all` | 当てる Forward の種類。`all` は全部 |
| `forward_kinds.listed.names` | 種類の名前の一覧 | 空 | `listed` のときに当てる種類。一覧に無い種類の依頼では、何もしない |

選ばれていないモードの値（`keep` のときの `set.values` など）も読む。数の一覧が数として読めなければ、エラーにする（§6.7 の 5）。`mode` が `set` なのに、その一覧（`set.values`・`set.names`）が空の設定も、エラーにする（§6.7 の 5）。空の一覧に、宣言を消す意味は持たせない（設定の特定の値に、別の意味を持たせない決まりによる。P36）。

検査は、変えた属性にだけ当てる。`keep` の属性は、届いた spec の宣言のまま通し、検査しない（範囲の宣言や `labels` が無いキーも、そのまま通る）。変えた属性には、次の検査を当てる（通らなければ、§6.7 の 9）。stack 軸があるキーは、軸を除いた 1 枚ぶんの形で見る。

- `scale` を `Unscaled` にするとき: 整数の dtype のキーと、離散のキーには書けない（観測の spec の検査 `StateSpec::AssertSanity` と同じ条件）。
- `labels` を `set` にするとき: 数は、`TensorSpec::HasValidLabels` の決まり（`Vector` は全要素の数、`Grid` と `Sequence` は先頭の次元の数）に合わせる。
- `min_values`・`max_values` を `set` にするとき: 数は、1 つか、1 枚ぶんの全要素の数にする。

値の尺度を、一部の Forward の種類にだけ当てると、統計を更新する流れと読む流れ、または行動選択と Sample のあいだで、正規化する対象がずれる。出力の spec が食い違えば、比較（D21）で止まる。`Normalize` が正規化したキーを `Scaled` にするので、出力の spec が同じになって、比較では止まらないことがある。そのときは、設定に書いたとおりに、ずれたまま動く（D19）。絞ったこと自体は、エラーにしない。既定（`all`）では起きない。

### 6.7 エラーにするもの

行は、DefaultDQN の Agent を組むときに起きる順に並べた。行を上から順に当て、最初に条件を満たした行で止まる。どの行にも当たらない設定は、エラーにしない。設定の値が、その設定キーの型として読めないとき（整数や、数の一覧として読めない）は、設定の読み取りがエラーにする（今ある仕組み。P34）。重ねる設定では 2 の行、部品の設定では 5 の行に入る。

| # | 時機 | 条件 |
|---|---|---|
| 1 | 部品の登録 | 同じ `type` の名前を、2 度登録した |
| 2 | Agent の構築（重ねる設定の読み取り） | `frame_stack.count` が、整数として読めないか、1 未満。`frame_stack.keys.mode` が値域の外。`listed` で一覧が空。`listed` の一覧に、Frame に無い名前がある。開始の印の鍵の名前が、Frame のキーと重なる |
| 3 | `ForwarderBuilder` の構築 | `structure` が名指す block の設定が無い。block の `type` が、登録に無い |
| 4 | `ForwarderBuilder` の構築 | 並びに選ばれた block の設定に、読まれないキーがある（今と同じ） |
| 5 | `ForwarderBuilder` の構築 | 部品の設定が、型か値域に反する（`Normalize` と `DropKeys` は、今の検査のうち spec に依らないもの。`SetSpec` は、`keys` が空、`scale`・各 `mode` が値域の外、全部の属性が `keep`、`forward_kinds.mode` が `listed` で一覧が空、`min_values.mode`・`max_values.mode`・`labels.mode` が `set` で、その一覧が空、数の一覧が数として読めない） |
| 6 | `ForwarderBuilder` の構築 | 並びで使える Forward の種類と、部品の設定の組み合わせが、部品の `ValidateForwardKinds` の検査に通らない。標準の部品では `SetSpec` が、`forward_kinds.mode` が `listed` のときに `forward_kinds.listed.names` に使えない名前がある場合を検査する。どの部品も種類を問わない並びでは、`SetSpec` はこの検査をしない |
| 7 | `BuildForwarder` | `input_specs` が空である（今と同じ） |
| 8 | `BuildForwarder` | 並びの部品のどれかが、依頼の Forward の種類を扱わない |
| 9 | `BuildForwarder` | 部品が、届いた spec か、依頼に載った値を扱えない。`FrameStack`: `PerceptionFromFrame` か `PerceptionFromSequence` で、重ねるキーが既に stack 軸を持つ。`PerceptionFromStack` で、枚数が 2 以上なのに、重ねるキーに stack 軸が無い。`PerceptionFromSequence` で、開始の印の鍵が届いた spec に無いか、その鍵の spec が各行について、shape が空で stack 軸を持たないスカラーの `bool` を表さない。依頼に載った枚数が 1 未満。`PerceptionFromSequence` で、依頼に載った行の数が 1 未満。`DropKeys`: `keys` に、届いた spec に無いキーがある。全部のキーを落とす。`Normalize`: 定数の数が、キーの 1 枚ぶんの要素数と合わない。統計つきで、Perception を作る依頼に届いた `Unscaled` の float のキーが、統計の入れ物に無いか、1 枚ぶんの形が違う（D28）。統計つきで、経験の受入れ用の依頼を 2 度目に受けたときに、届いた `Unscaled` の float のキーの集合か、1 枚ぶんの形が、入れ物と違う（D28）。`SetSpec`: `keys` に、届いた spec に無いキーがある。変えた属性が、§6.6 の検査に通らない |
| 10 | Agent の構築（経験の受入れ用を作ったあと） | 経験の受入れ用の Forwarder の出力の spec が、F とキー・形・dtype のどれかで違う（D14） |
| 11 | Agent の構築（Learner の構築を含む） | 行動選択用か Sample 用の Forwarder の出力の spec（D11 の軸を足し、D12 の印を除いたあと）が、Network の入力の Perception の spec と違う（D21） |
| 12 | Agent の構築（図の記録） | `ForwarderBuilder` が作った可視化用の Forwarder の図が、空である |

今の検査のうち、無くなるものは、並びの決まりの 3 つ（`FrameStack` が先頭でない、`FrameStack` が 2 つ以上、`Normalize` が 2 つ以上）である。「並びの入力の spec に、既に stack 軸を持つキーがある」の検査は、並びの入口では行わず、`FrameStack` の部品の検査（9 の行）になる。Sample 用の依頼では、束が stack 軸を持つのが正しいためである。

エラーの文面には、設定キーを完全キーで書く（今の決まりのまま）。

### 6.8 図

| 図の名前 | 中身 |
|---|---|
| `perception.structure` | 題（Forward の種類の名前）、入力のノード（`input`。キーの一覧）、部品のノード（block の名前と `type`）を並びの順につないだもの、出力のノード（`output`。キーの一覧） |
| `perception.detail` | 上に加えて、ノードごとの出口の spec（キーごとの shape・dtype・値の尺度・stack 軸の有無）、部品ごとの seed、部品が答えた値（`GetGraphVizDetails`） |

- Agent が記録するのは、`ForwarderBuilder` が作った可視化用の Forwarder（`PerceptionFromFrame`）の図である。記録する時機と出る場所は、今と同じである（Network の構造図と同じ時機。Run の `dot/` の下）。図が空なら、エラーにする（§6.7 の 12）。
- 図は、`ForwarderBuilder` が作った Forwarder が、自分の中の部品の Forwarder から作る。block の名前・`type`・seed は、`ForwarderBuilder` が、作るときに渡す。
- stack 軸を足す処理と、開始の印を取り除く処理（§6.5）は、図に描かない。
- `metrics.jsonl` の `scalar` と `trace` には何も足さないので、等価性の比べ方（§7）には影響しない。

### 6.9 移行対象

追跡しているファイルの移動・改名・削除は無い（置き場を変える扱いには当たらない）。検索は 2026-11-24 の `git grep -F`（大文字小文字を区別する固定文字列）による。検索の文字列（S1〜S21）は §7 にある。

既存の `apps/11_batch_run.bat` の Montezuma 起動では、出荷設定の既定（`frame_stack.count=4`、`frame_stack.keys.mode=all`）を使う。

**直すもの**

| 種類 | 対象 | 直し方 |
|---|---|---|
| コード | `forwarder.hpp`・`forwarder.cpp` | §6.1〜§6.4・§6.7・§6.8 |
| コード | `default_dqn_agent.cpp`・`.hpp`、`dqn_actor.cpp`・`.hpp`、`dqn_assembler.cpp`・`.hpp`、`dqn_learner.cpp`・`.hpp`、`dqn_config.hpp` | §6.5・§6.6 |
| コード | `rl.hpp` の `ReplayBuffer` と、継承する本体の 2 つ（`replay_buffer_impl.hpp` の `DefaultReplayBuffer`、`replay_buffer.hpp` の `PrefetchingReplayBuffer`） | `GetSampleObsSpecs` を足す（D14） |
| テスト | `ReplayBuffer` を継承する、テスト用の差し替え 3 つ（`dqn_fixture_test.hpp` の `RecordingReplayBuffer`、`replay_buffer_test.cpp` の `BlockingReplayBuffer`、`trainer_test.cpp` の `HintRecordingReplayBuffer`） | `GetSampleObsSpecs` を実装する |
| テスト | `forwarder_test.cpp`、`replay_buffer_test.cpp`、`dqn_learner_test.cpp`、`dqn_concrete_test.cpp`、`dqn_fixture_test.hpp` | 新しい口と、新しい設定キーに書き換える。並びの決まりを確かめているテストは、§7 の受入に合わせて書き直す |
| 設定 | 出荷設定の 4 ファイル（`agent.txt`・`GridMaze.txt`・`LunarLander.txt`・`DropMerge.txt`。S9 が 4 ファイル、S10 が 1 ファイル） | §6.6 の対応のとおりに移す。コメントにした行も移す |
| 検証資材 | [rfd0072/baseline/](../../src/core/base/testdata/rfd0072/baseline) の 17 ファイル | 旧いキーの項目を、新しいキーの項目に入れ替える。丸ごとは採り直さない。変えてよい範囲と手段は、§7 の「基準の直し方」 |
| 検証資材 | [固定入力の移行](../../src/core/base/testdata/rfd0072/perception_migration.py) | 固定した過去の入力を移す先を、新しいキー（`frame_stack.*`）にする |
| 検証資材 | [README.md](../../src/core/base/testdata/rfd0072/README.md) の「golden の更新方針」 | 本 RFD で移したキーと、直し方（RFD0161 と同じ）を 1 文足す |
| 設計文書（jp） | [135_forwarder_and_perception.jp.md](../design/135_forwarder_and_perception.jp.md) | 全体を、新しい口で書き直す |
| 設計文書（jp） | [010_framework_overview.jp.md](../design/010_framework_overview.jp.md) の「4.3 拡張点」の表 | Forwarder の部品の行を足す。interface は `ForwarderModule`・`ForwarderModuleFactory`、部品は `anet/forwarder.hpp` の `SpecForwarder` と `MakePassThroughForwarder`、登録口は `RegisterForwarderModuleFactory`、基盤の外で書ける（D6） |
| 設計文書（jp） | `110_agents_and_learning`、`130_neural_networks`、`140_observability`、`150_replay_buffer`、`200_dqn_agents`、`220_atari_env` の、Perception・`FrameStack`・実行器・並びの図・Replay の枚数とキーの受け渡しに触れる箇所 | 新しい契約に合わせる |
| 用語集 | [CONTEXT.md](../../CONTEXT.md) の 3 項目と、足す 2 項目 | 起草のときに直した（§5.7） |
| ADR | 0084・0086 | 本文は変えず、先頭に注記を足した。新しい 3 本を起こした（D29） |
| wiki | `314_tensor`、`318_rl_core`、`321_agent_core`、`324_dqn` | 起草のときに書き戻した。実装のあとの取り込みで、概要と経緯を直す |

**直さないもの**

| 種類 | 対象 | 理由 |
|---|---|---|
| 過去の記録 | `docs/memo/` のほかの RFD と実装記録（RFD0161 を含む）、`docs/management/issues/` の解決済みの課題（RFD0161 の 2 件） | 当時の記録 |
| 英訳 | `docs/design/*.en.md` | 翻訳の道具が作り直す |
| 本 RFD | 本 RFD が旧い名前を書いている箇所 | 問題の説明、移行対象、検索の文字列としての引用 |
| 変更履歴 | `CHANGELOG.md` | リリースの準備で書く。RFD0161 のキーの移行も、今は載っていない |

検索の当たりの割り振り（2026-11-24。ファイル数）は次のとおりで、どの行にも入らないファイルは無い。

- S1（`ForwarderChain`）: 11 ＝ 直す 8（`src` 7、設計文書 jp 1）＋ 直さない 3（`docs/memo` 2、課題 1）。
- S2（`ForwarderProvider`）: 3 ＝ 直す 2（`src`）＋ 直さない 1（`docs/memo`）。
- S3（`anet::ForwardExecutor`）: 4 ＝ 直す 4（`src`）。
- S4（`GetStackCount`）: 5 ＝ 直す 5（`src`）。S5（`GetStackKeys`）: 5 ＝ 直す 5（`src`）。
- S6（`ForwardRequest`）: 7 ＝ 直す 5（`src`）＋ 直さない 2（`docs/memo` 1、課題 1）。
- S7（`ForwardSpec(`）: 4 ＝ 直す 4（`src`）。S8（`MakeForwarder(`）: 2 ＝ 直す 2（`src`）。
- S9（`block.[stack].count`）: 28 ＝ 直す 26（`src` のテスト 4、`apps` 4、`rfd0072/baseline` 17、固定入力の移行 1）＋ 直さない 2（`docs/memo`）。
- S10（`block.[stack].keys`）: 25 ＝ 直す 22（`src` のテスト 3、`apps` 1、`rfd0072/baseline` 17、固定入力の移行 1）＋ 直さない 3（`docs/memo` 2、課題 1）。

数は、追跡しているファイルの、いまの作業ツリーの中身にかけた検索による。CONTEXT.md は、起草のときに直したので、どの検索にも当たらない（直す前は、S1 と S6 に当たっていた）。本 RFD と、一緒に起こした ADR 3 本は、まだ追跡していないので、数に入っていない。追跡したあとは、本 RFD が、旧い名前の引用（直さないものの「本 RFD」の行）で当たる。

### 6.10 ほかの RFD との順序

| RFD | 触る同じ箇所 | 扱い |
|---|---|---|
| [0161_observation_pipeline_10rfd.md](0161_observation_pipeline_10rfd.md) | forwarder、Perception | 実装済み。本 RFD はその上に書く。RFD0161 の本文は直さない |
| [0142_agent_prev_action_obs_10rfd.md](0142_agent_prev_action_obs_10rfd.md) | Actor、経験の受入れ、Perception に足す部品 | 本 RFD のあと。RFD0142 は、部品を登録する形（D6）と、依頼ごとの spec（D10）を前提に、再グリルで書き直す |
| [0191_rnd_raw_error_and_recent_learning_10rfd.md](0191_rnd_raw_error_and_recent_learning_10rfd.md) | `default_dqn_agent.cpp` の経験の受入れ | 実装済み。本 RFD は RND の観測加工に触れない（NG3） |

## 7. 受入

検索は `git grep -n -F`（大文字小文字を区別する固定文字列）で、範囲は項目に書く。検索の文字列は表の下に置く。

| # | 受入 |
|---|---|
| 1 | 検索 S1〜S10 が、`src`・`apps`・`data` で 0 行（今は、S1 が 7 ファイル 34 行、S2 が 2 ファイル 5 行、S3 が 4 ファイル 5 行、S4 と S5 が 5 ファイル 6 行ずつ、S6 が 5 ファイル 34 行、S7 が 4 ファイル 14 行、S8 が 2 ファイル 5 行、S9 が 26 ファイル 71 行、S10 が 22 ファイル 45 行） |
| 2 | 検索 S11 と S12 が、`src/core/base/include/anet/forwarder.hpp` と `src/core/base/src/forwarder.cpp` の 2 ファイルで 0 行（今は、S11 が 4 行と 9 行、S12 が 1 行と 5 行） |
| 3 | `[forwarder]` のテスト: テストが定義した Forward の種類の名前と、テストが登録した部品だけを使う並びで、`ForwarderBuilder` から Forwarder を作って通せる。作った Forwarder の `GetInputSpecs` は、渡した spec と同じである |
| 4 | `[forwarder]` のテスト: 同じ `type` の名前の 2 度目の登録、登録に無い `type` の設定、並びの部品が扱わない Forward の種類の依頼、空の `input_specs` が、それぞれエラーになる。`[perception]` のテスト: 並びに選ばれた block に、その部品が読まないキーを書いた設定は、`ForwarderBuilder` の構築がエラーになり、文面にそのキーの完全キーが入っている（今あるテストを、新しい口に書き換えたもの。キーの名前は、今のテストと同じく、どの部品も読まない名前にする）。同じ block の未読キーと値の不正では、未読キーが先に報告される。標準の工場とテストが登録した工場で確かめる。別々の block では、先行 block の値の不正より後続 block の未読キーが先、先行 block の種類の組み合わせの不正より後続 block の値の不正が先に報告される |
| 5 | `[perception]` のテスト: 4 つの Forward の種類のそれぞれで、作った Forwarder の `GetOutputSpecs` の各キーの shape と dtype が、`GetInputSpecs` どおりの形の仮の入力（行の数は 2）を通した出力の、行の軸を除いた形と dtype に一致する（4 枚と 1 枚、`DropKeys` を含む並び）。枚数 1 の `stack > norm` で、`Normalize` に届く spec が、3 つの Perception の種類のどれでも stack 軸つきである |
| 6 | `[perception]` の往復のテスト: 行動選択用の Forwarder が作った Perception と、同じ経験を ReplayBuffer に入れて Sample 用の Forwarder で作り直した Perception が一致する。条件は、4 枚と 1 枚、episode の先頭、打ち切りの終端、n より手前で episode が終わる next_obs、skip の窓の終点（RFD0161 の受入 22 と同じ条件） |
| 7 | `[perception]` のテスト: 並び `norm > stack` と `stack > norm` のどちらも組める。`Unscaled` と宣言した float のキーに、統計つきの正規化を使う設定で、どちらの並びでも、統計を更新し終えたあと（以後、経験の受入れ用の Forwarder を通さない）に作った、行動選択用と Sample 用の Perception が一致する |
| 8 | `[perception]` のテスト: `FrameStack` を書かない並び（`norm` だけ）で、枚数 1 では、行動選択・Sample・可視化の Perception の重ねる対象のキーが `[B, 1, *shape]` になり、通常の `Actor::MakeAction` が開始の印を受け取って行動を返す。開始の印を出力 spec から除いたと宣言した外部部品が実際には開始の印を返す場合、`Actor::MakeAction` が行動選択の前にエラーを返す。枚数 4 では、Agent の構築がエラーになる |
| 9 | `[perception]` のテスト: `frame_stack.keys.mode` が `listed` の設定で、一覧に書いたキーだけが stack 軸を持つ。`all` では、全キーが持つ。一覧に Frame に無い名前を書いた設定、`listed` で一覧が空の設定、`frame_stack.count` が 0 の設定、`frame_stack.count` が `2.5` の設定、Frame に開始の印の鍵と同じ名前（`episode_start`）のキーがある構成は、それぞれ Agent の構築がエラーになる |
| 10 | `[replay_buffer]` のテスト: `GetSampleObsSpecs` が、枚数 4 では、重ねるキーに stack 軸つきの spec（先頭の軸の長さが 4）を答え、枚数 1 では、軸の無い spec を答える。どちらも、Sample が返す obs の、先頭の軸を除いた形と一致する |
| 11 | `[perception]` のテスト: `SetSpec` で `scale` を `Unscaled` に変えた float のキーが、あとの `Normalize` の対象になる。`forward_kinds.mode` が `listed` のとき、一覧に無い種類の依頼では、spec が変わらない。範囲の宣言と `labels` の無い float のキーに、`scale` だけを変える `SetSpec` を当てても、エラーにならない。§6.7 の 5・6・9 の `SetSpec` の条件のそれぞれで、エラーになる。`FrameStack` の後ろに置いた `SetSpec` は、stack 軸つきの spec を受けても、1 枚ぶんの形で検査して通る |
| 12 | `[perception]` のテスト: テストが登録した部品（`PerceptionFromStack` のときだけ、キーを 1 つ落とす）を入れた並びで、Learner の構築がエラーになる（§6.7 の 11）。テストが登録した部品（`FrameWithStatsUpdate` のときに、キーを 1 つ落とす）を入れた並びで、Agent の構築がエラーになる（§6.7 の 10）。どちらの部品も、テスト（`src/algo/dqn` の側）が、公開ヘッダ `anet/forwarder.hpp` だけを使って書いて登録する（基盤の外で部品を書けることの確かめを兼ねる） |
| 13 | `[perception]` のテスト: 並び `stack > norm` の、可視化用の Forwarder の図の dot の文字列に、題（`PerceptionFromFrame`）、`input`、`output`、2 つの部品の名前と `type`、並びの順の edge が入っている。詳しい図には、ノードごとの shape と、枚数の値も入っている。素通しの Forwarder（`MakePassThroughForwarder`）の `MakeGraphViz` は、空を返す |
| 14 | 設定の基準の検証: 基準の 17 ファイルを、表の下の「基準の直し方」のとおりに直す。直したあとに、(a)「基準の直し方」の 1〜5 が、17 ファイルのどれでも成り立つ。(b) [prepare.py](../../src/core/base/testdata/rfd0072/prepare.py) の `prepare` と `compare` を順に実行し、`compare` の終了コードが 0 である（`compare` は、比べるテストの終了コードを、そのまま返す） |
| 15 | 等価性: 表の下の「等価性の比べ方」で、変更の前後が一致する。変更のあとの Run の Run フォルダの `dot/` の下に、`perception.structure` と `perception.detail` の dot ファイルがある |
| 16 | Debug のビルドが通り、ビルドが作るテスト実行体（`out\test\Debug\` の `*-test.exe`）を引数なしで走らせて、どれも失敗が 0 件である。`git diff --check` の出力が空である |
| 17 | 検索 S1・S2・S4・S5・S6・S7・S9・S10 が、`docs/design` の `*.jp.md` と [CONTEXT.md](../../CONTEXT.md) で 0 行（今は、S1 が `135_forwarder_and_perception.jp.md` の 1 行で、ほかは 0 行。CONTEXT.md は、起草のときに直したので 0 行である）。`135_forwarder_and_perception.jp.md` に、検索 S13〜S20 のそれぞれが 1 行以上ある（今は、どれも 0 行）。`010_framework_overview.jp.md` に、検索 S21 が 1 行以上ある（今は 0 行） |
| 18 | `[perception]` のテスト: 統計つきの正規化の並びで、経験の受入れ用の依頼より先に、Perception を作る依頼を出すと、エラーになる。並び `drop > norm` で、落とすキーが `Unscaled` の float のとき、経験の受入れ用の Forwarder に観測を通すと、落とすキーの統計も更新される。そのキーを、`DropKeys` の前に置いた `SetSpec` で `Scaled` にすると、更新されない |

```text
S1: ForwarderChain
S2: ForwarderProvider
S3: anet::ForwardExecutor
S4: GetStackCount
S5: GetStackKeys
S6: ForwardRequest
S7: ForwardSpec(
S8: MakeForwarder(
S9: block.[stack].count
S10: block.[stack].keys
S11: ForwardExecutor
S12: episode_start
S13: ForwarderBuilder
S14: BuildForwarder
S15: ForwarderRequest
S16: ForwardKind
S17: PerceptionFromSequence
S18: frame_stack.count
S19: SetSpec
S20: GetSampleObsSpecs
S21: RegisterForwarderModuleFactory
```

- S1〜S12 は、無くなることを確かめる旧い名前で、S13〜S21 は、文書に入ったことを確かめる新しい名前である。
- S6 は、新しい名前 `ForwarderRequest` には当たらない（`Forward` の直後が `Request` でないため）。S8 は、`MakePassThroughForwarder(` には当たらない。S11 は、`NetworkForwardExecutor`・`DqnForwardExecutor`・`CreateForwardExecutor` に当たるので、範囲を forwarder の 2 ファイルに限る（この 2 ファイルに、それらの名前は無い）。新しい名前（`ForwarderBuilder`・`ForwarderModule`・`ForwarderModuleFactory`・`ForwarderModuleContext`・`ForwarderRequest`・`ForwardKind`・`BuildForwarder`・`CreateForwarder`・`GetInputSpecs`・`GetOutputSpecs`・`GetSampleObsSpecs`・`frame_stack.count`・`frame_stack.keys.mode`・`frame_stack.keys.listed.names`・`RegisterForwarderModuleFactory`・`PerceptionFromSequence`・`SetSpec`）に S1〜S12 を当てて、どれにも当たらないことを確かめた。S12 は、DefaultDQN が開始の印の鍵の名前に使う `episode_start` に当たるが、その名前を書くのは `src/algo/dqn` の側で、範囲の 2 ファイルには書かない（鍵の名前は、依頼に載せて渡す）。
- S9 と S10 は、block の名前が `stack` の設定だけに当たる。`src`・`apps`・`data` で、block の `count` と `keys` を書く行を引くと、`count` を持つ block の名前は `stack` だけである（`block.[stack].count` が 71 行、`block.[stack].keys` が 45 行。ほかに当たるのは、`DropKeys` の `keys` を書く `block.[drop].keys` の 2 行と `block.[norm].keys` の 1 行で、重ねる設定ではない。2026-11-24 の検索）。

**基準の直し方**（受入 14）: 基準の 17 ファイル（[rfd0072/baseline/](../../src/core/base/testdata/rfd0072/baseline)）は、解決結果を丸ごと採り直さずに、RFD0161 と同じやり方で直す（P35）。固定コミット（`manifest.json` の `commit`）は変えない。固定入力の移行（`perception_migration.py`）の移す先を新しいキーにし、`prepare.py` の `prepare` で、固定入力と、移したあとの入力を用意する。17 構成を解決した結果は、基準と比べるテスト（`[rfd0072-baseline]`。`prepare.py` の `compare` が走らせる）が、準備した場所の `actual/` の下に書く。その結果から、新しいキーとその値だけを採り、各ファイルの旧いキーの項目と入れ替える。手順の細部は、RFD0161 の実装記録（[0161_observation_pipeline_20impl.md](0161_observation_pipeline_20impl.md) の「固定 17 構成」の行）にならって、実装計画に書く。`prepare.py` の `capture` は使わない（既存の基準の上書きは、道具が拒否する）。

変えてよいのは、各ファイルの `map_order`（キーの並び）と `values`（キーと値）のうち、名前が `.perception.block.[stack].count` か `.perception.block.[stack].keys.mode` で終わるキー（以下、旧いキー）の項目だけである。旧いキーは、同じ接頭辞に `.frame_stack.count`・`.frame_stack.keys.mode` を付けた名前のキー（以下、新しいキー）に替わり、値は変わらない。基準に、`keys.listed.names` のキーは無い。

受入 14 の (a) は、変更の直前のコミットの 17 ファイルと、直した 17 ファイルを比べて、次の 1〜5 が成り立つこととする。比べた手順と結果は、実装計画に書く。

1. `values` から、旧いキーと新しいキーを除いた残りが、キーも値も一致する。
2. 変更の前の `values` の旧いキーのそれぞれについて、直したあとの `values` で、対応する新しいキーが同じ値を持つ。直したあとの `values` に、旧いキーは無く、対応する旧いキーの無い新しいキーも無い。
3. `map_order` から、旧いキーと新しいキーを除いた残りの並びが一致する。直したあとの `map_order` に、2 の新しいキーが 1 回ずつあり、旧いキーは無い。
4. `baseline_commit` と `resolution` が一致する。
5. `git diff --stat <変更の直前のコミット> -- src/core/base/testdata/rfd0072/manifest.json src/core/base/testdata/rfd0072/resolution` の出力が空である。

1〜5 のどれかが成り立たないときは、原因を調べて、実装を止めて開発者に示す。

**等価性の比べ方**（受入 15）: 変更の直前の HEAD の exe と、変更のあとの exe で、同じ設定・同じ seed の Run を 1 本ずつ流す。対象は、出荷の設定ファイルを、`run.$` の行もほかの行も書き換えずに流した Run である。LunarLander（[LunarLander.txt](../../apps/runner/config/LunarLander.txt)。4 枚）、GridMaze（[GridMaze.txt](../../apps/runner/config/GridMaze.txt)。1 枚）、DropMerge（[DropMerge.txt](../../apps/runner/config/DropMerge.txt)）、Atari（[Atari.txt](../../apps/runner/config/Atari.txt)）を 1 本ずつ流す。変更のあとの Run は、新しい設定キーへ移した同じファイルを使う。実装計画に、Run ごとに解決された `run.$` と Atari のゲームを、Run の設定の写しから書く。長さは、learn の step が 1,000 に達するまでとする（そのための exp の step 数は実装計画に書く）。

全環境で、変更の前後の両方の起動時に `backend.$=backend.@deterministic` を指定する。Atari では、前後の起動時に `E1.game=qbert` も指定し、実効設定の `AtariEnv.game` が `qbert` であることを確かめる。比較に使う前後両方の Run で、`backend.deterministic_algorithms` と `backend.cudnn_deterministic` の実効値がともに `true` であることを確認し、実装計画に記録する。

合否は、道具（[compare_metrics.py](../../src/core/base/testdata/rfd0097/compare_metrics.py)）の出力で決める。次の 2 つをどちらも満たせば一致、どちらかを満たさなければ不一致とする。

1. 変更の前の Run が、比べる中身を持つ。道具に `--output <ファイル>` を付けて書き出した JSON の `old.counts` で、`scalar:37_agent_qtd/01_td_mean`（学習の TD 誤差）と `scalar:20_eps/10_train_reward`（学習側の episode の return）の件数が、どちらも 1 以上である。満たさなければ、Run の長さを learn の step 10,000 まで伸ばして、前後の両方を取り直す。それでも満たさなければ、受入は不成立とし、実装を止めて開発者に示す。
2. 道具の `exact_match` が真である。道具は、`metrics.jsonl` の `scalar` と `trace` の全レコード（tag・step・値）を比べ、片方にだけあるレコードが 1 つでもあれば偽にする。値が NaN どうし、値なしどうしのレコードは、同じ tag・同じ step なら一致とする。

除くのは、実時間と資源に依る `90_perf/`・`91_resource/`・`92_resource/` で始まる tag である（RFD0161 の比べ方と同じ）。道具の `--allow-lives-values` と `--allow-migrated-edges` は付けない。checkpoint の raw checksum は使わない。

不一致のときは、原因を調べて、実装を止めて開発者に示す。

## 8. 成功の測り方

- 公開の型のうち、使う側が触るものが、`Forwarder`・`SpecForwarder`・`ForwarderRequest`（と `ForwardKind`）・`ForwarderBuilder` になり、使い方が「依頼を出して、受け取った Forwarder の `Forward` を呼ぶ」の 1 つになる（受入 1〜3）。
- forwarder.hpp の `anet` の層の口に、RL の概念が残らない（受入 2・3）。nn.hpp の呼び出しごとの情報にある開始の印は、範囲に入れない（[issue-20261124-f9f964](../management/issues/issue-20261124-f9f964.md) が持つ）。
- 出荷設定で、学習の挙動が変わらない（受入 15）。
- 並べ方の自由と、成り立たない設定の検出が、両立している（受入 7・8・12）。

## 9. 未決

なし。

## 10. グリルの材料

決定ではなく、決定に至る材料である。

### 10.1 開発者の発言

日付を書いていない行は 2026-11-23〜24 の、hpp の草案の検討の中の発言である。

- 2026-11-23「実装されました。一旦確認。ただ、forwarder.hpp見るとなんか思ってたのと違うというかごちゃごちゃしてる感じがするのでリファクタリングしたい」（起点）。
- 「ForwardRequest：リクエストそのものじゃないのにそのような名前になってる」「Forwarder/ForwardExecuter/ForwardProvider、これらの関係がごちゃごちゃしてる。とりあえずForwarderとForwardExecuterは統合できないかな」「全部がanet名前空間、つまりRLに依存しない層。少なくともhppでみてepisode_startというRL依存概念が入ってる」「こういうのは最初からhppファイル内容ベースで調整すべきだったね」（§1、D1、D5）。
- 「設定次第で使われる具象クラス側がRLに依存する事は問題ありません」「NNみたいに登録性にはしたが良いかも」「episode_startは入力Dictに含める」（D1、D6）。
- 「外向け（Agent向け）のクラスと、内向け（部品の具象クラス）で分けたが良いかも。hppまでは分けなくていいけど」「forwarderを使いたい、もしくは作りたい時はとりあえずanet/forwarder.hppをインクルードする、がフレームワーク方針です」（D7）。
- 「ForwarderChain、Forwarerでもあり、Forwarderを作るものでもある？両方で良いんだっけ？責務大丈夫？」「分ける」（D2）。
- 「ForwarderBuilderが素直じゃないですかね」「SpecForwarder のインスタンスが出来ているという事自体が、…」「GetInputSpecs()もつけておきたいです。生成メソッドは、クラス名に合わせて、BuildForwarder()としておきたいです」（D3、D4）。
- 「ForwardRequestの指定が無いコンテキストでGetOutputSpecs()を呼べるのは本質的になんか変じゃないですか？」（D4、D9）。
- 「渡ってくる≠統計が必要、ってことでしょう？」「後で困って小さな実装を追加することになったら萎えるから先に最低限の策は入れておきたい」「SetSpecModuleで、対象とするForwardRequestのkindの配列を指定できるようにしといたが無難じゃない？」「ForwardKindとか？」（D5、D18、D19）。
- 「なんでstructureの並びに決まりを設ける必要があるんです？せっかく設定で自由に出来るようにするのに。やるとしても落ちるようなケースとかWARNログ出すところまでのような」（D8）。
- 「「必ず stack 軸を持つ」はForwarder自体じゃなくてRL側の制約だったよね？」「Stack軸を確保するのはForwarderじゃなくてその呼び出し元の責務じゃないですか？」「Stack軸を確保する方法、FrameStackのModuleにこだわる必要あるんですか？」（D11）。
- 「べき論でいうなら、実際にINPUTされるTensor形状と従ったSpecを入れるべき」（D10）。
- 「norm > stack　と　stack > norm　で動きが違って良くて、意図しないであろう動きだとしてもそういう設定にしたんなら正しい動きであって」（D8、D13）。
- 「RBにスタック情報を教えるんじゃなくて、RBがSpecの形で教える、じゃないですか?主従関係で言うとRBが主でForwarderは従だから。PerceptionがEnv情報をベースにするのと同じ関係」（D14）。
- 2026-11-24「直感ではスタック数は、Actor（行動選択のPerception）とLearner（RB）の両方で共有すべき情報だから、Agent全体の設定が正本になるべきな気はする」（D15）。
- 2026-11-24「Forward()じゃなくてBuildForder()向けの情報だとしたらBulildForwarderRequestじゃない？」（D5。型の名前を選び直すきっかけ。選んだのは `ForwarderRequest`）。
- 2026-11-24「Builderは工場であって、そこに作る製品の情報を書かせるのは予定情報になるのでちょっと回り道。出来た製品の情報を書きたいんだから、実際に出来た製品側に聞く方がスマートだと思う」（D22）。
- 2026-11-24「MakeGraphViz()って、未対応ならnull返す、に対応してるっけ？」（D23。この問いのあと、未対応は空を返す形を選んだ）。
- 2026-11-24 選択: 依頼の型の名前は `ForwarderRequest`、重ねる設定は Agent 全体の 1 か所、設定キーは `frame_stack`、図の口は `SpecForwarder` に足す、新しい RFD として起こす、細部の一覧は推奨の既定で埋める（D5、D15、D16、D23、D26）。
- 2026-11-24 選択: `SetSpec` の、当てる Forward の種類を絞る口は残す（U1。別モデルの検討が外す案を出したあとの問いへの答え。D19）。

### 10.2 AI の案と、置いた既定

- AI が初めに推して、開発者の発言で改めたもの: 並びの決まりを残す案、stack 軸を `FrameStack` の部品で確保する案、束の spec を依頼の種類で聞く案（R1）、重ねる設定を 2 か所に書く案（S1）、図を `ForwarderBuilder` が描く案（G1）、図の口のために型を分ける案（J2）。
- 聞かずに置いて、開発者に示したもの（番号で挙げてもらう形で示し、挙がらなかった）: D10 の開始の印の spec、D14 の 1 枚の spec、D17、D27、D28、§6.3 の device の扱い。
- 細部の一覧（10 件）として示し、「推奨で埋める」の答えを得たもの: D6 の登録の時機、D11 の軸を足す対象、D12、D20、D21、D24、D25、§6.6 の値域と既定、`SetSpec` の設定キーとエラー。
- 別モデルの検討（2026-11-24。会話は渡さず、本 RFD の草案だけを読ませた）を受けて直したもの:
  - 事実の訂正: 並びに選ばれた block の中の読まれないキーは、今も構築でエラーになる（P22、D27、§6.7 の 4）。`ReplayBuffer` の口は rl.hpp にあり、継承するクラスは 5 つある（P31、§6.1、§6.9）。MuZero は、同じ ReplayBuffer を使っていない（NG4）。
  - 起草のときに AI が置いた形を改めたもの: ReplayBuffer が受ける 1 枚の spec（D14）、`FrameStack` の Sample での動き（D11、§6.4）、出力の spec を比べる時機（D21）、統計の入れ物を作る時機と、入れ物に無いキーの扱い（D28）、重ねるキーを選ぶ先（D17）、`SetSpec` の検査の中身（§6.6）。
  - 足した決まり: 開始の印の device と、鍵の名前の重なり（D1）、部品が扱わないキーを残すこと（§6.3）、図が空のときの扱い（§6.5）、経験の受入れ用の出力の spec の確かめ（D14）。
  - 理由を補ったもの: 並べ方の自由の使い道と、重ねるのを並びの外へ出す案との比較（D13、§5.6）。Network の側との形の違い（D1）。
  - レビュー（1 回目）を受けて、今ある決まりと名前から導いて足したもの: 開始の印の鍵の名前を `episode_start` にすること（D1。Actor が今使っている名前にそろえた）。`SetSpec` の `set` の一覧が空の設定をエラーにし、`keep` の属性は検査しないこと（D19、§6.6。設定の特定の値に別の意味を持たせない決まりによる）。基準の 17 ファイルを、RFD0161 と同じやり方で直すこと（§7 の「基準の直し方」）。拡張点の表に行を足すこと（D6、§6.9）。
  - 自己点検で、決定から導いて足したもの: 経験の受入れ用の依頼を 2 度目に受けたときの、統計の入れ物の扱い（D28。入れ物に無いキーをエラーにする決まりを、同じ依頼にも当てた）。旧いキーの名前を書くテストを足さないこと（D27。クリーンブレーク方針による）。
  - 前に決めた形への反対として、開発者に聞いたもの: `SetSpec` の絞る口を外す案（D19）。残す（U1）・外す（U2）・残して、Perception の種類のあいだで違う当て方を止める（U3）を示し、AI は U2 を推した。開発者は U1 を選んだ。U2 と U3 を退けた理由（§5.6）は、比べた表から AI が書いた。
  - 採らなかったもの: 依頼の付帯の情報に、型つきの読み口を公開の口として足す案（付帯の情報のキーの名前は、依頼を作る関数と標準の部品の中に閉じ、実装計画で決める）。stack 軸を足す処理などの手助けを、`anet::rl` の公開の関数にする案（使う側は DefaultDQN だけなので、公開の口を最小に保ち、DefaultDQN の中に置く。§6.5）。

### 10.3 hpp の草案

2026-11-24 の第 19 版をもとに、2026-11-25 に設定の検査を行う公開の口を追記した。

名前と意味の正本は §6.2・§6.3 である。引数の型と、口の細かい形は、実装する側が決めてよい。

```cpp
namespace anet {

    // 節 1: 使う側（Agent など）が触るもの

    class ForwardCapture; // 今のまま

    class Forwarder {
    public:
        virtual ~Forwarder() = default;
        virtual TensorDict Forward(const TensorDict& input, ForwardCapture* capture = nullptr) = 0;
    };

    class SpecForwarder : public Forwarder {
    public:
        virtual const TensorSpecMap& GetInputSpecs() const = 0;
        virtual const TensorSpecMap& GetOutputSpecs() const = 0;
        virtual std::unique_ptr<graphviz::GraphViz> MakeGraphViz(bool detail) const; // 既定は空を返す
        virtual std::vector<std::pair<std::string, std::string>> GetGraphVizDetails() const { return {}; }
    };

    using ForwardKind = std::string;

    class ForwarderRequest {
    public:
        explicit ForwarderRequest(ForwardKind forward_kind, ConfigData options = {});
    public:
        const ForwardKind& GetForwardKind() const;
        const ConfigData& GetOptions() const;
    };

    class ForwarderBuilder final : public ModuleBase {
    public:
        ForwarderBuilder(const ConfigData& config_data, const std::string& config_prefix, seed_t seed);
    public:
        std::shared_ptr<SpecForwarder> BuildForwarder(const ForwarderRequest& request,
            const TensorSpecMap& input_specs) const;
        std::optional<float> GetScalar(const std::string& key, int64_t index = -1) const override;
        std::optional<torch::Tensor> GetTensor(const std::string& key, int64_t index = -1) const override;
    };

    // 節 2: 部品を作る側だけが触るもの

    class ForwarderModule : public ModuleBase {
    public:
        virtual void ValidateForwardKinds(const std::optional<std::vector<ForwardKind>>&) const {}
        virtual std::shared_ptr<SpecForwarder> CreateForwarder(const ForwarderRequest& request,
            const TensorSpecMap& input_specs) const = 0;
    };

    std::shared_ptr<SpecForwarder> MakePassThroughForwarder(const TensorSpecMap& specs);

    struct ForwarderModuleContext {
        seed_t seed = 0;
        std::optional<std::vector<ForwardKind>> forward_kinds;
    };

    class ForwarderModuleFactory {
    public:
        virtual ~ForwarderModuleFactory() = default;
        virtual std::optional<std::vector<ForwardKind>> GetForwardKinds() const = 0;
        virtual void ReadConfigKeys(const ConfigData&) const {}
        virtual std::shared_ptr<ForwarderModule> CreateModule(const ConfigData& config_data,
            const ForwarderModuleContext& context) const = 0;
    };

    void RegisterForwarderModuleFactory(const std::string& type_name,
        std::shared_ptr<ForwarderModuleFactory> factory);

}

namespace anet::rl {

    inline constexpr std::string_view kForwardKindFrameWithStatsUpdate = "FrameWithStatsUpdate";
    inline constexpr std::string_view kForwardKindPerceptionFromStack = "PerceptionFromStack";
    inline constexpr std::string_view kForwardKindPerceptionFromFrame = "PerceptionFromFrame";
    inline constexpr std::string_view kForwardKindPerceptionFromSequence = "PerceptionFromSequence";

    ForwarderRequest MakeFrameWithStatsUpdateRequest();
    ForwarderRequest MakePerceptionFromStackRequest(int stack_count, std::vector<std::string> stack_keys);
    ForwarderRequest MakePerceptionFromFrameRequest(int stack_count, std::vector<std::string> stack_keys);
    ForwarderRequest MakePerceptionFromSequenceRequest(int row_count, seed_t seed, std::string start_key,
        int stack_count, std::vector<std::string> stack_keys);

}
```

ReplayBuffer に足す口（`rl.hpp` の `ReplayBuffer`）:

```cpp
    /// Sample が返す obs（next_obs も同じ形）の spec。
    virtual const TensorSpecMap& GetSampleObsSpecs() const = 0;
```

### 10.4 起票のあとに出た材料（2026-11-24）

起票とレビューのあとの会話で出た材料である。決定ではない。本 RFD の確定事項（§5）と実装契約（§6）は変えておらず、開発者は、今の形のまま先に実装すると述べた。次に本 RFD か [0142_agent_prev_action_obs_10rfd.md](0142_agent_prev_action_obs_10rfd.md) を再グリルするときの材料にする。

**開発者の発言**

- 「イメージは、とりあえずActionも入力に入れておいて、ActionConcatForwarder（仮）をかましたらそれがPerceptionに混ぜられるイメージ。本流として入れるDictと必要に応じて横道から混ぜるネタにするDictの区別が必要？」
- 「Stackingもprev_actionも、共通して、直近の系列情報(状態・行動・報酬・終端情報）を貯めて置く必要があると思うけど、入力じゃなくてそのバッファを横からいつでも部品が見れるようにしたらいいんじゃないかなあ」
- 「記録だけ。明日のリセット前にPPO試作まで終わらせたいから、今のRFD0200で一旦実装します」

**AI の整理 1: RFD0142 を本 RFD の口に照らした結果**（机上の確かめで、試作はしていない）

口を変えずに載る。仕事は、次の 2 つに分かれる。

- 行動を入力の Dict に入れる仕事は、並びの手前（Actor と Agent）が行う。行動選択のとき、直前の行動は観測に無く、Sample で作り直すには Frame に入っている必要があるためである。入口は、行動選択・経験の受入れ・可視化の 3 つになる。経験の受入れのときに、部品がキーを作り出す形は、D14 の確かめ（受入れ用の出力の spec が Frame の spec と同じ）で止まる。
- 入った行動を Perception に混ぜる仕事は、並びの部品（登録した部品。経験の受入れでは素通しにする）でも、NN の `bind` でもできる。どちらにするかは、RFD0142 の再グリルで決める。

入力の Dict は 1 つでよい。区別するのは、キーごとの 3 点（Frame に保存するか、重ねるか、正規化するか）で、それぞれ、依頼に渡す spec、`frame_stack.keys`、spec の宣言で決まる。

RFD0161 の D2 は「Agent が足す情報は、Actor の実行器が作る」と書く。本 RFD は実行器を無くすので、「並びの手前で、Actor と Agent が Frame に足す」と読み替える。

**AI の整理 2: 直近の系列を、並びの外のバッファに置く案**（開発者の 2 つ目の発言による）

- 形: 行動選択では、Actor が、lane ごとの直近の系列（観測・行動・報酬・開始の印）を持つ。Sample では、ReplayBuffer が、標本ごとに同じ形の束を返す。部品は、どちらでも同じ束を受ける、状態を持たない変換になる（重ねるは観測の列をそのまま使い、直前の行動は行動の列の最後を取る）。
- 今の持ち主: 行動選択の側の履歴は、`FrameStack` の部品が Actor ごとに作る実行器の中の `DictFrameStacker` が持ち、重ねるキーの観測だけを貯める（[forwarder.cpp](../../src/core/base/src/forwarder.cpp) 117〜127 行、[stacker.hpp](../../src/core/base/include/anet/stacker.hpp) 28〜39 行）。Sample の側は、ReplayBuffer が保存した slot から重ね直す。
- 得るもの: 直前の行動や報酬を、Frame に保存しなくて済む（ReplayBuffer には行動と報酬の列が既にある）。行動選択と Sample が同じ依頼になり、両者の一致が作りで決まる。[0173_intrinsic_reward_novelty_10rfd.md](0173_intrinsic_reward_novelty_10rfd.md) の入力（直前の行動・外的報酬・内的報酬）と、同 RFD の未決の「履歴の持ち主」に合う。
- 払うもの: ReplayBuffer の Sample の契約が変わる（行動と報酬の列を、obs・n-step 先の next_obs・skip の学習の窓の終点に返す。NG4 の線を越える）。エピソードの先頭の埋め方を、Actor の側と ReplayBuffer の側でそろえる必要がある。
- 本 RFD の決定のうち、変わる見込みのもの: D1 のうち開始の印を入力の鍵にすること、D11・D12（軸を足す、印を除く）、D13（`FrameStack` を並びの部品のままにする。§5.6 で退けた「重ねるのを、並びの外へ出す」を、行動と報酬まで広げた形に当たる）、D21 の比較、§6.4 の Forward の種類（`PerceptionFromSequence` が要らなくなる見込み）。
- 読む道: 部品がバッファを横から読む形と、バッファの持ち主が束を切り出して、入力の Dict として渡す形がある。AI は後者を推した（Sample の側は標本ごとに見る場所が違い、部品が読むものは、依頼に渡す spec に現れている必要があるため）。開発者の答えは得ていない。
- 確かめていないこと: 性能（Sample で足す列の取り出し）と、ReplayBuffer の変更の大きさ。

## 11. 関連

- [0161_observation_pipeline_10rfd.md](0161_observation_pipeline_10rfd.md)（観測加工の定義の一元化。本 RFD が、その公開の口を作り直す）
- [135_forwarder_and_perception.jp.md](../design/135_forwarder_and_perception.jp.md)（今の口の設計文書。本 RFD の実装で書き直す）
- [314_tensor.md](../wiki/categories/314_tensor.md)（トピック別まとめ）
- [issue-20261124-6b50cb](../management/issues/issue-20261124-6b50cb.md)（本 RFD の課題）
- [issue-20261124-f9f964](../management/issues/issue-20261124-f9f964.md)（範囲に入れなかった、nn.hpp の側の行の開始の印）
- [issue-20261123-e43866](../management/issues/issue-20261123-e43866.md)（可視化で 1 枚の観測を重ねる経路のテスト。本 RFD が、この経路の入力の作り方を変える）
- 範囲に入れない課題: [issue-20261122-258ef8](../management/issues/issue-20261122-258ef8.md)、[issue-20261113-b96b2c](../management/issues/issue-20261113-b96b2c.md)、[issue-20261122-ba42c9](../management/issues/issue-20261122-ba42c9.md)
