---
title: 外的報酬と内的報酬の価値分解
page_kind: サマリ
material: "[[docs/memo/0121_extrinsic_intrinsic_value_decomposition_10rfd.md|0121_extrinsic_intrinsic_value_decomposition_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
  - "[[docs/wiki/concepts/methods/intrinsic_novelty.md|内的報酬による探索の比較]]"
  - "[[docs/wiki/concepts/methods/reward_scale_discount.md|報酬尺度と割引率]]"
  - "[[docs/wiki/concepts/framework/315_nn.md|315 ニューラルネットワーク]]"
updated: 2026-11-26
---
# 外的報酬と内的報酬の価値分解

## 元の材料

[RFD0121　外的報酬と内的報酬の価値分解](../../../memo/0121_extrinsic_intrinsic_value_decomposition_10rfd.md) blob:9a9fa9d4c51ccb754fd48c1654ba71a9abafba2e

## 要点

- DefaultDQN の TD・QR・IQN で、混合報酬を一つの Q で学ぶ従来方式と、二つの Head で学ぶ方式を選べるようにする。
- 内的価値の Head は primary と同じ Network に置き、Body のどの出力を使うかで特徴の共有範囲を選ぶ。
- 分解時は内的報酬に β を掛けずに学習し、行動選択などで二つの Q を合わせるときに β を使う。
- 内的損失には別の重みを設け、共有 Body を通る勾配の大きさを調整できるようにする。
- 既定の混合方式の再現と、外的・内的の成分、評価時に使った値の対応を検証する。

## 内容

### 目的と問題

外的と内的の大きさの釣り合いに、結果が大きく左右される。 β 0.01・Munchausen 無効の Run は、Greedy の 2500 点が 20.2〜29.2M に 99/160と 81/160だった。β 0.1 の Run は Munchausen の有無によらず 0 本だった。seed だけを変えた Run で、内的報酬の平均が 8M 以降に 1.6〜3.2 倍違った。1 本の Q では、外的と内的の大きさ・分散の違いを、NN が 1 つの出力で吸収するしかない。
内的だけ、割引率や終端の扱いを変えられない。 Burda らは、内的の収益を game over で切らず、内的の割引率を外的より小さくして成績を上げた。1 本の Q では試せない。
Head ごとの価値を見られず、内的を外した行動を評価できない。 2500 点を取ったあと、外的報酬の残らない局面で Q が行動の間でほぼ同じ値になり、Greedy が動かなくなった。この局面で、外的の価値と内的の価値がそれぞれどうなっているかを分けて見る手段が無い。

### 背景

seed だけの違い。 seed 2 の Run は、b001 と同じ実行体・同じ設定で seed だけが違う。誤差の標準偏差は b001 の 0.55〜0.56 倍、内的報酬の平均は 8M 以降の 4M 窓で 1.6〜3.2 倍だった。
2500 点の前にも止まる。 seed 2 の Run の 29.94M の重みを EvalPanelで見ると、鍵を取った直後に Q が 18 行動とも 2.692〜2.698 で、命を失う行動を選んだ。剣を取ったあとは梯子の下で DOWN を押し続け、Q は 1.654〜1.663だった。外的の価値が大きい局面で、行動間の差だけが小さい型である。
見立て（未検証）。 2500 点のあとの局面では、外的報酬が近くに残らないので、Q は内的報酬（1 step 3e-4 を約 330 step 積んだ 0.1 前後）だけでほぼ一様になり、行動を選ぶ向きが無くなる。鍵の直後・剣のあとの型は、この見立てでは説明できない。
Taiga ら。 Rainbow＋RND は Montezuma の 5 seed すべてが 90M フレームまでに 4,400 点を超えた（図 7 を目で読んだ値）。NoisyNets の 5 seed は 35〜40M フレームから 200M フレームまで 2500 点にとどまった。同じ RND でも 2500 点は天井ではない。

### 起草時点の現状

取り込み: `DefaultDQNAgent::UpdateFromBatch` が、報酬スケーラを通した外的報酬に、内的報酬 β × (α − 1) を足して 1 本の報酬にし、内側の Learner へ渡す（default_dqn_agent.cpp）。`next_state` を採点するとき、`intrinsic.use_done_mask` が真なら done の lane の内的報酬を 0 にする。内的報酬を持つのは DefaultDQNAgent だけである。
終端と打ち切りが同時に真の遷移: 打ち切りの slot が挿入され、遷移は終端として保存される。返る次観測は打ち切り時点の観測で、この観測から始まる遷移は無い（学習されない）。
Learner: 1 回の更新の進行は、基底の `dqn::Learner::UpdateFromSamples` に 1 本である（dqn_based_agent.hpp。RFD0125）。分布の表し方ごとの違いは、目標値（`MakeTarget`）と要素誤差（`ComputeElementError`）の上書きである。目標値の式「収益 ＋ γ^n ×（終端でない）× bootstrap」は `MakeBellmanTarget` の 1 か所にある。
`q`・`q_dist` の読み手: forward の結果の `q`・`q_dist` を直接読む箇所が、dqn_based_agent.cpp に 17 か所ある。行動の方策 3 種、bootstrap の行動の選択、Munchausen の目標値、NoisyNet の診断、policy churn の probe、学習のメトリクス、Actor の hint である。TBO が有効のとき、これらは h で圧縮した値の `q` を前提にしている。

### 確定事項

β は、Q を合わせるときに掛ける。 Replay の内的の成分は、β を掛けない値にする。内的の Head が学ぶ値は β に依らない。`intrinsic.beta` の設定キーは今のまま使う。
bootstrap の行動。 内的の Head は、常に合わせた Q で選んだ行動を使う。外的の Head は、`learner.intrinsic.extrinsic_bootstrap_mode`（`combined`｜`extrinsic`）で選ぶ。`combined`（既定）は合わせた Q で選んだ行動、`extrinsic` は外的の Q で選んだ行動を使う。Double DQN が有効なら online 側の値で行動を選び、target 側の各 Head で評価する。無効なら target 側の値で選ぶ。
終端の扱い。 `learner.intrinsic.terminal_mode`（`episodic`｜`non_episodic`）で選び、既定は `episodic`（終端で切る）。`non_episodic` のとき、内的の Head は終端の遷移でもマスクせず、Replay が返す次観測から bootstrap する。この次観測は、終端のあとの最初の観測である。終端と打ち切りが同時に真の遷移だけは、`non_episodic` でも bootstrap しない。n-step の和は終端で切れたままにする。終端の扱いのために、Replay が返す次観測・終端・打ち切りは変えない。打ち切りの扱いは今のまま（両方の Head が bootstrap を続ける）。
TempoRLとの併用は ERROR にする。 `decomposed` と `skip.enabled = true` を同時に指定すると、構築時に止める。

### 仕様

`discount_mode = shared`・`terminal_mode = episodic`・`extrinsic_bootstrap_mode = combined` のとき、y_外 ＋ β·y_内 は、今の 1 本の Q の y と同じ値になる。TBO が有効のときは、h を掛ける前の値で成り立つ。
π_外 は、`extrinsic_bootstrap_mode` が `combined` なら π、`extrinsic` なら外的の Q から作った方策である。Munchausen項も、同じ方策から作る。
方策を online の値から作るか target の値から作るか（`log_policy_mode`）は、今の決まりのままで、使う値を合わせた Q（`extrinsic` のときの外的の Head は外的の Q）に替える。
損失の効く範囲。 損失の折れ目（Huber と分位点の κ）と TD 誤差の clip は、外的と内的で共用する。内的の Head は値の桁が大きいので、今の 1 本の Q とは違う範囲（誤差が折れ目より大きい側）で学ぶことが多くなる。

### ほかとの関係

入力側の分け方と、価値の分解は、別の軸である。j は「どの β・γ の組の価値を求めるか」を指定し、価値の分解は「外的と内的を別の出力で学ぶか」を決める。Agent57 は NGU に価値の分解を重ねた。したがって本 RFD は 173 と食い違わず、173 が j を足すときは、2 つの Head の両方に j を入力する。
価値を分けるだけで、全方策の γ や bootstrap の方策の整合が解決するわけではない。本 RFD は方策 1 つなので、この問題を持ち込まない。
127（Trace の内的報酬の列）と本 RFD は、どちらが先に入ってもよい。あとから入る側が、Trace のボーナスの列を「内的報酬用の報酬スケーラを通した値に β を掛けた値」に合わせる。

### 実装契約

内的の Head は、読む Body の出力の形を、外的の Head と同じ決まりで検査する（IQN なら τ を混ぜた出力であること、など）。
slot に、成分ごとの n-step の収益と、成分ごとの 1 step の報酬を持つ。
割引率が定数の成分は、Push の時点で収益を保存する。schedule の成分は、抽選のあとに保存済みの step の報酬から作り直す。
終端、打ち切り、次観測、実際に進んだ step 数、抽選の分布、item key、PER の completion と eviction 統計の ready の基準、Push と Extract の lock、index の昇順は変えない。

### 成功の測り方と受入

seed。 同じ構成でも seed 1 と 2 で Greedy の 2500 点が 99/160 と 81/160 に分かれたので、群ごとに 2 本以上にし、この幅を超える差だけを読む。
仮説 B の群。 分解した Q で、内的を外的と同じ扱いにした群と、`separate`・`non_episodic` にした群。内的報酬用の報酬スケーラを `return_std` にし、`intrinsic.use_done_mask` を偽にした群を足すと、Burda らの形に寄る。
読む指標。 評価側（Greedy と ε 0.01）の 2500 点の割合と平均点、訪れた部屋の種類数、Head ごとの Q（Trace とメトリクス）。行動の β を 0 にした評価を足すと、2500 点のあとの局面が外的の Head にもあるかを分けられる。
途中の時期の差。 鍵や扉に届いた時期の差は、Run ごとの揺れの内とみなし、判定に使わない（起点の方向 (1)）。

## 主要な概念

- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [内的報酬による探索の比較](../../concepts/methods/intrinsic_novelty.md)
- [報酬尺度と割引率](../../concepts/methods/reward_scale_discount.md)
- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
