---
title: PPO 系と方策ベースの学習の共通部品
page_kind: サマリ
material: "[[docs/memo/0209_ppo_family_and_on_policy_core_parts_10rfd.md|0209_ppo_family_and_on_policy_core_parts_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/framework/326_ppo.md|326 Agent個別実装 PPO系列]]"
  - "[[docs/wiki/concepts/framework/321_agent_core.md|321 Agent 共通基盤]]"
  - "[[docs/wiki/concepts/framework/318_rl_core.md|318 強化学習 共通基盤]]"
  - "[[docs/wiki/concepts/framework/315_nn.md|315 ニューラルネットワーク]]"
  - "[[docs/wiki/concepts/framework/314_tensor.md|314 テンソル処理]]"
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
updated: 2026-11-28
---
# PPO 系と方策ベースの学習の共通部品

## 元の材料

[RFD0209　PPO 系（方策ベースの Agent の系統）と、方策ベースの学習に要る core の共通部品](../../../memo/0209_ppo_family_and_on_policy_core_parts_10rfd.md) blob:b9be7ca2cdd7d7ede1a4b468dd0e9079ced627f0

## 要点

- 暫定の起票（2026-11-25）で、グリルはしておらず、確定した決定は無い。決めることはすべて §12 の未決表に「暫定の既定」として置き、実装の入口のレビュー、実装で分かった点、実装のあとのレビューで改訂した。2026-11-28 に番号を 0202 から 0209 へ振り直した。
- DQN 系に続く 2 本目の Agent の系統「PPO 系」を `src/algo/ppo`（ライブラリ `anet-algo-ppo`、初期化 `InitPpo()`、class id `PPOAgent`）に立て、最初は離散行動の PPO だけを作る。
- core に、行動の分布（`CategoricalDistribution`）、advantage と value の目標の推定（`ComputeGae`）、Head の合成（`CompositeHeadFactory`）、`LaneTimeTensorStore` の全体の読み出し（`GetAll`）、活性化の module `Tanh` を足す。損失、rollout の並べ替え、optimizer の組み立ては PPO 系に置く。
- Runner の今の契約（`UpdateFromBatch` は 0 件以上の更新結果を返せる）のまま、Learner が rollout をため、たまったら optimizer の step 1 回につき 1 件の更新結果を返す。Runner・Observer・`rl.hpp` の口と DQN 系は変えない。
- Network は 1 つで、Head が受け取る 2 つのキー（`policy_feature`・`value_feature`）を向ける branch の設定だけで、Body を共有にも完全に別にもできる。出荷の CartPole は別にする。
- 打ち切りは既定で次の観測の value で bootstrap し、終端と同じに扱う形も選べる。rollout の方策の古さの最大が 2 以上なら止め、写しを持つ Actor を Train に使った構成を通さない。
- CPU と CUDA の両方で動かし、device 間の同期の転送を経路ごとに数える。PPO の診断のメトリクスは、群 78・79 のプロファイル `metrics.scalar.@ppo` 1 つにそろえる。
- 出荷の既定値は CleanRL の `ppo.py` に合わせる。受入は、テストで確かめる 18 項と、CartPole の Run で Greedy の評価の step 数が上限 500 に届くこと（Run は開発者の了承のあと）である。
- DQN 系の中にあってほかの系統でも要る 8 つの部品は今回は移さず、core へ上げる条件と無回帰の確かめ方を表にした。連続行動、RNN、RND、A2C、PPG、SAC などは、差し込み位置と作る条件だけを書く。

## 内容

### 解く問題と現状（§1・§2）

強化学習の Agent はどれも価値を学んで行動を選び、方策を直接学ぶ Agent が無い。方策ベースの学習に要る部品（行動の分布、advantage の推定、ためてから学ぶ器、方策と価値の 2 つの Head）のどれが core にあり、どれが DQN 系の中にあるかが整理されておらず、系統どうしは依存できないので、DQN 系の中の部品は PPO 系から使えない。

コードで確かめた現状は次のとおり。Runner は 1 env step ごとに `UpdateFromBatch` を呼び、返った件数だけ `learn_step` を進め、結果が空なら `LearnEvent` を出さない。定期評価は `LearnEvent` でだけ発火し、1 回の `LearnEvent` に複数の結果があれば LEARN 軸は 1 件ずつ、ほかの軸は平均で記録する。Pipeline の Train Runner では行動選択と学習は重ならず、まとまりの最初の 1 step だけ更新の前の Network で行動が選ばれる。Agent は Runner の種類を受け取らず、Train の Runner は Actor の `Sync()` を呼ばない。経験は CPU 上にあり、Actor は `BatchActionInfo` の `info` で値を Learner へ運べる。2 つの Head を並べる汎用の factory と `Tanh` は core に無く、`LaneTimeTensorStore` にはためた範囲の全体を複製せずに読む口が無い。Forwarder の並びの仕上げは DQN 系の中にある。anet の CartPole は Gymnasium の CartPole-v1 と上限の step、報酬、角度の上限が違い、評価の開始の状態が毎回同じなので、Greedy の評価は同じ重みなら同じ 1 本になる。CleanRL の `ppo.py` と SB3 の原文を読み、既定値と式を表にした。GAE・比・approx KL・方策の損失・advantage の標準化の式は同じで、違うのは既定値（収集の長さ、ミニバッチ、epoch、学習率、entropy の係数）、打ち切りの扱い（CleanRL は終端と同じに扱い、SB3 は γ·V を報酬に足す）、value の損失の clip と 0.5 倍、KL で止める位置（CleanRL は epoch の終わり、SB3 はミニバッチごとに閾値の 1.5 倍）である。CleanRL は方策と価値を別の MLP にし、直交初期化を使う。

### ゴール・非ゴール・成功の測り方（§3）

ゴールは、系統の新設、今の Runner の契約のままの学習、共通部品の core への配置と分担の表、設定だけで共有にも別にもできる Network、CPU と CUDA の両対応と同期の転送の最小化、PPO の診断のメトリクスを既定のプロファイル 1 つでそろえること、あとで足すものの差し込み位置と作る条件の 7 つ。非ゴールは、DQN 系のコードと動き、Runner・Observer・`rl.hpp` の口、連続行動、Forwarder の並びの結線、報酬と観測の正規化・AMP・可塑性と勾配ノイズの診断・方策の確率の GUI 表示、正式な Run と実験記録の 6 つで、それぞれに退けた案と、本当の制約かを書く。成功は、テストの受入、CartPole の Run、§10 の表の各行がそろうことで測る。

### 前提と用語（§4・§5）

前提は 29 行で、既存部品の働き、守る線（系統のライブラリは兄弟に依存しない、汎用層は特定の Agent の名前を持たない、毎 step の同期の転送は見積もってループに置かない、購読の無い診断は計算しない、共有の Network は通常 eval）、過去の判断（Perception の作り方、stack 軸、強制行動、Network の所有と合成 Head、interval の発火、メトリクスの群の番号）、参照実装の式と既定値を挙げる。用語は、rollout（学ぶ前にためる続いた intake のまとまり。用語集の「rollout（補助学習）」を広げる）、epoch、方策の世代、方策の古さ（`policy_lag`）、advantage、value の目標で、DQN 系の `replay_ratio` や policy churn とは区別する。

### 設計の概要（§6）

core に `distribution`・`advantage` の機能グループを新設し、`nn_head`・`nn_modules`・`tensor_store` に足す。PPO 系は `ppo`・`ppo_config`・`ppo_network_model`・`ppo_math`・`ppo_actor`・`ppo_learner`・`ppo_agent` の 7 つに分ける。Actor は毎 step、観測を device へ移して長さ 1 の stack 軸を足し、Network を 1 回通して分布から行動を選び、log 確率・value・Network の入力・方策の世代を行動の情報に載せる。Learner は device 上の値と CPU 上の報酬・印を別にため、rollout の長さに届いたら、報酬と印を 1 回で device へ送って GAE で advantage と value の目標を作り、epoch ごとに並べ替えたミニバッチで更新する。次の観測の value は、終端、打ち切り（`bootstrap` と `terminal`）、rollout の最後、そのほかの 5 行の表で決める。Train の Actor は Network を共有し、評価の Actor は写しを `Sync()` で写し直す。Pipeline の Runner では、まとまりの最初の 1 step の経験だけ方策の古さが 1 になるが、記録した log 確率は行動を選んだ方策の値なので比の計算は成り立つ。写しを持つ Actor を Train に使うと重みが古いままになるので、古さの最大が 2 以上の rollout で止める。評価は rollout の更新の `LearnEvent` でだけ発火し、間隔は `learn_step` で数え、倍数で書く決まりは置かない。出荷の CartPole は Serial の Runner で、Greedy と抽選の 2 本の評価を持ち、target の評価は休止にする。

### 実装契約（§7）

分布は logits と同じ device で float32 にしてから正規化し、値を CPU へ読み戻さない。`ComputeGae` は時刻 × lane の入力から同じ形の advantage と value の目標を返し、勾配を作らない。`CompositeHeadFactory` は名前・受け取るキー・factory の組から 1 つの Head を作り、組が 0 個、名前の空・重複・区切りの「.」、キーの不足、出力の重複で止まる。設定キーは接頭辞 `PPOAgent` の下に表で定め、型・値域の外は構築時に完全キー・指定値・期待する範囲を示して止める。使わない側のキーも検証し、値を丸めない。`rollout.length × lane 数` がミニバッチの数で割り切れないときと、標準化するミニバッチが 2 件未満のときも止める。時間の値（学習率、比の clip の幅、entropy の係数）は rollout が埋まった intake の exp step で 1 回読む。KL で止める判定は epoch の終わりだけで、返る件数は実行した epoch の数 × ミニバッチの数である。メトリクスは `78_agent_ppo`（損失、KL と clip の割合、比の統計、value と advantage の統計、勾配の内訳と clip、schedule の実効値）と `79_agent_policy`（entropy、行動ごとの確率と頻度、終端と打ち切り、方策の古さ）で、購読があるときだけ計算し、rollout に 1 回まとめて CPU へ写す。同期の転送は、毎 step の観測と行動、打ち切りの value、rollout の終わり、更新結果の読み出し、KL の判定の 6 経路で、ループの中には置かない。保存は形式名・設定・Network・optimizer で、共有の Network は Learner の更新の区間だけ train にする。

### 移行対象と受入（§8・§9）

core と PPO 系のコードとテスト、ビルドと Runner への登録、検証資材（ライブラリの依存の検査、設定キーの期待一覧、設定の既定の検査が読む env の設定の一覧）、出荷の設定（`agent.txt`・`nn.txt`・`metrics_scalar.txt`・`CartPole_ppo.txt`）、設計文書 115・230 の新設とほかの設計文書の追随、用語集、`categories.json`、wiki を挙げる。英語版の設計文書は直さず、旧い契約のための仕組みは足さない。受入は 19 項で、ビルド、分布・GAE・Head の合成・`GetAll`・`Tanh` の手計算との一致と不正な入力で止まること、PPO 系のテストの CPU と CUDA、共有と別の切り替え、rollout の件数、更新前の最初のミニバッチで比が 1、打ち切りの扱い、写しの Actor で止まること、小さな課題の学習（hidden）、メトリクスの鍵、設定の不正、設定ファイルの検査、既定スイート、`src/algo/dqn` の差分が空であること、CartPole の Run（3 seed・500,000 step で Greedy の step 数が 500 に届き、400,000 step から先の評価の点の 8 割以上が 500）である。

### DQN 系から core へ上げる候補（§10）

rollout の並べ替え、Actor の写しと同期、optimizer の組み立て、購読ごとの診断の間引き、2 つの Head を並べる factory、Perception の仕上げと spec の照合、設定の検証の補助、RND の段取りの 8 つについて、行数、PPO 系の今回の扱い、移す条件、移したときに DQN 系で変わる所を表にした（移す条件の例: rollout の並べ替えは 3 つ目の利用者が出たときか RND の補助学習を PPO 系で使うとき、Actor の写しは写しを持つ Train の Actor を PPO 系で使うとき）。移すときの無回帰は、dqn と core のテスト、設定の基準、同じ seed の Run の全点の照合（4 環境、決定的なバックエンド。RND に触る行は RND を有効にした構成を足す）、ライブラリの依存の検査で確かめる。

### あとで足すもの（§11）

連続行動、Actor と Critic の別の入力、Forwarder の並び、RNN、RND、A2C、PPG、V-trace・TD(λ)、SAC（別の系統）、報酬の正規化、AMP、写しを持つ Train の Actor の同期、可塑性と勾配ノイズの診断、方策の確率の GUI 表示について、差し込み位置、作る条件、先送りの理由（別の関心か、実験の結果が要るか）を書く。可塑性と勾配ノイズの診断では、RFD0092 が置いた再開条件（対象の Learner が具体化）を本 RFD で満たすとした。

### 未決表と材料（§12〜§15）

U1〜U30 に暫定の既定を置く。主なものは、core の範囲（U2）、Network の形（U4）、Serial と Pipeline の両方で動かすこと（U5）、打ち切りの既定（U6）、rollout に Actor が作った Network の入力を運ぶこと（U10）、Forwarder を結線しないこと（U11）、DQN 系を移さないこと（U17）、機能カテゴリ 326 と Topic（U18）、CartPole の測り方（U19）、既定値（U20）、KL で止める形と閾値 0.015（U23。OpenAI Spinning Up の値に合わせた）、勾配の clip の上限の名前（U30）である。再グリルでは U4・U10 を先に見て、U2・U17・U19 を続ける。材料は、既存のフレームワークの共通部品の切り方の調査と core・algo の下調べ（どちらもリポジトリに無い）、別の計画の文書の 3 つ。先行と並行の RFD（0200・0142・0108・0201）とは触る箇所が重ならない。

## 主要な概念

- [326 Agent個別実装 PPO系列](../../concepts/framework/326_ppo.md)
- [321 Agent 共通基盤](../../concepts/framework/321_agent_core.md)
- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
- [314 テンソル処理](../../concepts/framework/314_tensor.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
