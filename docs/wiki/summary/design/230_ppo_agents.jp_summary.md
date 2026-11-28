---
title: PPO系Agent
page_kind: サマリ
material: "[[docs/design/230_ppo_agents.jp.md|230_ppo_agents.jp]]"
material_kind: 設計文書
concepts:
  - "[[docs/wiki/concepts/framework/326_ppo.md|326 Agent個別実装 PPO系列]]"
  - "[[docs/wiki/concepts/framework/315_nn.md|315 ニューラルネットワーク]]"
  - "[[docs/wiki/concepts/framework/318_rl_core.md|318 強化学習 共通基盤]]"
  - "[[docs/wiki/concepts/framework/321_agent_core.md|321 Agent 共通基盤]]"
  - "[[docs/wiki/concepts/framework/314_tensor.md|314 テンソル処理]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
updated: 2026-11-28
---
# PPO系Agent

## 元の材料

[PPO系Agent](../../../design/230_ppo_agents.jp.md) blob:3305198a9c288889ad6e2e719da3f54da092186d

## 要点

- PPO系は方策を直接学ぶAgentの系統で、最初の`PPOAgent`が離散行動のPPOを実装する。経験をReplayにためず、今の方策で集めたrolloutを1回の更新で使い切る。
- `src/algo/ppo`に置いてcoreにだけ依存し、公開するのは初期化関数`InitPpo()`だけである。`PPOAgent`はRunnerの種類と評価の間隔を知らない。
- Networkは1つで、方策と価値の2つのLinear Headをcoreの`CompositeHeadFactory`で並べる。Bodyの共有と分離は、2つのキーを向けるbranchの設定だけで決まる。
- Actorは選んだ時点のlog確率・value・Networkの入力・方策の世代を行動の情報に載せる。Learnerはそれをためて、rolloutが埋まったらadvantageを推定し、epoch×ミニバッチで更新して、optimizerのstep 1回につき1件の更新結果を返す。
- 方策の古さの最大が2以上のrolloutは止める。Serialでは古さが0、Pipelineではrolloutの最初の1 stepだけ1になる。
- 設定の既定はCleanRLの`ppo.py`に合わせ、メトリクスは`metrics.scalar.@ppo`の群78・79にまとめる。
- CPUとCUDAで動き、device間の同期の転送は6つの経路に限られ、laneや行を回すループの中には無い。

## 内容

### 1. はじめに

PPOは、新旧の方策の確率の比を`[1 − ε, 1 + ε]`にclipした目的関数で、1つのrolloutで方策が動く幅を抑え、同じrolloutを複数のepochに使えるようにする。PPO系は損失の式と更新の進め方を持ち、行動の分布・advantageの推定・Headの合成はcoreの部品を使う。

### 2. 部品の構成

`PPOAgent`（生成・資源の所有・保存）、`NetworkModel`（Networkの組み立て・写し・方策の世代）、`Actor`（行動選択）、`Learner`（rolloutの蓄積・advantageの推定・更新・optimizer）、`UpdateResult`、`PPOAgentConfig`、損失と統計の純粋な関数（`ppo_math`）に分ける。NetworkはAgentのResourceで、学習のActorは共有し、評価のActorは写しを`Sync()`で写し直す。

### 3. Network

出力は`logits`（`[行, 行動数]`）と`value`（`[行, 1]`）で、方策のHeadは`policy_feature`、価値のHeadは`value_feature`を受け取る。共有すると価値の損失の勾配が方策のBodyにも流れ、`learner.value_coef`がその強さを決める。入力は観測の各キーに長さ1のstack軸を足した`[lane, 1, …]`で、各branchはstack軸を畳むmoduleで始め、出力の形が合わない設定は構築時に止まる。Headは直交初期化で、方策はgain 0.01、価値は1.0である。

### 4・5. 1 stepの流れと行動選択

毎stepにNetworkを1回通し、`action_mode = sample`は分布から抽選し、`greedy`は確率が最大の行動を選ぶ。強制行動は方策の選択のあとに反映し、強制した行動のlog確率を記録する。行動の情報の鍵は`log_prob`・`value`・`policy_version`・`input.<観測キー>`である。

### 6. rolloutとadvantage

deviceの器にNetworkの入力・行動・log確率・valueを、CPUの器に報酬と印をためる。Learnerは経験の観測をdeviceへ送り直さないので、収集時と更新時のlog確率が同じ入力から作られる。ほかの系統のActorの行動や、形とdeviceの合わない経験は器に入れる前に止まる。次の観測のvalueは、終端、打ち切り（`bootstrap`か`terminal`）、rolloutの最後、そのほかの5行の表で決める。方策の世代はrolloutの更新ごとに1進み、Learnerは更新の前に古さを数える。

### 7. 更新

報酬と印を1回でdeviceへ送ってadvantageを作り、`T × N`個の遷移をepochごとに並べ替えてM個のミニバッチにする。時間の値はそのintakeのexp stepで1回読む。損失は`L_policy − c_entropy · H + c_value · L_value`で、方策の損失はclipした比を使い、valueの損失は任意でclipし、advantageはミニバッチごとに標準化できる。勾配は全体のノルムでclipし、optimizerは方策と価値で1つのAdamWである。`kl_stop.mode = epoch`は、epochの最後のミニバッチのapprox KLが閾値を超えたら残りのepochをやめる。返る件数は`K × M`か、実行したepochの数×`M`である。

### 8. 設定

接頭辞`PPOAgent`の下のキーの型・値域・既定を表にする。値域の外は構築時に止め、使わない側のキーも検証する。学習率を0へ下げるscheduleはenvの設定が書き、出荷のCartPoleは500,000 stepで線形に下げる。

### 9. メトリクス

`78_agent_ppo`は損失、approx KL、clipの割合、比の統計、explained variance、valueとadvantageの統計、勾配の内訳とclip、scheduleの実効値、`79_agent_policy`はentropy、確率が最大の行動の確率、終端と打ち切りの割合、episodeの数、方策の古さを持つ。行動ごとの鍵`action_prob_<n>`・`action_freq_<n>`のtagはenvの設定が書く。統計は購読があるときだけdeviceの上で作り、rolloutに1回CPUへ写す。成績はRunnerの鍵で、Greedyと抽選の2本の評価を読む。

### 10〜13. 評価・device・排他・保存

評価はrolloutの更新の`LearnEvent`でだけ発火し、intervalはlearn stepで数え、`[eval_target]`は休止にする。同期の転送は毎stepの観測と行動、打ち切りのvalue、rolloutの終わり、更新結果の読み出し、KLの判定で、小さなNetworkと少ないlaneではCPUのほうが速い見込みである（測っていない）。Learnerはミニバッチ1つの区間ごとにだけ排他のロックを持ってtrainにする。保存はNetworkとoptimizerで、rollout・方策の世代・乱数は保存せず、checkpointの引き継ぎで続ける使い方は確かめていない。

### 14. テスト

`anet-algo-ppo-test`のタグで確かめ、deviceを受けるテストはCPUとCUDAで走る。既定の`[ppo]`は損失、Network、Actor、Learner、設定、保存、製品のRunnerを通す確かめ、小さな課題の学習、出荷の設定を含み、hiddenの`[ppo_learn]`と`[ppo_shipped_cartpole]`がある。小さな課題は長さ8の一本道（`CorridorEnv`）である。

## 主要な概念

- [326 Agent個別実装 PPO系列](../../concepts/framework/326_ppo.md)
- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
- [321 Agent 共通基盤](../../concepts/framework/321_agent_core.md)
- [314 テンソル処理](../../concepts/framework/314_tensor.md)
- [評価と比較の条件](../../concepts/measurement/evaluation.md)
