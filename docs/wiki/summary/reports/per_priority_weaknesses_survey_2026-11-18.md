---
title: "Survey: PER の優先度の弱点と代替"
page_kind: サマリ
material: "[[reports/per_priority_weaknesses_survey_2026-11-18.md|per_priority_weaknesses_survey_2026-11-18]]"
material_kind: 報告
concepts:
  - "[[docs/wiki/concepts/methods/prioritized_replay.md|優先度付き経験再生（PER）]]"
  - "[[docs/wiki/concepts/methods/replay_ratio_lane_window.md|Replay ratioとlane窓]]"
  - "[[docs/wiki/concepts/methods/tau_sampling.md|τ配置とIQN比較]]"
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
updated: 2026-11-26
---
# Survey: PER の優先度の弱点と代替

## 元の材料

[Survey: PER の優先度の弱点と代替](../../../../reports/per_priority_weaknesses_survey_2026-11-18.md) blob:b4e82a6d386768a49c86756f42a9c15c82ee311f

## 要点

- PER は TD 誤差の大きい遷移を多く引き、偏りを重要度重み（IS、指数 β）で補正する。優先度の強さは α で決める。本冊は、TD 誤差の大きさが「学ぶ値打ち」を表さない場面、優先度が古くなる仕組み、抽選が実は損失の形を変えていること、分布型 RL での定義の違い、そして TD 誤差に代わる優先の仕方を、原典の記述で並べる。
- ReLo は、固有のノイズで TD 誤差が何度引かれても高いままの遷移に PER が高い優先度を与え続けると指摘し、7×7 の格子で一様乱数の報酬の状態を置くと、PER が序盤その状態への遷移を引き続けて本来のゴールを学べなかったことを示した。報酬にノイズを足した DMC でも、SAC+PER の TD 誤差が最も高いままだった。[Sujit ReLo (2023/11), §3、§4.1〜4.3、Figure 2・4a]
- LAP は、優先度で抽選した損失と一様に抽選した別の損失の期待勾配が一致する条件を示し、損失 |δ|^τ に PER を組むと一様抽選で |δ|^(τ+α−αβ) に比例する損失を使うのと等価だとした。MSE の最小化は目標の平均、L1 は中央値を与えるので、τ+α−αβ が 2 でなければ PER の目的には偏りがあり、MSE と組んで β<1 なら冪が 2 を超えて外れ値に寄る。[Fujimoto LAP (2020/10), §4 定理1、§5.1 定理3]
- EBU: 優先度を付けず、Replay からエピソード 1 本を一様に選び、全遷移の target を後ろから再帰的に作る（拡散係数 β で過大評価を抑える）。49 ゲーム・10M フレームで平均 253.55%・中央値 51.55%（PER は 156.57%・40.86%）、適応版の 20M フレームで 347.99%・92.50% だった。Montezuma's Revenge は DQN 比でほぼ 0 だった（図からの読み取り）。[Lee EBU (2019/12), §3.2〜3.3、Algorithm 2、表2、図4]
- Mattar & Daw は、backup の期待価値（EVB）を、方策が変わることによる期待収益の増加（gain）と、その状態をこの先訪れる見込み（need、successor representation から計算）の積に分解した。Prioritized Sweeping との違いとして、gain は方策を変える更新だけを数え、need は訪れそうな状態へ backup を偏らせると述べる。表形式・モデルベースで、gain を実行前に知る前提は非現実的だと自ら認める。[Mattar & Daw (2018/10), Methods §4.2〜4.5 式4〜6・13、Discussion]

## 内容

### 2. TD 誤差の優先度の弱点

原論文の付録は、|δ| は報酬・遷移の確率性や部分観測、関数近似の容量の限界を無視するので、学びようのない遷移があると問題になると述べ、|δ| の変化分を代案に挙げたが、予備実験では |δ| を上回らず、調べた環境がほぼ決定的だったためかもしれないとする。[Schaul PER (2016/02), 付録 A]

### 3. α と β の選び方

原論文は β を学習の終盤に 1 へ上げる理由を、不偏であることが最も重要になるのは収束の近くだからとし、α と β は相互作用し、IS は実効的な step size を下げるとも書く。完全な IS は序盤の学習を遅くしたが、早すぎる収束の危険は小さかった。[Schaul PER (2016/02), §3.4、付録 B.2.1、Figure 12]

### 4. 分布型 RL での優先度

関連する事実として、BTR の実装は τ を forward のたびに引き直し、その学習 step の τ から作った値で優先度を毎回上書きする（既定 taus=8）。Toromanoff らは IQN 系で損失の分布が広がり引かれすぎる遷移が出ると報告するが、原因を τ のサンプリングに帰してはいない。[BTR 実装 (2026/11), networks.py `calc_cos`、Agent.py；Toromanoff Rainbow-IQN (2019/08), §4]

### 7. 優先度を計器として出す例

LaBER は、代理の抽選分布・一様分布と最適分布の全変動距離を毎 step 計算してヒストグラムにした。[Lahire LaBER (2022/07), §4、Figure 2]

## 主要な概念

- [優先度付き経験再生（PER）](../../concepts/methods/prioritized_replay.md)
- [Replay ratioとlane窓](../../concepts/methods/replay_ratio_lane_window.md)
- [τ配置とIQN比較](../../concepts/methods/tau_sampling.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
