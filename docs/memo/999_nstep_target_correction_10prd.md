# n-step の目標値の補正（窓の途中の Munchausen の項・方策オフの補正）暫定 PRD

> 2026-10-06 起票。詳細設計は別途グリルで行う前提で、決めることを §5 に残した。原因の候補が違う 2 つの補正（§3.1 と §3.2）を 1 本にまとめてある。詳細設計の段階で分けてよい。
> 2026-10-23 改訂: §2 の Q の水準と softmax のエントロピーを、経験数の窓で引き直した値に直し、比較の 2 本の Run 名を今のフォルダ名に直した（[2026-10-01_atari-feedback.md](../experiments/default-dqn/dropmerge/2026-10-01_atari-feedback.md) の探索ブロック 07 の判断更新）。
> 起点: DropMerge で n-step 3 を試した Run が n = 1 より約 1 割低く、NEET が大幅に悪化したこと（2026-10-06）。DropMerge では以前から、n-step を増やすと局所解に寄る懸念があった（[2026-08-09_iqn.md](../experiments/default-dqn/dropmerge/2026-08-09_iqn.md) の「P2: NEETへの介入」）。
> 関連: [done/067_MunchausenRL_10prd.md](done/067_MunchausenRL_10prd.md)（D3: bonus は n-step return の先頭へ 1 回だけ）、[999_decision_aligned_nstep_10prd.md](999_decision_aligned_nstep_10prd.md)（窓を延ばす n-step。本 PRD の上に載る）、[999_smdp_transition_duration_10prd.md](999_smdp_transition_duration_10prd.md)、[CONTEXT.md](../../CONTEXT.md) の「NEET」。

## 1. 目的と問題

n > 1 の n-step return を、Munchausen と強い探索（DropMerge の spatial exploration）がある設定でも、n = 1 より悪くならない形で使えるようにする。n-step は報酬の遡りを速めるはずだが、DropMerge では逆に成績を下げている。原因の候補は、互いに独立な次の 2 つである。

1. **窓の途中の Munchausen の項が抜けている:** 今の実装は、Munchausen の bonus を n-step return の先頭の遷移にだけ足し、soft value は窓の終わりの bootstrap にだけ入れる（PRD067 の D3、BTR 互換）。1-step の再帰では、途中の各 step にも bonus と soft value のエントロピーの項が入る。そのため、今の n-step の目標値は、M-DQN の作用素を n step に展開した形と一致しない。先頭の bonus は replay の各遷移に掛かるので、判断の時点で行動の差を広げる働き自体は残る。
2. **探索の手が目標値に混ざる:** 補正のない n-step return は、窓の途中で挙動方策が実際に打った手の報酬と、その結果の状態をそのまま含む。ε の大きい lane のランダムな手が混ざると、精密な続きが要る組み立ての価値が低く見積もられ、安全で伸びない打ち方に寄る（局所解）。n = 1 の Q 学習では、探索の手は「どの状態を見たか」の広さにだけ効き、目標値には入らない。

どちらがどれだけ効いているかは、まだ切り分けていない。

## 2. 現状

- n-step の積み上げは replay で行い、学習側の γ を使う（`core/anet-core/src/dqn_based_agent.cpp:2558`）。目標値の割引は `pow(γ, n_steps)` で、分布の目標値が `:3253`、Munchausen の分布の目標値が `:3334`、スカラーの目標値が `:3622`・`:3654`。
- Munchausen の bonus は `alpha * clip(scaled_log_policy, clip_value_min, 0)`（同 `:92`）。集約済みの n-step return の先頭へ 1 回だけ加える（同 `:3333` のコメント、PRD067 の「N-step target」と D3）。PRD067 は、原論文が M-DQN を 1-step、M-IQN を 3-step で評価したことを記している。
- DropMerge の学習用 Actor は UQE（`uqe_tau` 0.95 → 0.85）で行動を選び、spatial exploration（log）で lane ごとの ε を 1e-4〜0.6 の段に固定している。ランダムな手は 59 行動から一様に選ぶので、ほとんどが DROP になる。
  - 全 lane の平均で、約 7% の手がランダムになる。
  - n = 3 の窓の後ろ 2 手にランダムな手が混ざる割合は、全 lane の平均で約 12%、ε 0.6 の lane では 84% になる。
- 2026-10-06 の比較（IQN + Munchausen τ0.03、soft target、NoDropTimeout の罰 −20、seed 1）。n-step 3 は `run_20261006-033243_dmfb_19_nstep3`、n = 1 は `run_20261005-130039_dmfb_16_ndt20`。学習中の評価の 80〜90M は次のとおり。

| 80〜90M | n = 1 | n-step 3 |
|---|---|---|
| return（UQE target / Greedy） | 1,741 / 1,732 | 1,546 / 1,532 |
| NOOP の選択率（UQE） | 32% | 64% |
| NOOP と最善 DROP の差（UQE） | −0.062 | +0.002 |
| エピソード長（UQE / Greedy） | 416 / 580 step | 740 / 1,057 step |
| maxrank の平均 | 10.64 | 10.49 |

  - **NEET:** n-step 3 では、締切の手前まで粘る長考型の NEET が、手の 3 分の 2 を占めた。
  - **NOOP と最善 DROP の差:** Munchausen 無しの QR51 の NEET 期（−0.001〜−0.002）と同じ同点まで戻っている。
  - **Q の水準:** 90〜100M の `37_agent_qtd/31_q_max_real_mean` は n = 1 の 7.5 に対し n-step 3 が 4.1 で、n-step 3 の方が低い。開いたのは Q の差（h 空間）で、0.019 に対し 0.038 と約 2 倍だった（[2026-10-01_atari-feedback.md](../experiments/default-dqn/dropmerge/2026-10-01_atari-feedback.md) の探索ブロック 07）。当初は 80〜90M を 0.38 / 0.79 と書き「全区間で約 2 倍」としたが、経験数の軸の指標を学習ステップの窓で切った値で、学習の最初の約 0.4M 経験を見ていた（同ブロックの判断更新で訂正）。
  - **target 方策の softmax のエントロピー:** 90〜100M で n = 1 が 0.95、n-step 3 が 0.45 だった（同ブロック）。当初は「n = 1 が 3.9〜4.0 で横ばい、n-step 3 は 3.93 から 3.44 へ下がった」と書いたが、Munchausen の指標（`36_agent_munchausen/*`）も経験数の軸で記録されるので、同じ窓の取り違えによる値だった（同じ記録の「成績の 3 つの物差し」の注記）。

## 3. 提案の骨子

### 3.1 窓の途中の Munchausen の項

1. n-step return を積むときに、窓の途中の遷移の Munchausen の項も入れられるようにする。設定で選べる形にし、既定は今の D3 のままとする。
2. 狙いは、M-DQN の作用素を n step に展開した形に揃えること。途中の手を target の方策が打ったとみなすと、途中の step の項は、bonus（α·τ·log π）と soft value のエントロピーの項（−τ·log π）を合わせた (α − 1)·τ·log π になる。実際の途中の手は挙動方策（UQE と ε）なので、その扱いは §3.2 と一緒に決める。
3. 途中の状態の log π は、学習時に target net で求めるか、行動時に保存するかを選ぶ。前者は、窓の長さのぶん forward が増える。

### 3.2 方策オフの補正

1. 窓の途中の手が target の方策から外れているとき、その先を目標値に入れない、または重みを下げる補正を入れる。
2. 候補は次の 3 つ。
   - ランダムな手で窓を切る: いちばん安い。行動時に「ランダムだったか」の旗を replay に保存する。
   - Retrace(λ): 挙動方策の確率と target の方策の比で、続きを入れる重みを決める。
   - Peng の Q(λ): 補正を緩めた多段の目標値。
3. UQE で選んだ手（平均で選ぶ greedy と違う手）を方策外とみなすかも決める。

## 4. User Stories

1. 研究者として、n-step を使っても NEET が悪化しない形で、報酬の遡りを速くしたい。
2. 研究者として、探索の手が混ざった目標値のせいで組み立ての価値が低く見積もられないようにし、局所解に寄るのを避けたい。
3. 研究者として、窓の扱い（途中の項の有無、補正の方式）を設定で切り替え、n = 1 と比べたい。

## 5. 決めること（詳細設計で詰める）

1. 原因の切り分けの実験。たとえば、Munchausen 無しの n-step 3、ε の段を下げた n-step 3、§3.1 だけ、§3.2 だけ。
2. §3.1 の式（(α − 1)·τ·log π を足すか、別の形にするか）と、clip の扱い。
3. 途中の状態の log π を学習時に求めるか、行動時に保存するか。
4. §3.2 の方式と、そのために replay に保存する量（ランダムだったかの旗、行動時の確率）。
5. PER の優先度、IQN と QR の分布の目標値、TBO の h 変換との組み合わせ。
6. 既定（今の D3、補正なし）で今と metrics checksum が一致すること。
7. Atari（BTR 互換の D3 で動いている）への影響。既定は変えず、DropMerge で切り替える案を既定にする。
8. [999_decision_aligned_nstep](999_decision_aligned_nstep_10prd.md) との関係。窓を延ばす区間は待ちの区間なので、§3.1 と §3.2 の両方が効く。

## 6. 対象外

- 判断の区切りで窓を延ばす n-step（[999_decision_aligned_nstep](999_decision_aligned_nstep_10prd.md)）。
- 経過時間を割引に通す仕組み（[999_smdp_transition_duration](999_smdp_transition_duration_10prd.md)）。
- ε の段（spatial exploration）の設計の変更。

## 7. 依存と順序

単独で作れる。[999_decision_aligned_nstep](999_decision_aligned_nstep_10prd.md) は本 PRD の上に載る。順番待ちの候補として置く。

## 8. 参照

- [Vieillard et al., 2020] Nino Vieillard, Olivier Pietquin, Matthieu Geist. "Munchausen Reinforcement Learning." NeurIPS 2020. https://arxiv.org/abs/2007.14430
- [Munos et al., 2016] Rémi Munos, Tom Stepleton, Anna Harutyunyan, Marc G. Bellemare. "Safe and Efficient Off-Policy Reinforcement Learning." NeurIPS 2016. https://arxiv.org/abs/1606.02647（Retrace）
- [Precup et al., 2000] Doina Precup, Richard S. Sutton, Satinder Singh. "Eligibility Traces for Off-Policy Policy Evaluation." ICML 2000.（tree-backup）
- [Peng & Williams, 1996] Jing Peng, Ronald J. Williams. "Incremental Multi-Step Q-Learning." Machine Learning 22.
- [Hernandez-Garcia & Sutton, 2019] J. Fernando Hernandez-Garcia, Richard S. Sutton. "Understanding Multi-Step Deep Reinforcement Learning: A Systematic Study of the DQN Target." https://arxiv.org/abs/1901.07510
