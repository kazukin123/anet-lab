# 次の判断まで遡る n-step（判断単位の可変長 n-step）暫定 PRD

> 2026-10-03 起票。詳細設計は別途グリルで行う前提で、決めることを §5 に残した。
> 2026-10-23 改訂: Run 名を今のフォルダ名に合わせた（2026-10-15〜17 の改名。対応は DropMerge の workspace の改名の対応表）。中身は変えていない。
> 起点: DropMerge の NEET の分析で、NOOP の連続（最長はショットクロックの 100 step）の結果が、n-step = 1 では 1 回の更新で 1 step ずつしか遡らないことが分かったこと。
> 関連: [999_nstep_target_correction_10prd.md](999_nstep_target_correction_10prd.md)（n-step の目標値の補正。本 PRD はその上に載る）、[999_smdp_transition_duration_10prd.md](999_smdp_transition_duration_10prd.md)（判断の区切りを経過時間 τ から決める）、[086_dropmerge_macro_actions_10prd.md](086_dropmerge_macro_actions_10prd.md)、[098_temporl_skip_policy_10prd.md](098_temporl_skip_policy_10prd.md)（続けた区間をまとめた遷移を使う点が共通）、[CONTEXT.md](../../CONTEXT.md) の「NEET」「NoDropTimeout」。

## 1. 目的と問題

待っている間の遷移では n-step の窓を延ばし、次に判断した時点で区切る。待ち始めた step の価値を、次に DROP を決める step の価値で直接答え合わせできるようにする。

DropMerge で起きている問題は次のとおり。

- 現行は n-step = 1（`DefaultDQNAgent.learner.n_step = 1`）。NOOP を選んだ step の価値は、すぐ次の step の価値で答え合わせする。
- 90 step 待ってから落とした場合、「待った結果どうなったか」は 90 回の更新を経ないと待ち始めまで届かない。
- NEET の長考型では、締切の直前だけ DROP し、それ以外は NOOP と最善の DROP がほぼ同点のまま残る形に固まりやすい。QR51 100M（`run_20261002-031314_dmfb_qr`）の NEET 期（8〜30M）では、評価のエピソード長が 580〜1,180 step に伸びた（2026-10-03 集計）。

最適な手の定義は変えない。割引は今と同じ（step 単位、または [999_smdp_transition_duration](999_smdp_transition_duration_10prd.md) の経過時間 τ）で、学習の遡り方だけを変える。

## 2. 現状

- replay の n-step は固定長で、学習側の割引は `pow(γ, n_steps)`（`core/anet-core/src/dqn_based_agent.cpp:3253`、`:3334`、`:3622`、`:3654`）。n-step の積み上げは学習側の γ をそのまま使う（同 `:2558`）。
- 固定長の n-step 3 を DropMerge で試すと、n = 1 より約 1 割低く、長考型の NEET が手の 3 分の 2 を占めた（2026-10-06、[999_nstep_target_correction](999_nstep_target_correction_10prd.md) の §2）。窓の途中の Munchausen の項が抜けていることと、探索の手が目標値に混ざることが、原因の候補として挙がっている。
- DropMerge では、行動 0 が NOOP、1〜58 が DROP（`core/envs/dropmerge1/src/DropMergeEnv.cpp:993-995`）。ショットクロックは DROP 以外の行動で進み、100 で NoDropTimeout になる（同 `:1018-1022`、`:1192-1203`）。
- QR51 + Munchausen τ0.03 100M（`run_20261003-081944_dmfb_qr_munch`）でも、10〜22M に長考型の NEET が出た。14M 付近の録画（EvalPanel、target ネット + Greedy）では、NoDropTimeout にならない範囲で NOOP を続けてから DROP していた。評価のエピソード長は 12〜14M に 1,458 step まで伸びたが、NoDropTimeout で終わった回は 10〜15M に約 1 割だけで、大半は締切の手前で DROP していた（2026-10-03 集計）。締切の直前で落とすので、NoDropTimeout の罰を受ける回は少ない。

## 3. 提案の骨子

1. n-step の窓の終わりを「判断の遷移」で決める。判断でない遷移（待ち）が続く間は窓を延ばし、次の判断の遷移、終端、または上限の長さで区切る。
2. 判断かどうかは、[999_smdp_transition_duration](999_smdp_transition_duration_10prd.md) の経過時間から決める案を既定にする（DROP 単位で数えるとき、τ = 0 の遷移は待ち）。別の旗を持たないので、時間の数え方と区切りが食い違わない。
3. 割引は、窓の中の γ^τ の積で行う。
4. n-step に特有の偏り（目標値が「その後も待ち続けた前提の値」になる）は、無駄な待ちに値段を付ける向きに働く。これを許容する前提で、検証で効き方を確かめる。
5. 本 PRD の「判断」は待ちでない遷移（DropMerge の DROP など）を指し、[098 TempoRL](098_temporl_skip_policy_10prd.md) の「判断」（Actor が行動とスキップ長を選ぶ step）とは別の語である。詳細設計で語を分ける。

## 4. User Stories

1. 研究者として、NEET の長考型（締切の直前だけ落とす）が、少ない更新で崩れるようにしたい。
2. 研究者として、時間の数え方（[999_smdp_transition_duration](999_smdp_transition_duration_10prd.md)）を変えたときに、n-step の区切りも自動でそれに合ってほしい。

## 5. 決めること（詳細設計で詰める）

1. 窓の上限の長さ（ショットクロックの 100 step に合わせるか）。
2. step 単位で数えるとき（τ = 1 のまま）にも使えるよう、Env が判断かどうかの旗を返す案を残すか。
3. PER の優先度、Munchausen の項、IQN の分布の目標値との組み合わせ。Munchausen の項の積み方と方策オフの補正は、[999_nstep_target_correction](999_nstep_target_correction_10prd.md) の決定に従う。
4. replay の保存量（可変長の窓の終点の持ち方）。

## 6. 対象外

- 経過時間を割引に通す仕組みそのもの（[999_smdp_transition_duration](999_smdp_transition_duration_10prd.md)）。
- 続ける長さの学習（[098_temporl_skip_policy](098_temporl_skip_policy_10prd.md)）。

## 7. 依存と順序

[999_smdp_transition_duration](999_smdp_transition_duration_10prd.md) と [999_nstep_target_correction](999_nstep_target_correction_10prd.md) の上に載る。固定長の n-step で NEET が悪化した問題が解けるまでは、窓を延ばしても同じ問題が出るおそれがある。順番待ちの候補として置く。

## 8. 参照

- [Biedenkapp et al., 2021] André Biedenkapp, Raghu Rajan, Frank Hutter, Marius Lindauer. "TempoRL: Learning When to Act." ICML 2021. https://arxiv.org/abs/2106.05262（続けた区間をまとめた遷移の目標値。詳細解説: `reports/biedenkapp2021_temporl_paper_2026-10-03.md`）
- [Sutton et al., 1999] Richard S. Sutton, Doina Precup, Satinder Singh. "Between MDPs and semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning." Artificial Intelligence 112. https://people.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf
- [Tallec et al., 2019] Corentin Tallec, Léonard Blier, Yann Ollivier. "Making Deep Q-learning Methods Robust to Time Discretization." ICML 2019. https://arxiv.org/abs/1901.09732
