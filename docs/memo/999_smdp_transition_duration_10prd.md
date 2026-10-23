# SMDP 対応（遷移ごとの経過時間を割引に通す）暫定 PRD

> 2026-10-03 起票。詳細設計は別途グリルで行う前提で、決めることを §5 に残した。フレームワークの仕組み（§3.1）と、DropMerge での時間の数え方（§3.2）を 1 本にまとめてある。詳細設計の段階で分けてよい。
> 2026-10-23 改訂: §2 の Munchausen の正味の項を、経験数の窓で引き直した値に直し（[2026-10-01_atari-feedback.md](../experiments/default-dqn/dropmerge/2026-10-01_atari-feedback.md) の探索ブロック 04 と判断更新）、§1・§2 の Run 名を今のフォルダ名に直した。
> 起点: DropMerge の NEET の分析で、1 つの割引率 γ が「待つことに値段を付ける役目」（γ を小さくしたい）と「長い計画を見通す役目」（γ を 1 に近づけたい）を同時に負っていることが分かったこと。あわせて、[086](086_dropmerge_macro_actions_10prd.md) のマクロ行動は何フレーム続いても 1 step として割り引かれる。
> 関連: [085_dropmerge_eval_score_axes_10prd.md](085_dropmerge_eval_score_axes_10prd.md)（待ち時間の罰を入れると return で比べられなくなる）、[086_dropmerge_macro_actions_10prd.md](086_dropmerge_macro_actions_10prd.md)、[098_temporl_skip_policy_10prd.md](098_temporl_skip_policy_10prd.md)、[999_decision_aligned_nstep_10prd.md](999_decision_aligned_nstep_10prd.md)、[ADR 0035](../adr/0035-munchausen-target-learner-local-real-space.md)（Munchausen の目標値）、[CONTEXT.md](../../CONTEXT.md) の「NEET」「NoDropTimeout」。

## 1. 目的と問題

各遷移に「経過時間 τ」を持たせ、学習の目標値の割引を γ^n から γ^τ の積に変える。行動ごとにかかる時間が違う問題の標準的な枠組みである semi-MDP（SMDP）に対応させる。既定は τ = 1 で、今の挙動と変わらない。

DropMerge で起きている問題は次のとおり。

- 割引は Agent の step 単位で掛かる（`DefaultDQNAgent.learner.gamma = 0.997`、`n_step = 1`）。現行の `use_instant_drop = true` では、DROP の 1 step は落とした果物が着地するまで物理を進め、NOOP の 1 step は 1 フレーム（1/60 秒）しか進めない。それでも割引はどちらも 1 回である。
- 1 step の NOOP は盤面をほとんど変えないので、その価値は「次の step で最善手を打つ価値」を γ で 1 回割り引いたものとほぼ同じになる。最善の DROP との本来の差は (1−γ)·V ≈ 0.3% しかなく、推定の揺れで順位が入れ替わる。QR51 100M（`run_20261002-031314_dmfb_qr`）の NEET 期には、この差が −0.020〜−0.026 から −0.001〜−0.002 に潰れ、NOOP が評価 step の 74〜84% で選ばれた（2026-10-03 集計）。
- 待つ損を大きくしようとして γ を下げると、大きい果物を育てる長い計画（ダブルスイカなど）が見えにくくなる。γ = 0.997 の半減期は約 231 step で、1 ゲーム（評価で 250〜400 step 程度）の終わりの報酬は今の 0.3〜0.5 倍にしか見えない。
- [086](086_dropmerge_macro_actions_10prd.md) のマクロ行動は、何フレーム続いても 1 step として割り引かれる。

時間の刻みが細かいと、Q 学習は「今やる」と「少し後にやる」を区別できなくなる、という指摘がある（Tallec et al. 2019）。遷移ごとに割引を変える定式化（White 2017）、状態と行動に依存する割引の理論（Pitis 2019）、可変時間の行動を扱う SMDP と options の枠組み（Bradtke & Duff 1995、Sutton et al. 1999）が理論の裏づけになる。

## 2. 現状（コードで確認したこと）

- 学習側は n-step の割引を `pow(γ, n_steps)` で作っている。分布の目標値（`core/anet-core/src/dqn_based_agent.cpp:3253`）、Munchausen の目標値（同 `:3334`）、スカラーの目標値（同 `:3622`、`:3654`）。
- replay の n-step の積み上げは、学習側の γ をそのまま使う（同 `:2558` の `rep_config.gamma = config_.gamma`）。
- Env の Step の結果（`SingleStepResult`、`core/anet-core/include/anet/rl.hpp:373`）が持つのは報酬と次の状態だけで、経過時間を運ぶ欄は無い。
- DropMerge には、DROP 以外の step に掛かる `time_penalty` と、NOOP に掛かる `noop_penalty` がある（`core/envs/dropmerge1/src/DropMergeEnv.cpp:1152-1155`、`:1245-1248`、現行どちらも 0）。DirectNoop では両者が同じ step に掛かる。`noop_penalty` は `action == kActionNoop` で判定しているため、Direct モードでは列 0 の DROP にも掛かる。
- QR51 + Munchausen τ0.03 100M（`run_20261003-081944_dmfb_qr_munch`、2026-10-03 集計）では、評価セッションの「NOOP の UQE スコア − 最善 DROP の UQE スコア」が、健全な時期に −0.10〜−0.16 だった。QR51 素の −0.02〜−0.03 の約 5 倍である。NEET 期（10〜22M）にはこの差が +0.024（12〜14M）まで振れ、NOOP が最善の手になった。QR51 素の NEET 期は −0.001（同点）止まりだった。同じ Run の 14M 付近の録画（EvalPanel、target ネット + Greedy）では、余裕のある静止した盤面で NOOP の平均 Q が全 DROP を上回り、NOOP の分位点の広がりは 0.005〜0.007 と、DROP の 0.017〜0.045 より細かった。
- 同じ Run で Munchausen の指標（`36_agent_munchausen/04_bonus_mean` と `07_soft_gap`）を記録した。1 step ごとに目標値へ足される正味の項（bonus + γ·soft gap、報酬 ×0.01 の単位）は、soft gap を記録した 3 本（この Run を含む）の 90〜100M で −0.037〜−0.045 で、全窓で負だった（[2026-10-01_atari-feedback.md](../experiments/default-dqn/dropmerge/2026-10-01_atari-feedback.md) の探索ブロック 04）。平均で見る限り、Munchausen の項が「生き延びるほど得」にはしていない。当初は「0〜10M で −0.125、10〜20M で −0.072、30〜40M で −0.029」と書いたが、経験数の軸の指標を学習ステップの窓で切った値で、学習の最初の約 0.4M 経験を見ていた（同ブロックの判断更新で訂正）。

## 3. 提案の骨子

### 3.1 フレームワーク: 遷移ごとの経過時間

1. Env が遷移ごとに経過時間 τ ≥ 0（Env が定める単位）を返す。既定は 1。
2. replay は τ を遷移と一緒に保存する。n-step の積み上げは、報酬の割引に γ^τ の積を使い、ブートストラップの係数も γ^τ の積にする。
3. 学習側のすべての目標値（DQN、QR、IQN、Munchausen、TBO の h 変換を含む）で、`pow(γ, n_steps)` の代わりに保存した係数を使う。
4. 既定の τ = 1 で、今の挙動と metrics checksum が一致することを受け入れ条件にする。
5. Bellman の目標値の契約が変わるので、詳細設計で ADR を起こす。

γ は Agent が持つ「時間 1 単位あたりの割引」のまま残し、時間の意味は Env が持つ。γ の定義が Env と Agent の 2 か所に分かれないようにするため、Env には割引そのものではなく経過時間を返させる。

### 3.2 DropMerge: 時間の数え方

1. 時間の数え方を設定で選べるようにする。
   - step 単位（現行と同じ）: 各遷移で τ = 1。
   - フレーム単位: τ = その step で進めた物理のフレーム数。
   - DROP 単位: 実際に果物が落ちた遷移で τ = 1、それ以外（NOOP、dropper が busy の間の DROP 指示）は τ = 0。待っている間は割り引かない。
2. DROP 単位で数えるときは、待つ値段を明示の罰で付ける。既存の `time_penalty`（DROP 以外の step に掛かる）に一本化し、`noop_penalty` との重複と、Direct モードでの列 0 への誤った加算を整理する。
3. ショットクロック（100 step）と投了の裁定は、NEET の安全網として残す。
4. [086](086_dropmerge_macro_actions_10prd.md) のマクロ行動は、数え方に応じた τ を返す（フレーム単位なら実行したフレーム数、DROP 単位なら 0）。

## 4. User Stories

1. 研究者として、待つことの値段（明示の罰）と、長い計画の見通し（γ）を別々に決めたい。NEET を抑えながら、ダブルスイカのような長い計画を学ばせるため。
2. 研究者として、何フレーム続くかが行動ごとに違うマクロ行動を、進めた時間どおりに割り引きたい。
3. 研究者として、既定の設定では今の学習と同じ結果になることを確かめてから、DropMerge の時間の数え方だけを変えて比べたい。
4. フレームワークの利用者として、Atari など他の env では何も変えずに今までどおり動いてほしい。

## 5. 決めること（詳細設計で詰める）

1. τ を運ぶ場所（`SingleStepResult` の欄か、aux か）と型（整数のフレーム数か、実数の時間か）。
2. DROP 単位で γ^0 = 1 が続く NOOP の連鎖の扱い。ショットクロックで長さは有限だが、学習の安定性を確かめる。
3. Munchausen の項（取った行動の log-policy の加点と、次の状態の soft value）を、τ = 0 の遷移でどう扱うか（ADR 0035 との整合）。
4. PER の優先度と IS 重み、n-step の窓の長さとの関係（[999_decision_aligned_nstep](999_decision_aligned_nstep_10prd.md) と一緒に決める）。
5. 待つ罰の大きさ、γ の値（調整の範囲なので本 PRD では決めない）。
6. 比較の物差し。待つ罰を入れると return が変わるので、[085](085_dropmerge_eval_score_axes_10prd.md) のスコア軸で比べる。
7. Rainbow など、DQN 系以外の Agent の扱い（τ を無視してよいか、対応させるか）。

## 6. 対象外

- 続ける長さを学習すること（TempoRL、[098_temporl_skip_policy](098_temporl_skip_policy_10prd.md)）。
- 次の判断まで遡る可変長の n-step（[999_decision_aligned_nstep](999_decision_aligned_nstep_10prd.md)）。
- γ や待つ罰の値の調整。

## 7. 依存と順序

比較には [085](085_dropmerge_eval_score_axes_10prd.md) の物差しが要る。[086](086_dropmerge_macro_actions_10prd.md) のマクロ行動とは独立に作れるが、マクロの時間を正確に数えるには本 PRD が要る。[098_temporl_skip_policy](098_temporl_skip_policy_10prd.md) と [999_decision_aligned_nstep](999_decision_aligned_nstep_10prd.md) は本 PRD の上に載る。

## 8. 参照

- [Tallec et al., 2019] Corentin Tallec, Léonard Blier, Yann Ollivier. "Making Deep Q-learning Methods Robust to Time Discretization." ICML 2019. https://arxiv.org/abs/1901.09732（詳細解説: `reports/tallec2019_time_discretization_paper_2026-10-03.md`）
- [White, 2017] Martha White. "Unifying Task Specification in Reinforcement Learning." ICML 2017. https://arxiv.org/abs/1609.01995（詳細解説: `reports/white2017_unifying_task_specification_paper_2026-10-03.md`）
- [Pitis, 2019] Silviu Pitis. "Rethinking the Discount Factor in Reinforcement Learning: A Decision Theoretic Approach." AAAI 2019. https://arxiv.org/abs/1902.02893
- [Bradtke & Duff, 1995] Steven J. Bradtke, Michael O. Duff. "Reinforcement Learning Methods for Continuous-Time Markov Decision Problems." NIPS 7 (1994 年の会議、1995 年刊行)。URL 未確認
- [Sutton et al., 1999] Richard S. Sutton, Doina Precup, Satinder Singh. "Between MDPs and semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning." Artificial Intelligence 112. https://people.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf
