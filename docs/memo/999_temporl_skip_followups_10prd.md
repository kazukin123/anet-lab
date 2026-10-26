# TempoRL の後続（長い j の不確かさ・閉ループの割り込み・全 k の一括学習）暫定 PRD

> 2026-10-25 起票（暫定。質疑を省き、既定を置いて、決めることを §9 に残した）。グリル前。行番号は起票時点のもので、実装時は近傍のシンボル名で探し直す。
> 起点: [PRD098](098_temporl_skip_policy_10prd.md)（TempoRL 対応）の段 1 の途中経過と、2026-10-25 の会話。外部の対話 AI との壁打ちで挙がった後続の手法（UTE、Lazy-MDP、t-DQN の安定化）を、PRD098 の実装と論文解説に照らして仕分けた。安定化の 3 項目のうち、目標値の Double DQN 化と Dueling は PRD098 で実装済み、全 k の一括は PRD098 §8 で先送り済みだったので、本書は先送りの 1 つと新しい 2 つを扱う。スキップ中の forward の省略は作らず、作るときの形と条件を残す（D9）。
> 改訂: 2026-10-25。起票候補の 2 件を記録した。UTE と Lazy-MDP の論文解説を survey のキューに積み（§9 の 1）、PRD098 §8 の 3 行を本書へ向けた（§5.7・§9 の 8）。
> 改訂: 2026-10-26。§7 に段 0 の結果（(a) UQE の楽観。作るのは (I) だけ）と、今のビルドでの対照を記録した。
> 改訂: 2026-10-26（2 回目）。5×5 の既定 seed の件を「崩壊」から「ゴールに届かない」に直した（§1・§2.3・D11・§7。評価は一度も届かず、学習側もほぼ届いていなかった）。§7 の段 0 の「前半を読む」理由が、この前提に依っていたことを書き添えた。
> 関連: [999_bootstrapped_multihead_q](999_bootstrapped_multihead_q_10prd.md)（primary の複数の Head。本書のメンバーは skip Head の中だけ）、[999_observation_pipeline](999_observation_pipeline_10prd.md)（強制行動を MakeAction の入力にする。PRD098 §9 の 1）、[999_decision_aligned_nstep](999_decision_aligned_nstep_10prd.md)（判断に揃える n-step）、[行動の時間尺度（wiki）](../wiki/mechanisms/temporal_abstraction.md)。

## 1. 目的と問題

PRD098 で入れたスキップ（選んだ行動を続ける長さの学習）には、次の 4 つの弱点がある。本書は弱点 1〜3 に手を打つ拡張を、効き目を確かめられる順に足す。弱点 4 は作らない理由と、作るときの形を残す。

1. **長い j の価値の不確かさ**: スキップ長 j が長いほど replay の窓に現れる回数が少なく（k は起点の k_max 以下から一様に引き、k_max は判断の区間の残り）、価値の推定が粗いまま選ばれうる。5×5 の既定 seed で有効の 2 型がゴールに届かなかった件（PRD098 §9 の 2）の、起票時の候補の 1 つだった（段 0 で、原因はスキップ長を UQE の楽観で選ぶことと分かった。§7）。
2. **開ループの硬さ**: 判断で j を決めると、途中で状況が変わっても j step 続ける。
3. **長い j の学習の遅さ**: 1 つのサンプルから 1 つの k しか学ばない。
4. **計算の節約が得られない**: スキップ中も全 lane で毎 step forward している。

## 2. 現状（2026-10-25 に確認したこと）

### 2.1 PRD098 の実装

- skip Network（Head を含む）は primary と別の Network である。スキップの目標値の終点の価値は行動価値（primary）で測り、スキップ価値自身の target は Learner で使わない（[PRD098](098_temporl_skip_policy_10prd.md) D3・§5.4）。このため、skip Head の出力を何組に増やしても、目標値はどの組でも同じになる。
- スキップの目標値の終点の価値は、Q と同じ target 方策に従う。Double DQN が有効なら online の argmax と target の値、Munchausen なら soft value（PRD098 §5.4）。
- skip Head の出力は、分離型が全行動分のスキップ長ごとの値、包含型が入力した行動のスキップ長ごとの値で、分布型では分位の分布（`skip_q_dist`）とその平均（`skip_q`）を出す（[dqn_based_heads.cpp:87-125](../../core/anet-core/src/dqn_based_heads.cpp)、PRD098 §5.3）。
- Actor は lane ごとに「続けている行動」と「残りの step 数」を持つ。`episode_start` か残り 0 の lane だけが判断し、続けている lane は方策の提案を捨てて続けている行動で上書きする。j は skip の方策が選ぶ（[dqn_based_agent.cpp:2469-2526](../../core/anet-core/src/dqn_based_agent.cpp) の `Actor::ApplySkip`）。skip の方策は、既定で行動の方策と同じ規則・同じ値、`actor.[key].skip.policy.*` を書けば専用（PRD098 D9）。
- replay の窓は、判断の旗で区切った区間の中だけから作る。サンプルは起点の k_max 以下から 1 点 k を一様に引く（PRD098 D13）。全 k をマスク付きで一括するサンプルは、target の forward が J 倍になるので先送り（PRD098 §8）。k_max に比例する重みも先送り（同）。
- スキップ中も全 lane で primary を forward する（PRD098 §5.5）。全 lane の行を前提にする読み手（step 診断、Actor Q ヒントと PER の初期優先度、margin 診断、spatial、Noisy の share=sample、stacker、replay の Push、QValuePanel、Evaluation Trace）がいて、一部の lane だけ forward を省く口は無い（PRD098 §2.3）。省略は範囲外（PRD098 §3・§8）。
- skip Head の Dueling は、primary の Q(s,a) を基準にする形が実装済み（`skip.head.use_dueling`、PRD098 D8）。
- Evaluation Trace は aux の `q_values` から、実行した行動の Q（q_sa）と状態価値 V を作る（PRD098 §2.3）。スキップを続けている step では、実行した行動は続けている行動なので、Munchausen が無効なら V − q_sa が D5 の G に当たる（Munchausen が有効なとき V は soft value になる。[dqn_based_agent.cpp:372-381](../../core/anet-core/src/dqn_based_agent.cpp)）。

### 2.2 論文と後続の手法について分かっていること

- TempoRL の論文の 3 構成（連結・文脈化・重み共有）は、どれも行動をスキップ側の入力に取る。重み共有は、行動出力の argmax をスキップ出力の入力にする（[論文解説](../../reports/biedenkapp2021_temporl_paper_2026-10-03.md) 3.3）。全行動分を出して行動で絞る形（anet の分離型）は論文に無い。
- 論文は構成どうしを MountainCar と LunarLander で比べ、重み共有は MountainCar で最良、LunarLander で最悪だった。Atari では重み共有だけを使い、構成どうしは比べていない（論文解説 4.2.2・4.2.3）。Atari の実装は double DQN（同 4.2.3）。
- 論文解説の範囲では、判断の少なさは「方策を学びやすい」ことと解釈のしやすさの利点として述べられている（論文解説 4.1・5）。
- UTE（AAAI 2024、Learning Uncertainty-Aware Temporally-Extended Actions）と Lazy-MDP（AAMAS 2022）は、壁打ちで挙がった手法で、**原文を確かめていない**。本書の (U) と (I) は、壁打ちで説明された考え方（スキップ価値のアンサンブルの平均と標準偏差で j を選ぶ。行動を変える価値の差が閾値を超えたときだけ切り替える）を、PRD098 の構造に合わせて置き直したものである。原文と食い違えば本書を直す（§9 の 1）。

### 2.3 実測と推定

実測:

- GridMaze 6×10（agent seed 1 本）で、評価 target 報酬が +0.93 に届いたのは分離型×遮断が 251k step。包含型×遮断は +0.93 に届くが、正になるのは 311k step（PRD098 §7 の段 1 の途中経過）。
- 5×5 の既定 seed で、有効の 2 型がゴールに届かなかった（PRD098 §9 の 2）。評価では一度も届かず、学習側もエピソードの報酬の最大が序盤に一度 +0.03 になっただけだった。起票時は PRD098 に合わせて「序盤に見つけたゴールを保持できず崩壊した」と書いていたが、2026-10-26 に記録を読み直して改めた。有効の checkpoint は `run_20261024-060039` の `agent_close.anet` に残り、Evaluation Trace の列 `$action_info skip_q.[j]` で j ごとのスキップ価値を読める（PRD098 §5.7）。
- skip 有効の学習 1 回の更新の実時間は、無効に対して、初回実装の Debug（OFF→ON→ON→OFF の順）で 1.299 倍、最適化後の逆順の測定で Debug 1.379 倍・Release 1.511 倍（Atari BTR、N=128、B=256、BF16、prefetch。[PRD098 20impl](098_temporl_skip_policy_20impl.md)）。PRD098 の受入の 1.3 倍は、最適化後の測定で満たしていない。
- Atari で lane 数 N を 64 から 128 に上げると throughput が 9% 上がり、128 から先は伸びなかった（[2026-08-17 baseline](../experiments/default-dqn/atari/2026-08-17_baseline.md) の考察）。RR4 で envs64 は envs128 より 50M の実時間が長かった（11.960h 対 9.450h。[2026-09-15 lane 窓](../experiments/default-dqn/atari/2026-09-15_replay-lane-window.md)）。

入力元の推定と、本書の推定:

- 2026-08-17 の考察は、N を上げて速くなる分を「actor forward が batch N で 1 回にまとまることによる step あたりコストの低下」と考え、128 で頭打ちになるのは学習側が律速に回るためと推定している。lane 窓の記録は、envs64 が遅いのを「actor forward の回数が倍・バッチが半分になるぶん」の損と説明している。
- 本書の推定: GPU で学習する構成では、Actor の forward の費用は行の数より呼び出しの回数に強く依る。D9 の判断はこの推定に依る。確かめるには、Actor の forward の実時間を行の数を変えて測る（§8）。
- 本書の計算: 判断する lane が 1 本でもあれば forward は 1 回残る。全 lane が同時にスキップ中になる確率は、判断率を d として (1−d)^N で、d=0.3・N=128 なら約 10 の −20 乗である。

## 3. ゴール・非ゴール・成功の測り方

ゴール: PRD098 のスキップに、§1 の弱点 1〜3 へ手を打つ 3 つの拡張を、それぞれ設定で有効にでき、無効では今と同一になる形で足す。どれを作り、どの順で作るかは、段 0 の診断の結果で決める（D11、§7）。

非ゴール（消える案と、それが本当の制約かどうか）:

- 連続行動の手法（TAAC、SDAR など）とモデルベースの手法: anet の DQN 系は離散行動で、別の Agent になる。範囲の違いによる非ゴール。
- Lazy-MDP の「既定の方策に任せる行動」を足し、制御を取るたびに罰を払う形: 行動空間と報酬を変える別の関心（Env 側のマクロ行動 [086](086_dropmerge_macro_actions_10prd.md) に近い）。制約ではなく、別 PRD。
- 行動を選ぶ Q（primary）の学習を変えること: 本書の 3 つの拡張は、どれも skip 側の学習か Actor の選び方で閉じるので要らない。クリーンブレークで移せるので制約ではなく、ゴールに要らないことによる非ゴール。
- NEET の抑制: PRD098 §3 と同じく、行動を選ぶ Q の 1 step の差を変えないので効かない見込みによる非ゴール。
- スキップ中の forward の省略（弱点 4）: 本書では作らない（D9）。全 lane の行を前提にする読み手の契約はクリーンブレークで移せるので制約ではなく、効くのが 1 env のリアルタイム実行だけで学習のゴールに寄与しない別の関心による非ゴール。理由と作るときの形は §8。

成功の測り方（判定の定義は §7）:

- (U) 5×5 の既定 seed で、評価の Actor を悲観側の λ にした腕がゴールを保持する。6×10 で、メンバーを持つ 2 つの腕の評価の AUC が無効以上。
- (I) 6×10 で、評価の AUC が割り込み無し以上で、評価の判断率の上がり幅が分かる。
- (K) 同じ step 数で、評価の AUC が 1 点 k 以上。学習 1 回の更新の実時間の比が分かる。

## 4. 確定事項（暫定の既定）

各決定に理由を書き、ほかの決定や事実に依存するときはそれを名指す。

| # | 決定 | 理由と依存 |
|---|---|---|
| D1 | 作る候補の拡張は 3 つ。(U) skip Head のメンバーと、不確かさを足した j の選択。(I) 閉ループの割り込み。(K) 全 k の一括学習。どれを作るかは D11 の条件で決まる | §1 の弱点 1〜3 に 1 対 1 で当たる。弱点 4 は D9 |
| D2 | (U) は skip Head の中に E 組の射影（メンバー）を持ち、Body は共有する。全メンバーが同じ目標値で学ぶ。メンバーの違いは初期値だけ（既定） | 目標値は primary で決まるので、メンバー別の target は要らない（§2.1）。primary の複数の Head（[999_bootstrapped_multihead_q](999_bootstrapped_multihead_q_10prd.md)）は行動の方策を分ける目的で、スキップ価値の不確かさを測る本書とは目的が違うので別に持つ |
| D3 | (U) の選択は、メンバーの値の平均 μ_j と標準偏差 σ_j から μ_j + λ·σ_j を作り、それを既存の skip の方策（PRD098 D9）に渡す。分布型は、メンバーの分布の平均を、分位ごとに λ·σ_j だけずらして渡す。λ は Actor の用途ごとに持つ | 方策の規則（ε-greedy・UQE・Thompson）を変えずに値だけを変えれば、選び方は 1 経路のまま。学習の Actor は 0 か正（探索）、評価の Actor は負（悲観）を選べる。D2 に依存 |
| D4 | (U) の Learner は、メンバーごとの loss の平均を skip の loss にする。PER の合成に使う skip の TD 誤差は、メンバー平均の値の TD 誤差。IS 重みは全メンバーに同じ値を掛ける | PER の優先度の定義を 1 つに保つ。D2 に依存 |
| D5 | (I) は、スキップを続けている lane で、その step の primary の forward から G = max_a Q̄(s,a) − Q̄(s, 続けている行動) を作り（Q̄ は分布型なら分位平均）、G が閾値 η を超えたら、その step を判断にする。行動は方策の提案を使い、j を選び直す | スキップ中も全 lane で forward しているので（§2.1）、forward を足さずに済む。判断の旗が立つので replay の窓は割り込みの手前で切れ（PRD098 D13）、学習側は変えなくてよい |
| D6 | (I) の η は学習した Q の単位（報酬のスケーリングのあとの単位）の絶対値で、Actor の用途ごとに与える | 最小の形。尺度を揃える形は実測が要る（§9 の 2）。段 I の値は Trace から決める（§7） |
| D7 | (K) は、サンプルした起点ごとに k = 1..k_max のすべての目標値を作り、有効な k の loss の平均を、その起点の skip の loss にする。既定は今の 1 点 k | 長い j の学習の遅さを直接減らす。全接続を等しく数えるので、k_max に比例する重み（PRD098 §8 の先送り）もこの形で扱える。target の forward が B×J 行になる（§5.5）ので既定にしない |
| D8 | (K) の PER に使う skip の TD 誤差は、有効な k の TD 誤差の絶対値の平均 | 起点ごとに優先度を 1 つにする。D7 に依存 |
| D9 | スキップ中の forward の省略（弱点 4）は本書で作らない | 効くのは lane が 1〜数本で、1 env をリアルタイムで動かす実行（実機の制御など）だけで、学習とは別の関心である。学習（GPU、N=128 前後）では呼び出しの回数が減らず効かない見込み（§2.3 の本書の推定）。作るときの形と再開条件は §8 |
| D10 | 3 つとも、無効のとき（E=1、割り込み off、1 点 k）は今と同一（forward の回数・順序、RNG の消費、loss、優先度、checkpoint の形）。E=1 の skip Head は今と同じ形 | PRD098 の段 0 と同じ等価性の受入を保つ |
| D11 | 作る拡張と順は、段 0 の診断の結果で決める（§7 の段 0 の表）。(I) はどの結果でも作る。(U) は段 0 が (b)「長い j に寄る」、(K) は (c)「長い j を学べていない」に当たったときだけ作り、当たらなければ作らない | (I) は 5×5 で届かない件と関係なく弱点 2 に当たる。(U) と (K) が要るかは、届かない原因という実験の結果が無いと決められない |
| D12 | 新しい語は 2 つ。**メンバー**は skip Head の中の射影の 1 組（設定キーは `ensemble_size`）。**割り込み**は、スキップを続けている lane を、その step で判断に戻すこと | 「Head」は primary の複数の Head（999_bootstrapped_multihead_q）と紛れ、Body を共有して射影だけを複数持つ本書の形を言い表せない。「判断」は割り込みの結果で、続けている区間を途中で切る原因の側を指す語が無い。用語集（[CONTEXT.md](../../CONTEXT.md)）へ足すかはグリルで決める（§9 の 6） |

## 5. 実装契約（暫定）

### 5.1 設定キー

すべて `DefaultDQNAgent.` の下。不正値（型・範囲・列挙値の外）はすべて Agent の構築時に、キーと値と期待値を示して `ANET_SYSTEM_ERROR` で止める（値を丸めない、既定へ置き換えない）。`skip.enabled = false` のときは検証だけして使わない（PRD098 §5.6 と同じ）。

| キー | 型・値域 | 既定 | 意味 |
|---|---|---|---|
| `skip.head.ensemble_size` | 整数、1 以上 | 1 | skip Head のメンバーの数 E。1 は今と同一 |
| `actor.[key].skip.uncertainty_weight` | 有限の実数 | 0 | λ。E=1 のときは検証だけして使わない（σ が無い）。0 以外なら対象ごとに 1 回だけ WARN |
| `actor.[key].skip.interrupt.mode` | `off` ／ `q_gap` | `off` | 割り込みの方式 |
| `actor.[key].skip.interrupt.threshold` | 有限、0 以上 | 無し（`q_gap` なら必須。無ければ構築時にエラー） | η。`off` のときは検証だけして使わない |
| `learner.skip.sample_mode` | `single_k` ／ `all_k` | `single_k` | (K)。`all_k` は全 k の一括 |

E=1 の λ と、`off` の η をエラーにしないのは、AGENTS.md「Fail-Fast 原則」の表の「意図された休止・無効化」（利用者が止めた機能と、それを参照する設定）に当たるためである。λ だけ WARN を出すのは、E を戻し忘れた設定の取り違えに気づけるようにするためである。

設定例（コロン記法）:

```
# agent.txt: 無効のまま、既定をべた書き（?=）
DefaultDQNAgent.@baseline : skip.head.ensemble_size ?= 1
DefaultDQNAgent.@baseline : learner.skip.sample_mode ?= single_k

# CLI・bat: 変種は葉で
DefaultDQNAgent.skip.head.ensemble_size=5 DefaultDQNAgent.actor.[eval].skip.uncertainty_weight=-1.0
DefaultDQNAgent.actor.[train].skip.interrupt.mode=q_gap DefaultDQNAgent.actor.[train].skip.interrupt.threshold=0.05
DefaultDQNAgent.learner.skip.sample_mode=all_k
```

### 5.2 skip Head のメンバー（U）

- E>1 の skip Head は、メンバーごとの値を `skip_q_members`（メンバーごとの分位平均）と、分布型では `skip_q_dist_members`（メンバーごとの分布）で出す。形は、今の `skip_q`・`skip_q_dist` の行の次にメンバーの軸を挟んだものである。`skip_q`・`skip_q_dist` にはメンバーの平均を出すので、平均を読む既存の読み手（Actor の選択、Trace の `skip_q.[j]`）は変えない。E=1 では新しい鍵を出さない（D10）。
- メンバーの射影は、同じ初期化設定（`skip.head.init.*`）で、別々の乱数で初期化する。Noisy の射影ならメンバーごとに Noisy のパラメータを持つ。
- `use_dueling` の基準 Q(s,a) は全メンバーに同じ値を足すので、σ は変わらない。
- Actor は、選んだ行動のスキップ長ごとの σ を aux `skip_q_std` に載せる。
- E が違う checkpoint は、読み込み時に `ANET_SYSTEM_ERROR`（skip Network の形が違う。PRD098 §5.8）。
- MODEL SHAPE DUMP と GraphViz に新しい出力の鍵が出る。

### 5.3 Actor の 1 step（U・I）

スキップが有効な Actor は、1 step で次の順に振る舞う。

1. 判断する lane を決める: `episode_start` か残り 0（今と同じ）。
2. 割り込み（`q_gap`）: 続けている lane で G（D5）を作り、G > η の lane も判断にする。aux `skip_interrupt` に載せる。
3. 判断する lane は方策の提案、ほかは続けている行動を、この step の行動にする（今と同じ）。
4. skip Network を forward し、メンバーの平均 μ と標準偏差 σ を読む。
5. λ ≠ 0 なら μ + λ·σ（分布型は分位ごとにずらす）を skip の方策に渡し、j を選ぶ。
6. lane の状態を更新し、info の `decision`（replay の判断の旗）は 1 と 2 のどちらかに当たる lane で立てる。

G は、行動の方策が使った forward の出力の平均 Q で作る（UQE の risk score は使わない。§9 の 3）。評価の Actor でも、設定すれば割り込む。

### 5.4 Learner（U・K）

- (U): skip の loss はメンバーごとの loss（今と同じ定義、同じ目標値）の平均。勾配のクリップは今と同じく skip Network 全体で 1 つ。
- (K) `all_k`: replay の判断の窓の規則が、起点ごとに k = 1..J の窓の割引報酬・終わり方・k 先の観測と、有効な k（k ≤ k_max）の印を出す。Learner は primary の target を B×J 行で 1 回 forward し（Double DQN が有効なら online も B×J 行）、k ごとの目標値と loss を作り、有効な k で平均してから IS 重みを掛ける。PER は D8。
- `single_k` では今と同一。

### 5.5 性能の見積もり

「単位」は、B 行の forward 1 回分の計算である（PRD098 §5.10 と同じ数え方。今の学習 1 回の更新は current 1、target 1〜2、backward 2 の 4〜5 単位）。

| 経路 | 頻度 | 単価の増分 | 見込み |
|---|---|---|---|
| (U) Actor の 1 step | 1 回／step | skip Head の最終層が E 倍。積和は 1 行あたり 128×A×J×E（分位の数を掛ける）で、Atari（A=18、J=10、E=5、IQN の τ 8 本）なら約 0.9M 回 | primary の畳み込みの Body より小さい見込み。受入 4 で測る |
| (U) Learner の 1 回の更新 | 1 回／更新 | skip Head の最終層の forward と backward と、loss が E 倍。target の forward は増えない | 上と同じ規模。受入 4（1.1 倍以内）で確かめる |
| (I) Actor の 1 step | 1 回／step | G の計算（行動の数の最大値と、1 つの値の取り出し） | 無視できる |
| (K) Learner の 1 回の更新 | 1 回／更新 | target の forward が B 行から B×J 行へ（J−1 単位の増加）。Double DQN が有効なら online の forward も同じだけ増える（さらに J−1 単位） | J=7 で Double DQN 無しなら 4〜5 単位が 10〜11 単位（約 2〜2.5 倍）。小さい env 向け |
| (K) 抽出 | B 件／更新 | k 先の観測を J 個組む | Atari では転送が J 倍。既定にしない |

### 5.6 メトリクスと trace

群は `50_agent_skip`（PRD098 D16）。tag は 40 文字以内。取得元は既存の skip の tag と同じく、Actor 側は学習の slot（`@train $action_info`）、Learner 側は `@learn $learn_step $update_result`。

| tag | 元の鍵（取得元） | 意味 |
|---|---|---|
| `50_agent_skip/05_skip_q_std` | `skip_q_std_mean`（`@train $action_info`） | 判断した lane の、選んだ j のメンバーの標準偏差の平均（E>1） |
| `50_agent_skip/06_interrupt_rate` | `skip_interrupt_rate`（`@train $action_info`） | 続けていた lane のうち、割り込んだ割合（学習の Actor で割り込みが有効なとき） |
| `50_agent_skip/16_valid_k_mean` | `skip_valid_k_mean`（`@learn $learn_step $update_result`） | `all_k` で、起点あたりの有効な k の数の平均 |

Evaluation Trace の列は `$action_info skip_interrupt`、`$action_info skip_q_std.[j]` を宣言できる（範囲外の j は `skip_q.[j]` と同じ扱い）。評価の Actor の判断率や割り込みは、既存の評価 slot の読み方（PRD098 §7 の `$eval.[tag]`）で読む。

### 5.7 移行対象と等価性

- 設定キーの改名は無い。新しいキーを agent.txt の `DefaultDQNAgent.@baseline` に `?=` で足す（[agent.txt](../../apps/runner/config/agent.txt)）。
- 設計文書:
  - [110](../design/110_agents_and_learning.jp.md)（Actor の skip の節。158 行の「`episode_start` で継続を解除し、`decision` で新しい判断を示す」に、割り込みで判断に戻ることを足す）。
  - [130](../design/130_neural_networks.jp.md)（209 行の skip Head の設定キーの並びに `skip.head.ensemble_size` を足し、メンバーの出力の鍵を書く）。
  - [200](../design/200_dqn_agents.jp.md)（skip の節）。
  - [140](../design/140_observability.jp.md)（指標と Trace の列）。
  - [150](../design/150_replay_buffer.jp.md)（判断の窓の規則の `all_k` の出力）。
- [PRD098](098_temporl_skip_policy_10prd.md) §8 の「全 k を mask 付きで一括するサンプル」「k_max に比例する重み」「skip 中の lane の forward を省く」の行を、本書へ向けた（PRD098 の 2026-10-25 の改訂。§9 の 8）。
- 等価性: 3 つとも無効のとき、§7 の受入 1 の 6 つの比較で metrics checksum が一致する。

### 5.8 クラス一覧

> 従えない事情があれば、実装する側が理由を実装計画に書いて変えてよい（質問しない）。

| 名前 | 区分 | 責務 | 置き場 |
|---|---|---|---|
| `SkipHead` | 変更 | メンバー E 組の射影と、平均とメンバーごとの出力 | DQN 共通 component（dqn_based_heads） |
| `SkipHeadFactory` | 変更 | メンバーの数の受け渡しと出力の鍵 | DQN 共通 component（dqn_based_heads） |
| `SkipConfig` | 変更 | `ensemble_size` を持つ | Agent の設定（agent） |
| `DQNActorConfig` | 変更 | `uncertainty_weight` と割り込みの設定を持つ | Agent の設定（agent） |
| `dqn::Actor` | 変更 | 割り込みと、不確かさを足した選択 | DQN 共通 component（dqn_based_agent） |
| `dqn::Learner`、`TDLearner`、`QRLearner`、`IQNLearner` | 変更 | メンバー平均の loss、`all_k` の目標値と loss | DQN 共通 component（dqn_based_agent） |
| `SampleWindowRule`（判断の窓の規則） | 変更 | `all_k` の窓の出力 | ReplayBuffer |
| `DQNActionInfo` | 変更 | aux の新しい鍵と、Trace の列の読み出し | DQN 共通 component（dqn_based_agent） |

## 6. 細部（推奨の既定）

1. E を使うときの推奨は 5。壁打ちで説明された UTE の範囲（5〜10）の下端で、最終層だけなので費用は小さい。
2. λ の既定は 0。評価の Actor の悲観の腕は −1.0。
3. 割り込んだ step の行動は方策の提案（探索を含む）。
4. `all_k` の loss は、有効な k で平均してから IS 重みを掛ける。
5. `all_k` の J 個の target の forward は、1 回の呼び出しにまとめる。
6. 指標の番号は Actor 側 05・06、Learner 側 16。
7. 新しい指標は、該当する機能が有効な Run の腕で metrics のチェーンに足す（既定の `metrics.scalar.@skip` に入れるかは §9 の 5）。
8. 「長い j」は j ≥ 4（J=7 の GridMaze の場合。J の半分より長い j）。
9. 段 0 の判定の閾値（長い j の選択率 2 割・1%、`14_kmax_mean` 2）は §7 のとおり。

## 7. 検証と受入

評価の値は PRD098 §7 と同じ評価 slot から読む。用語の定義:

- **AUC**: 評価 slot の評価報酬の、Run 全体での平均。
- **A が B 以上**: 腕 A の 3 seed の AUC の平均が、腕 B の 3 seed の AUC の最小値を下回らない（3 seed のブレ幅を超えて悪化しない。PRD098 §7 の段 1 と同じ考え方）。
- **ゴールを保持**: Run の最後の 100k step の評価報酬の平均が正。3 seed の段では 3 seed のうち 2 seed 以上で満たし、1 seed の段ではその seed で満たす。

段 0（届かない原因の診断。コードを変えない）: 5×5 の既定 seed（PRD098 §9 の 2）で、skip の方策を平均に対する ε-greedy にした腕（`actor.[key].skip.policy.*`。PRD098 §7 の手当て）を 1 seed・400k で流す。あわせて、届かなかった Run（`run_20261024-060039`）の記録を次のとおり読む。

- **読む区間**: Run の前半。tag ごとに、その tag が記録した step の軸のまま読み、その軸の最後の値の半分までの点を使う。tag どうしの軸は換算しない（`03_skip_length_rate.[j]`・`04_skip_q_gap` は `@train $action_info`、`14_kmax_mean` は `$learn_step` で、軸が違う）。起票時は「崩壊は序盤に起きる」を前半を読む理由にしたが、実際には最初から届いていなかった。段 0 は (a) で決まり、(b)(c) の判定は使わなかったので、この区間の決め方は確かめていない。
- **長い j の選択率**: 前半の `50_agent_skip/03_skip_length_rate.[j]`（判断した lane のうち j を選んだ割合。学習の Actor）の、長い j（§6 の 8）の和の平均。母数は判断の数である。
- 有効の checkpoint（同じ Run の `agent_close.anet`。Run の終わりのもの）は、Runner で読んで Evaluation Trace の `skip_length`・`skip_q.[j]` を見る目視にだけ使い、判定には使わない。

結果ごとに次のとおり分かれる（上から順に当てる）。

| 結果 | 判定 | 作るもの（順） |
|---|---|---|
| (a) UQE の楽観 | ε-greedy の腕がゴールを保持する（1 seed） | (I) だけ。あわせて PRD098 の skip の方策の既定を見直す（PRD098 の改訂にするか env ごとの設定にするかは §9 の 9） |
| (b) 長い j に寄る | (a) に当たらず、前半の長い j の選択率が 2 割以上で、前半の `50_agent_skip/04_skip_q_gap`（選んだ j と j=1 のスキップ価値の差）の平均が正 | (U) → (I) |
| (c) 長い j を学べていない | (a) に当たらず、前半の長い j の選択率が 1% 未満なのに、前半の `50_agent_skip/14_kmax_mean` の平均が 2 以上（長く続いた区間が replay にある） | (K) → (I) |
| (d) どれでもない | 上のどれにも当たらない | (I) だけ。届かない原因は PRD098 §9 の 2 として別に調べる |

(a) と (d) で作らない (U)・(K)、(b) で作らない (K)、(c) で作らない (U) の再開条件は、PRD098 の段 2 以降の env で、それぞれ (b)・(c) の判定に当たる兆候が出たときである。

段 0 の結果（2026-10-26）: **(a) UQE の楽観**に当たった。
- ε-greedy の腕（`run_20261026-041740_t98_gm_eps_s1`。ε は行動の UQE と同じ 0.60→0.05、200k）は、最後の 100k の評価 target 報酬が +0.93 で、ゴールを保持した。
- 比べ相手の届かなかった Run は 10/24 のビルドだったので、同じ設定で今のビルド（PRD101 まで）の対照（`run_20261026-082425_t98_gm_uqe_s1`。skip の方策は既定の UQE）を流した。最後の 100k は −1.00 で同じく届かず（評価の EMA の最大も −1.00）、評価の値も前半の skip の指標も 10/24 の Run と一致した。2 本の違いは skip の方策だけである（どちらも 1 seed）。ε-greedy の腕は 305k に届いたので、届くのは無効（124k）より遅い。
- 参考に、届かなかった Run の前半の指標は、長い j の選択率 29%（j=7 が 14%）、`04_skip_q_gap` の平均 −0.11、`14_kmax_mean` の平均 2.76。ε-greedy の腕は、長い j 25%（j=7 が 11%）、`04_skip_q_gap` −0.09、`14_kmax_mean` 2.56。(b) の「`04_skip_q_gap` の平均が正」には、届かなかった Run でも当たらない。
- D11 により、作るのは (I) だけで、(U) と (K) は作らない（再開条件は上のとおり）。PRD098 の skip の方策の既定の見直しは §9 の 9 で決める。

Run は GridMaze-01 にある。

作る段:

| 段 | env と構成 | 腕 | seed・長さ | 判定 | 満たさなかったとき |
|---|---|---|---|---|---|
| U | GridMaze 5×5 の既定 seed、GridMaze 6×10 | 無効、E=5・λ=0、E=5・評価の λ=−1 | 3 seed、5×5 は 400k、6×10 は 500k | 5×5 で E=5・評価の λ=−1 の腕がゴールを保持する。6×10 で E=5 の 2 つの腕がどちらも無効以上 | 既定は無効（E=1）のまま残し、結果を §7 と wiki に記録する。5×5 で届かないままなら PRD098 §9 の 2 へ戻す。次の段は続ける |
| I | GridMaze 6×10（既定のチェーン。Munchausen は無効） | 割り込み無し、η 2 値（割り込みは学習の Actor と評価の Actor の両方に同じ η で有効にする） | 3 seed、500k | 2 値のうち少なくとも 1 値の腕が、割り込み無し以上。評価の判断率の上がり幅を記録する | 既定は `off` のまま残し、結果を記録する。§9 の 2（η の尺度）を決め直してから 1 回だけやり直す |
| K | GridMaze 6×10、LunarLander（現行の LunarLander.txt の既定。IQN、J=4） | `single_k`、`all_k` | 3 seed、6×10 は 500k、LunarLander は 1M | 両方の env で `all_k` が `single_k` 以上。学習 1 回の更新の実時間の比を記録する | 既定は `single_k` のまま残し、結果を記録する |

段 I の η の 2 値は、先に割り込み無しの腕（3 seed、500k）を流し、各 seed の Run の終わりの checkpoint（`agent_close.anet`）を Runner で読んで、EvalPanel で GridMaze の既定の評価のエピソード数だけ流し、その Evaluation Trace から、スキップを続けている step（`skip_decision` が偽）の V − q_sa（§2.1 のとおり G に当たる）を 3 seed 分まとめて集め、その 50 パーセンタイルと 90 パーセンタイルにする。EvalPanel の Actor（`actor.[eval_panel]`）は target の網で行動するので、η をかける学習と評価の Actor（online の網）とは網が違う。target の網は online の網を追うので Q の尺度は近いとみなし、尺度の決め方そのものは §9 の 2 で見直す。

無効のまま残すのは、段 2 以降の env で結果が変わりうるためで、無効では今と同一（D10）なので残しても既存の Run に影響しない。

受入（外から観測できるもの）。受入 1 は常に当て、受入 2〜4 は段 0 の結果で作った拡張の分だけに当てる（作らなかった拡張の単体テスト・tag・性能は問わない）。

1. 3 つとも無効で、次の 6 つの比較の metrics checksum が、本書の前のビルドと一致する。比べ方と短縮条件は PRD098 の検証資材（[README](../../core/anet-core/testdata/prd098/README.md)）の段 0 と同じ（`90_perf/`・`91_resource/`・`92_resource/` を除く全 scalar 点の完全一致）。
   - skip 無効: GridMaze、LunarLander、Atari BTR 基準、DropMerge nzrep 基準（PRD098 §7 の段 0 の 4 構成）。
   - skip 有効: GridMaze（現行の GridMaze.txt の既定。QR、分離型×遮断）、LunarLander（現行の LunarLander.txt の既定。IQN、分離型×遮断）。
2. 単体テスト: E=1 の skip Head の形が今と同じ。E>1 の出力の形と平均。λ の加算（分布型のずらし）と E=1 の WARN。割り込みの判定（G と η、`episode_start`、残り 0 との重なり）と判断の旗。`all_k` の目標値の数値例（有効な k の印、done、truncated、判断の旗での区切り）。§5.1 の不正値と、`q_gap` で η が無いときのエラー。
3. smoke Run（GridMaze の短縮条件、skip 有効。作った拡張について、(U) は E=5、(I) は学習の Actor で `q_gap`・η 0.05、(K) は `all_k`）の `metrics.jsonl` と `metrics.scalar.defs` に、作った拡張の §5.6 の tag（(U) は 05、(I) は 06、(K) は 16）が出る。
4. 性能（(U) を作ったとき）: Atari BTR の構成（N=128、B=256、BF16、prefetch、Release）で、skip 有効の E=5 と E=1 を逆順の 4 Run（PRD098 の検証資材の `performance` と同じ測り方）で比べ、学習 1 回の更新の実時間の比が 1.1 倍以内。(K) は比を記録するだけで閾値を置かない。

## 8. 作らないもの・先送り

| 項目 | 裁定 | 理由と再開条件 |
|---|---|---|
| スキップ中の forward の省略（弱点 4） | 範囲外 | 別の関心（D9）。作るときの形: その step で判断する lane が 1 本も無いとき、primary と skip の forward を丸ごと飛ばす。replay へ遷移を送らない Actor にだけ置き、割り込みとは併用しない（PER の初期優先度と割り込みは毎 step の Q が要る）。飛ばした step の aux は直前の値を持ち越す。再開条件: 1 env をリアルタイムで動かす実行（実機の制御など）の PRD が起きたとき。EvalPanel は表示の速さが律速でないので対象にしない |
| 学習の Actor で、判断する lane だけを forward する形 | 範囲外 | 呼び出しの回数が減らず効かない見込み（§2.3 の本書の推定）。全 lane の行を前提にする読み手の契約を変えるが、クリーンブレークで移せるので制約ではなく、効かない見込みによる範囲外。再開条件: Actor の forward の実時間を行の数を変えて測り、行の数に比例して効く構成（CPU の forward、または IQN で lane×分位の行が律速）で、Actor の forward が全体の 2 割を超えると分かったとき |
| メンバーの事前関数と bootstrap の印 | 先送り | 実験の結果が要る。[999_bootstrapped_multihead_q](999_bootstrapped_multihead_q_10prd.md) の段階 2 と同じ仕組み。再開条件: (U) でメンバーの σ が学習の早い時期に 0 近くへ潰れたら |
| λ の自動調整 | 先送り | 実験の結果が要る（固定の λ で env ごとに最良の値が違うかは、段 U と PRD098 の段 2 以降でしか分からない）。再開条件: 固定の λ で、env ごとの最良の値が大きく違ったら |
| Lazy-MDP の任せる行動と罰 | 範囲外 | 行動空間と報酬を変える別の関心（§3）。別 PRD |
| skip Head の独立 Dueling、スカラーのスキップ価値、skip 専用の lr とクリップ閾値 | PRD098 §8 のまま | 本書で扱わない |
| 連続行動の手法、モデルベースの手法 | 範囲外 | §3 |

## 9. 未決

| # | 事項 | 扱い |
|---|---|---|
| 1 | UTE と Lazy-MDP の原文の確認 | 論文解説を survey のキューに積んだ（[reports/INDEX.md](../../reports/INDEX.md) のキューの 14・15。2026-10-25）。解説が出たら、λ の符号の意味、全メンバーで同じ目標値を使うこと、割り込みの閾値の定義を原文と照らし、違えば D2・D3・D5 を直す |
| 2 | η の尺度 | 既定は学習した Q の単位の絶対値（D6）。報酬のスケーリングで値が変わるので、Q の大きさか TD 誤差の大きさで割った相対値にするかを、段 I の結果で決める |
| 3 | 割り込みの G に UQE の risk score を使うか | 既定は平均 Q（§5.3）。学習の Actor の探索と割り込みの関係を段 I で見る |
| 4 | 段を満たさず無効のまま残した機能を、いつ消すか | 既定は残す（§7）。PRD098 の段 4 まで終えても使う env が無ければ、消す改訂を起こす |
| 5 | 新しい指標を `metrics.scalar.@skip` に既定で入れるか | 既定は入れない（§6 の 7）。機能が無効の Run で欠けた値をどう扱うかを実装で確かめて決める |
| 6 | 用語（メンバー、割り込み。D12）を CONTEXT.md に入れるか | グリルで決める |
| 7 | (K) の Atari での転送量 | 連続する観測を共有して組む実装で減らせるか。Atari を対象にするかと、性能の閾値を置くかを決める |
| 8 | PRD098 §8 との結び | PRD098 §8 の該当 3 行を本書へ向けた（PRD098 の 2026-10-25 の改訂）。残りは無い |
| 9 | 段 0 が (a) のときの PRD098 の既定（2026-10-26 の段 0 で (a) になった） | skip の方策の既定（今は行動の方策と同じ）を平均に対する ε-greedy に改める改訂を PRD098 に起こすか、env ごとの設定にするかを決める |

## 10. 依存と順序

- [PRD098](098_temporl_skip_policy_10prd.md) の実装に乗る。段 0 はいつでもできる（コードを変えない）。
- 作るものと順は段 0 の結果で決まる（D11、§7）。どの結果でも (I) を作る。
- [999_bootstrapped_multihead_q](999_bootstrapped_multihead_q_10prd.md) とは独立。多様性の仕組みを共有する余地は §8。
- [999_observation_pipeline](999_observation_pipeline_10prd.md) の単位 A（強制行動を MakeAction の入力にする）が入ったら、割り込みと同じ「判断の旗を立てて状態を捨てる」経路を使える（PRD098 §9 の 1）。

## 11. 参照

- [PRD098 TempoRL 対応](098_temporl_skip_policy_10prd.md)、[PRD098 実装メモ](098_temporl_skip_policy_20impl.md)、[PRD098 の検証資材](../../core/anet-core/testdata/prd098/README.md)
- [TempoRL 論文解説](../../reports/biedenkapp2021_temporl_paper_2026-10-03.md)
- [Atari の baseline（lane 数）](../experiments/default-dqn/atari/2026-08-17_baseline.md)、[Atari の lane 窓](../experiments/default-dqn/atari/2026-09-15_replay-lane-window.md)
- [行動の時間尺度（wiki）](../wiki/mechanisms/temporal_abstraction.md)
