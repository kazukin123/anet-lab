# 外的と内的の価値の分解（Q を外的報酬と内的報酬の 2 つの Head で学ぶ） 暫定 PRD

> 2026-11-10 起票（暫定）。質疑を省いた。確定した決定は持たず、AI が置いた既定と決めきれない点を §8 の未決表に置いた。§5・§7・§9 は §8 の暫定の既定を前提に書き、前提にした行の番号を名指す。
> 起点: Montezuma の RND の Run（§2.1）を見た 2026-11-10 の会話。会話で出た方向は次の 3 つで、どれも決定ではない。(1) 主題は 2500 点が評価側（Greedy）で安定するかで、途中の時期の差は Run ごとの揺れの内とみなす。(2) Body を分けるか共有するかは Network の設定で決まり、Agent57 型でも Burda 型でも Head の分離になるので、2 つの型の実装の違いは小さい見込み。(3) Q の計算を分けると、Learner だけでなく Replay にも影響が出る（§4）。
> 関連: [924_intrinsic_reward_novelty_10prd.md](924_intrinsic_reward_novelty_10prd.md)（NGU 型の内的報酬と方策の組。本 PRD との関係は §6）、[116_rnd_lifelong_novelty_10prd.md](116_rnd_lifelong_novelty_10prd.md)（RND による一生の新しさ。本 PRD の内的報酬の出どころ）、[098_temporl_skip_policy_10prd.md](098_temporl_skip_policy_10prd.md)（TempoRL）、[943_nstep_target_correction_10prd.md](943_nstep_target_correction_10prd.md)（n-step の目標値の補正）、[ADR 0012](../adr/0012-replay-initial-priority-hint-completion.md)・[ADR 0058](../adr/0058-approx-priority-column-computed-by-agent-step-diagnostics.md)・[ADR 0062](../adr/0062-networkmodel-holds-auxiliary-network-fed-by-primary-outputs-with-target-used-only-for-acting.md)・[ADR 0063](../adr/0063-replay-slot-keeps-per-step-reward-and-end-and-samples-variable-windows.md)・[ADR 0065](../adr/0065-scheduled-gamma-recomputes-return-at-sample.md)、[CONTEXT.md](../../CONTEXT.md) の「報酬スケーラ」「価値ストリーム」。

## 1. 目的と問題

### 問題

DQN 系の Agent は、内的報酬（RND の新しさ）を外的報酬に足した 1 つの報酬で、1 つの Q を学ぶ（§4）。Montezuma の RND の Run（§2.1）では、次の 2 つが見えた。

1. 外的報酬と内的報酬の大きさの釣り合いで、評価の結果が大きく分かれた。s1（β 0.1、Munchausen 有効）は 29.2M まで Greedy で 2500 点が 1 本も無く、b001（β 0.01、Munchausen 無効、Double DQN 有効）は 14.2M 以降の Greedy の約 2/3 が 2500 点だった。β と Munchausen を同時に変えたので、どちらが効いたかは切り分けていない（b01 で切り分け中）。
2. 2500 点を取ったあと、外的報酬の残らない局面で Q が行動の間でほぼ同じ値になり、Greedy が動かなくなった（b001。§2.1）。

1 本の Q では、外的と内的の大きさ・分散の違いを、NN が 1 つの出力で吸収するしかない。内的の割引率や終端の扱いを、外的と別にすることもできない。

### ゴール

- DQN 系 Agent が、Q を外的の Head と内的の Head の 2 つで学べる（Q = Q_外 ＋ β·Q_内）。行動は合わせた Q で選ぶ。既定は無効で、無効のときは今と同じ学習系列になる（同 seed・決定的モードの metrics checksum が一致する）。
- 内的の Head の割引率と終端の扱いを、外的と同じにする（Agent57 型）か、別にする（Burda 型: 割引率を別に持ち、game over で切らない）かを、設定のモードで選べる（§8 の 3・4）。
- 外的報酬と内的報酬を、取り込みから Replay・PER の初期優先度・Learner の目標値まで別々に運ぶ。Replay の契約は「報酬の成分」として汎用に書き、内的報酬という名前を Replay に持ち込まない（§8 の 6）。
- 2 つの Head の Q を、診断（Trace の列とメトリクス）で別々に見られる（§8 の 13・14）。
- [924](924_intrinsic_reward_novelty_10prd.md)（NGU 型と方策の組）が、この分解の上に方策の番号の入力（UVFA）を重ねられる形にする（§6）。
- 効果を Montezuma で確かめる。主な指標は、2500 点の Greedy での安定（§9.1）。

### 非ゴール

- エピソード内の新しさ（NGU・E3B）、方策の組（方策ごとの β_j・γ_j と、方策の番号の入力）、β=0 の評価の方策の学び方。別の関心として 924 で扱い、本 PRD は方策 1 つ（β 1 つ）のまま扱う。技術の制約ではなく、未決が 23 件ある 924 と切り離して先に試すための順序の選択。本 PRD の形の上に 924 が重ねる（§6）。
- Retrace などの方策オフの補正。別の関心で、924 の 14 と [943](943_nstep_target_correction_10prd.md) の範囲。
- 内的の収益を次のエピソードへつなぐこと（Burda らの本来の非エピソードの形）。今の Replay の n-step はエピソードの終わりで切れる（§4）ので、本 PRD は終端の観測から bootstrap する近似にとどめる（§8 の 4）。技術の制約ではない。先送りの理由は実験の結果が要ることで、作る条件は「`non_episodic` の群が `episodic` の群を上回り、かつ終端の観測からの bootstrap の粗さが効いていると分かったとき」。そのときは Replay の作りを変える別 PRD にする。
- Agent57 の方策の選択（バンディット）。別の関心で、924 §6 の範囲。
- Atari のゲームの仕様を変えること、RAM の値を報酬にすること。924 の非ゴールと同じ裁定による。
- 価値の分解と Munchausen・TempoRL を同時に有効にすること。本 PRD では設定の誤りとして止める（§8 の 9・10）。技術の制約ではなく、どちらの Head に何を足すかを決めずに併用させないための順序の選択。Munchausen は別の関心（Munchausen の項の扱いは 943 と 924 の 15 で決める）として先送りし、作る条件は 943 か 924 の 15 で項の扱いが決まったとき。TempoRL は実験の結果が要るとして先送りし、作る条件は「Atari で分解の効果が出て、TempoRL が既定で有効な LunarLander でも使いたくなったとき」。

## 2. 背景

### 2.1 Montezuma の RND の Run（2026-11-08〜10、seed 1）

3 本は同じ構成（workspace AtariRnd-01、[apps/11_batch_run.bat](../../apps/11_batch_run.bat) の Run 名に `mz_tg_` を持つ群）から、β・Munchausen・Double DQN だけを変えた。内的報酬は RND の既定の式（`clipped`・`whiten`。1 step の内的報酬は β × (α − 1) で、α − 1 は 0〜4）で、外的報酬に足して 1 本の Q で学ぶ（[116](116_rnd_lifelong_novelty_10prd.md) D2・D4）。割引率は 0.997、n-step は 3。

| Run | β | Munchausen | Double DQN | 評価側: Greedy（evalg・evalgs、各 16 ゲーム、約 1M ごと） | 学習側 |
|---|---|---|---|---|---|
| s1（`run_20261108-232006_mz_tg_rnd_s1`） | 0.1 | 有効 | 無効 | 29.2M まで 2500 点は 0 本。1 回の平均の最高は 462.5（27.2M）で、10.2M 以降はおおむね 400 点（鍵と扉）、ときどき 500 点。ε 0.01 の評価（10 ゲーム、約 2M ごと）も 10.2〜28.21M に 400〜490 | 最高は 2500 点（12・13・17・28M 台に出た） |
| b001（`run_20261110-003002_mz_tg_nmdd_b001_s1`） | 0.01 | 無効 | 有効 | 14.2M に初めて 2500 点。14.2〜29.2M の 16 回で、2500 点は evalg 170/256（66%）、evalgs 172/256（67%）。1 回あたり 0/16〜16/16 で揺れ、落ちた回（20.2M・27.2M・28.2M）は 500 点（剣を取り、敵を倒さない）が主。前半（14.2〜21.2M）64%、後半（22.2〜29.2M）69%。ε 0.01 の評価は 14.21〜28.21M に 74/80（92%） | 15M 以降、終わったゲームの 83〜91% が 2500 点。最高 2500 点。29.71M で手で止めた |
| b01（`run_20261110-120903_mz_tg_nmdd_b01_s1`） | 0.1 | 無効 | 有効 | 走行中（2026-11-10 16:51 に 11.5M）。Greedy は 9.2M に 6/16 が 400 点、11.2M は 0 点。2500 点はまだ無い | 最高 500 点（9M 台から） |

集計は各 Run の metrics の episode trace（`53_evalg`・`54_evalgs`・`52_eval2`・`42_env`）から行った。

**b001 の 2500 点のあと。** 2026-11-10 11:35〜11:53 に Training を止めて EvalPanel で見ると、2500 点を取ったあと部屋 1 の開始の足場で UP を押し続け、27,000 step の打ち切りまで動かなかった（lives 1）。このときの Q は 18 行動とも 0.055〜0.059（差 0.004 で、τ の標準偏差 0.009〜0.010 より小さい）、V_s は 0.056〜0.063 だった。評価側の指標では 29.2M の Greedy の 16 ゲームがすべて 2500 点・打ち切りで終わり、学習側の指標では 15M 以降のゲームが 5〜8 千 step で game over により終わっていた（ε のある lane は動き続けて死ぬ）。

**b001 の内的報酬の大きさ。** 13M 以降の 1 step の内的報酬は 2〜4e-4 で、正の値は 13〜19%、RND の生の誤差は 0.021 から 0.013 へゆっくり下がった。

**見立て（未検証）。** 外的報酬が近くに残らない局面では、Q は内的報酬（1 step 3e-4 を約 330 step 積んだ 0.1 前後）だけでほぼ一様になり、行動を選ぶ向きが無くなる。1 本の Q では、この一様な内的の価値と、外的の価値の差を分けて学べない。

### 2.2 文献

- **Taiga ら（ICLR 2020、[arXiv 2109.11052](https://arxiv.org/abs/2109.11052)）。** Rainbow＋RND は Montezuma の 5 seed すべてが 90M フレームまでに 4,400 点を超えた（図 7 を目で読んだ値）。NoisyNets の 5 seed は 35〜40M フレームから 200M フレームまで 2500 点にとどまった。表 5 の 200M フレームの値は RND 5,199、ε-greedy 780。同じ RND でも 2500 点は天井ではない。b001 は 29.71M exp step（1 exp step = 4 フレームで 119M フレーム）で止めた。
- **NGU（[Badia ら 2020a](https://arxiv.org/abs/2002.06038)）。** 方策の番号 j（β_j と γ_j の組）を one-hot で NN に入力し、j ごとに外的報酬と β_j 倍の内的報酬を足した報酬で、1 本の価値を学ぶ（[Agent57](https://arxiv.org/abs/2003.13350) §2 の要約による）。Agent57 §2 は、外的と内的の報酬の大きさや疎らさが違うと NGU が不安定になると書いている。
- **Agent57（Badia ら 2020b）§3.1。** Q(x, a, j) = Q(x, a, j; θ_e) ＋ β_j·Q(x, a, j; θ_i) に分ける。θ_e と θ_i は同じ形の別々の NN で、どちらも j を入力に取る。2 つは外的報酬と内的報酬で別々に学び、bootstrap に使う方策は、合わせた Q の argmax で共通にする。単純な勾配降下では、合わせた報酬で 1 本を学ぶのと目的が等価になる（付録 B）。分ける理由は、各 NN がそれぞれの報酬の大きさと分散に合わせられることと、optimizer の状態を分けられること。transformed Bellman の h を使う分け方もあるが、実験では単純な和を使った。
- **Agent57 §4.2。** 分けると、内的報酬の重みの広い範囲で学習が安定した。外すと成績が 20% 以上落ちた。
- **Burda ら（[RND、ICLR 2019](https://arxiv.org/abs/1810.12894)）§2.3。** 内的の収益を game over で切らない（非エピソード）。外的は game over で切る。性質の違う 2 つを組むために、価値の Head を 2 つ（V_E と V_I）にし、和を価値にする。
- **Burda ら §3.2（図 4）。** 内的を非エピソードにすると、探索した部屋の数が増えた。両方エピソードのままなら、Head を 2 つにしても 1 つより良くならなかった。
- **Burda ら §3.3（図 5）。** 外的の割引率 0.999・内的 0.99 にすると成績が大きく上がった。内的を 0.999 にすると下がった。

### 2.3 2 つの型

| | Agent57 型 | Burda 型 |
|---|---|---|
| 割引率 | 外的と内的で同じ | 内的を別に持つ（Burda らは外的 0.999・内的 0.99） |
| 終端 | 両方 game over で切る | 内的は切らない |
| 目的 | 合わせた報酬で 1 本を学ぶのと同じ。変わるのは学び方（それぞれの大きさに合わせられる） | 目的そのものが変わる |
| 効く先の見立て | 大きさの違い（§1 の 1） | 内的の価値が近くの新しさへ向く（§1 の 2）。未検証 |
| 実装の違い | NN は同じ（Head の分離。§8 の 2）。違いは Replay の収益の割引率と、Learner の終端のマスクだけ（§5） | 同左 |

## 3. 用語

| 語 | 意味 |
|---|---|
| 価値の分解 | Q を外的の Head と内的の Head の和で表し、それぞれを別の報酬で学ぶこと。[CONTEXT.md](../../CONTEXT.md) の「価値ストリーム」（Dueling の V の流れ）とは別のもので、本 PRD ではストリームの語を使わない |
| 外的の Head・内的の Head | 外的報酬で学ぶ Q の出力と、内的報酬で学ぶ Q の出力。Head・Body は訳さない |
| 報酬の成分 | Replay が 1 step ごとに別々に記録する報酬の並び（外的と内的）。Replay の汎用の語として使い、内的報酬の意味を持たせない |

既存の語では言い換えられない。「価値ストリーム」は Dueling の V の流れに限った語で、報酬の種類で分けた出力を表さない。「報酬スケーラ」は報酬の大きさを変える部品で、報酬を複数の並びで持つことを表さない。用語集への登録は、924 の 20（用語の確定）とあわせて決める（§8 の 18）。暫定の起票なので、本 PRD では [CONTEXT.md](../../CONTEXT.md) を変えない。

## 4. 現状（2026-11-10 にコードで確認）

- 取り込み: `DefaultDQNAgent::UpdateFromBatch` が、報酬スケーラを通した外的報酬に、RND の内的報酬（β × (α − 1)）を足して 1 本の報酬にし、内側の Learner へ渡す（[default_dqn_agent.cpp:711-736](../../core/anet-core/src/default_dqn_agent.cpp)、足すのは 735 行）。[116](116_rnd_lifelong_novelty_10prd.md) D2 はこれを「報酬へ畳む（仮）」とし、外的・内的を別に保存する契約を 924 に送っている（116 §1.3）。
- 受け渡しの形: `BatchExperience` の報酬は 1 本（[rl.hpp:615-627](../../core/anet-core/include/anet/rl.hpp)）。
- Replay の記録: Push は割引率を 1 つ受け取る（[replay_buffer.hpp:121](../../core/anet-core/include/anet/replay_buffer.hpp)）。1 step ごとに報酬 1 つを待ち行列と step の記録へ入れる（[replay_buffer_impl.cpp:1634-1640](../../core/anet-core/src/replay_buffer_impl.cpp)、[replay_buffer_impl.hpp:49](../../core/anet-core/src/replay_buffer_impl.hpp)・[236-262](../../core/anet-core/src/replay_buffer_impl.hpp)）。step の報酬・終わり方・判断の旗を slot に持つのは [ADR 0063](../adr/0063-replay-slot-keeps-per-step-reward-and-end-and-samples-variable-windows.md) による。
- n-step の収益: Push の時点で、1 つの割引率で逆順に足して slot に 1 つ保存する（[replay_buffer_impl.cpp:326-352](../../core/anet-core/src/replay_buffer_impl.cpp)・[711](../../core/anet-core/src/replay_buffer_impl.cpp)）。収益の計算はエピソードの終わり（done）で切れ、打ち切り（truncated）は bootstrap を続ける。割引率が schedule のときは、抽選のあとに保存済みの step の報酬から作り直す（[ADR 0065](../adr/0065-scheduled-gamma-recomputes-return-at-sample.md)）。
- 標本: n-step の収益、窓の収益、抽選時の step の報酬は、どれも 1 本（[rl.hpp:828-870](../../core/anet-core/include/anet/rl.hpp)）。
- PER の初期優先度: 見積もりの入力は、確定済みの n-step の収益 1 つと、bootstrap の割引と終端（[replay_buffer.hpp:41-48](../../core/anet-core/include/anet/replay_buffer.hpp)）。ACTOR_APPROX では Actor の Q_sa・V_s・Munchausen の項を hint に詰めて運ぶ（[dqn_based_agent.cpp:351](../../core/anet-core/src/dqn_based_agent.cpp)、見積もりは [427](../../core/anet-core/src/dqn_based_agent.cpp) の `DqnInitialPriorityEstimator`）。
- Learner の目標値: 分位点の目標は「収益 ＋ γ^n × (1 − 終端) × 次の分布」（[dqn_based_agent.cpp:4347-4362](../../core/anet-core/src/dqn_based_agent.cpp)）。Munchausen の目標は [4438](../../core/anet-core/src/dqn_based_agent.cpp) から、TD の目標は [4758-4851](../../core/anet-core/src/dqn_based_agent.cpp)、TempoRL の窓の収益は [4651](../../core/anet-core/src/dqn_based_agent.cpp) で Q の目標と並べる。TD・QR・IQN の 3 つの学習器がある（[dqn_based_agent.hpp:1424-1459](../../core/anet-core/src/dqn_based_agent.hpp)）。
- Head: Linear・Dueling・分位点・skip の Head の生成部品がある（[dqn_based_heads.hpp](../../core/anet-core/src/dqn_based_heads.hpp)）。Head の設定は `DefaultDQNAgent.head.*`（[agent.txt](../../apps/runner/config/agent.txt)）。Head の入力は Body の名前つきの出力で、Dueling は `net.body.output.[value_feature]`・`[adv_feature]` を受ける（[Atari.txt](../../apps/runner/config/Atari.txt) の 1276〜1277 行）。skip は `skip.net.*`・`skip.head.*` と `net.body.output.[skip_context]` で primary の出力を受ける。
- Trace: 評価の Trace の Q_sa・V_s の列は、Agent の step の診断から読む（[ADR 0057](../adr/0057-eval-trace-columns-declared-apart-from-metrics-and-read-at-transition-time.md)・[ADR 0058](../adr/0058-approx-priority-column-computed-by-agent-step-diagnostics.md)）。内的報酬は Trace に無い（[issue-20261107-1226e5](../management/issues/issue-20261107-1226e5.md)）。

**触る既存部品と、実装の地図に書かれた選ばなかった形・変えるときの条件**

| 地図の項目 | 本 PRD との関係 |
|---|---|
| 「特徴空間・行動価値・スキップ価値を別 Network にする 3 分割 — 不採用」（[324 DQN](../wiki/categories/324_dqn.md)。出典 [098](098_temporl_skip_policy_10prd.md) §4 D3、ADR 0062） | 切れ目は Network の中の branch で表せ、別 Network にすると checkpoint の形と、Network 1 つを前提にする診断・可視化へ波及する、という理由。§8 の 2 の暫定の既定（同じ Network に内的の Head を足す）の根拠にした。不採用の形を開き直すものではない |
| 「γ の schedule で常に抽選時に収益を作り直す形 — 不採用」（[318 RL 共通基盤](../wiki/categories/318_rl_core.md)。出典 ADR 0065） | 定数の割引率では Push 時の保存値の経路を保つ。§8 の 6 の暫定の既定は、成分ごとにこの規則をそのまま当てる |
| 「初期優先度のヒントを TensorDict で任意の named tensor として運ぶ形 — 不採用」（318。出典 ADR 0012） | hint は 1 つに詰めて運ぶ。§8 の 11 で hint の値が増えても、この形を保つ |
| 変えるときの条件: 「スキップが無効のとき、forward の回数・順序、RNG の消費、loss、優先度、checkpoint の形を今と同一に保ち、metrics checksum の一致で確かめる」（324。出典 098 §5.9・§7） | 価値の分解が無効のときも同じ形で確かめる（§9.2 の 1） |
| 変えるときの条件: 「slot に 1 step の記録を足しても、sampling 分布・item key・PER の completion と eviction 統計の ready 基準・Push と Extract の lock・index の昇順を変えない」（318。出典 098） | 報酬の成分を足すときも守る（§9.2 の 2） |

## 5. アルゴリズム（§8 の暫定の既定を前提）

**今の 1 本の Q。** r = r_外 ＋ β·r_内 として、

y = Σ_{h=0}^{n−1} γ^h r_{t+h+1} ＋ γ^n (1 − d) Q_target(s_{t+n}, a*)

**分解した Q。** Q(s, a) = Q_外(s, a) ＋ β·Q_内(s, a)。bootstrap の行動 a* は、合わせた Q の argmax で選ぶ（Double DQN なら online の合わせた Q で選び、target の各 Head で評価する。§8 の 7）。

y_外 = Σ_{h=0}^{n−1} γ_外^h r_外,{t+h+1} ＋ γ_外^n (1 − d) Q_外,target(s_{t+n}, a*)

y_内 = Σ_{h=0}^{n−1} γ_内^h r_内,{t+h+1} ＋ γ_内^n (1 − d_内) Q_内,target(s_{t+n}, a*)

- Agent57 型（`discount_mode = shared`・`terminal_mode = episodic`。§8 の 3・4）: γ_内 = γ_外、d_内 = d。y_外 ＋ β·y_内 は、今の 1 本の Q の y と同じ式になる。
- Burda 型（`separate`・`non_episodic`）: γ_内 は別の値、d_内 = 0。終端の遷移でも内的の Head は終端の観測から bootstrap する（近似。§1 の非ゴール）。n-step の和はエピソードの終わりで切れる。
- β は Replay に記録する内的報酬に掛けず、Q を合わせるときに掛ける（§8 の 5）。Replay の内的報酬は α − 1 のまま、外的報酬は報酬スケーラを通したもの（今と同じ）。
- 分布型（QR・IQN）では、各 Head が自分の目標で分位点の損失を学ぶ。行動を選ぶときは、各 Head の分位点の平均を β で合わせる。分位点ごとの和を合わせた分布とはみなさない（2 つの確率変数の和の分位点は、分位点の和にならないため）。IQN の τ は Body で結ぶので、2 つの Head は同じ τ の標本を使う。
- 損失は 2 つの Head の損失を 1:1 で足す（§8 の 12）。Body を共有しない構成なら、別々に最適化するのと勾配は同じになる。
- 優先度は、外的と内的を β で合わせた値に今の優先度の計算を当てる（§8 の 11）。

## 6. NGU（924）との関係

### 6.1 アルゴリズム

| | NGU | Agent57 | 本 PRD |
|---|---|---|---|
| 方策の番号 j の入力（UVFA） | あり（one-hot） | あり（2 つの NN の両方に入力） | なし（方策 1 つ） |
| 価値の分解 | なし（j ごとに合わせた報酬で 1 本） | あり（外的・内的の 2 つの NN） | あり（外的・内的の 2 つの Head） |
| 割引率 | j ごとの γ_j | j ごとの γ_j を 2 つの NN で共通に使う | 外的と同じか、内的を別に持つ（§8 の 3） |
| 方策の選択 | 均等 | バンディット | なし |

入力側の分け方（j）と、価値の分解は、別の軸である。j は「どの β・γ の組の価値を求めるか」を指定し、価値の分解は「外的と内的を別の出力で学ぶか」を決める。Agent57 は NGU に価値の分解を重ねた。したがって本 PRD は 924 と食い違わず、924 が j を足すときは、2 つの Head の両方に j を入力する（Body の入力に j を足す Network の設定で表せる見込み。924 の 4）。

924 の論点への効き方:

- **924 の 2（β=0 の方策を全 lane の経験で直接学ぶ形）。** 外的の Head の bootstrap の行動を、合わせた Q ではなく外的の Head 自身の argmax にすると、外的の Head は「内的報酬を使わない点数の方策」の価値を、全 lane の遷移から方策オフで学ぶ形になる。924 の 2 の候補「条件付きの外的・内的価値分解」の一つの実現である。ただし、合わせた Q が 1 つの方策の価値ではなくなる（924 §4.5 の「bootstrap policy の整合」）。本 PRD の暫定の既定は Agent57 と同じ共通の行動にし、外的の Head 自身の argmax は候補として残す（§8 の 7）。
- **924 の 16（評価の方策）。** Actor ごとに行動に使う β を変えられるようにすると（§8 の 8）、評価で β=0（外的の Head だけ）の行動を見られる。924 が方策の番号で評価する形に決めたら、この葉は方策の番号の選択へ置き換える。
- **924 §4.5 の注意。** 価値を分けるだけで、全方策の γ や bootstrap の方策の整合が解決するわけではない。本 PRD は方策 1 つなので、この問題を持ち込まない。

### 6.2 実装

本 PRD が作り、924 がそのまま使う見込みのもの:

| 部品 | 本 PRD | 924 で足すもの |
|---|---|---|
| Replay の報酬の成分 | 外的・内的の 2 成分を 1 step ごとに記録し、成分ごとの割引率で n-step の収益を作る（§8 の 6） | 924 の 11（収集時の内的報酬と外的報酬を分けて保存する案）はこれで満たす見込み。方策の番号の保存と、方策ごとに違う γ を同じバッチで使う契約（924 §3）は 924 で足す。本 PRD の割引率は成分ごとで、標本ごとではない |
| 取り込み | 価値の分解が有効なら、内的報酬を報酬へ畳まず、別の成分で渡す。無効なら 116 D2 のまま | 924 の採点点（Actor か facade か）で取り込みの場所が変わっても、成分で渡す形は同じ |
| Q の Head | 内的の Head を足す（§8 の 2） | j の入力を Body に足す。2 つの Head の両方が j を受ける |
| Learner の目標値 | Head ごとの目標と、Munchausen・skip との併用を止める判定（§8 の 9・10） | 924 の 15（QR／IQN・Munchausen・PER・UQE の揃え方）で、併用の仕方を決める |
| 優先度 | 合わせた値で 1 つ（§8 の 11） | 924 の 15 の「複数 i の loss を使う場合の優先度集約」 |
| 診断 | 2 つの Head の Q_sa を Trace とメトリクスへ | 方策の番号、エピソード内の新しさの値 |

[116](116_rnd_lifelong_novelty_10prd.md) の部品（一生の新しさの口、facade の採点と補助学習）は変えない。116 D2 の「報酬へ畳む（仮）」は、価値の分解が無効のときの運び方として残る。

### 6.3 課題の関係

| 課題 | 関係 |
|---|---|
| [issue-20261110-0de9e7](../management/issues/issue-20261110-0de9e7.md) | 本 PRD の仕事 |
| [issue-20261103-736e39](../management/issues/issue-20261103-736e39.md)（924） | 924 の 2・11・15 の一部を本 PRD が先に作る。924 の再グリルは本 PRD の契約を前提にできる。`depends` に本 PRD の課題を足すかは開発者が決める（本 PRD では変えない） |
| [issue-20261106-16b7af](../management/issues/issue-20261106-16b7af.md)（116） | 内的報酬の出どころ。116 の部品は変えない |
| [issue-20261107-1226e5](../management/issues/issue-20261107-1226e5.md)（Trace で内的報酬を見られない） | 同じ Trace に列を足すが、評価の Actor では RND の採点が走らない（116 §1.3）ので、内的報酬の列は本 PRD では解けない。別のまま |
| [issue-20261103-ab98b7](../management/issues/issue-20261103-ab98b7.md)（TD・QR・IQN の UpdateFromSamples の重複） | 本 PRD は 3 つの学習器に Head ごとの目標を足す（§8 の 1）ので、重複のまま入れると同じ変更を 3 回書く。先に片づけるか、同時に片づけるかは未決（§8 の 1） |
| [issue-20261103-75b8da](../management/issues/issue-20261103-75b8da.md)（943） | Munchausen の項の扱いが重なる。本 PRD は併用を止める（§8 の 9） |

### 6.4 順序

本 PRD は RND（116、実装済み）だけで動き、Montezuma で効果を試せる。924 の再グリルの前に入れてよい。924 の再グリルでは、本 PRD の契約（報酬の成分、内的の Head）を前提に、j の入力と方策の組を重ねる。順序の判断は開発者が行う。

## 7. 実装契約（暫定の既定を前提）

### 7.1 設定と失敗の扱い

名前はすべて暫定（§8 の 18）。

| 葉 | 型・値 | 読む条件 | 前提にした未決表の行 |
|---|---|---|---|
| `DefaultDQNAgent.intrinsic.value_mode ?= combined` | `combined`（今の 1 本の Q）｜`decomposed`（価値の分解） | 常に | §8 の 1 |
| `DefaultDQNAgent.intrinsic.discount_mode ?= shared` | `shared`（外的と同じ割引率）｜`separate` | `decomposed` のとき | §8 の 3 |
| `DefaultDQNAgent.intrinsic.gamma` | 実数、0 より大きく 1 より小さい。ProfiledValue（今の外的の割引率 `DefaultDQNAgent.learner.gamma` と同じ形で、schedule も書ける） | `separate` のとき | §8 の 3 |
| `DefaultDQNAgent.intrinsic.terminal_mode ?= episodic` | `episodic`（game over で切る）｜`non_episodic`（切らない） | `decomposed` のとき | §8 の 4 |
| `DefaultDQNAgent.intrinsic.value.head.*` | 内的の Head の設定。葉は `DefaultDQNAgent.head.*` と同じ（`use_dueling`・`init.*`・`projection.noisy.*`） | `decomposed` のとき | §8 の 2 |
| `DefaultDQNAgent.actor.[<名前>].intrinsic.action_beta_mode ?= learner` | `learner`（学習と同じ β で行動）｜`value` | `decomposed` のとき | §8 の 8 |
| `DefaultDQNAgent.actor.[<名前>].intrinsic.action_beta` | 実数、0 以上 | `value` のとき | §8 の 8 |

判定は、次の表を上から順に当て、最初に当たった行で扱う。どの組み合わせも最後の行までに必ず当たる。

| # | 条件 | 扱い |
|---|---|---|
| 1 | `value_mode`・`discount_mode`・`terminal_mode`・`action_beta_mode` のどれかが列挙の外 | ERROR（キーと値と列挙を示す） |
| 2 | `value_mode = combined` | 分解の葉（`discount_mode`・`gamma`・`terminal_mode`・`value.head.*`・`action_beta_mode`・`action_beta`）を読まない。書いてある場合の扱いは [117](117_warn_policy_and_unread_config_10prd.md) の読まない設定の決まりに従う。学習は今と同じ |
| 3 | `value_mode = decomposed` かつ `intrinsic.enabled = false` | ERROR（内的報酬が無いので分解できない） |
| 4 | `decomposed` かつ Munchausen が有効 | ERROR（§8 の 9） |
| 5 | `decomposed` かつ `skip.enabled = true` | ERROR（§8 の 10） |
| 6 | `decomposed` かつ `discount_mode = separate` で、`gamma` が無いか範囲の外 | ERROR（キーと値を示す） |
| 7 | `decomposed` かつ `action_beta_mode = value` で、`action_beta` が無いか 0 未満か有限でない | ERROR |
| 8 | それ以外 | 有効。`shared` なら `gamma` を読まない（扱いは 117 に従う）。`learner` なら `action_beta` を読まない（同上） |

黙った代替はしない。`decomposed` を、内的の Head が組めない構成で今の 1 本の Q に戻さない。

### 7.2 Replay の契約（汎用）

前提: §8 の 6。

- Replay の設定で、報酬の成分の数を決める。既定は 1 成分で、今と同じ記録・収益・標本・checksum になる。
- 1 step ごとに、成分の数だけ報酬を受け取って記録する。
- n-step の収益を成分ごとに作る。割引率は成分ごとに受け取る。定数なら Push の時点で保存し、schedule なら抽選のあとに保存済みの step の報酬から作り直す（ADR 0065 の規則を成分ごとに当てる）。
- 窓の収益と抽選時の step の報酬も、成分ごとに返す。
- PER の初期優先度の見積もりの入力は、成分ごとの n-step の収益と割引にする。
- 終端の扱い（bootstrap を切るか）は Replay の契約に入れない。Replay は今と同じ終端と打ち切りを返し、成分ごとのマスクは Learner が決める。
- 成分の名前と意味（外的・内的）は Replay に持たせない。どの成分を何に使うかは DQN 側が決める。

### 7.3 Learner と Actor

- Q の Network に内的の Head を足す（§8 の 2）。target 同期と checkpoint は今の Network の仕組みに乗る。
- 目標値は §5 の式で Head ごとに作る。bootstrap の行動は合わせた Q（§8 の 7）。
- 損失は 2 つの Head の和（§8 の 12）。優先度は §8 の 11。
- Actor は合わせた Q で行動を選ぶ。`action_beta_mode = value` の Actor は、その β で合わせた Q で選ぶ（§8 の 8）。ε・τ・NoisyNet の扱いは今と同じ。
- ACTOR_APPROX の hint は、2 つの Head の Q_sa と V_s を運ぶ（ADR 0012 の 1 つに詰める形のまま。§8 の 11）。

### 7.4 診断

- Trace に、外的の Head と内的の Head の Q_sa を列で足す（§8 の 13）。今の Q_sa・V_s は合わせた値のまま。
- メトリクスに、2 つの Head の損失と Q の平均を足す（§8 の 14）。

### 7.5 クラス一覧

> 従えない事情があれば、実装する側が理由を実装計画に書いて変えてよい（質問しない）。

| 名前 | 区分 | 責務 | 置き場（機能グループ） |
|---|---|---|---|
| DefaultDQNAgent | 変更 | `decomposed` のとき、内的報酬を報酬へ畳まずに別の成分で内側の Learner へ渡す。設定の判定（§7.1） | DefaultDQNAgent |
| BatchExperience・ExperienceSamples | 変更 | 報酬・収益を成分ごとに運ぶ | RL 共通基盤（rl） |
| ReplayBufferConfig・PrefetchingReplayBuffer・ReplayExperienceStorage・DefaultExperienceBuilder | 変更 | 成分ごとの step の記録と n-step の収益、窓・抽選時の報酬 | ReplayBuffer |
| InitialPriorityEstimator・DqnInitialPriorityEstimator | 変更 | 成分ごとの収益と、2 つの Head の hint から初期優先度を見積もる | ReplayBuffer・DQN 系 |
| Learner・QuantileLearnerBase・TDLearner・QRLearner・IQNLearner | 変更 | Head ごとの目標値・損失・優先度。Munchausen・skip との併用の判定 | DQN 系（dqn_based_agent） |
| IntrinsicValueHeadFactory（候補） | 新規 | 内的の Head を `intrinsic.value.head.*` から組む。既存の Head の生成部品を使い、複製しない | DQN 系の Head（dqn_based_heads） |
| 診断の提供部品（step の診断・Trace の列） | 変更 | 2 つの Head の Q_sa を出す | DQN 系・評価の Trace |

### 7.6 移行対象

経路の検索（2026-11-10、`git grep -i -F`、範囲は `core`・`apps`・`docs`）: `target_returns`・`InitialPriorityEstimateInput`・`sampled_rewards`・`window.returns`・`FinalizeReturns`・`GetStepReward`・`PackActorQHint`・`報酬へ畳む` で 25 ファイル（過去の PRD・実装計画・実験記録・ADR・課題を含み、それらは書き換えない）。そのうち現用のものに、取り込みと Head の箇所（§4 でコードを読んで特定）を足すと次のとおり:

- コード: [rl.hpp](../../core/anet-core/include/anet/rl.hpp)・[rl.cpp](../../core/anet-core/src/rl.cpp)・[replay_buffer.hpp](../../core/anet-core/include/anet/replay_buffer.hpp)・[replay_buffer_impl.hpp](../../core/anet-core/src/replay_buffer_impl.hpp)・[replay_buffer_impl.cpp](../../core/anet-core/src/replay_buffer_impl.cpp)・[dqn_based_agent.hpp](../../core/anet-core/src/dqn_based_agent.hpp)・[dqn_based_agent.cpp](../../core/anet-core/src/dqn_based_agent.cpp)・[dqn_based_heads.hpp](../../core/anet-core/src/dqn_based_heads.hpp)・[default_dqn_agent.cpp](../../core/anet-core/src/default_dqn_agent.cpp)・[default_dqn_agent.hpp](../../core/anet-core/include/anet/default_dqn_agent.hpp)。
- テスト: [replay_buffer_test.cpp](../../core/anet-core/src/replay_buffer_test.cpp)・[dqn_based_agent_test.cpp](../../core/anet-core/src/dqn_based_agent_test.cpp)・[dqn_munchausen_test.cpp](../../core/anet-core/src/dqn_munchausen_test.cpp)・[intrinsic_test.cpp](../../core/anet-core/src/intrinsic_test.cpp)。Push の口が変わるので、Push を呼ぶテストはすべて直す。断定が変わるテストは、1 成分のときに今と同じ値になる断定だけなので、既存の断定は変わらない見込み（実装計画で確かめる）。
- 設定: [agent.txt](../../apps/runner/config/agent.txt)（§7.1 の葉の既定）。Atari.txt の Run プロファイルへ群を足すかは実験のとき（§8 の 16）。
- 設計文書: [150 ReplayBuffer](../design/150_replay_buffer.jp.md)・[200 DQN 系 Agent](../design/200_dqn_agents.jp.md)・[212 RND](../design/212_rnd_lifelong_novelty.jp.md) の jp と en。価値の分解は新しい仕組みなので、200 の節にするか、2xx の設計文書を新設するかを決める（§8 の 19）。
- 互換層: 旧い Push の口の別名や互換の分岐は作らない（クリーンブレーク）。過去の Run・実験記録・ADR は書き換えない。

## 8. 決めること（未決表）

暫定の起票なので、どの行も決まっていない。「暫定の既定」は AI が置いた既定で、§5・§7・§9 はこれを前提に書いた。再グリルですべて開き直す。

| # | 事項 | 暫定の既定 | 理由と依存 | ほかの候補 |
|---|---|---|---|---|
| 1 | 入れる範囲 | DefaultDQNAgent の TD・QR・IQN の 3 つの学習器。有効化は `intrinsic.value_mode` の 1 つ。課題 ab98b7（3 つの学習器の骨格の重複）は先に片づけない | 後で要るものは同時に作る。Atari の今の構成は IQN、LunarLander も IQN だが、TD・QR だけ未対応にすると設定の組み合わせで黙って動かないものができる（§7.1 で止める行が増える） | IQN だけにして TD・QR は ERROR。ab98b7 を先に片づけてから入れる（重複のまま 3 回書く手間が減る） |
| 2 | Head の入れ方 | Q の Network に内的の Head を足す。同じ Network・同じ target 同期・同じ checkpoint。Head の入力（Body を共有するか、内的用の branch を持つか）は Network の設定（`net.body.output`・`net.branch`）で選ぶ。既定は Body を共有する | 324 の地図の不採用の理由（別 Network は checkpoint と診断へ波及、branch で表せる。§4）。会話の方向 (2)。Agent57 の別々の 2 つの NN は、branch で Body まで分ければ同じ形になる。パラメータが分かれていれば Adam の状態もパラメータごとに分かれるので、Agent57 の「optimizer の状態を分ける」理由の多くは同じ Network でも満たせる。共有されるのは、勾配の大域ノルムの clip と学習率の schedule。既定を共有にするのは計算量（Body の forward が 1 回）のため | 別の Network（Agent57 そのまま）。既定を Body の分離にする |
| 3 | 割引率 | `discount_mode ?= shared`。`separate` の `gamma` に既定値は置かない（指定必須） | `shared` は目的が今と等価（Agent57 付録 B）なので、効果を「分け方」だけに絞って測れる（§9.1）。Burda らの値 0.99（§2.2）は外的 0.999 との組の値で、anet の外的 0.997 との組では決まらないので、既定値にしない | `separate` を既定にする |
| 4 | 終端 | `terminal_mode ?= episodic`。`non_episodic` は終端の観測から bootstrap する | 3 と同じ理由。`non_episodic` の近似は Replay がエピソードをまたがない今の作りで作れる最小の形（§1 の非ゴール）。終端の観測の Q_内 は学習で見る回数が少ないので、値が粗い可能性がある | 次のエピソードの最初の観測から bootstrap する（Replay がエピソードをまたぐ必要がある） |
| 5 | β を掛ける位置 | Replay には β を掛けない内的報酬（α − 1）を記録し、β は Q を合わせるときに掛ける。外的報酬は報酬スケーラを通したもの（今と同じ） | Agent57 §3.1 の形。β の schedule（`intrinsic.beta` は ProfiledValue）を変えても過去の遷移を作り直さずに済み、Q_内 の大きさが β に依らない。`combined` のときは今のまま（116 D2） | Replay に β 倍した値を記録する（`combined` と同じ尺度。β の schedule が過去の遷移に効かない） |
| 6 | Replay の口 | 汎用の「報酬の成分」を足し、成分ごとの割引率で n-step の収益を作る。Push に割引率を成分の数だけ渡す。定数は Push 時の保存値、schedule は抽選時に作り直す | 汎用層に内的報酬の名前を持ち込まない（[AGENTS.md](../../AGENTS.md) §汎用機構と利用側の責任境界）。定数の経路を保つ（ADR 0065、§4 の地図の行）。1 成分のときの checksum を保てる | Replay は step の報酬だけを持ち、収益は Learner が作る（口が大きく変わり、定数の経路の checksum が変わる）。成分を 2 に固定する（汎用でなくなる） |
| 7 | bootstrap の行動 | 両 Head とも、合わせた Q で選んだ行動。Double DQN なら online の合わせた Q で選び、target の各 Head で評価 | Agent57 §3.1 と同じで、`shared`・`episodic` のとき目的が今と等価になる（§5）。924 の 2 に依存 | 外的の Head は自分の argmax で bootstrap する（§6.1。外的の Head が β=0 の点数の方策の価値になるが、合わせた Q が 1 つの方策の価値でなくなる） |
| 8 | Actor ごとの行動の β | 作る（`actor.[<名前>].intrinsic.action_beta_mode`・`action_beta`）。既定は `learner` | 評価で β=0（外的の Head だけ）の行動を見ると、2500 点のあとに動かない局面（§2.1）が外的の Head にもあるかを分けられる。924 の 16 で後で要るので同時に作る。924 が方策の番号で評価する形に決めたら置き換える | 作らない（924 に任せる） |
| 9 | Munchausen との併用 | 止める（§7.1 の 4） | Munchausen の項（τ·log π）は合わせた方策から作るので、どちらの Head の目標に足すかが決まらない。s1 では Munchausen が有効だった（§2.1）。943 の窓の途中の Munchausen の項と重なる | 外的の Head に足す。両方に足す |
| 10 | TempoRL（skip）との併用 | 止める（§7.1 の 5） | skip の目標は primary の Q で bootstrap する（§4）ので、skip の Head も 2 つにするかが決まらない | skip は合わせた報酬の窓の収益と、合わせた Q で学ぶ |
| 11 | PER の優先度 | 外的と内的の目標・推定を β で合わせた値に、今の優先度の計算を当てる。分布型では、同じ τ の分位点を β で合わせた値に当てる。ACTOR_APPROX の hint は 2 つの Head の Q_sa と V_s を運び、見積もりは成分ごとの収益と割引で合わせる | 優先度は行動に使う Q にとっての驚きを表す、という今の意味を保つ（`combined` のときと同じ量になる）。分位点ごとの和は優先度のためだけの近似 | 2 つの Head の損失の和（内的の大きさ α − 1 は最大 4 で、外的の clip 1 より大きくなりうるので、内的が優先度を支配しうる） |
| 12 | 損失の重み | 2 つの Head の損失を 1:1 で足す。重みの葉は作らない | Body を共有しない構成なら、別々に最適化するのと勾配が同じ（Agent57）。重みを足すと目的が変わる | 重みの葉を足す |
| 13 | Trace の列 | 外的の Head と内的の Head の Q_sa を足す（列の名前は 18 で決める）。今の Q_sa・V_s は合わせた値のまま | §2.1 の局面を Head ごとに見るため。ADR 0057 の列の宣言に乗る | V_s も Head ごとに足す |
| 14 | メトリクス | 2 つの Head の損失と Q の平均を足す。群の番号は [108](108_metrics_tag_numbering_10prd.md) の決まりで、実装のときに空きを確かめる | 2 つの大きさの釣り合い（§1 の 1）を学習中に見るため | 合わせた Q に占める β·Q_内 の割合も足す |
| 15 | checkpoint | 内的の Head は Network の一部として保存する。価値の分解の有無が設定と checkpoint で違えば、今の Network の形の検証で ERROR になる。Replay の保存は今の範囲の外のまま | 2 の帰結。互換の読み替えは作らない | — |
| 16 | 実験の組み立て | Montezuma、seed 1、b001 と同じ構成に `decomposed`・`shared`・`episodic` を足し、β 0.01 と 0.1 の 2 群、30M。判定は §9.1 | 主題（2500 点の Greedy での安定）を、今の比較相手（b001・b01）と同じ条件で見る。β 0.1 の群は b01 の結果と並べてスケールの効き方を見る。`separate`・`non_episodic` は、`shared` の結果を見てから足す | `separate`・`non_episodic` を同時に回す。seed を増やす |
| 17 | 924 の課題との依存 | 本 PRD では 924 の課題を変えない。924 の課題の `depends` に本 PRD の課題を足すかは、開発者が決める | 課題の監理の項目は開発者の指示で変える（[anet-create-issue](../../.agents/skills/anet-create-issue/SKILL.md) の監理の項目） | 924 の課題の `depends` に足す |
| 18 | 名前と用語 | 葉と列の名前は §7.1 の暫定の名前。「価値の分解」「外的の Head・内的の Head」「報酬の成分」を用語集へ入れるかは、924 の 20 とあわせて決める | 924 の 20 が内的報酬の用語を決める場所。CONTEXT の「価値ストリーム」と紛れないように、ストリームの語を使わない | — |
| 19 | 設計文書 | 価値の分解を [200](../design/200_dqn_agents.jp.md) の節にする。Replay の報酬の成分は [150](../design/150_replay_buffer.jp.md) に書く | 新しい仕組みには設計文書が要る。価値の分解は DQN 系の Learner の中の仕組みなので 200 に置く | 2xx の設計文書を新設する |
| 20 | ADR | 再グリルで決めたら、「Replay は報酬の成分ごとの割引率で収益を作る」を ADR にする | 戻しにくい（Replay の口）、文脈なしでは驚く、本当のトレードオフ（Learner が収益を作る案との比較）の 3 条件を満たす見込み | ADR を起こさない |
| 21 | 性能 | 受入に閾値を置かない。追加は Head 1 つの forward・backward と、Replay の step あたり報酬 1 つ・slot あたり収益 1 つ。実装後に Atari の実時間を記録する | ホットパスの順番は変えない見込み（同じ forward に Head が 1 つ増えるだけ）。Body を分ける構成では Body の forward が 2 倍になる | 実時間の予算を置く |

## 9. 成功の測り方と受入の候補

### 9.1 効果

前提: §8 の 16。

- **主な指標（2500 点の Greedy での安定）。** 2500 点が評価で初めて出たあとの、Greedy（evalg・evalgs）の 2500 点の割合と平均点。基準は b001（evalg 66%・evalgs 67%、1 回あたり 0/16〜16/16。§2.1）。seed は 1 本ずつなので、b001 の揺れを超える差だけを読む。
  - 毎回 14/16 以上で、落ちる回が無い: 安定した。
  - 30M まで Greedy で 2500 点が出ない: 悪くなった（s1 と同じ形）。
  - その間: 差なし。
- **副の指標。** 2500 点の先（評価で 2500 点を超える点、`room_max` が 7 より先）。2500 点のあとの局面での、内的の Head の Q の行動間の差（Trace。§8 の 13）。
- **スケール。** β 0.1 の群を b01 と並べる。b01 が 2500 点に届かず、分解した β 0.1 の群が届けば、分解が大きさの違いに効いた（§1 の 1）と読む。b01 も届けば、この比較では差が出ない。
- **途中の時期の差。** 鍵や扉に届いた時期の差は、Run ごとの揺れの内とみなし、判定に使わない（起点の方向 (1)）。

### 9.2 正しさの受入候補

契約を再グリルで確定したあとに固める。

| # | 観点 | 観測する結果 |
|---|---|---|
| 1 | 無効のときの同一性 | `intrinsic.value_mode` が既定（`combined`）のとき、同 seed・決定的モードの短い Run で、実装前の実行体と metrics checksum が一致する。対象の構成は Atari の `run.@rnd`、LunarLander の RND あり・なし、GridMaze（区間は実装計画で決める） |
| 2 | 1 成分の Replay | 成分が 1 つの Replay で、[replay_buffer_test.cpp](../../core/anet-core/src/replay_buffer_test.cpp) の既存の断定がすべて通る。sampling 分布・item key・PER の completion・index の昇順が変わらない |
| 3 | 等価 | `shared`・`episodic` で、既知の値を入れた標本について、外的と内的の目標を β で合わせた値が、合わせた報酬の 1 本の目標と一致する（浮動小数の誤差の範囲） |
| 4 | 成分ごとの割引 | 既知の step の報酬で、成分ごとの n-step の収益が手計算と一致する。定数と schedule の両方、終端と打ち切りで短くなる場合を含む |
| 5 | 非エピソード | `non_episodic` のとき、終端の遷移で、内的の Head の目標に bootstrap が入り、外的の Head には入らない |
| 6 | 設定の判定 | §7.1 の判定の表の各行が、表のとおりに扱われる。ERROR の文に対象のキーと値が入る |
| 7 | Trace | EvalPanel の Trace に、2 つの Head の Q_sa の列が出る |
| 8 | 初期優先度 | ACTOR_APPROX で、hint が 2 つの Head の値を運び、`shared` のとき見積もりが `combined` の式と一致する |
| 9 | 行動の β | `action_beta_mode = value`・`action_beta = 0` の Actor の greedy の行動が、外的の Head の argmax と一致する |

## 10. 参照

- Badia ら、[Agent57: Outperforming the Atari Human Benchmark](https://arxiv.org/abs/2003.13350)、ICML 2020。§2（NGU の要約と不安定さ）、§3.1（価値の分解）、§4.2（分解の効果）、付録 B（等価）。
- Badia ら、[Never Give Up: Learning Directed Exploration Strategies](https://arxiv.org/abs/2002.06038)、ICLR 2020。
- Burda ら、[Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894)、ICLR 2019。§2.3（内的と外的の収益の組み方）、§3.2（図 4）、§3.3（図 5）。
- Taiga ら、[On Bonus-Based Exploration Methods in the Arcade Learning Environment](https://arxiv.org/abs/2109.11052)、ICLR 2020。図 7、表 5。
