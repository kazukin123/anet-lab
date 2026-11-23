# Evaluation Trace で RND の内的報酬を見る（統計を変えない採点と step 診断の鍵）PRD

> 2026-11-12 起票。グリルを行い（問い 9 つと細部 13 件）、別モデルの検討で出た論点を反映した。決定は §4。
> 2026-11-12 改訂。共有乱数を使う層は推定器の写しだけでは隔離されないため、G2・D11の学習不変性を、写しを持つActorでも乱数を引く層を持たない構成に限定した（開発者の選択）。
> 2026-11-13 改訂。受入5・10の学習報酬の照合は、偽のLearnerの受信値に加え、実際のLearnerが保存したn-step無効時の公開Replay収益でも確認できると明記した。観測する報酬の列と一致条件は変えない。
> 2026-11-13 改訂。開発者が変更した列配置を採り、rnd_bonusをrewardの直後の235番へ移した。240番のTempoRL decisionとの重複を避ける番号は開発者が選択した。rnd_err・rnd_normは600・610番を維持する。受入12・13と検証資材を同じ配置へ合わせた。
> 2026-11-13: D2 のうち β の出どころ（題、本文の 2 文目の前半、本文の最後の文）、D7 の 2 文目（学習の入口をまだ通っていない Run ではボーナスを NaN にする）、§5.2 の表の行 5 と表の下の 1 つ目の箇条は、[PRD129](0129_eval_trace_bonus_beta_before_learning_10rfd.md) が置き換える（範囲の正本は PRD129 の §9。Agent の構築時に β を exp step 0 で評価し、学習を止めた Run でも学習の入口で β を評価する）。D7 の 1 文目（ボーナスが 0 になる条件）と 3 文目（β は checkpoint に保存しない）は変わらない。§2.6 の「学習の入口は呼ばれない」は誤りで、学習の入口は train step ごとに呼ばれ、学習を止めた分岐で内的報酬の部分を飛ばしている（PRD129 §2.3）。
> 2026-11-14 改訂。§8 の U1 を決め、未決表から外した。§7 の 14 の測定では、pane を開いた状態の学習の速さが閉じた状態より通常 RND で 13%、Taiga 式で 6% 下がり、どちらも閉じた状態のぶれを超えた。推定器と β の写しは 1 回 152〜189 ms で、見積もり（数ミリ秒）の約 50 倍だった（値は[設計文書 212](../design/212_rnd_lifelong_novelty.jp.md) の §4.6）。U1 の分かれ方では 3 列を外す側だったが、開発者は Trace を開いている間の学習の遅れを許容とみて、3 列を `run.@rnd` の既定に残すと決めた。写しを減らす直しは課題 [issue-20261114-9468cb](../management/issues/issue-20261114-9468cb.md) に分けた。
> 起点: 課題 [issue-20261107-1226e5](../management/issues/issue-20261107-1226e5.md)。2026-11-07 の Montezuma の RND の Run で、Train では鍵を取るのに評価（EvalPanel）では止まる場面があり、どの画面で新しさのボーナスが大きいかを Evaluation Trace で 1 step ずつ確かめたいという話が出た。同じ日に「Greedy がたどる状態のうち Train でめったに通らないものは、Q が直らずそこで止まる」という仮説が立ち、その検証にも使う。
> 関連: [116_rnd_lifelong_novelty_10prd.md](0116_rnd_lifelong_novelty_10rfd.md)（RND の本体。評価での採点と診断を非ゴールにした）、[123_rnd_intrinsic_reward_scale_10prd.md](0123_rnd_intrinsic_reward_scale_10rfd.md)（誤差の統計と warm-up）、[173_intrinsic_reward_novelty_10prd.md](0173_intrinsic_reward_novelty_10rfd.md)（採点の場所と保存する報酬の問い）、[done/094_evalpanel_step_history_10prd.md](done/0094_evalpanel_step_history_10rfd.md)（Evaluation Trace と step 診断）、[184_eval_session_step_trace_10prd.md](0184_eval_session_step_trace_10rfd.md)（同じ鍵を評価 session で使う）、[170_nn_trace_capture_unification_10prd.md](0170_nn_trace_capture_unification_10rfd.md)（要求を引数で渡す考え方）。ほかの PRD との関係は §9。

## 1. 目的と問題

### 1.1 問題

RND を有効にした Run で、評価の 1 step ごとに新しさを見る手段が無い。

- 内的報酬は、Agent が Train の遷移を受け取って Replay へ渡す入口でだけ計算する（§2.1）。評価の Runner はこの入口を呼ばないので、評価の step には内的報酬の値が無い。
- 学習側の値は群 `56_agent_in` の集計だけで、step ごと・画面ごとには見えない。
- そのため、評価で止まる場面で新しさがどれだけ出ているかを確かめられず、「ボーナスが小さいのか、方策が使えていないのか」を切り分けにくい。起点の仮説（Greedy の止まり場は Train のデータに少ない場所か）も測れない。

### 1.2 ゴール

- **G1**: Evaluation Trace の列で、その step の観測に対する RND の値（誤差、標準化した値、ボーナス）を見られる。
- **G2**: 見ても学習が変わらない。同じ遷移の列を学習へ与えたとき、predictor の重み、誤差と観測の統計、採点の件数、群 `56_agent_in` の診断値、学習へ渡る報酬が、見ない場合と同じである。列に入れていなければ、追加の計算をしない。
- **G3**: 学習中の Run でも、checkpoint を読んだ評価専用の Run でも出る（評価専用の Run で出ない値は §5.2 の表による）。
- **G4**: 題材の課題が片付く。仮説の検証に使える状態になる。

G2 の保証は、写しを持つ Actor では forward で乱数を引く層を持たない構成、写しを持たない Actor ではそれに加えて内部の統計を更新する層も持たない構成に限る（D11）。出荷設定はこの条件を満たす。G2 は値の同一性をいう。実際の Run では、写しを取る間の待ち（§5.4）で実時間のタイミングは変わりうる。

### 1.3 非ゴール（退けた案と、本当の制約か）

| 退けた案 | 扱いと理由 |
|---|---|
| 学習用の内的報酬を Actor 側で計算し、外的報酬と分けて保存する | 範囲外。統計の更新順、補助学習の刻み、終端の扱い、Replay の保存形式が変わる。173 の問い 7・11 が持つ。本当の制約ではなく順序の選択で、移しても評価の値が可視化用であることは変わらない（§2.5） |
| 仮説の検証そのもの（Montezuma・DropMerge の Run と分析） | 範囲外。本 PRD は道具までで、検証は実験記録に書く。制約ではなく、PRD の受入を実装とテストで閉じるための線引き |
| DropMerge 用の RND のプロファイル（`run.@rnd`） | 範囲外。DropMerge の観測は盤面とベクトルの組で、RND の Network の形を選ぶ必要がある。形の選択は検証の実験の設計に属する。再開条件は §4.1 |
| 仮説が当たったときの改善策（Train の一部の lane を評価と同じ方策にする、方策の組） | 範囲外。別の関心（173）。制約ではなく、検証の結果が出てから決める順序の選択 |
| 評価 session の step trace を Run の記録に残すこと | 範囲外。PRD184 が持つ。制約ではなく、記録の口を別の PRD が持つための線引きで、本 PRD の鍵は 184 でそのまま使える（§9） |
| Greedy の軌跡の Q と、その後の割引収益の比較 | 範囲外。Evaluation Trace のダンプと集計でできて、実装が要らない。制約ではない |

### 1.4 機構とゴールの対応

| | 機構 | すること | 効くゴール |
|---|---|---|---|
| A | 読むだけの採点 | 推定器が、統計も重みも変えずに観測を採点する口を持つ（D1・§5.1） | G1・G2・G3 |
| B | step 診断の内的報酬の鍵 | DefaultDQN の step 診断が 3 つの鍵を出す（D2・D5・§5.2） | G1 |
| C | 作るときの入力 | Evaluation Trace が、列の鍵と Runner の Actor を step 診断の部品を作るときに渡し、内的報酬の鍵があるときだけ採点する（D3・§5.3） | G2 |
| D | 推定器の写し | step 診断の部品が推定器と β の写しを持ち、Actor が network を写し直すのと同じロックの中で写し直す（D4・D20・§5.4） | G2・G3 |
| E | 列の設定 | `run.@rnd` が 3 列を宣言する（D12・§5.5） | G1・G4 |

先送りした機構は無い。

### 1.5 成功の測り方

- G1・G3: RND を有効にした Run で Evaluation Trace を開くと 3 列に値が入り、止まった画面の値を読める（§7 の 13）。評価専用の Run でも誤差と標準化した値が入る（§7 の 8）。
- G2: 採点を挟んだ学習と挟まない学習の結果が一致する（§7 の 1・10）。Trace を使わない Run の metrics checksum が実装の前後で一致する（§7 の 11）。
- G4: 上の 2 つを満たし、課題の「仮説の検証での使い方」の最初の項（止まった画面の誤差を Train でよく通る画面と比べる）を、Montezuma の `run.@rnd` の Run で人が行える。
- 費用は記録にとどめ、合否の閾値を置かない（D14）。

## 2. 現状（2026-11-12 にコードと文書で確認）

作業ツリーには並行する作業の変更があり、行が動きうる。実装時はシンボル名で探し直す。

### 2.1 学習の採点

- `DefaultDQNAgent::UpdateFromBatch` が、Agent の書き込みロックの中で、報酬スケーラのあとに推定器の `Estimate` を呼ぶ。採点する観測は `intrinsic.observation` で選ぶ（既定は遷移のあとの観測 `next_state`、`state` なら行動した観測）。ボーナスは β × (倍率 − 1) で、`next_state` を採点して `intrinsic.use_done_mask` が真のとき、done の lane は 0 にする。β は `intrinsic.beta`（ProfiledValue）を、学習の Runner が渡した exp step で評価する（[default_dqn_agent.cpp](../../src/algo/dqn/src/default_dqn_agent.cpp) の 713〜741 行付近）。
- 書き込みロックは、採点・補助学習・Replay への追加・Q の更新が終わるまで持ち続ける（同 711〜794 行付近）。
- 推定器を書き換えるのは、この入口と、checkpoint の読み込みだけである。読み込み（`LoadNetwork`）は Agent のコンストラクタからだけ呼ばれ、その時点に step 診断の部品はまだ無い。
- Taiga 式の Atari（`run.@rnd_taiga`。128 lane、`learn_batch_size = 1`、`learn_ratio = 1`）では、入口 1 回に採点 1 回と補助学習 128 回が入る（[Atari.txt](../../apps/runner/config/Atari.txt) の `run.@rnd_taiga` と `run.train.num_envs`）。

### 2.2 RND の採点が変えるもの

`RndNoveltyEstimator::Estimate`（[intrinsic_rnd.cpp](../../src/core/base/src/intrinsic_rnd.cpp)）は、呼ぶたびに次を更新する。

- 誤差の統計（whiten では warm-up の条件による。`variance_feedback` では lane の順に 1 件ずつ）。
- 観測の統計（whiten は採点のあとに更新する。`variance_feedback` は lane の順に 1 件ずつ更新しながら正規化する）。
- 採点の件数（warm-up の進み）、target の出力の統計、診断値。

whiten の採点は、今回の batch を入れる前の統計で倍率を作る。`variance_feedback` は、lane ごとに、観測をその lane を入れる前の統計で正規化し、誤差をその lane を入れたあとの統計で標準化する（`VarianceFeedbackStatistics::NormalizeAndUpdate`・`UpdateErrors`）。誤差の統計の状態は、件数・平均・分散の 3 つの数である。

どちらの方式にも、統計を変えずに観測を正規化する口がすでにある（`VarianceFeedbackStatistics::Normalize`、`ObservationNormalizer::Normalize`。補助学習が使う）。

### 2.3 Evaluation Trace と step 診断

- Evaluation Trace の行は `StepTraceBuilder::OnTrain`（[metrics.cpp](../../src/core/base/src/metrics.cpp)）が作る。step ごとに Agent の step 診断（`AgentStepDiagnosticsProvider::Observe`）を 1 回呼び、`$agent_step <鍵>` の列を `GetScalar(鍵, lane)` で読む。値は行を作るときに行へ持たせ、あとで行を選んでも計算し直さない。
- step 診断の部品は `Agent::CreateStepDiagnosticsProvider(num_lanes)` で作る（[rl.hpp](../../src/core/base/include/anet/rl.hpp)）。列に何が宣言されているかも、どの Actor の Runner かも受け取らない。DQN の部品（[dqn_based_agent.cpp](../../src/core/base/src/dqn_based_agent.cpp) の `DqnAgentStepDiagnosticsProvider`）は DefaultDQN と Rainbow が共用し、`q_sa`・`state_value`・`approx_priority` を必ず計算する。
- 知らない鍵は、最初の読み出しで列番号と鍵つきで止まる。この検査は 5 つの target（`$runner`・`$actor`・`$env_step`・`$action_info`・`$agent_step`）に共通の 1 か所にある（metrics.cpp の `OnTrain`）。既知で値が成立しない鍵は NaN で、表は空欄になる（[140 §6.1](../design/140_observability.jp.md)）。
- 行を作る部品は Runner の起動時に作る（[RunnerFrame.cpp](../../src/apps/runner/src/RunnerFrame.cpp) が `EvalTracePanel::Initialize` を呼ぶ）。行を作るのは Evaluation Trace の pane が開いている間だけである（[EvalTracePanel.cpp](../../src/apps/runner/src/EvalTracePanel.cpp) の `FunctionTrainObserver` の中の `IsPaneOpen`）。列の宣言は設定（`app.eval_trace_panel.columns.[番号]`）で、Run の間は変わらない。
- pane の開閉の切り替わりは EvalTracePanel が知っていて、切り替わるたびに行を消し、行を作る部品をやり直す（EvalTracePanel.cpp の `SetActive`）。EvalTracePanel は EvalPanel への参照を持つ。
- 行の画面は、行動の結果の画面である。Q の列は、その行動を決めた時点の値である（[020 §6.5](../design/020_user_guide_run.jp.md) の注）。
- 表の数値は有効数字 4 桁で出る（[gui.cpp](../../src/core/base/src/gui.cpp) の `FormatTraceValue`）。1e-4 台の値も読める。

### 2.4 学習と EvalPanel の関係、Actor の写し

- 学習の Runner と EvalPanel の Runner は別だが、Agent は 1 つで、Evaluation Trace は同じ Agent から step 診断の部品を作る（EvalTracePanel.cpp が `manager->GetAgent()` を渡す）。Runner は自分の Actor を公開している（rl.hpp の `Runner::GetActor`）。
- Actor ごとの設定 `actor.[鍵].clone_model` が、Actor が network の写しを持つかを決める。評価用の Actor（`actor.@eval_base`）の既定は true、学習用の Actor（`actor.[train]`）の既定は false である（[agent.txt](../../apps/runner/config/agent.txt)）。写しは Actor が持つ（[ADR 0013](../adr/0013-actor-network-resource-policy.md)）。
- 写しを持つ Actor は、同期のとき（`Actor::Sync`、および `sync_interval` による定期の写し直し）だけ、Agent の読み取りロックの中で network を写し直す（dqn_based_agent.cpp の `Actor::CopySourceNetwork`。skip の network も同じロックの中で写す）。最初の写しは、Actor を作るときに同じロックの中で作る（default_dqn_agent.cpp の `CreateActor`）。写しを持たない Actor は、forward のたびに Agent の読み取りロックを取る（同 `Actor::MakeAction`）。
- EvalPanel は、`app.eval_panel.model_sync.*`（既定は 10 秒ごと。[common.txt](../../apps/runner/config/common.txt)）で Actor を同期する。既定の設定では、最初の frame でも同期する。評価 session は開始のときに同期する。同期のとき、評価の Runner は学習側の step も控え、以後の行動選択にその step を渡す（[trainer.cpp](../../src/core/base/src/trainer.cpp) の `EvalRunner::Sync` と `DoStepInternal`）。γ などの時間で変わる値は、この「同期した時点の学習の step」で評価されている。表の `step` の列は EvalPanel 自身のカウンタである。
- EvalPanel の同期は GUI のタイマーから呼ばれる（[EvalPanel.cpp](../../src/apps/runner/src/EvalPanel.cpp) の `OnTimer` から `SyncModel`）。network を写すのは UI スレッドで、UI スレッドが Agent の読み取りロックを取る。
- 同じ Actor の `MakeAction` と `Sync` は、並行に呼ばない契約である（rl.hpp の `Actor`）。Evaluation Trace の行は、Runner の step の通知の中で同期に作る。そのため、Actor の同期、行動選択、step 診断の `Observe` は、その Runner を進める 1 つのスレッドで順に走る。
- EvalPanel の既定の進みは毎秒 15 step である（common.txt の `app.online.eval_panel.fps`・`app.batchrun.eval_panel.fps`）。
- 既存の step 診断は Agent のロックを待たない。報酬スケーラは、学習が使った標準偏差を値 1 つとして公開し、診断側はそれを読む（[scaler_impl.cpp](../../src/core/base/src/scaler_impl.cpp) の `RunningStdRewardScaler::ScaleWithoutUpdate`）。テスト `[diagnostic_no_wait]`（[dqn_based_agent_test.cpp](../../src/core/base/src/dqn_based_agent_test.cpp)）がこれを確かめている。[PRD094](done/0094_evalpanel_step_history_10rfd.md) は「step trace は UI スレッドで同期に作り、ロックを持たない」と書いている。
- 規約は、毎 step・毎 tick の経路に、別のスレッドのロック待ちと device 間の同期の転送を足さないこと、足すなら頻度と 1 回の時間の見積もりを書くことを求める（[impl.md](../quality/impl.md) の「計測」の規約）。[212 §5.1](../design/212_rnd_lifelong_novelty.jp.md) も「画面表示のために新しい学習ロック待ちを追加しない」と書いている。
- 規約は、UI スレッド（GUI のイベントとタイマー）から、学習のスレッドが更新の間ずっと持つロックを取らないこと、学習側の可変状態は、学習のスレッドで作った写しか、atomic に公開した値を通して読むことも求める（impl.md の「C++ コード」の規約）。PRD094 の実装で UI スレッドが学習のロックを待った反省から入った。上の EvalPanel の Actor の同期は、この規約より前からある動きで、UI スレッドから同じロックを取っている。

### 2.5 評価の値は可視化用にしかならない

- 評価の遷移は Replay に入らず、学習されない。評価の step には「実際に使った内的報酬」が、採点をどこに置いても無い。実際に使った値があるのは Train の遷移だけである。
- Actor の `MakeAction` は行動を選ぶ時点で呼ばれる。その時点では、既定で採点する遷移のあとの観測も、ボーナスに要る done も無い。終端の観測は `MakeAction` を通らない。[PRD116](0116_rnd_lifelong_novelty_10rfd.md) §1.3 が Actor での採点を退けた理由（1 step 遅れの採点を運ぶ契約が要る）と同じである。step 診断は、遷移の 4 つ組と done を受け取る。
- 方策が内的報酬を入力に取る形（NGU の「直前の内的報酬」。[173](0173_intrinsic_reward_novelty_10rfd.md) §4.5）では、評価の Actor も行動のために内的報酬を計算する。その値は Actor が出す値で、本 PRD の列とは別のものになる。

### 2.6 評価専用の Run

`learner.enabled=false` でも、`intrinsic.enabled=true` なら推定器は作られ、checkpoint から predictor と統計を読む。採点と補助学習は休止する（default_dqn_agent.cpp の構築と `LoadNetwork`）。学習の入口は呼ばれないので、β は評価されない。

### 2.7 forward が変えるもの

- スペクトル正規化の内部状態は、学習モードで勾配が有効な forward でだけ更新する（[nn_modules.cpp](../../src/core/base/src/nn_modules.cpp) の `MakeEffectiveWeight`）。採点は勾配なしで forward するので、状態は動かない。
- predictor は学習モードのままである。dropout・droppath は、学習モードなら勾配なしでも乱数を引く。BatchNorm（同ファイルの `BatchNorm2dModule`）は、学習モードなら勾配なしでも内部の統計を書き換える。RND の Network にこれらを入れた構成では、学習の採点もそう動いている。出荷設定の RND の Network（Atari の Conv の構成、`RndMlp`）には無い。

### 2.8 設定

- `run.@rnd` を持つのは [Atari.txt](../../apps/runner/config/Atari.txt)・[GridMaze.txt](../../apps/runner/config/GridMaze.txt)・[LunarLander.txt](../../apps/runner/config/LunarLander.txt) である。DropMerge には無い。
- Evaluation Trace の既定の列は [common.txt](../../apps/runner/config/common.txt) にあり、プロファイルが列を足す形の前例は `run.@skip` である（Atari.txt ほか）。列番号は 600〜999 が空きである（[020 §6.5](../design/020_user_guide_run.jp.md) の番号の表）。
- Rainbow は内的報酬の設定を持たない。

## 3. 用語

- **読むだけの採点**: 推定器の重み・統計・件数・診断値を変えずに、その推定器が持つ predictor と統計で観測を採点すること。公開の口は `EstimateWithoutUpdate`。名前は、同じ役割の `RewardScaler::ScaleWithoutUpdate` の形にそろえた。
- **推定器の写し**: 学習側の推定器と同じ状態を持つ、別の推定器。写しを採点しても、学習側の推定器は変わらない。step 診断の部品が持ち、Actor が network を写し直すときに一緒に写し直す（D4）。Actor が持つ network の写しと同じ考え方である。
- **同期に従うもの**: Actor が source から写し直すとき、source を守るロックの中で一緒に写し直すもの。公開の口は `ActorSyncFollower`（§6.1）。推定器と β の写しがこれに当たる。既存の語では言い換えられない。Observer は、`Notifier` に付けて Runner の出来事（step や episode の終わり）の通知を受ける部品で、通知は source を守るロックの外で届く。同期に従うものは、Actor が写し直す時機に、Actor がロックを持ったまま呼ぶことが契約である。用語集の「Train Actor network snapshot」は、Actor が持つ network の時点コピーそのものを指し、Actor の外の持ち主が持つ写しも、それを写す時機に呼ばれる相手も指さない。
- **誤差・倍率・ボーナス**: [212 §1.2・§3.3](../design/212_rnd_lifelong_novelty.jp.md) の語のまま使う。誤差は predictor と target の出力の差（予測誤差）、倍率は 1 を基準とする採点の結果、ボーナスは β × (倍率 − 1) である。
- **標準化した値**: 倍率を作るときの「倍率 − 1」を、上下で切る前の値。212 が式の中で記号 z と書いている値である。鍵の名前は `normalized_error` にする（D5）。

用語集（`CONTEXT.md`）は変えない。RND の語は用語集に無く、212 の用語の節が持っている。本 PRD が足す値の名前は、どれも 212 の語と既存の鍵（`intrinsic.error_mean` など）の言い換えで足りる。「同期に従うもの」は、上のとおり既存の語で言い換えられないので、新しい呼び名として決めた。Actor の実装の口の呼び名で、分野の語ではないので、用語集には入れず、設計文書 200 に書く。

## 4. 確定事項

2026-11-12 のグリルで決めた。D1〜D4 と D10 は問いへの答え、D5〜D9・D11〜D14 は推奨の既定を採った細部、D15〜D18 は規約・現行コード・ほかの決定から決まるもの、D19 はユーザーの要望である。D4・D8・D10・D11・D14 は、別モデルの検討で出た論点を受けて、同じ日に決め直すか書き直した。D4 は、そのあとのレビューで実装の規約（UI スレッドのロック。§2.4）との衝突が分かり、問いを立て直して決め直した。D20 は、D4 を決め直したあとの細部である。1 は起草側の既定を開発者に示して変える指定が無かったもの、2 と 3 は開発者の答えである。

### D1. 学習用の採点は動かさず、Trace の値は可視化用に別に計算する

学習用の採点は、Agent の学習の入口のままにする。Evaluation Trace の値は、同じ predictor と統計を読むだけの採点で、可視化用に別に計算する。学習用の採点を Actor 側へ移すか、方策が内的報酬を入力に取るかは、173 の問いとして開いたままにする。

理由: 評価の値が可視化用になるのは、採点の場所の選び方ではなく構造から決まる（§2.5）。今回の目的（Greedy が止まった画面を、Train で育った predictor がどれだけ見慣れていないか）は、学習側から見た値を測ることなので、可視化用の値で目的に合う。Actor で計算しない形は、評価 session を含むすべての Actor の step に forward を足さずに済む。課題の依存関係では、173 の課題が本課題を待っている。依存: §2.5。

### D2. 出す値は誤差・標準化した値・ボーナスの 3 つ。ボーナスの β は、学習の入口が直近に使った値

ボーナスは「学習側がこの遷移を受け取ったら報酬に足す値」とし、学習と同じ式（β、上下の切り方、done の lane の扱い、warm-up）で作る。β は、学習の入口が直近に評価した値を Agent が覚え、推定器と同じ時機に写す（D4）。学習の入口をまだ通っていない間は、ボーナスだけが空欄になる。

理由: RND の調整で繰り返し問題になったのは、内的報酬が外的報酬に比べてどれだけの大きさか、だった（[PRD123](0123_rnd_intrinsic_reward_scale_10rfd.md) §2.1）。学習の報酬と同じ単位の列があれば、止まった画面でその大きさを直接読める。β を学習の入口の値にすると、β を schedule にしても学習と同じ値になる。成立しないときは空欄にし、別の値で埋めない（[impl.md](../quality/impl.md) の「GetScalar」の規約）。依存: D1・D4、§2.4（学習と EvalPanel は同じ Agent を持つ）。

### D3. 列の鍵と Runner の Actor は、step 診断の部品を作るときに渡す

Evaluation Trace の行を作る部品が、列の宣言から `$agent_step` の鍵の一覧を作り、その Runner の Actor とあわせて、`Agent::CreateStepDiagnosticsProvider` の入力として渡す。DefaultDQN の step 診断は、一覧に内的報酬の鍵があるときだけ、呼ばれた時点で読むだけの採点を行う。

理由: 計算するかどうかを、部品を作る 1 か所で決められる。値を返す部品は、今と同じく値を持つだけになる。読み出しが forward を隠し持つ形は、読み手が増えたとき（PRD184）に計算の時点のずれを生みやすい。「何が要るかを、進める側が引数で渡し、core に登録を持たない」という考え方は PRD170（[ADR 0072](../adr/0072-make-action-inputs-are-passed-per-call-through-runner-step-not-registered.md)）と同じである。列の宣言は Run の間変わらないので、渡すのは作るときの 1 回で足りる（§2.3）。Actor を渡すのは、D4 が Actor の設定と同期に従うためである。D4 は、Actor に「同期に従うもの」をつなぐ。これは登録の形だが、ADR 0072 が退けた登録（step の結果を、引数に現れない値で変えるもの）には当たらない。つないでも、行動選択の出力も step の結果も変わらない。つなぐのは Agent で、呼び出す側（Evaluation Trace）は入力を渡すだけである。依存: D4、§2.3。

### D4. 推定器の写しは、Actor が network を写し直すのと同じロックの中で写す

step 診断の部品は、その Runner の Actor の設定に従って、次のどちらかで採点する。表は `clone_model` の値で 1 行に決まる。

| Actor の `clone_model` | 採点の相手 | 推定器と β を写す・読む時機 | step 診断が Agent の読み取りロックを取るか |
|---|---|---|---|
| true（評価用の Actor の既定） | 部品が持つ推定器の写し | 部品を使っている間（D20）に、Actor が network を写し直すとき。Actor が取っている読み取りロックの中で、network に続けて推定器と β を写す | 取らない |
| false | 学習側の推定器そのもの | step ごと。β もその場で読む | step ごとに取る |

- 「Actor が network を写し直すとき」は、`Actor::Sync` と、`sync_interval` による定期の写し直しの両方である（§2.4）。
- 写し始める時機と、Evaluation Trace の pane の開閉との関係は、D20 による。
- 学習の入口に足すのは、評価した β を覚えることだけである（D2）。採点、補助学習、ロックの取り方は変えない。新しいロックも、新しい設定キーも作らない。

この形は、実装の規約「UI スレッドから、学習のスレッドが更新の間ずっと持つロックを取らない。学習側の可変状態は、学習のスレッドで作った写しか、atomic に公開した値を通して読む」（§2.4）の例外である。EvalPanel では、UI スレッドが、Agent の読み取りロックの中で学習側の推定器を読む。例外とすることは、開発者がグリルで決めた。

理由:

- 既存の Actor の同期が、すでに同じロックを UI スレッドから取っている（§2.4）。同じロックの中で一緒に写せば、UI スレッドがロックを取る箇所は増えない。回数が増えるのは、pane を開いて写しが無かったときの同期 1 回だけである（D20）。ロックを持つ時間が、推定器を写す分（数ミリ秒の見込み。§5.4）だけ延びる。
- network・推定器・β を同じロックの中で写すので、同じ行の Q・γ と内的報酬の値の時点が、仕組みの上でそろう。評価側は「同期した時点の学習側」を見る、という既存の作りにそのまま乗る。
- 足す分だけを規約どおりにする形（学習のスレッドが写しを作って公開する。§4.1）にしても、network の同期の待ちは残る。そのうえ、写しが 2 つの仕組みに分かれ、値の時点がずれ、学習の入口が走っていない間は写しが届かない。
- 写しを持たない Actor は、forward がすでに step ごとに読み取りロックを待っているので、同じ待ち方を足しても、その設定を選んだ利用者の想定を外れない。
- 推定器を書き換えるのは Agent の書き込みロックの中の学習の入口だけなので、読み取りロックの中で写せば足りる（§2.1）。

依存: §2.1・§2.4。

帰結:

- 行の値の時点は、写しを持つ Actor では「Actor が最後に network を写し直した時点の、学習側の predictor・統計・β」で、同じ行の Q を出した network と同じ時点である。持たない Actor では「その step の瞬間」である。行は、それが学習の何 step 目の状態だったかを持たない。写し直しをまたいだ行どうしでは、採点した predictor の学習の進みが違う。
- [PRD094](done/0094_evalpanel_step_history_10rfd.md) の「step trace はロックを持たない」は、写しを持つ Actor では、内的報酬の列を宣言してもそのまま成り立つ。step ごとに Agent の読み取りロックを取るのは、写しを持たない Actor で内的報酬の列を宣言した構成だけである。
- PRD184 の評価 session では、session の開始の同期 1 回で、session の全体を同じ predictor で採点することになる。
- 変えるときの条件: Actor の network の同期を、規約どおりの形（学習のスレッドが写しを作って公開する）へ直すときは、推定器と β の写しも一緒に移す。既存の同期が規約に合っていないことは、別の課題にする（§8）。

### D5. 鍵の名前

`$agent_step` の鍵は `intrinsic.error`（誤差）、`intrinsic.normalized_error`（標準化した値）、`intrinsic.bonus`（ボーナス）とする。

理由: 既存の `$agent` の鍵（`intrinsic.error_mean`・`intrinsic.bonus_mean`）と、設定の `intrinsic.rnd.error_norm.*` に語をそろえる。記号の z は鍵にしない（[impl.md](../quality/impl.md) の「名前」の規約の 8）。`error` と `normalized_error` は RND の推定器が名乗る鍵で、`bonus` は DefaultDQNAgent が作る（§6.1）。

### D6. 標準化した値の定義と、空欄になる条件

標準化した値は、学習が倍率を作るときの「倍率 − 1」を、上下で切る前の値とし、統計の取り方の設定（`statistics_mode`、`error_norm.use_centering`）に従う。誤差の統計が空のときと、除数が 0 のときは NaN にする。warm-up 中でも統計があれば出す。

理由: 切ったあとの値はボーナスの列から読める。値が成立しないときは NaN にする規約による。依存: D2。

### D7. ボーナスが 0 になる条件と、空欄になる条件

ボーナスは、学習が倍率を 1 にする条件（warm-up 中、統計が空、除数が 0）と、done の lane（`next_state` を採点し、`intrinsic.use_done_mask` が真のとき）で 0 になる。学習の入口をまだ通っていない Run（起動直後、評価専用、checkpoint から再開した直後）では NaN にする。β は checkpoint に保存しない。

理由: 0 は学習が実際に足す値と同じで、偽りではない。β が無いのは値が成立しない状態なので NaN にする。β を保存しないのは、形式を変えられないからではない（クリーンブレークで変えられる）。再開した Run は最初の学習の入口で β を評価し直すので、保存して埋まるのは、再開から最初の入口までの間のボーナスの列だけである。そのために checkpoint の書式の識別子を上げ、今ある checkpoint を読めなくする不利益に見合わない。依存: D2。

### D8. Taiga 式（`variance_feedback`）も、学習に入れたときと同じ値にする

読むだけの採点は、誤差の統計の 3 つの数を手元に写し、その観測を「次の 1 件」として入れてから標準化する。推定器の統計は変えない。lane が複数あるときは、lane ごとに独立に「次の 1 件」として計算する。

理由: 学習は、その誤差を統計に入れてから標準化する（§2.2）。手元の写しで同じ計算をすれば、統計を変えずに、その観測を次に学習へ入れたときの値と一致する。ボーナスの意味（D2）と合う。依存: D2、§2.2。

### D9. 部品を作るときの入力に既定値を付けない

`CreateStepDiagnosticsProvider` の入力（lane の数、鍵の一覧、Actor）は、呼び出しの全部が書く。

理由: 呼び出しは本体 1 か所とテスト数か所で、書き換えが少ない。PRD170 が `MakeAction` の引数に既定値を付けたのは、テストの呼び出しが約 65 か所あるためで、事情が違う。依存: D3。

### D10. 知らない鍵で止まる時機は変えない。出荷の列はテストで守る

知らない `$agent_step` の鍵は、今と同じく最初の読み出しで止まる。部品を作る時点の検査へは広げない。出荷の `run.@rnd` の列は、その宣言で実際に行を作るテストで守る（§7 の 12）。

理由: 知らない鍵の検査は 5 つの target に共通の 1 か所にあり、`$env_step` と `$action_info` は、実際の step の結果が無いと鍵を知っているか答えられない（§2.3）。`$agent_step` だけを早めると、止まる時機が target ごとに分かれる。内的報酬の鍵を書くのは出荷の `run.@rnd` なので、テストで綴り違いを止められる。Run の途中で pane を開くと列の誤りで Run が止まる問題は全 target に共通で、別の課題にする（§8）。依存: D3、§2.3。

### D11. forward で乱数を引く層・状態を変える層は検出しない

RND の Network に、forward で乱数を引く層（dropout・droppath）や内部の統計を書き換える層（BatchNorm）を入れた構成は、検出も WARN もしない。設計文書 212 に制約として書く。

- 写しを持つ Actor（既定）: BatchNorm の統計の書き換えは写しの中だけで起きる。一方、dropout・droppath の乱数は共有の乱数列を進めるため、推定器を写しても学習が変わりうる。これらの乱数を引く層を持つ構成は G2 の保証に含めない。値は学習の採点と一致しないことがある。
- 写しを持たない Actor: 学習側の推定器を直接通すので、乱数の列や BatchNorm の統計が変わり、学習が変わる。

理由: 学習の採点も同じ forward で、今も検出していない（§2.7）。出荷設定には無い。G2 は、写しを持つ Actor では乱数を引く層を持たない構成、写しを持たない Actor では乱数を引く層も内部の統計を更新する層も持たない構成で成り立つ。乱数を分離する設計は今回に含めず、共有乱数の状態の保存・復元も足さない。依存: D4、§2.7。

### D12. 列を足すプロファイルと、番号・見出し

Atari・GridMaze・LunarLander の `run.@rnd` に、235 `rnd_bonus`、600 `rnd_err`、610 `rnd_norm` の 3 列を足す（§5.5）。`rnd_bonus` は230 `reward`の直後へ置き、600 番台を「内的報酬の誤差」の分類にする。RND を選ばない Run の列は変えない。

理由: RND を持つ env の全部に入れる。ボーナスを環境の報酬と並べて比べられる配置を、開発者が選んだ。240番はTempoRLの`decision`なので、既存列を動かさず直前へ差し込むため、番号の一の位を空ける原則の例外として235番を使う。プロファイルが列を足す形は `run.@skip` と同じであり、両方の順序で重ねても互いの列を上書きしない。誤差2列は空いている600番台を使う（§2.8）。既定に入れたままにすることは、§7 の 14 の費用の測定のあとに開発者が決めた（冒頭の 2026-11-14 の改訂）。依存: §2.8。

### D13. 百分位の列は既定で足さない

`display:pct` の列は既定では足さず、足し方をユーザーガイドに書く。

理由: 百分位は評価の行の中での順位で、Train との比較は標準化した値が担う。列も増えすぎる。

### D14. 費用は測って記録し、合否の閾値を置かない

Evaluation Trace を閉じた状態と、開いて 3 列を出した状態の、学習の速さ、step 診断 1 回の時間、Actor の同期のうち推定器と β を写す時間を、実装後に測って設計文書 212 に書く。構成は Atari の `run.@rnd` と、Taiga 式（`run.@rnd_taiga`）の 2 つである（§7 の 14）。

理由: 採点の費用と、推定器を写す費用がかかるのは、pane を開いて列を出している間だけである（D20 の帰結）。課題でも未測定で、閾値を置く材料が無い。§5.4 の見積もりを、この測定で確かめる。Taiga 式は学習の入口が長く（§2.1）、Actor の同期の待ちが最も大きくなる構成である。依存: D4。

### D15. 採点する観測は `intrinsic.observation` に従う

既定（`next_state`）は遷移のあとの観測を、`state` は行動した観測を採点する。

理由: 学習と同じ観測を採点しないと、ボーナスが「学習が足す値」にならない（D2）。値がその遷移のどの時点に属するかは設定で決まるので、[ADR 0057](../adr/0057-eval-trace-columns-declared-apart-from-metrics-and-read-at-transition-time.md) の「時点が決まっている値だけを読む」に合う。依存: D2、§2.1。

帰結: 既定では、3 列は行に表示している画面（行動の結果の画面）の値になる。`state`（Taiga 式）では、行動を決めた画面の値になり、表示している画面より 1 つ前である（§2.3）。設計文書 212 とユーザーガイドに書く。

### D16. RND が無効のときと、Rainbow

DefaultDQN で `intrinsic.enabled=false` のとき、3 つの鍵は認識して NaN を返す。Rainbow は内的報酬の鍵を知らないので、列に書くと既存の規則どおり最初の読み出しで止まる。

理由: 設定で無効にした機能は鍵を保って NaN を返す（[impl.md](../quality/impl.md) の「GetScalar」の規約。`$agent intrinsic.*` も同じ扱いをしている）。Rainbow は内的報酬の設定そのものを持たないので、鍵は契約に無い。依存: §2.3・§2.8。

### D17. 値は行を作るときに 1 回計算する。checkpoint の形式は変えない

値は、行を作るときに 1 回計算して行に持たせる。過去の行を選んでも計算し直さない。checkpoint に足す状態は無い。

理由: 既存の列と同じ作りである（§2.3）。保存が要る状態が無い。推定器の写しは、学習側の推定器からいつでも作り直せる。β は、学習の入口が評価し直す（保存しない判断の理由は D7）。

### D18. ADR は起こさない

理由: 足すだけの診断で、あとから変えにくい形を含まない。部品を作るときの入力を足す D3 は、PRD170 の ADR 0072 と同じ考え方に沿う。写しの扱い（D4）は実装の規約の例外だが、戻しにくい形ではない。写す場所を学習側の公開へ移しても、鍵、値の意味、設定は変わらない。例外にした理由と変えるときの条件は、D4 と wiki のトピック別まとめに残す。推定器の写しの持ち主は step 診断の部品で、Actor は写し直す時機に呼ぶだけである。Actor が forward に使う写し（ADR 0013 が Actor の所有と決めたもの）とは別で、ADR 0013 は変えない。採点の場所の判断（D1）は 173 に開いたまま残している。

### D19. 流れをシーケンス図で設計文書に書く

実装のとき、設計文書 140 の処理フローの章に、Evaluation Trace の行を作る流れ（pane を開いたときの同期、Actor の同期での network・推定器・β の写し直し、step 診断の呼び出し、読むだけの採点、行への書き込み、pane を閉じたときの休ませ方まで）のシーケンス図を足す。

理由: 学習のスレッド、GUI のスレッド、Agent、Actor、推定器にまたがる流れで、文だけでは追いにくい（ユーザーの要望）。

### D20. 推定器を写し始める時機と、pane の開閉

1. 部品を作る時点では写さない。部品に写しが無い間、3 つの値は NaN である（§5.2 の表の行 3）。
2. Evaluation Trace の pane を閉じている間は、推定器と β を写さない。step 診断の部品は「使っている」「休んでいる」の 2 つの状態を持つ。作った直後は「使っている」である。Evaluation Trace が、pane を閉じたときに休ませ、開いたときに使い始める。休んでいる間に Actor が network を写し直したら、部品は持っていた写しを捨てる（network と時点が合わなくなるため）。
3. pane を開いたとき、部品に写しが無ければ、EvalPanel がその場で Actor を同期する。この同期は EvalPanel の通常の同期と同じ処理で、network・推定器・β を同じロックの中で写す（D4）。写しが残っていれば（閉じている間に Actor が写し直していない）、同期しない。

1 は、起草側が置いた既定を開発者に示し、変える指定が無かったものである。2 と 3 は開発者の答えである（「閉じている間は写さない。開いたとき、同期が古ければすぐに同期する」）。「同期が古い」は、部品の写しが Actor の今の network と同じ時点のものでないこと（写しが無いこと）と読んだ。

理由:

- 1: 部品を作る時点で写すと、UI スレッドがロックを取る箇所が 1 つ増え、Actor の network の最初の写し（Actor を作ったとき）とは時点が別になる。
- 2: 列を宣言しただけで見ていない Run に、推定器を写す費用をかけない。
- 3: 閉じている間に写さないと、開いた直後の行が、次の定期の同期（既定で最大 10 秒）まで空欄になる。その場で同期すれば、最初の行から値が出る。推定器だけを写さずに Actor ごと同期するのは、同じ行の Q と値の時点をそろえるためである（D4）。
- 状態の切り替えは、行動選択の出力も step の結果も変えない。開いたときの同期は、EvalPanel の通常の同期を前倒しするもので、同期の時機は今も実時間で決まっている。

依存: D4、§2.3（pane の開閉は EvalTracePanel が知っている）、§2.4。

帰結:

- UI スレッドが Agent の読み取りロックを取る回数は、pane を開いて写しが無かったときに 1 回増える。取る箇所は、既存の EvalPanel の同期と同じである。
- 同期を episode の単位にした設定（`app.eval_panel.model_sync.mode`）でも、pane を開いたときの同期は、episode の途中で起きる。
- 「写しが無い」の NaN が出るのは、部品を作ってから Actor がまだ写し直していない間と、休んでいる間に写しを捨てたあと、Actor がまだ写し直していない間である。RunnerApp では、pane を開いたときの同期があるので、ほぼ出ない。Runner を直接使う構成（テスト、PRD184 の評価 session）では、最初の同期までの間に出る。
- RunnerApp は、起動時に pane が閉じていれば、行を作る部品を作った直後に休ませる。開閉の判定は、行を作るかどうかを決めている EvalTracePanel の既存の判定に従う。
- 採点の費用と、推定器を写す費用がかかるのは、pane を開いている間だけである（D14 で測る）。

### 4.1 選ばなかった形

| 形 | 裁定 | 理由 |
|---|---|---|
| PRD170 の `MakeActionOptions` に欄を足し、Actor に採点させる | 不採用 | `MakeAction` の時点では遷移のあとの観測も done も無く、終端の観測は通らない。Actor は推定器を持たない。RND の採点は Q の forward の副産物ではなく、capture で運べない（§2.5） |
| ボーナスの β を、Actor が行動のたびに評価する（γ と同じ扱い） | 不採用 | Actor に内的報酬の設定を持たせることになり、D1 に反する。写しと一緒に β を写せば、同じ時点の値になる（D2・D4） |
| 誤差と標準化した値の 2 つだけを出す | 不採用 | 学習の報酬の単位で読むには、利用者が β と切り方を設定から当てはめる必要がある |
| 内的報酬の鍵を読まれたときに計算する（課題に書いた形） | 不採用 | 値を返すだけに見える読み出しが forward を通す。読む時機で計算の時点がずれる（D3） |
| RND が有効なら常に採点する | 不採用 | PRD184 で `q_sa` だけを記録する宣言をしたとき、評価の全 lane・全 step で使われない採点が走る |
| step 診断を呼ぶたびに鍵を渡す | 不採用 | 列の宣言は Run の間変わらない（§2.3） |
| Actor の設定に依らず、Agent の読み取りロックを step ごとに取る | 不採用 | 写しを持つ Actor の Runner でも、Trace の step のたびに GUI のスレッドが Q の更新の終わりを待つ。規約と 212 の記述に反する（§2.4） |
| 内的報酬だけのロックを作り、学習の採点・補助学習と Trace の採点を排他する | 不採用 | 学習の入口にロックが入る。Taiga 式では入口 1 回に補助学習が 128 回あり（§2.1）、Trace の step が採点か補助学習 1 回ぶんを毎回待ちうる。BatchNorm を入れた構成では学習が変わる |
| Trace 側が、Actor の設定に依らず自前の間隔で写し直す | 不採用 | 間隔の設定キーが増え、同じ行の Q と内的報酬の値の時点が別になる |
| step 診断の部品が、Actor の写し直しを知ったあとの最初の採点の前に、自分で Agent の読み取りロックを取って写す | 不採用 | 既存の Actor の同期とは別にロックを取るので、UI スレッドの待ちが続けてもう 1 回起きうる。写す時点が network の写しより遅れ、同じ行の Q と値の時点がずれる場合がある |
| 学習のスレッドが、求めに応じて推定器と β の写しを作って公開し、step 診断は Agent のロックを取らない（足す分だけを規約どおりにする） | 不採用 | 既存の network の同期が同じロックを UI スレッドから取り続けるので、UI スレッドの待ちは無くならない。写しが、UI スレッドが取りに行く network と、学習のスレッドが渡す推定器の 2 つの仕組みに分かれ、値の時点がずれる。学習の入口が走っていない間（学習が終わったあとなど）は写しが届かない。写しを持たない Actor では値を出せない |
| Actor の network の同期ごと、学習のスレッドが写しを作って公開する形へ直す | 範囲外 | 規約どおりの形だが、Actor の同期の作りを変える別の関心である。制約ではなく、順序の選択である。再開条件: 既存の同期が規約に合っていない課題（§8）に取り組むとき。そのとき、推定器と β の写しも一緒に移す（D4） |
| 部品を作るときに、Agent の読み取りロックを取って最初の写しを作る | 不採用 | UI スレッドがロックを取る箇所が 1 つ増える。Actor の network の最初の写し（Actor を作ったとき）とは時点が別になる。RunnerApp では、pane を開いたときの同期があるので、空欄の行はほぼ出ない（D20） |
| pane の開閉に依らず、Actor が写し直すたびに推定器を写す | 不採用 | 列を宣言しただけで見ていない Run にも、Actor の同期のたびに写す費用がかかる。開いた直後の空欄は、開いたときの同期で埋められる（D20。開発者の判断） |
| pane を開いたとき、Actor は同期せず、推定器と β だけをその場で写す | 不採用 | 既存の同期とは別に、UI スレッドがロックを取る箇所が増える。写す時点が network の写しと別になり、同じ行の Q と値の時点がずれる（D4） |
| pane を開いたら、写しが残っていても必ず同期する | 不採用 | 閉じている間に Actor が写し直していなければ、写しは network と同じ時点のままで、同期は要らない（D20。開発者の判断） |
| pane を閉じたら行を作る部品ごと壊し、開いたら作り直す（部品に状態を持たせない） | 不採用 | 列の宣言の読み取りと、部品を作るときの契約違反の検出（§6.2 の行 3）が、起動時から pane を開いた時点へ遅れる。写しが閉じるたびに失われ、開くたびに必ず同期になる |
| 評価専用の Run でだけ出し、学習中の Run では空欄にする | 不採用 | ロックも写しも要らないが、学習中の Run を見ながら止まった画面を調べる使い方ができない。G3 を縮める |
| 知らない `$agent_step` の鍵を、部品を作るときに検査して起動時に止める | 不採用 | D10。止まる時機が target ごとに分かれ、エラーの文に列番号を入れるには共通の口が 1 つ増える |
| DropMerge の `run.@rnd` を足す | 範囲外 | §1.3。再開条件: DropMerge で仮説を確かめる実験を起こすとき |
| 評価 session の新しさを scalar で残す | 範囲外 | 別の関心。step ごとの記録は PRD184 が担う。再開条件: 184 の記録だけでは Greedy と Train の誤差の比較が重いと分かったとき |
| β を checkpoint に保存する | 不採用 | 本当の制約ではない（形式は変えられる）。保存して埋まるのは、再開から最初の学習の入口までの間のボーナスの列だけで、書式の識別子を上げて今ある checkpoint を読めなくする不利益に見合わない（D7） |
| 乱数を引く層・状態を変える層を持つ RND の Network を検出して WARN する | 不採用 | D11 |

## 5. 仕様

### 5.1 読むだけの採点

推定器は、渡された観測について、次を lane ごとに返す。重み・統計・件数・診断値は変えない。forward 自体が乱数を引く層や内部の統計を更新する層を持つ場合の制約は D11 に従う。

1. 観測を、今の観測の統計で正規化する（統計は更新しない）。
2. target と predictor に通し、誤差を作る。勾配は使わない。
3. 標準化した値を、今の誤差の統計で作る。
4. 倍率を、学習の `Estimate` と同じ式で作る。

RND の標準化した値は、次の表を上から順に当て、最初に当たった行で決める。

| # | 条件 | 標準化した値 |
|---|---|---|
| 1 | 誤差が有限でない | NaN |
| 2 | whiten で、誤差の統計が空（誤差を 1 件も入れていない） | NaN |
| 3 | `variance_feedback` で、観測の統計に分散 0 の成分がある | NaN |
| 4 | 除数が 0 以下（whiten は誤差の標準偏差。`variance_feedback` は、行 7 の「入れたあと」の誤差の分散） | NaN |
| 5 | whiten で `error_norm.use_centering` が真 | (誤差 − 誤差の平均) ÷ 誤差の標準偏差 |
| 6 | whiten で `error_norm.use_centering` が偽 | 誤差 ÷ 誤差の標準偏差 |
| 7 | `variance_feedback` | 誤差の統計の写しに、その誤差を次の 1 件として入れたあとの、(誤差 − 誤差の平均) ÷ 誤差の分散（D8） |

倍率は、次の表を上から順に当てる。

| # | 条件 | 倍率 |
|---|---|---|
| 1 | 採点の件数が `warmup_steps` より少ない | 1 |
| 2 | 標準化した値が NaN | 1 |
| 3 | `multiplier_mode = clipped` | 1 ＋ 標準化した値を、1 以上 `multiplier_max` 以下に切った値 |
| 4 | `multiplier_mode = unclipped` | 1 ＋ 標準化した値 |

- 読むだけの採点の倍率は、直後にその観測 1 件を学習の `Estimate` へ渡したときの倍率と一致する。whiten は batch を入れる前の統計で採点し、`variance_feedback` は D8 の計算をするためである。複数の lane をまとめて `Estimate` へ渡した場合、`variance_feedback` で一致するのは先頭の lane だけである（2 つ目以降は、前の lane を入れたあとの統計で採点される）。
- `variance_feedback` の標準化した値は分散で割るので、whiten の値と単位が違う。方式の違う Run どうしで `rnd_norm` の列は比べられない。
- 誤差が有限でない lane は、エラーにせず NaN で返す。学習の `Estimate` が非有限の誤差で止まる動きは変えない。

### 5.2 step 診断が出す値

DefaultDQN の step 診断は、鍵の一覧に `intrinsic.` で始まる鍵が 1 つでもあるとき、step ごとに 1 回、全 lane をまとめて読むだけの採点にかける。採点する観測は D15、採点の相手は D4 による。写しを持つ Actor で、写しがまだ無い間は採点しない（表の行 3）。

3 つの鍵の値は、次の表を上から順に当て、最初に当たった行で決める。

| # | 条件 | `intrinsic.error` | `intrinsic.normalized_error` | `intrinsic.bonus` |
|---|---|---|---|---|
| 1 | Agent が Rainbow | 知らない鍵（最初の読み出しで止まる） | 同左 | 同左 |
| 2 | `intrinsic.enabled=false` | NaN | NaN | NaN |
| 3 | 写しを持つ Actor で、部品に写しが無い（部品を作ってから、Actor がまだ network を写し直していない。または、休んでいる間に写しを捨てたあと、Actor がまだ写し直していない。D20） | NaN | NaN | NaN |
| 4 | 誤差が有限でない lane | NaN | NaN | NaN |
| 5 | β が無い（β を写した時点で、学習の入口がまだ 1 回も通っていない。写しを持たない Actor では、読んだ時点。起動直後、評価専用の Run、checkpoint から再開した直後） | 誤差 | §5.1 の標準化した値 | NaN |
| 6 | `next_state` を採点し、`intrinsic.use_done_mask` が真で、done の lane | 誤差 | §5.1 の標準化した値 | 0 |
| 7 | それ以外 | 誤差 | §5.1 の標準化した値 | β × (§5.1 の倍率 − 1) |

- β は、学習の入口が直近に評価した値である。学習の入口は、評価するたびに覚える値を置き換える。β を写す・読む時機は、推定器と同じである（D4 の表）。
- 行 4 に当たる lane があったら、Run に 1 回 WARN を出す（§6.2）。
- 鍵の一覧に `intrinsic.` で始まる鍵が無いときは、採点せず、推定器の写しも作らず、Actor にもつながず、Agent のロックも取らない。この一覧に無い鍵を読まれたときも、既知の鍵なら NaN を返す（一覧は計算するかどうかだけを決め、鍵の認識は変えない）。
- `intrinsic.` で始まる鍵のうち上の 3 つ以外は、DefaultDQN でも知らない鍵である。

### 5.3 部品を作るときの入力

- Evaluation Trace の行を作る部品は、列の宣言のうち target が `$agent_step` の列の鍵を、重複を除いて集める。
- 行を作る部品は、lane の数、鍵の一覧、その Runner の Actor を、`CreateStepDiagnosticsProvider` へ渡す。
- 入力は、部品を作ったあとは変わらない。
- 既存の 3 つの値（`q_sa`・`state_value`・`approx_priority`）は、入力に依らず今と同じく計算する。

### 5.4 推定器の写しと、待ちの見積もり

写しを持つ Actor（`clone_model = true`）の Runner では、推定器と β の写しを次のとおり扱う。

| 時機 | すること | Agent の読み取りロック |
|---|---|---|
| 内的報酬の鍵を入力に持つ部品を作るとき | 写しの入れ物を、同期に従うものとして、その Actor につなぐ。この時点では写さない。部品は「使っている」状態で始まる（D20） | 取らない |
| 使っている間に、Actor が network を写し直すとき（`Actor::Sync`、`sync_interval` による写し直し） | Actor が、network を写したのと同じロックの中で、続けて学習側の推定器と β を写す | Actor がすでに取っているものを使う。取り直さない |
| 休んでいる間に、Actor が network を写し直すとき | 写さない。持っていた写しを捨てる | 学習側を読まない |
| 休ませるとき（`Suspend`） | 状態を変えるだけである。写しは持ったままにする | 取らない |
| 使い始めるとき（`Resume`） | 状態を変え、写しが無ければ「Actor の同期が要る」と答える。自分では写さない | 取らない |
| step 診断の採点 | 写しがあれば、写しを採点する。無ければ採点せず、3 つの値を NaN にする | 取らない |

RunnerApp は、この状態を次のとおり切り替える（D20）。

- 起動時: 行を作る部品を作り、pane が閉じていれば休ませる。
- pane を開いたとき: 使い始める。「Actor の同期が要る」と答えられたら、EvalPanel がその場で Actor を同期する。通常の同期と同じ処理で、学習側の step の控えと、次の定期の同期までの数え直しも同じに行う。
- pane を閉じたとき: 休ませる。

状態が決めるのは、Actor が写し直すときに写すかどうかだけである。採点は、状態に依らず、写しがあれば行う。

- 写しは部品が持ち、checkpoint には入れない。部品 1 つにつき predictor 1 つぶんのメモリを使う。
- Actor は、つながれたものを所有しない。部品が無くなったら、Actor はそれを写すのをやめる。
- 写しを書くのは Actor の写し直しで、読むのは step 診断の `Observe` である。どちらも、その Runner を進める 1 つのスレッドで順に走る（§2.4）。写しを守るロックは作らない。
- 表示をやり直したとき（`Reset`）は写し直さない。Actor の network もそのとき写し直さないためである。
- 写しを持たない Actor（`clone_model = false`）の Runner では、写しを作らず、Actor にもつながない。step ごとに Agent の読み取りロックの中で、学習側の推定器を採点し、β を読む。

見積もり（未測定。D14 で確かめる）:

- 採点 1 回（GUI のスレッド）: 既定で毎秒 15 step、1 lane。CPU での正規化、観測の device への転送 1 回、forward 2 回、誤差の読み戻し 1 回である。転送と読み戻しは同期の転送で、tick ごとに 1 回ずつ足される（lane のループの外でまとめて行う）。Atari の 1 lane で、1 回あたり 1〜3 ミリ秒の見込み。
- 定期の Actor の同期（写しを持つ Actor。EvalPanel の既定で 10 秒に 1 回）: GUI のスレッドが学習の入口 1 回の終わりを待つのは、今と同じ 1 回である。定期の同期では、待つ回数は増えない。ロックを持つ時間が、predictor と統計を写す分（数ミリ秒の見込み）だけ延び、学習の入口はその分だけ長く待つことがある。Taiga 式の Atari は入口 1 回が長いので（§2.1）、同期の待ちが最も大きくなる構成だが、この待ちは今もある。
- pane を開いたとき: 写しが無ければ、同期が 1 回入る。GUI のスレッドは、定期の同期と同じく、学習の入口 1 回の終わりを待つ。人の操作 1 回につき、最大 1 回である。
- 写しを持たない Actor: step ごとに、GUI のスレッドが学習の入口 1 回の終わりを待ちうる。その Actor の forward が、すでに同じ待ち方をしている。
- 内的報酬の列が無いときと、Evaluation Trace を閉じているときは、採点も写しも足されない。

### 5.5 列の設定

Atari.txt・GridMaze.txt・LunarLander.txt の `run.@rnd` に、次の 3 行を足す。

```text
run.@rnd : app.eval_trace_panel.columns.[235] = $agent_step intrinsic.bonus label:rnd_bonus
run.@rnd : app.eval_trace_panel.columns.[600] = $agent_step intrinsic.error label:rnd_err
run.@rnd : app.eval_trace_panel.columns.[610] = $agent_step intrinsic.normalized_error label:rnd_norm
```

百分位を見たいときは、利用者が次のように足す（既定には入れない。D13）。

```text
app.eval_trace_panel.columns.[605] = $agent_step intrinsic.error display:pct label:rnd_err_pct
```

## 6. 実装契約

### 6.1 公開の口と鍵

- `LifelongNoveltyEstimator::EstimateWithoutUpdate`: §5.1 の読むだけの採点。返すのは、lane ごとの倍率と、実装が名乗る lane ごとの診断値である。RND は `error` と `normalized_error` を名乗る。汎用の口は、RND の名前を持たない。
- `LifelongNoveltyEstimator::Clone`: 同じ状態の推定器を作る。写しを採点しても、元の推定器は変わらない。呼び出し側は、元の推定器が書き換えられない状態（Agent の読み取りロックの中）で呼ぶ。
- `LifelongNoveltyEstimatorFactory::LaneDiagnosticKeys`: 推定器を作らない構成でも、lane ごとの診断値の鍵の一覧を返す（既存の `InitialDiagnostics` と同じ考え方）。DefaultDQNAgent は、これに接頭辞 `intrinsic.` を付けた鍵と `intrinsic.bonus` を、step 診断の既知の鍵にする。
- `AgentStepDiagnosticsRequest`: step 診断の部品を作るときの入力。lane の数、`$agent_step` の列の鍵の一覧、その Runner の Actor を持つ（D3・D9）。名前は、同じ仕組みの `AgentStepDiagnosticsProvider`・`AgentStepDiagnosticsResult` と、Actor を作るときの入力 `ActorRequest` の形にそろえた。渡す Actor は、部品の `Observe` を呼ぶ Runner の Actor で、その Agent の `CreateActor` が作ったものとする。部品の `Observe` は、その Actor の `MakeAction`・`Sync` と並行に呼ばない（Actor の既存の契約と同じ）。
- `AgentStepDiagnosticsProvider::Suspend`・`Resume`: step 診断の部品を休ませる口と、使い始める口（D20・§5.4）。`Resume` の戻り値は、採点に要る写しが無く、Actor の同期が要るかどうかである。基底の既定は、`Suspend` が何もせず、`Resume` が偽を返す（同期に従う写しを持たない部品の答え）。DQN の部品は、RND が有効で、内的報酬の鍵を入力に持ち、写しを持つ Actor のときだけ、状態を持つ。それ以外のとき、`Resume` は偽を返す。`Observe` と同じく、その Actor の `MakeAction`・`Sync` と並行に呼ばない。
- `StepTraceBuilder::Suspend`・`Resume`: 同じ呼び出しを、step 診断の部品へ伝える。
- `ActorSyncFollower`: 同期に従うものの口（§3）。Actor は、source から写し直すとき、source を守るロックの中でこれを呼ぶ。呼ばれた側は、同じ source から自分の写しを写し直す。
- `Actor::AttachSyncFollower`: 同期に従うものを Actor につなぐ。戻り値は、その Actor が写し直すたびに呼ぶかどうかである。基底の既定は、つながずに偽を返す（同期に従うものを呼ばない Actor の答えで、`MuZeroActor`・`ImageClsActor`・テスト用の Actor はこのままにする）。DQN 系の Actor は、写しを持つときはつないで真を返し、写しを持たないときはつながずに偽を返す。Actor は相手を所有しない。相手が無くなったら呼ばない。`MakeAction`・`Sync` と並行に呼ばない。
- `Agent::CreateStepDiagnosticsProvider`: 引数を `AgentStepDiagnosticsRequest` に替える。DefaultDQNAgent と RainbowAgent の実装、テスト用の Agent も同じ引数を受ける。
- `$agent_step` の鍵: `intrinsic.error`・`intrinsic.normalized_error`・`intrinsic.bonus`（D5）。

### 6.2 設定と失敗の扱い

新しい設定キーは無い。足すのは列の宣言（§5.5）だけで、列の文法と検証は今のままである。

次の表は上から順に当て、最初に当たった行で扱う。

| # | 状態 | 挙動 |
|---|---|---|
| 1 | Rainbow の Run で内的報酬の列を宣言した | 既存の規則どおり、最初の読み出しで列番号と鍵つきで止まる |
| 2 | RND が無効の DefaultDQN の Run で内的報酬の列を宣言した | 3 列とも空欄。通知は出さない（設定で無効にした機能の値が無いのは、利用者が意図した状態である） |
| 3 | RND が有効の DefaultDQN で、内的報酬の鍵を求められたのに、入力の Actor が無い | 部品を作る時点で `ANET_SYSTEM_ERROR`。文に、求められた鍵と、Actor が要ることを入れる（部品の呼び出しの契約違反） |
| 4 | 読むだけの採点で、誤差が有限でない lane があった | その lane の 3 つの値を NaN にし、WARN を Run に 1 回出す。文には lane と値を入れる。学習と評価の値を変えずに表示の一部が欠ける異常に当たる（[AGENTS.md](../../AGENTS.md) §Fail-Fast 原則の表） |

別の Agent が作った Actor を渡すのは契約違反だが、検出しない。基底の `Actor` には、どの Agent が作ったかを確かめる口が無く、`dynamic_cast` は実装の規約で使えない。Run の中の Agent は 1 つなので、起きるのはテストの書き誤りに限られる。写しを持たない Actor と、DQN 系でない Actor は、`AttachSyncFollower` が偽を返すので、step ごとに自分の Agent の読み取りロックを取る形になり、別の Agent のロックの中で写すことは起きない。

### 6.3 checkpoint

形式も中身も変えない（D17）。

### 6.4 クラス一覧

> 従えない事情があれば、実装する側が理由を実装計画に書いて変えてよい（質問しない）。

| 名前 | 区分 | 責務 | 置き場（機能グループ） |
|---|---|---|---|
| `LifelongNoveltyEstimator` | 変更 | 読むだけの採点の口 `EstimateWithoutUpdate` と、写しを作る口 `Clone` を足す | `intrinsic`（`include/anet/intrinsic.hpp`） |
| `LifelongNoveltyEstimatorFactory` | 変更 | lane ごとの診断値の鍵の一覧 `LaneDiagnosticKeys` を返す | `intrinsic` |
| `RndNoveltyEstimator` | 変更 | 読むだけの採点を §5.1 の式で実装し、写しを作る | `intrinsic_rnd`（`src/intrinsic_rnd.hpp`・`.cpp`） |
| `AgentStepDiagnosticsRequest` | 新規 | step 診断の部品を作るときの入力（lane の数、鍵の一覧、Actor）を運ぶ | `rl`（`include/anet/rl.hpp`） |
| `Agent` | 変更 | `CreateStepDiagnosticsProvider` が `AgentStepDiagnosticsRequest` を受ける | `rl` |
| `ActorSyncFollower` | 新規 | 同期に従うものの口。Actor が source を守るロックの中で呼び、呼ばれた側が自分の写しを写し直す | `rl` |
| `Actor`（基底） | 変更 | 同期に従うものをつなぐ口 `AttachSyncFollower` を足す。既定は、つながずに偽を返す | `rl` |
| `DefaultDQNAgent` | 変更 | 学習の入口で評価した β を覚える。鍵の一覧に内的報酬の鍵があるとき、推定器と β の写しの入れ物（`IntrinsicSnapshot`）を作って Actor につなぎ（§5.4。この時点では写さない）、step 診断の部品へ 3 つの値の計算（§5.2）を渡す | `default_dqn_agent`（`include/anet/default_dqn_agent.hpp`・`src/default_dqn_agent.cpp`） |
| `IntrinsicSnapshot` | 新規（非公開） | 学習側の推定器と β の写しを持つ。`ActorSyncFollower` として、Actor が写し直すときに、使っている間は写し直し、休んでいる間は写しを捨てる。写しが無いかを答え、step 診断の採点に写しを貸す | `default_dqn_agent`（`.cpp` の無名名前空間） |
| `AgentStepDiagnosticsProvider` | 変更 | 休ませる口 `Suspend` と、使い始める口 `Resume`（Actor の同期が要るかを返す）を足す。基底の既定は、何もせず、偽を返す | `rl` |
| `RainbowAgent` | 変更 | `AgentStepDiagnosticsRequest` を受ける。鍵の一覧と Actor は使わない | `rainbow_agent`（`include/anet/rainbow_agent.hpp`・`src/rainbow_agent.cpp`） |
| `Actor`（DQN 系） | 変更 | `AttachSyncFollower` を実装する。写しを持つとき、つながれたものを、network を写し直すのと同じ読み取りロックの中で、network に続けて呼ぶ。相手を所有しない | `dqn_based_agent`（`src/dqn_based_agent.hpp`・`.cpp`） |
| `DqnAgentStepDiagnosticsProvider` | 変更（非公開） | Agent が内的報酬の採点を渡したときだけ、step ごとに 1 回それを呼び、値を既知の鍵として返す。`Suspend`・`Resume` を、Agent が渡した内的報酬の採点へ伝える | `dqn_based_agent` |
| `StepTraceBuilder` | 変更 | 列の宣言から `$agent_step` の鍵を集め、Runner の Actor とあわせて渡す。コンストラクタが Actor を受ける。`Suspend`・`Resume` を step 診断の部品へ伝える | `metrics`（`include/anet/metrics.hpp`・`src/metrics.cpp`） |
| `EvalTracePanel` | 変更 | 行を作る部品へ、Runner の Actor を渡す。起動時に pane が閉じていれば、部品を休ませる。pane を開いたら使い始め、「Actor の同期が要る」と答えられたら EvalPanel に同期させる。閉じたら休ませる | RunnerApp（`apps/runner/src/EvalTracePanel.cpp`） |
| `EvalPanel` | 変更 | Evaluation Trace の求めで、その場で Actor を同期する公開の口を足す。処理は通常の同期と同じである | RunnerApp（`apps/runner/src/EvalPanel.hpp`・`.cpp`） |

### 6.5 移行対象

検索は 2026-11-12 に `git grep`（追跡しているファイルの全体。大文字小文字を区別しない）で行った。検索語と件数は、`CreateStepDiagnosticsProvider` が 16 ファイル 22 行（コード 8 ファイル、テスト 3 ファイル、設計文書 140 の日英、課題、PRD094、本 PRD の書き戻しで足した wiki の 1 行）、`StepTraceBuilder` がコードとテストで 7 ファイル 21 行、`agent_step` がコード・設定・設計文書で 13 ファイル、`CopySourceNetwork` が 5 ファイル 7 行（コード 2 ファイル、課題、過去の PRD 2 つ）、`anet::rl::Actor` を継ぐクラスが 9 つ（DQN 系の `Actor`、`MuZeroActor`、`ImageClsActor`、テスト用の 6 つ）、ほかに `eval_trace_panel.columns`、`ロックを持たない`、`ロック待ち` である。過去の記録（`docs/memo/done`、実装記録、`docs/experiments`、課題記録、Run の設定の写し）は書き換えない。

- コード: §6.4 の各ファイル。`CreateStepDiagnosticsProvider` の宣言と実装（rl.hpp、default_dqn_agent.hpp・.cpp、rainbow_agent.hpp・.cpp、dqn_based_agent.hpp・.cpp）と、呼び出し（metrics.cpp）。`StepTraceBuilder` のコンストラクタ（metrics.hpp・.cpp）と、それを作る EvalTracePanel.cpp。`ActorSyncFollower` と `Actor::AttachSyncFollower`（rl.hpp）、DQN 系の `Actor` の実装と `CopySourceNetwork`（dqn_based_agent.hpp・.cpp）。`MuZeroActor`・`ImageClsActor`・テスト用の Actor は、基底の既定（つながない）のままで、変えない。`AgentStepDiagnosticsProvider` の `Suspend`・`Resume`（rl.hpp）と、DQN の部品の実装（dqn_based_agent.cpp）。`AgentStepDiagnosticsProvider` を継ぐクラスは 3 つ（DQN の部品と、テスト用の 2 つ）で、テスト用は基底の既定のままでよい。`StepTraceBuilder` の `Suspend`・`Resume`（metrics.hpp・.cpp）。pane の開閉の切り替え（EvalTracePanel.cpp の `SetActive` と `Initialize`）と、EvalPanel の同期の公開の口（EvalPanel.hpp・.cpp）。
- 設定: Atari.txt・GridMaze.txt・LunarLander.txt の `run.@rnd` に §5.5 の 3 行を足す。common.txt の既定の列は変えない。
- テスト（引数が変わるので直すもの）: [dqn_based_agent_test.cpp](../../src/core/base/src/dqn_based_agent_test.cpp) の `CreateStepDiagnosticsProvider` の呼び出し 4 か所（`[step_diagnostics]`・`[diagnostic_replay]`・`[diagnostic_no_wait]` と、`dqn::CreateStepDiagnosticsProvider` を直接呼ぶ `[prd101]` のテスト）、[metrics_test.cpp](../../src/core/base/src/metrics_test.cpp) と [EvalTracePanel_test.cpp](../../src/apps/runner/src/EvalTracePanel_test.cpp) のテスト用の Agent の override（各 1 か所）、`StepTraceBuilder` を作る箇所（dqn_based_agent_test.cpp の 2 か所、[trainer_test.cpp](../../src/core/base/src/trainer_test.cpp) の 3 か所、metrics_test.cpp の 8 か所）。どれも断定は変わらない。
- テスト（足すもの）: §7 の 1〜10・12・15。タグ `[prd127]` を付ける。15 は [EvalTracePanel_test.cpp](../../src/apps/runner/src/EvalTracePanel_test.cpp) に足す。同ファイルの既存のテストは `SetActive` を呼んで pane の開閉を切り替えており（呼び出し 11 か所）、その断定は変わらない。
- 出荷設定のテスト: `[shipped_config]` に §7 の 12 を足す。既定の列の検査（[config_test.cpp](../../src/core/base/src/config_test.cpp) の `app.eval_trace_panel.columns` の一致）は変わらない。
- 検証資材: `core/anet-core/testdata/prd127/README.md` を新設し、§7 の 11 の比較の手順と、§7 の 14 の測定の手順を書く。[PRD123 の検証資材](../../src/core/base/testdata/rfd0123/README.md)の比較の道具を使えるなら使う。
- 設計文書 [140](../design/140_observability.jp.md): §5 に、Evaluation Trace の行を作る流れのシーケンス図を足す（D19）。§6.1 の target の表の `$agent_step` の行（DQN の鍵に内的報酬の 3 つを足す）と、`CreateStepDiagnosticsProvider` の段落（入力の型、鍵の一覧は計算するかどうかだけを決めること）を直す。英訳 [140 en](../design/140_observability.en.md) の同じ箇所。
- 設計文書 [212](../design/212_rnd_lifelong_novelty.jp.md): Evaluation Trace で見る値の節を足す（3 つの値の意味、§5.1・§5.2 の表、値の時点（D4 の帰結）、どの画面の値か（D15 の帰結）、`variance_feedback` の値の単位（§5.1）、forward で乱数を引く層・状態を変える層の制約（D11））。§5.1 の「画面表示のために新しい学習ロック待ちを追加しない」の文を、D4・D20・§5.4 に合わせて直す（写しを持つ Actor では、Actor の同期と同じロックの中で写すので、定期の同期のロック待ちの回数は増えない。pane を開いて写しが無かったときに、同期が 1 回入り、GUI のスレッドが学習の入口 1 回の終わりを待つ。写しを持たない Actor では、step ごとに待つ。実装の規約の例外であることと、変えるときの条件も書く）。§7 の 14 の測定値を書く。英訳 [212 en](../design/212_rnd_lifelong_novelty.en.md) の同じ箇所。
- 設計文書 [200](../design/200_dqn_agents.jp.md): §6.2 の RND の段落（読むだけの採点と、step 診断が持つ写し）、§9.1 の群 56 の段落の近くに step 診断の鍵を書く。Actor の `clone_model` を説明する箇所（同書に 2 か所）に、同期に従うもの（`ActorSyncFollower`）を network と同じ読み取りロックの中で呼ぶことと、step 診断の推定器の写しがそれに当たることを足す。英訳 [200 en](../design/200_dqn_agents.en.md) の同じ箇所。
- 設計文書 [100](../design/100_runtime_and_configuration.jp.md): `Actor::Sync()` の契約を書いた段落に、同期に従うものをつなぐ口（`AttachSyncFollower`）を 1 文足す。英訳 [100 en](../design/100_runtime_and_configuration.en.md) の同じ箇所。
- 設計文書 [160](../design/160_applications_and_tools.jp.md): `EvalTracePanel` の節に、pane を閉じている間は step 診断の部品を休ませること、開いたときに写しが無ければ EvalPanel がその場で Actor を同期することを足す（D20）。英訳 [160 en](../design/160_applications_and_tools.en.md) の同じ箇所。
- ユーザーガイド [020](../design/020_user_guide_run.jp.md): §6.5 の列番号の表に235 `rnd_bonus`と600〜699「内的報酬の誤差」を足し、`run.@rnd` の Run で出る 3 列の読み方（どの画面の値か、値の時点、方式による単位の違い）と、百分位の列の足し方（§5.5）を書く。EvalPanel の同期を説明する箇所（`app.eval_panel.model_sync.mode`）に、内的報酬の列を宣言した Run では、Evaluation Trace を開いたときにも同期が入ることがあると足す（D20）。英訳 [020 en](../design/020_user_guide_run.en.md) の同じ箇所。
- wiki: トピック別まとめの [324 DQN 系列](../wiki/categories/324_dqn.md)・[321 Agent 共通基盤](../wiki/categories/321_agent_core.md)・[317 可視化・メトリクス](../wiki/categories/317_metrics.md)・[318 強化学習 共通基盤](../wiki/categories/318_rl_core.md)・[RunnerApp](../wiki/apps/110_runner_app.md)、[内的報酬のページ](../wiki/mechanisms/intrinsic_novelty.md)の「関連」（起票の書き戻しで直す）。
- 直さないもの: `$agent_step` の既存の 3 つの鍵だけを使うテストの断定と、PRD094 の検証資材（`core/anet-core/testdata/prd094/compare_configs.py`。既定の列だけを比べる）。既定の列も既存の鍵も変えない。`StepTraceBuilder` に触れる過去の検証資材の説明（`core/anet-core/testdata/prd094/README.md`・`prd098/README.md`）は、当時の手順の記録なので書き換えない。
- PRD116 の §1.3 の「評価での採点と診断」の行は、当時の判断として残す。診断を本 PRD で決めたことは、本 PRD の §9 と wiki に書く。
- 実装の規約（impl.md の UI スレッドのロックの規約）は変えない。D4 が開発者の決めた例外であることを、実装計画と、実装のレビューの依頼に書く。

## 7. 成功の測り方と受入

1〜12 と 15 はテストか検証資材で確かめる。13・14 は人が確かめる。新しいテストはタグ `[prd127]` を付ける。

| # | 観点 | 観測する結果 |
|---|---|---|
| 1 | 読むだけの採点が推定器を変えない | 同じ seed と設定の推定器を 2 つ作り、同じ観測の列で `Estimate` と `Learn` を繰り返す。片方だけ、各呼び出しの間に `EstimateWithoutUpdate` を挟む。2 つの推定器で、`Estimate` の戻り値、`Learn` の戻り値、`Diagnostics` の値がすべて一致する。構成は whiten の `cumulative`、whiten の `ema`、`variance_feedback` の 3 つ |
| 2 | 読むだけの採点の式 | 既知の統計で、誤差・標準化した値・倍率が §5.1 の 2 つの表の手計算と、浮動小数点の丸めの範囲で一致する。whiten の中心化あり・なし、`clipped`・`unclipped`、`variance_feedback` を通す。3 つの方式（whiten の `cumulative`・`ema`、`variance_feedback`）のそれぞれで、読むだけの採点の倍率が、直後にその観測 1 件を `Estimate` へ渡したときの倍率と一致する |
| 3 | 標準化した値の NaN | §5.1 の標準化した値の表の行 1〜4 のそれぞれで NaN になり、倍率が 1 になる。warm-up 中で統計がある構成（`warmup_mode = collect_errors`）では、標準化した値が有限で、倍率が 1 になる |
| 4 | 写しが元を変えない | 推定器の `Clone` で作った写しに `EstimateWithoutUpdate` を繰り返しても、元の推定器の `Estimate` の戻り値と `Diagnostics` の値が、写しを作らなかった場合と一致する。写しの `EstimateWithoutUpdate` の値は、写した時点の元の値と一致する |
| 5 | step 診断の値 | §5.2 の表の行 2・5・6・7 のそれぞれで、3 つの鍵の値が表のとおりになる。行 7 の `intrinsic.bonus` は、同じ遷移 1 件を、推定器と β を写した時点の直後に学習の入口へ渡したときに報酬へ足される値（偽の Learner の受信値、または n-step 無効時の公開 Replay 収益で確認する）と一致する。`intrinsic.observation = state` では、行動した観測を採点した値になる |
| 6 | 写しは Actor の写し直しに従う | `clone_model = true` の Actor で、内的報酬の鍵を入力に入れた step 診断を作る。(a) Actor がまだ写し直していない間は、3 つの値が NaN である（§5.2 の表の行 3）。(b) Actor を同期すると、次の `Observe` で、同期した時点の学習側の値になる。(c) そのあと学習の入口を進めて predictor を変えても、Actor が写し直すまでは、同じ観測の 3 つの値が変わらない。(d) `sync_interval` による写し直しでも、(b) と同じく、写し直した時点の値に入れ替わる。(e) step 診断の部品を破棄したあとに Actor を同期しても、エラーにならない。(f) 値が出ている状態で `Suspend` を呼び、Actor を同期せずに `Resume` を呼ぶと、偽が返り、同じ観測の 3 つの値は休ませる前と同じである。(g) `Suspend` を呼んだあと、学習の入口を進めて Actor を同期すると、`Resume` が真を返し、次の `Observe` の 3 つの値は NaN である。そのあと Actor を同期すると、同期した時点の値が入る。(h) 内的報酬の鍵を入力に入れていない step 診断、`clone_model = false` の Actor の step 診断、`intrinsic.enabled=false` の Agent の step 診断では、`Resume` が偽を返す |
| 7 | ロックの待ち方 | 別のスレッドが Agent の書き込みロックを持っている間に、内的報酬の鍵を入力に入れた step 診断の `Observe` を呼ぶ。(a) `clone_model = true` の Actor では、Actor の同期の前でも後でも、1 秒以内に返る。同期の後なら、同期した時点の値を返す（写すのが Actor の同期の中で済んでいる）。(b) `clone_model = false` の Actor では、1 秒待っても返らず、ロックを離すと返る。(c) 内的報酬の鍵を入力に入れていない step 診断は、どちらの Actor でも 1 秒以内に返る（既存の `[diagnostic_no_wait]` も通る） |
| 8 | 評価専用の Run | 学習した Agent の checkpoint を、`learner.enabled=false` の Agent が読む。`clone_model = true` の Actor を同期したあと、step 診断の `intrinsic.error` と `intrinsic.normalized_error` が有限で、`intrinsic.bonus` が NaN になる |
| 9 | RND が無効、Rainbow、Actor の欠け、非有限の誤差 | §6.2 の表の行 1〜4 のそれぞれが、表のとおりになる。行 1 と行 3 の ERROR の文に、表が挙げるものが入る。行 4 の WARN は 1 回だけ出て、文に lane と値が入る |
| 10 | Trace を挟んでも学習が変わらない | 同じ seed と設定の DefaultDQNAgent を 2 つ作り、同じ遷移の列を学習の入口へ渡す。片方だけ、毎 step、内的報酬の鍵を入力に入れた step 診断の `Observe` を呼び、途中で Actor を同期する。2 つの Agent で、Learner が受け取った報酬の列（偽の Learner の受信値、または n-step 無効時の公開 Replay 収益で確認する）と、`$agent intrinsic.*` の診断値がすべて一致する。`clone_model` が true と false の両方で確かめる |
| 11 | Trace を使わない Run が今と同じ | 着手前に保存した実行体と、実装後の実行体に、着手前の設定の写しを渡して比べる。構成は次の 3 つで、どれも seed 1 本・決定的モードである。(a) LunarLander の既定の連鎖に `run.@rnd` を足した 100k exp step の Run。(b) 同じ連鎖に `run.@rnd` と、Atari の `run.@rnd_taiga` の 8 つの設定値を足した 100k exp step の Run。(c) `intrinsic.enabled = false` の GridMaze・LunarLander・Atari・DropMerge の既定の連鎖の Run、各 8,704 exp step（PRD123 の検証資材が RND 無効で比べた 4 本と同じ条件）。比べるのは全 scalar 点の checksum と `metrics.scalar.defs` で、除くのは時刻・資源の群だけである（実時間に依るため）。3 つとも一致する |
| 12 | 出荷設定 | Atari・GridMaze・LunarLander の `run.@rnd` を解決すると、`app.eval_trace_panel.columns.[235]`・`[600]`・`[610]` が §5.5 の 3 行の値になる。235は230 `reward`の直後に並ぶ。`run.@rnd`単独、`run.@skip > run.@rnd`、`run.@rnd > run.@skip`で確かめ、RNDの3列以外の全キー・文字列値は着手前の同じ選択の解決と一致する。`run.@rnd`を選ばない空の選択と`run.@skip`単独では、この3つの設定キーが無く、全キー・文字列値が着手前と一致する（`verify_configuration.py`）。LunarLander の `run.@rnd` を解決した列の宣言で、DefaultDQN の行を 1 行作ると、3 つの鍵がエラーなく読める（`[shipped_config]`） |
| 13 | 画面（人が確かめる） | LunarLander か Atari の `run.@rnd` の Run で Evaluation Trace を開くと、`rnd_bonus`が`reward`の直後、`rnd_err`・`rnd_norm`が`prio_pct`の右に出る。`rnd_err` は、pane を開いた直後の最初の行から値が入る。`rnd_norm` は誤差の統計ができたあと、`rnd_bonus` は学習の入口を通ったあとに、Actor が同期した行から値が入る。学習が進んでいる Run で、View メニューか Panel ツールバーの操作で pane を閉じ、10 秒以上おいてから同じ操作で開き直しても、最初の行から 3 列に値が入る（利用者の操作から、開いたときの同期までがつながっていること）。Trace モードで過去の行を選んでも、その行の 3 列の値が変わらない。`run.@rnd` を選ばない Run では 3 列が出ない |
| 14 | 費用の記録（人が測る。合否なし） | Atari の実時間優先の構成に `run.@rnd` を足した Run と、さらに `run.@rnd_taiga` を足した Run のそれぞれで、Evaluation Trace を閉じた状態と、開いて EvalPanel を再生した状態を、交互に 3 巡以上測る。各状態の、学習側の 1 秒あたりの exp step の中央値、step 診断の `Observe` 1 回の時間の中央値と最大、Actor の同期 1 回のうち推定器と β を写す時間の中央値と最大を、設計文書 212 に書く。GUI を動かせない枠で実装を終えるときは、測らずに終え、本 PRD の課題に残りの作業として書く（課題の判定は「残件有り」） |
| 15 | pane の開閉と同期 | EvalTracePanel のテストで、テスト用の step 診断と Actor を使う。pane の開閉は、`EvalTracePanel::SetActive` を直接呼んで切り替える。これは公開の口で状態を作る準備で、利用者の操作から `SetActive` までのつながり（フレームの pane の表示・非表示の処理）は、人が 13 で確かめる。(a) `SetActive` を呼ばずに `Initialize` を終えると（pane が閉じた状態）、行を作る部品を作った直後に `Suspend` が 1 回呼ばれている。(b) `SetActive(true)` で `Resume` が呼ばれ、それが真を返すときは、Actor の `Sync` がちょうど 1 回呼ばれる。偽を返すときは呼ばれない。(c) `SetActive(false)` で `Suspend` が呼ばれる |

## 8. 決めること（未決表）

残る未決は無い。U1（`run.@rnd` の既定に 3 列を入れたままにするか）は 2026-11-14 に決めた（冒頭の改訂と D12）。

本 PRD で直さないが、課題として残す候補は次の 3 つである。工程の最後に起票の候補として挙げる。

- DropMerge 用の RND のプロファイル（§1.3）。
- Evaluation Trace の列に知らない鍵があると、Run の途中で pane を開いた時点で Run ごと止まること（全 target に共通。D10）。
- EvalPanel の Actor の同期が、UI スレッドから学習のロックを取っていること（実装の規約より前からの動き。§2.4）。直すときは、推定器と β の写しも一緒に移す（D4）。

## 9. ほかの PRD との関係

- [PRD116](0116_rnd_lifelong_novelty_10rfd.md): §1.3 が「評価での採点と診断」を非ゴールにし、「評価で内的報酬を見たくなったら 173 の採点点の問いで決める」とした。本 PRD は、採点の場所を動かさずに診断だけを足す形で、この再開条件に応えた。学習用の採点の場所の問いは 173 に残る。
- [PRD123](0123_rnd_intrinsic_reward_scale_10rfd.md): 誤差の統計の取り方、warm-up、平均を引かない形、done の lane の扱いは、本 PRD の読むだけの採点がそのまま従う（§5.1・§5.2）。
- [173](0173_intrinsic_reward_novelty_10rfd.md): 問い 7（ボーナスの合成と尺度）と 11（保存する報酬）は開いたままである。Actor 側の採点や、方策が内的報酬を入力に取る形が入ると、Actor が出す値を読む列が別にできる。評価の Actor は学習用の統計を更新できないので、そのときも読むだけの採点の口と写しは使える。
- [PRD121](0121_extrinsic_intrinsic_value_decomposition_10rfd.md)（外的と内的の価値の分解。暫定）: 内的の価値を別の Head で学ぶ形が入ると、`intrinsic.bonus` の列を内的の価値の列と並べて読める。本 PRD は 121 を待たない。
- [PRD094](done/0094_evalpanel_step_history_10rfd.md): 「step trace はロックを持たない」は、写しを持つ Actor ではそのまま成り立つ。写しを持たない Actor で内的報酬の列を宣言した構成だけが例外になる（D4 の帰結）。PRD094 の再発防止として入った実装の規約（UI スレッドのロック）に対しては、D4 が例外である。
- [PRD184](0184_eval_session_step_trace_10rfd.md)（暫定・未実装）: 同じ `$agent_step` の鍵と `StepTraceBuilder` を評価 session で使う。本 PRD の 3 つの鍵は、184 が入ればそのまま記録に使える。184 の再グリルでは、`CreateStepDiagnosticsProvider` の入力（D3）と、session の全体を開始の同期の時点の predictor で採点すること（D4 の帰結）を前提に入れる。step 診断の部品は、作った直後から「使っている」状態なので、session の記録は `Suspend`・`Resume` を呼ばなくてよい（D20）。部品を session の開始の同期より前に作っておけば、最初の step から値が入る。
- [PRD170](0170_nn_trace_capture_unification_10rfd.md)（未実装）: `MakeAction` と Runner の step に引数を足す。本 PRD とは別の口だが、同じ基底（rl.hpp）とテスト用の Agent を触るので、実装の順によって書き換えが重なる。意味はぶつからない。
- [PRD117](0117_warn_policy_and_unread_config_10rfd.md): §6.2 の行 4 の WARN は、「構築後に、学習と評価の値を変えずに表示の一部だけが欠ける異常」の行に当たる。

## 10. issue の対応表

| ID | 題 | area | 現在のコードで確かめた根拠の位置 | 直す確定事項 | 確かめる受入 |
|---|---|---|---|---|---|
| issue-20261107-1226e5 | Evaluation Traceで内的報酬（RNDの新しさ）を見られない | 324_dqn、110_runner_app | [default_dqn_agent.cpp](../../src/algo/dqn/src/default_dqn_agent.cpp) の `UpdateFromBatch`（採点は学習の入口だけ）、[intrinsic_rnd.cpp](../../src/core/base/src/intrinsic_rnd.cpp) の `Estimate`（呼ぶたびに統計を更新）、[dqn_based_agent.cpp](../../src/core/base/src/dqn_based_agent.cpp) の `DqnAgentStepDiagnosticsValues`（内的報酬の鍵が無い） | D1〜D5・D12 | §7 の 1・5・10・12・13 |

## 11. 参照

- 設計文書 [212 RND による新しさの評価と内的報酬](../design/212_rnd_lifelong_novelty.jp.md) §3.3（倍率とボーナスの式）、§5.1（計算する場所と費用）。
- 設計文書 [140 可観測性](../design/140_observability.jp.md) §6.1（Evaluation Trace の列と読み取り）。
- [ADR 0013](../adr/0013-actor-network-resource-policy.md)、[ADR 0057](../adr/0057-eval-trace-columns-declared-apart-from-metrics-and-read-at-transition-time.md)、[ADR 0058](../adr/0058-approx-priority-column-computed-by-agent-step-diagnostics.md)、[ADR 0072](../adr/0072-make-action-inputs-are-passed-per-call-through-runner-step-not-registered.md)。
