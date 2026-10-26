# TempoRL 対応（スキップ: 選んだ行動を続ける長さの学習）PRD

> 2026-10-03 起票（暫定。質疑を省き、決めることを §5 に残していた）。
> 2026-10-23 グリル（`/anet-create-prd`、Claude Code）で全面改訂した。論文解説、外部の壁打ちで作った引継ぎ資料と図、下調べ、現行コードの再確認（2 体の調査）に照らし、暫定版の「区間をまとめた遷移を replay に入れ、割引は SMDP の経過時間で行う」「網の構成は詳細解説を確かめてから選ぶ」を、次の決定で置き換えた: ゴールは判断の回数と学習の速さで、NEET の抑制は非ゴール（D1）。NetworkModel は primary とスキップ価値の Network の 2 つで、skip の target は bootstrap に使わず target 側の Actor の行動にだけ使う（D3、[ADR 0062](../adr/0062-networkmodel-holds-auxiliary-network-fed-by-primary-outputs-with-target-used-only-for-acting.md)）。replay は slot に 1 step の reward・終わり方・判断の旗を残し、サンプル時に判断で決めた区間の中の窓を組む（D11・D13、[ADR 0063](../adr/0063-replay-slot-keeps-per-step-reward-and-end-and-samples-variable-windows.md)）。skip Head は分離型と包含型の両方（D5）、勾配は合算／遮断を設定で切り替え、クリップは Network 別（D6）、スキップ価値は Q と同じ型（D7）、Dueling は無しを既定に Q 共有型も選べる（D8）、探索は行動の方策と同じ規則（D9）、サンプルは 1 点 k（D13）、PER は Q の TD を既定に合成も選べる（D14）。同日に実装へ着手するため 999 から 098 へ採番した。起草後に別モデル（Opus）の検討を受け、窓を判断で区切ること（D13）、skip の target の複製（D3）、sampleable の遅れの範囲（D11）、Actor の複製と同期（§5.3・§5.5）、群番号（59）、FP16 との併用の禁止、§7 の判定の細部を直した。
> 起点: DropMerge の NEET の分析で、待ちを 1 step ずつ判断することが問題の一部だと分かったこと。[086](086_dropmerge_macro_actions_10prd.md) は「繰り返し回数を離散化した行動」を Env 側で足し、Agent の改修なしで試せるようにする。本 PRD はその次の段階として「どの行動か」と「何 step 続けるか」を Agent が分けて学ぶ TempoRL を扱う。
> 関連: [086_dropmerge_macro_actions_10prd.md](086_dropmerge_macro_actions_10prd.md)、[999_smdp_transition_duration_10prd.md](999_smdp_transition_duration_10prd.md)（経過時間の割引。非ゴール）、[999_decision_aligned_nstep_10prd.md](999_decision_aligned_nstep_10prd.md)・[999_nstep_target_correction_10prd.md](999_nstep_target_correction_10prd.md)（§5.1 の土台に乗る 2 本）、[999_observation_pipeline_10prd.md](999_observation_pipeline_10prd.md)（強制行動を判断の入力にする単位 A。§9）、[done/067_MunchausenRL_10prd.md](done/067_MunchausenRL_10prd.md)（D3 の n-step の形）、[done/078_replay_frame_history_integrity_10prd.md](done/078_replay_frame_history_integrity_10prd.md)、[ADR 0044](../adr/0044-replay-frame-history-start-from-episode-start-at-push.md)、[CONTEXT.md](../../CONTEXT.md) の「スキップ」「スキップ長」「スキップ価値」「判断」「NEET」。
> 改訂: 2026-10-23。実装レビュー対応。強制行動をtrainへ接続するときの判断記録と、§5.11の実装名を明記した。
> 改訂: 2026-10-24。段 1 の途中経過（GridMaze の迷路依存）を §7 に、既定 seed での崩壊の機構を §9 に追記。
> 改訂: 2026-10-24（2 回目）。設定見直し第 2 依頼の実装に合わせ、§5.3 に skip Head の出力の形と包含型 × IQN の broadcast 連結、§5.7 に Trace 列 `skip_q.[j]`、§7 に段 1 の包含型の取り直しの結果を追記。
> 改訂: 2026-10-24（3 回目）。特徴量の説明上の記号 z と設定の公開出力名を区別し、§5.6 の IQN の配線を現行の `skip_context` と融合済み branch の対応へ合わせた。
> 改訂: 2026-10-25。§8 の「全 k を mask 付きで一括するサンプル」「k_max に比例する重み」「skip 中の lane の forward を省く」の行を、後続の暫定 PRD（[999_temporl_skip_followups](999_temporl_skip_followups_10prd.md)）へ向けた。
> 改訂: 2026-10-26。§7 に段 1 の 3 seed と段 2（LunarLander）の結果、§9 の 2 に 5×5 の既定 seed で届かない原因（UQE でスキップ長を選ぶこと）、§9 の 3（評価 slot の判断率）を追記。
> 改訂: 2026-10-26（2 回目）。5×5 の既定 seed の件を「序盤に見つけたゴールを保持できず崩壊」から「スキップ長を UQE で選ぶとゴールに届かない」に直した（評価は一度も届かず、学習側もほぼ届いていなかった）。§7 の段 1 の結論を、届く速さと終盤の安定さに書き分けた。
> 改訂: 2026-10-26（3 回目）。段 2 の結果を受けて LunarLander の既定を有効（分離×合算）にしたことを §6 の 21 と §7 の段 2 に追記。`run.@skip` の metrics の連鎖が Munchausen の指標を落としていた件と、その直しも §7 の段 2 に記した。
> 改訂: 2026-10-26（2モデルレビュー対応）。実装契約を観測できる振る舞いに整理し、Agent側とRun/metrics側のprofileの区別、M1の意味、1step記録のメモリ内訳を明記した。観測パイプラインの既存ドラフトを参照先として同梱対象へ含める。
> 改訂: 2026-10-27。§7 に段 3 のふるい（実時間優先の qbert、4 腕 × 2 seed、10M）の結果を追記し、その結果で Atari の実時間優先の既定を有効（分離×合算）にしたことを §6 の 21 に追記した。

## 1. 目的と問題

行動と一緒に「その行動を何 step 続けるか」も選ぶ TempoRL（Biedenkapp et al. 2021）を、DefaultDQN で設定により有効にできるようにする。論文は、行動を選ぶ Q（behaviour Q）と、選んだ行動を条件に続ける長さを選ぶ Q（スキップ価値、skip Q）を分け、続けた区間の経験（1 回の区間から j(j+1)/2 本の sub-skip）でスキップ価値を学ぶ構成で、素の Q 学習より少ない判断で同じ方策に達し、グリッドワールドでは最大で 1 桁速く学ぶと報告している（[論文解説 §3.2](../../reports/biedenkapp2021_temporl_paper_2026-10-03.md#32-いつ意思決定するかの学習learning-when-to-make-decisions)、[同 4.2.3](../../reports/biedenkapp2021_temporl_paper_2026-10-03.md#423-atari-環境atari-environments)）。

DQN 系の Agent は今、Env の 1 step ごとに 1 つの行動を選ぶ。続ける長さを選ぶ仕組みは無く、続けた区間の経験は 1 step ずつの遷移としてしか使われない。[086](086_dropmerge_macro_actions_10prd.md) の「繰り返し回数の離散化」は、(行動, 回数) の組を平らな行動の一覧として Agent に見せる。Agent の改修が要らない代わりに、行動の数が回数の候補の数だけ増え、続けた区間の途中の遷移は学習に使われない。

本 PRD は、スキップ価値の学習に要る 6 つの機構を DefaultDQN と共通 ReplayBuffer に足す（§4 D2）。NEET の抑制は非ゴールである（§3）。

## 2. 現状（2026-10-23 にコードで確認したこと）

行番号は HEAD e885a5ae の作業ツリーの値。

### 2.1 NN と NetworkModel

- `Network` は Body 1 つと Head 1 つで構成され、Head は省略できる。Head があるとき forward は Head の出力だけを返し、Body の出力（特徴量）は外に出ない（[nn_impl.cpp:1913-1932](../../core/anet-core/src/nn_impl.cpp)）。Body の出力と Head の入力の対応付けは `net.body.output.[鍵] = branch` で、既存の Head が読む鍵は `features`・`value_feature`・`adv_feature` だけである（[nn_impl.cpp:1517-1525](../../core/anet-core/src/nn_impl.cpp)、[nn_heads.hpp:46](../../core/anet-core/src/nn_heads.hpp)、[dqn_based_heads.cpp:15-16](../../core/anet-core/src/dqn_based_heads.cpp)）。
- Body は forward のたびに全ブランチを実行し、bind した入力が無いとエラーになる（[nn_impl.cpp:1505-1511](../../core/anet-core/src/nn_impl.cpp)、[:1376-1381](../../core/anet-core/src/nn_impl.cpp)）。部分実行は可塑性 probe 用の `ForwardUpTo`（NoGrad・eval 固定、Head を通らない）だけである（[:1529-1548](../../core/anet-core/src/nn_impl.cpp)）。
- 登録済みの NN ブロックは 29 種で、勾配を止めるブロックは無い（[nn_modules.cpp:3265-3311](../../core/anet-core/src/nn_modules.cpp)）。
- Head は射影を複数持てる（Dueling は value と adv の 2 本。[dqn_based_heads.cpp:143-252](../../core/anet-core/src/dqn_based_heads.cpp)）。IQN の射影は分位ごとに同じ射影を当てる（[:501-515](../../core/anet-core/src/dqn_based_heads.cpp)）。射影の Noisy は `LinearProjectionModule` が担い、構築の設定 `DefaultDQNAgent.head_projection.*` は Head の全射影で共有する（[nn_modules.cpp:182-475](../../core/anet-core/src/nn_modules.cpp)、[default_dqn_agent.hpp:107-115](../../core/anet-core/include/anet/default_dqn_agent.hpp)）。
- Head の種類は `quantile_mode` と `use_dueling_net` で 6 種の factory に直書きされ、設定から差し替える登録の仕組みは無い（[default_dqn_agent.cpp:94-123](../../core/anet-core/src/default_dqn_agent.cpp)）。IQN の `taus` は Agent が入力 spec に能動的に足す（[:151-159](../../core/anet-core/src/default_dqn_agent.cpp)）。離散の spec は one-hot が自動になる（[nn_impl.cpp:1289-1324](../../core/anet-core/src/nn_impl.cpp)）。
- `NetworkModel` は online と target の Network を 1 つずつ持ち、Save/Load はその 2 つを順に書く（[dqn_based_agent.hpp:681-739](../../core/anet-core/src/dqn_based_agent.hpp)、[dqn_based_agent.cpp:783-799](../../core/anet-core/src/dqn_based_agent.cpp)）。可視化は `policy-net.`／`target-net.` の prefix でこの 2 つを引く（[:759-781](../../core/anet-core/src/dqn_based_agent.cpp)）。`GetOnlineNetwork` 系の呼び出しは本体 31 か所、テスト 73 か所ある。MuZero 試作の `MuZeroNetworkSuite` は 3 つの Network を束ねる Module で、target・checkpoint・Noisy の箱・診断は持たない（[muzero_based_agent.hpp:25-38](../../core/anet-core/src/muzero_based_agent.hpp)）。
- 行動は forward の後に NN の外の ActionPolicy（ε-greedy・UQE・Thompson、spatial exploration で lane ごとの ε・τ が固定）が決める（[dqn_based_agent.cpp:2079-2086](../../core/anet-core/src/dqn_based_agent.cpp)、[:1343-1357](../../core/anet-core/src/dqn_based_agent.cpp)、[:1612-1613](../../core/anet-core/src/dqn_based_agent.cpp)）。Greedy は ε=0 の EpsilonGreedy（[default_dqn_agent.cpp:253-260](../../core/anet-core/src/default_dqn_agent.cpp)）。

### 2.2 Learner

- 勾配クリップは online の全パラメータの global norm 1 本で、既定 on・閾値 30（[dqn_based_agent.cpp:3011-3021](../../core/anet-core/src/dqn_based_agent.cpp)、[nn_util.cpp:177-196](../../core/anet-core/src/nn_util.cpp)、[agent.hpp:184-185](../../core/anet-core/include/anet/agent.hpp)）。optimizer の param group は Noisy の σ の有無で 1〜2 つで、lr は共通。group 別の lr を設定する仕組みは無い（[dqn_based_agent.cpp:2769-2816](../../core/anet-core/src/dqn_based_agent.cpp)）。checkpoint の読み込みは group を添字で回し、設定から作った options を代入する（[:3497-3544](../../core/anet-core/src/dqn_based_agent.cpp)、[ADR 0053](../adr/0053-checkpoint-load-applies-current-config-to-optimizer-options.md)）。
- 目標値の割引は `pow(γ, n_steps)` で、`n_steps` は replay の実 n-step 数（[dqn_based_agent.cpp:3599](../../core/anet-core/src/dqn_based_agent.cpp)、[:3680](../../core/anet-core/src/dqn_based_agent.cpp)、[:3980](../../core/anet-core/src/dqn_based_agent.cpp)、[:4012](../../core/anet-core/src/dqn_based_agent.cpp)）。Munchausen の項は `MakeMunchausenTargetTerms` が作り、bonus は集約済み return の先頭へ 1 回、soft value は終端の bootstrap に入る（[:76-113](../../core/anet-core/src/dqn_based_agent.cpp)、[PRD067 D3](done/067_MunchausenRL_10prd.md)、[ADR 0035](../adr/0035-munchausen-target-learner-local-real-space.md)）。
- loss は要素ごとの loss に IS 重みを掛けて平均し、backward は 1 回。PER の優先度は |TD| + eps で、最適化の前に計算し後で反映する（[:4102-4109](../../core/anet-core/src/dqn_based_agent.cpp)、[:4020-4123](../../core/anet-core/src/dqn_based_agent.cpp)、[:299-314](../../core/anet-core/src/dqn_based_agent.cpp)）。優先度は 1 slot に 1 つ（[ADR 0011](../adr/0011-generation-aware-replay-item-key.md)）。
- 勾配を止める方式（detach）と optimizer 2 つの方式は、Adam の更新が各パラメータ自身の勾配の履歴で決まる一方、global norm クリップがパラメータ間をつなぐため、クリップを分けない限り同じ更新にならない。

### 2.3 Actor と Runner

- `MakeAction` は const で、可変の状態は mutable のメンバ（forward の時計、snapshot）とポインタ越しの部品（方策のスケジュール、stacker、RNG、NN 実行状態）が持つ。lane ごとの状態は lane 添字のテンソル（stacker の buffer、Noisy の保持ノイズ、spatial の ε・τ）で持ち、RNG は Actor に 1 つ（[dqn_based_agent.cpp:2046-2122](../../core/anet-core/src/dqn_based_agent.cpp)、[dqn_based_agent.hpp:840-842](../../core/anet-core/src/dqn_based_agent.hpp)）。`episode_start` は stacker の lane 別リセット、Noisy の hold=episode、診断の aux に使う。
- 全 lane の行数を前提にする読み手: step 診断（[:522-526](../../core/anet-core/src/dqn_based_agent.cpp)）、Actor Q ヒントの行数検査（[replay_buffer_impl.cpp:1335-1338](../../core/anet-core/src/replay_buffer_impl.cpp)）、margin 診断（[:1204-1273](../../core/anet-core/src/dqn_based_agent.cpp)）、spatial（[:1403-1418](../../core/anet-core/src/dqn_based_agent.cpp)）、Noisy の share=sample（[nn_modules.cpp:323-327](../../core/anet-core/src/nn_modules.cpp)）、stacker（書き込み位置は全 lane で 1 つ。[stacker.cpp:44-47](../../core/anet-core/src/stacker.cpp)）、replay の Push（[replay_buffer_impl.cpp:1318-1326](../../core/anet-core/src/replay_buffer_impl.cpp)）、QValuePanel。一部の lane だけ forward を省く口は無い。
- EvalPanel・QValuePanel・RunnerFrame の強制行動は、`MakeAction` のあとに `WithAction` で行動だけを差し替える。aux はそのままで、ヒントがあれば q_sa と bonus を取り直す（[dqn_based_agent.cpp:814-847](../../core/anet-core/src/dqn_based_agent.cpp)、[trainer.cpp:270-280](../../core/anet-core/src/trainer.cpp)、[EvalPanel.cpp:167-182](../../apps/runner/src/EvalPanel.cpp)）。
- 学習の 1 step は MakeAction、Env の Step、Push の順。Pipeline Runner では Push と学習が 1 step 遅れるだけで、MakeAction が Env より先に走ることは無い（[trainer.cpp:530-569](../../core/anet-core/src/trainer.cpp)、[:658-703](../../core/anet-core/src/trainer.cpp)）。本番の構成は DropMerge・Atari とも pipeline、Actor は clone_model。
- Evaluation Trace は毎 step の `$action_info` の列を読み、Observe が aux の `q_values` から q_sa と V を作る（[metrics.cpp:201-281](../../core/anet-core/src/metrics.cpp)、[dqn_based_agent.cpp:518-595](../../core/anet-core/src/dqn_based_agent.cpp)）。

### 2.4 ReplayBuffer

- n-step return は Push の時点で確定し、slot には `target_returns_`・`terminals_`・`actual_n_steps_` の結果だけが残る（[replay_buffer_impl.cpp:554-559](../../core/anet-core/src/replay_buffer_impl.cpp)）。1 step ごとの reward と done は n-step queue の一時値で、確定すると消える（[replay_buffer_impl.hpp:46-54](../../core/anet-core/src/replay_buffer_impl.hpp)、[replay_buffer_impl.cpp:169-173](../../core/anet-core/src/replay_buffer_impl.cpp)）。truncated は slot に書かれない（[:1500](../../core/anet-core/src/replay_buffer_impl.cpp)）。END を slot に持たないのは [ADR 0044](../adr/0044-replay-frame-history-start-from-episode-start-at-push.md) で意図して決めたこと。n_step=1 の構成（DropMerge の本番）では `target_returns_` と `terminals_` に 1 step の値が実質残り、truncated も次の dummy の印で分かるが、n ≥ 2（Atari の 3、LunarLander の 5）では残らない。
- truncated のときは次の slot に dummy を足して終端の観測を入れる。dummy の action は 0 で、DropMerge と Atari の NOOP と同じ値である（[:561-572](../../core/anet-core/src/replay_buffer_impl.cpp)、[:1386-1411](../../core/anet-core/src/replay_buffer_impl.cpp)）。
- ready range の未来側の遅れは max(1, n_step) で固定、容量の下限検査は max(1, n_step) + 1 + (stack_count − 1)（[:272-278](../../core/anet-core/src/replay_buffer_impl.cpp)、[:2313-2319](../../core/anet-core/src/replay_buffer_impl.cpp)、[ADR 0024](../adr/0024-replay-sampleable-range-excludes-overwritten-stack-history.md)）。
- `Sample(out, B, beta)` に長さを渡す引数は無く、prefetch は同じ引数で 1 つ先を読む（[rl.hpp:901](../../core/anet-core/include/anet/rl.hpp)、[replay_buffer.hpp:88-130](../../core/anet-core/include/anet/replay_buffer.hpp)）。`ExperienceSamples` は obs・actions・target_returns・次状態・terminals・n_steps・item key・IS 重み・優先度 source・info を持つ（[rl.hpp:825-845](../../core/anet-core/include/anet/rl.hpp)）。
- n-step を組む controller と builder は差し替えられる interface だが、factory が NStep と既定の builder に固定している（[replay_buffer_impl.hpp:76-101](../../core/anet-core/src/replay_buffer_impl.hpp)、[replay_buffer_impl.cpp:2338-2339](../../core/anet-core/src/replay_buffer_impl.cpp)）。次の観測はサンプルごとに `t + actual_n` から履歴開始の規則で stack を組む（[:1219-1249](../../core/anet-core/src/replay_buffer_impl.cpp)）。
- replay は checkpoint に入らない（[150 §7.6](../design/150_replay_buffer.jp.md#76-checkpoint)）。
- slot あたりの既存の負担は obs（Atari 7,056 B、DropMerge 2,700 B）のほか約 56 B。reward（float32）と終わり方（1 B）を足すと +5 B で、Atari +0.07%、DropMerge +0.18% である。

### 2.5 現行の設定と論文の値

- DropMerge の基準（ε ラダーを NoisyNet で置き換えた [run_20261008-015339_dmfb_21_nzrep](../experiments/default-dqn/dropmerge/config/run_20261008-015339_dmfb_21_nzrep.txt)）: n_step 1、γ 0.997、IQN（stratified 32）、Munchausen（target、α 0.9、τ 0.03）、TBO、PER（α 0.2、初期優先度 fixed 0.3）、batch 256、RR 1、256 lane、UQE τ 0.95→0.85 と spatial、NoisyNet σ0 0.5、行動は NOOP + DROP×58。
- Atari の BTR 基準（[run_20260927-035604_a5_qbert_btr50m_eps005_end0](../experiments/default-dqn/atari/config/run_20260927-035604_a5_qbert_btr50m_eps005_end0.txt)）: frame_skip 4、sticky 0.25、n_step 3、γ 0.997、IQN K=8、PER（max 初期化）、Munchausen、hard update 500、128 lane、ε 0.05→0 と spatial、NoisyNet 無し。
- LunarLander（[LunarLander.txt](../../apps/runner/config/LunarLander.txt)）: n_step 5、γ 0.99、IQN、Munchausen、UQE。暫定版が基準に挙げた `run_20260612-015116` はリポジトリに無い（§6 の 25）。
- GridMaze（[GridMaze.txt](../../apps/runner/config/GridMaze.txt)、[GridMazeEnv.hpp:26-31](../../core/envs/gridmaze1/src/GridMazeEnv.hpp)）: 5×5 の固定迷路、行動は上下左右の 4 つ（NOOP 無し）、goal +1・hole −1・壁 −0.1・step −0.01・上限 100 step、n_step 3、γ 0.995、QR、UQE。
- 論文の最大スキップ長 J: 表形式 7、MountainCar 10、LunarLander 4（連結構成で最良。改善は小さい）、Atari 10（qbert は最初の 1M step は遅れ、その後に追い越す）（[論文解説 4.2.2](../../reports/biedenkapp2021_temporl_paper_2026-10-03.md#422-特徴量ベースの環境における-dqnfeaturized-environments--dqn)、[4.2.3](../../reports/biedenkapp2021_temporl_paper_2026-10-03.md#423-atari-環境atari-environments)、[付録 E](../../reports/biedenkapp2021_temporl_paper_2026-10-03.md#付録-e-最大スキップ長の影響influence-of-the-maximum-skip-length)）。論文の 3 構成（連結・文脈化・重み共有）はどれも行動を入力に取る（[§3.3](../../reports/biedenkapp2021_temporl_paper_2026-10-03.md#33-深層強化学習でいつ意思決定するかの学習learning-when-to-make-decisions-in-deep-rl)）。スキップの目標値の終点の価値は行動価値で測る（[付録 B](../../reports/biedenkapp2021_temporl_paper_2026-10-03.md#付録-b-実装の詳細implementation-details)）。分布型・PER との組み合わせは論文に無い。

## 3. ゴール・非ゴール・成功の測り方

解く問題: DQN 系 Agent は 1 step ごとに行動を選び、「同じ行動を何 step 続けるか」を状態に応じて学ぶ仕組みが無い。

ゴール:

- DefaultDQN で、行動を選ぶ Q と、選んだ行動を条件にスキップ長 j ∈ {1..J} を選ぶスキップ価値を、設定で有効にできる。
- 無効（既定）では今と同じ結果になる（metrics checksum 一致）。
- 有効では判断の回数が減り、同じ step 数での学習の速さ（評価成績の立ち上がり）が落ちない、または上がる。GridMaze、LunarLander、Atari（qbert）の順に確かめ、DropMerge は最後に回す（§7）。

非ゴール（消える案と、それが本当の制約かどうか）:

- Env 側のマクロ行動（[086](086_dropmerge_macro_actions_10prd.md)）と、経過時間 τ を割引に通すこと（[999_smdp](999_smdp_transition_duration_10prd.md)）: 別の関心で、別 PRD。本 PRD の割引は k から γ^k を作る（§5.4）。τ は §5.1 の 1 step の記録に欄を足せば乗る。
- 次の判断まで遡る n-step（[999_decision_aligned_nstep](999_decision_aligned_nstep_10prd.md)）と n-step の目標値の補正（[999_nstep_target_correction](999_nstep_target_correction_10prd.md)）: 別の関心。本 PRD は窓の規則を差し替えられる土台だけを作る（§5.1）。
- NEET の抑制: 論文の形では行動を選ぶ Q は 1 step の TD のままで、静止した盤面での NOOP と最善 DROP の差 (1−γ)V ≈ 0.3%（γ 0.997）は変わらない。スキップ価値が決めるのは NOOP を選んだあとに何 step 続けるかだけである。NEET は 999_smdp・086・999_decision_aligned_nstep に任せる。制約ではなく、効かない見込みによる非ゴール。
- skip 中の lane の forward を省いて実時間を減らすこと: 全 lane の `q_values` を前提にする読み手（§2.3）の契約を変える別の関心。クリーンブレークで移せるが、Learner が律速の Atari では利きが小さい。再開条件: TempoRL が同じ step 数で効くと分かり、forward が律速の env がある。

成功の測り方（§5.7 の指標と §7 の判定）: `50_agent_skip/01_decision_rate` が無効時の 1.0 から下がること。評価成績の立ち上がり（AUC）が、同じ seed 数の無効の Run のブレ幅を超えて悪化しないこと。無効時の metrics checksum が一致すること。

## 4. 確定事項

各決定に理由を書き、ほかの決定や事実に依存するときはそれを名指す。

| # | 決定 | 理由と依存 |
|---|---|---|
| D1 | NEET の抑制は非ゴール。ゴールは判断の回数と、同じ step 数での学習の速さ | §3 の機序。J・続ける行動・受入の指標・最初の env はこの決定から決まる（§7） |
| D2 | 新しく作る機構は 6 つ: スキップ価値の Network（Head 含む）、Actor のスキップの実行、replay の可変長の区間サンプル、Learner のスキップの loss、スキップ長の探索、指標。skip 中の forward の省略は作らない | それぞれの寄与: 価値の本体、判断の回数を減らす実行、目標値の材料、学習、探索、ゴールの計測。forward の省略は §3 の非ゴール |
| D3 | NetworkModel は primary（今の Network）と、スキップ価値の Network（skip Network）の 2 つを持つ。skip Network も online と target の対を持つが、その target は Learner の bootstrap に使わず、target を元にする Actor（eval_target・EvalPanel）の行動にだけ使う。特徴空間・行動価値・スキップ価値を別 Network にする 3 分割と、1 つの Network に skip の branch と射影を足す形は採らない | 3 分割は checkpoint の object 数と、Network 1 つを前提にする診断（plasticity・churn・replay fit・GNS・weight norm・attention・Noisy σ）と GraphViz、呼び出し 31 か所とテスト 73 か所に波及し、特徴空間と行動価値の切れ目は primary の DAG の branch で表せる。1 つの Network は、包含型で Actor の forward が全 branch で 2 回になり、勾配を止めるブロックが要るうえ、`NetworkModel::Load` が module を丸ごと読むため無効の checkpoint から分岐できない。MuZero 試作が 3 つに分けるのは dynamics を再帰的に当てるためで、skip には当たらない。skip の target を持つのは、target 側の Actor が online の skip を遅れた z に当てる歪みを避け、DropMerge の eval_target と EvalPanel でスキップをそのまま測るため。[ADR 0062](../adr/0062-networkmodel-holds-auxiliary-network-fed-by-primary-outputs-with-target-used-only-for-acting.md) |
| D4 | 特徴量 z（説明上の記号。設定例の公開出力名は `skip_context`）の取り出しは、primary の `net.body.output.[鍵] = branch` に書いた鍵を Network の出力にも残すことで行う（Head の出力に Body の出力を合わせる。コピーは無く、dtype は Body のまま）。skip Network はその鍵を bind して読む。skip Network の入力 spec は、その bind が参照する鍵（primary の公開出力、obs の鍵、`action`、`taus`）だけから導く | 新しい記法を増やさない。全 obs の spec を渡すと未使用入力 WARN（[ADR 0018](../adr/0018-iqn-via-bind-product-dag.md)）が毎回出る。D3 に依存 |
| D5 | skip Head は分離型（全行動分のスキップ価値を出し、選んだ行動で絞る）を既定にし、包含型（行動を skip Network の入力に取り、その行動のスキップ価値だけを出す。論文の形）も設定で選べる。Learner と選択が読むスキップ価値は、どちらの形でも「選んだ行動の、スキップ長ごとの値」に統一する | 論文の主張を確かめるには論文の形が要り、行動数の多い env には分離型が向く。D3 により forward の回数はどちらも primary 1 回と skip 1 回。読み口を統一するので Learner と選択は 1 経路で済む |
| D6 | skip の loss の勾配を primary へ流すか（合算）止めるか（遮断）は設定 1 つで切り替える。optimizer は 1 つで、総 loss = L_Q + w·L_skip を 1 回 backward する。global norm クリップは primary と skip Network で別々に取る | 遮断時に Q 側が skip の勾配の影響を全く受けないようにするため（クリップを共有すると ‖g‖ に skip の勾配が入る）。Adam の更新は各パラメータ自身の勾配の履歴で決まるので、1 optimizer + Network 別クリップは optimizer 2 つと同じ更新になる。無効では group が primary だけで今と同一。optimizer 2 つは Save/Load・ADR 0053 の options 置換・GNS が 2 系統になるため採らない |
| D7 | スキップ価値の型は Q と同じ（TD ならスカラー、QR なら K 本の分位、IQN なら taus ごとの分位）。loss と bootstrap は既存の Learner の目標値の組み立てを、サンプルごとの k で流用する | UQE・Thompson がスキップ長の選択にも掛かる。D8 の Q 共有型で次元が揃う。既存の 3 Learner の目標値の関数を再利用できる。スカラーは §8 の先送り |
| D8 | skip Head の Dueling は無し（論文標準）を既定にし、Q 共有型（primary の Q(s,a) を切り離して基準にし、j=1 を引いた advantage を足す）も設定で選べる。独立 Dueling は作らない | スキップ長の選択は同じ行動の中で j を比べるので、Q(s,a) を足しても argmax は変わらず、違いは学習の目標値の尺度だけ。Q 共有型は尺度を Q が供給し、j=1 のスキップ価値が Q(s,a) に一致する。基準の Q(s,a) は常に切り離す（skip の loss が Q の目標値の定義を変えないため）。D7 に依存 |
| D9 | スキップ長の探索は、既定で行動の方策と同じ規則・同じ値（ε-greedy なら同じ ε で J から一様乱択、UQE なら skip の分位に同じ τ の risk score、Noisy は skip Network の Noisy、spatial の lane ラダーも同じ。乱数は行動と独立）。`actor.[key].skip.policy` を明示したときだけ専用の方策 | 論文は行動と skip に同じ ε のスケジュールを使う。評価 slot では skip も greedy になる。専用方策は policy_type の組み合わせと spatial の値を自分で持つ |
| D10 | EvalPanel などの強制行動（WithAction の事後上書き）と Actor の skip 状態のずれは許す。強制した step のあとも Actor は残りのスキップ長の間、元の行動を続ける | 事後 hook を Actor に足さない（決定点への入力を優先する）。[999_observation_pipeline](999_observation_pipeline_10prd.md) の単位 A（強制行動を MakeAction の入力にする）が入ったら、強制された lane の skip 状態を捨てて新しい判断にする（§9） |
| D11 | replay は slot に 1 step の reward（scale 後）と終わり方（なし／done／truncated）を残し、サンプル時に起点から次の判断の手前まで（上限 J）を走査して k_max を得て、割引報酬と k 先の観測を組む。sampleable の列挙の遅れを max(1, n_step, J) に広げる（PER の初期優先度の完成と eviction 統計は n_step の遅れのまま）。論文どおりの skip 用の別 replay と、Push 時に区間を確定して slot に足す形は採らない | 別 replay は PER・prefetch・ready・世代 key が 2 系統になり obs を複製するとメモリ 2 倍。Push 時の確定は同じ起点から k_max より短い sub-skip を取れない。+5 B/slot。他の 2 本も同じ記録に欄を足し、窓の規則を差し替えることで乗る。[ADR 0063](../adr/0063-replay-slot-keeps-per-step-reward-and-end-and-samples-variable-windows.md)、[ADR 0044](../adr/0044-replay-frame-history-start-from-episode-start-at-push.md) の END の部分を改める |
| D12 | 可変長の区間サンプルの土台（1 step の記録と、窓の規則を差し替えられる抽出）は、本 PRD の中に節を分けて書く（§5.1）。他の 2 本の窓の規則は作らない | 他の 2 本の要件が pin されていない。節と ADR で分離を保ち、グリルは 1 回で済ませる |
| D13 | スキップの窓は判断で決めた区間の中だけから作る（論文の接続と同じ）。Actor が出す判断の旗を 1 step の記録に残し、窓は次の判断で止まる。サンプルは 1 点 k で、起点の k_max 以下から k を 1 つ一様に引く。Sample の出力に k・窓の割引報酬・終端・k 先の観測を足す | 判断の境界を越えて同じ行動を数えると、境界の先で同じ行動を選び直すかが次の状態に依存するため、確率的な env では長い窓に「選び直したくなる状態」だけが入り、長い j が過大になる。判断で区切れば k_max は判断の時点で決まり、偏りは終端の打ち切りの分だけ。全 k を mask 付きで一括すると target forward が J 倍（Atari で 11 倍）。k を一様に引くので短い k は論文の「接続ごとに等しい重み」より多く入る（重み付けは §8 の先送り） |
| D14 | PER の優先度は Q の TD だけを既定（λ=0）とし、合成 \|δ_Q\| + λ\|δ_skip\| の λ も設定で選べる。IS 重みは Q と skip の両方の loss に同じ値を掛ける。Q と skip で独立にサンプルしない | λ=0 なら遮断モードで Q 側のサンプル分布が今とほぼ同じになる（sampleable が最新の J−n step 分だけ狭まる）。独立サンプルは forward と抽出が 2 倍。IS 重みはサンプルの確率から決まるので両方に掛ける |
| D15 | 用語は「スキップ（skip）」。スキップ長 j、最大スキップ長 J、スキップ価値（skip Q）、判断（decision）。設定キーは `skip`、クラス名は `Skip…`、メトリクスの群は `50_agent_skip` | 論文と wiki の語。Env の frame_skip（1 agent step の中のフレーム反復）とは層が違うことを用語集で定義し、action repeat・行動反復を Avoid にする（[CONTEXT.md](../../CONTEXT.md)） |
| D16 | 指標は群 `50_agent_skip` の 9 つ（§5.7）。Evaluation Trace に `skip_decision`・`skip_length` の列を足せる | 判断の回数（ゴール）と skip の学習の両方を scalar で追う。[999_replay_eviction_sample_count](999_replay_eviction_sample_count_10prd.md) の群番号は 59 へ移した（53〜58 は評価の群と他の 999 の予約で使用中） |
| D17 | 検証は段 0（等価性）→ 段 1 GridMaze（6×10）J=7 → 段 2 LunarLander J=4 → 段 3 qbert J=10 → 段 4 DropMerge（基準の checkpoint から分岐）。受入は §7 | 論文の J（表形式 7、LunarLander 4、Atari 10）と結果の型。DropMerge は判定に 50M 以上かかるので最後 |
| D18 | 細部は §6 の推奨の既定で埋める | グリルで「推奨で埋める」を選んだ |

## 5. 実装契約

確定事項と本節には実装の語（内部の型、入れ物の型、添字の決め方）を書かない。公開 API 名・設定キー・メトリクス名・クラス一覧の名前は書く。

### 5.1 replay: 1 step の記録と、窓の規則を差し替えられる抽出

この節は TempoRL に限らない土台で、[999_decision_aligned_nstep](999_decision_aligned_nstep_10prd.md) と [999_nstep_target_correction](999_nstep_target_correction_10prd.md) も同じ土台に乗る。

- `DefaultReplayBuffer` は Push のとき、slot ごとに 1 step の記録として「その遷移の reward（Agent が渡す scale 後の値）」「終わり方（なし／done／truncated）」「判断の旗（Actor が `BatchActionInfo` の info の `decision` で渡す。鍵が無ければその step を判断とみなす。鍵の名前は汎用で、skip の名を共通層に持ち込まない）」を保存する。skip の有効・無効によらず常に書く。Push時に確定する既存のn-stepの割引報酬・終端判定・実際のstep数と、履歴開始は変えない。終わり方と判断の有無は組み合わせを合わせて1 byteで記録し、rewardの4 byteと合わせた増分はslotあたり5 byteとする。
- 終わり方は、その遷移で lane の episode が終わったかどうかと、終わり方が done か truncated かを表す。truncated の次の dummy slot は、今までどおり終端の観測を入れるためだけに置き、窓の起点にも途中にもならない（dummy は sampleable から除外されている）。
- 抽出は「窓の規則」を通して、起点 t ごとに窓の長さ k を決める。窓の規則は、起点と slot の 1 step の記録（action・終わり方・判断の旗）、長さの上限を読み、起点から取れる最大の長さ k_max を返す。本 PRD が作る規則は §5.2 の「判断で決めた区間の中の窓」だけで、規則は差し替えられる形にする（n-step の controller と builder が interface を持つのと同じ）。固定の n で終端で切る今の n-step も、この窓の規則の 1 つとして書ける形にする（Push 時の n-step を置き換えるのは §10）。他の PRD が別の規則を足すときは、1 step の記録に必要な欄（経過時間、ランダムな手の旗、log π など）を足して同じ抽出に乗せる。
- 窓から組む値は、窓の中の reward の割引和（γ^m を m=0..k−1 で掛けた和。γ は Learner の γ）、窓の終端の終わり方、窓の k 先の観測（起点 t+k の観測を、今の `next_obs` と同じ履歴開始の規則で stack したもの。truncated なら dummy の観測）。
- sampleable の列挙（Sample、一様の非復元、抽選履歴の probe、sampleable の件数）の未来側の遅れを max(1, n_step, 窓の上限) にする。PER の初期優先度の完成（bootstrap のヒントが t + actual_n の step で届く検査）と eviction 統計は、今の max(1, n_step) の遅れのまま変えない。完成が sampleable より先に済むので「sampleable なのに優先度が未完成」は起きない。窓の規則と上限は Learner が構築時に渡す。窓の規則が渡されないとき（DefaultDQN では skip が無効のとき）、抽出は窓を組まず、Sample の窓の欄は空で、遅れは今と同じ max(1, n_step)。容量の下限検査は窓の上限を含む値で行う。[ADR 0024](../adr/0024-replay-sampleable-range-excludes-overwritten-stack-history.md) の「sampleable は ready に history margin を適用した集合」に、未来側の窓の margin を足す拡張であり、history margin の除外と、完成と eviction 統計を ready に置く判断は保つ。[CONTEXT.md](../../CONTEXT.md) の「sampleable range」の定義はこれに合わせて直す。
- 抽出側は、k が k_max を超えない、窓が終わり方の無い slot の先へ出ない、窓の終端より先の slot を読まない、を検査し、違反は `ANET_SYSTEM_ERROR`。
- sampling 分布、item key、PER の completion と eviction 統計の ready 基準、Push と Extract の lock、index の昇順は変えない（[done/078](done/078_replay_frame_history_integrity_10prd.md) §6.2 の条件）。

### 5.2 TempoRL の窓の規則（判断で決めた区間の中の窓）

- k_max は、起点 t から次の判断の旗の手前まで（判断で決めた区間の中）の長さで、上限 J、終わり方が done か truncated の遷移を含めたらそこで止まる。すなわち k_max = max { k ≤ J : t+1..t+k−1 に判断の旗が無く、t..t+k−2 の終わり方がすべて「なし」}。区間の中の action は同じ判断で決めた行動なので一致する（抽出はこれを検査し、違えば `ANET_SYSTEM_ERROR`）。k_max は判断の時点で決まるので、次の判断で同じ行動を選び直すかに依存しない。
- サンプルごとに、k を 1..k_max から一様に 1 つ引く（D13）。乱数は replay の sampler の RNG を使う。
- 判断の中で選んだ j が探索（ε や Noisy）によるものかは区別しない。スキップ価値の定義は「a を k step 続けたあと行動価値に従う」で、判断で決めた区間の中の遷移の分布は j を選んだ理由によらない（[論文解説 式 (4)](../../reports/biedenkapp2021_temporl_paper_2026-10-03.md#32-いつ意思決定するかの学習learning-when-to-make-decisions)）。Atari の sticky actions で保存した行動と実行した行動が違うのは、今の 1 step の学習と同じく Env の確率性として扱う（[CONTEXT.md](../../CONTEXT.md) の「sticky actions」）。
- Sample の出力は、今の欄に加えて、窓の欄として「k」「窓の割引報酬」「窓の終端の終わり方（done なら bootstrap 無し）」「k 先の観測」を持つ。欄の名前は窓（window）で、スキップの名を共通層に持ち込まない。窓の規則が渡されていなければ窓の欄は空（§5.1）。

### 5.3 NetworkModel と skip Network

- `NetworkModel` は primary（online と target。今と同じ）に加えて、skip が有効なら skip Network（online と target）を持つ。構築は primary を先に行い、その dummy forward の出力から skip Network の入力 spec を導く（D4）。target の複製と同期（hard／soft）は primary と skip に同じ時点で掛ける。skip の target は Learner の bootstrap には使わず、target を元にする Actor の行動にだけ使う（D3）。
- Actor の複製（clone_model）は skip Network も primary と同じ時点・同じロックで複製し、定期同期（sync_interval）も同時に行う。共有の Network を使う Actor は同じ shared lock の下で skip Network も読む。target を元にする Actor は primary の target と skip の target の組を使う。Rainbow は同じ Actor 実装を使うが、skip が無効の経路は今と同一。
- primary の `Network::Forward` は、Head の出力に `net.body.output` の鍵をすべて合わせて返す。既存の読み手は固定の鍵しか読まないので影響しない。コピーは作らず、dtype は Body の出力のまま（AMP なら BF16。Head に渡す FP32 化は今までどおり）。
- skip Network の構造は `DefaultDQNAgent.skip.net.*` に、primary の `net.*` と同じ記法（block・branch・bind・`body.output`）で書く。bind できる鍵は primary の公開出力（`net.body.output` の鍵）、obs の鍵、`action`（包含型のとき。離散 spec で one-hot は自動）、`taus`（IQN のとき。primary と同じ値）。特徴を共有しない構成は、skip Network が obs の鍵を bind して自前の抽出層を持つことで表す。包含型で IQN の融合済み特徴（[B,K,F]）と `action` の one-hot（[B,A]）を `,` で連結するときは、分位ごとに同じ行動のone-hotを共有し、分位の軸を保って末尾のfeature軸で連結する（[130](../design/130_neural_networks.jp.md) の bind の規則。2026-10-24 の設定見直しで追加。`bind_concat_dim` は既存のキー）。
- skip Head は `features` を読み、スキップ価値を出す。出力の鍵は `skip_q`（分離型 [B,A,J]、包含型 [B,J]）で、分布型（QR・IQN）はこれに加えて `skip_q_dist`（分離型 [B,A,J,K]、包含型 [B,J,K]）を出し、`skip_q` はその分位平均になる（primary の `q` と `q_dist` と同じ形）。Actor と Learner の読み口は `skip_q_dist` があればそれを使い、選択と目標値は分位で行う。GraphViz の dot には両方の出力が出る（IQN の K は primary と同じく −1 で示す）。
  - 分離型（既定）: 全行動分のスキップ価値（行動ごと・スキップ長ごと。分布型は分位ごと）を出し、Learner と Actor が選んだ行動で絞る。射影の幅は行動数×J。
  - 包含型（`skip.network_mode = action_in`）: 行動を入力に取り、スキップ長ごとの値だけを出す。射影の幅は J。状態と行動の交差項は skip Network の Body（bind の `*` か後段の MLP）が持ち、Head の射影だけでは持たない（IQN の fusion と同じく設定者の責任）。
  - 型は Q と同じ（D7）: TD ならスカラー、QR なら K 本の分位、IQN なら taus ごとの分位（IQN Head と同じく分位ごとに同じ射影を当てる）。IQN では、taus 込みの特徴（primary の fusion）を `skip_context` に取るか、`taus` を bind するかのどちらか（既定は前者。§6 の 21）。Head は iqn で rank 3 の特徴を要求し、違えば構築時に `ANET_SYSTEM_ERROR`（taus の bind 自体は要求しない）。
  - Dueling（`skip.head.use_dueling`、bool）: `false`（既定）は射影 1 本でスキップ価値を直接出す（分解しない）。`true` は射影 1 本で advantage を出し、primary の Q(s,a)（選んだ行動の値。勾配は切り離す）を V として、j=1 の advantage を引いた advantage を足してスキップ価値にする。分布型では分位ごとに同じ計算をする。V の取り方はこの 1 形だけなのでキーは作らない（自前の V を出す形は §8 の先送り。足すときにキーを足す）。
- skip Head の射影の構築設定は `DefaultDQNAgent.skip.head.projection.*`、初期化は `DefaultDQNAgent.skip.head.init.*`（それぞれ primary の `head.projection.*`・`head.init.*` と同じ項目。省略時は primary と同じ値。2026-10-24 の設定見直しで `head_projection`・`head_init` から改名）。Noisy の有無は skip Network の block と射影の設定で決まる。
- skip Network は、Actor（online 側・target 側）と Learner の current の用途ごとに NN 実行状態の箱を持ち、`nn_forward_key` は primary の同じ用途と同じキーを参照する（[ADR 0048](../adr/0048-noisynet-epsilon-in-caller-execution-state-and-nn-forward-key.md)、[ADR 0051](../adr/0051-nn-forward-prepares-concrete-calls-and-purpose-owned-execution.md) の規則）。Learner の target 構築の用途では skip を forward しない。
- GraphViz は skip Network を別のグラフとして出す。可視化の TensorDictFunction の `skip-net.` prefix は、入力（obs を primary に通してから渡す形）を決める読み手が無いので §8 の先送り。
- primary の DAG に勾配を止めるブロックは足さない。遮断は Learner が z を skip Network に渡す seam で行う（§5.4）。

### 5.4 Learner: スキップの目標値・loss・クリップ・PER

- current の forward: primary（train mode）を今までどおり forward し、その出力の z（と包含型ならサンプルの行動）で skip Network を forward する。`learner.skip.feature_gradient = stop` なら z を切り離してから渡し、`merge` ならそのまま渡す。
- スキップの目標値: サンプルの k・窓の割引報酬 R・終わり方・k 先の観測から、R + γ^k·V(s_{t+k}) を作る。done なら bootstrap 無し。V は Q の bootstrap と同じ関数で、Q と同じ target 方策に従う（Double DQN なら online の argmax と target の値、Munchausen なら soft value、`use_optimistic_target` なら risk score）。Munchausen の bonus は起点に 1 回、soft value は終端に入れる（[PRD067 D3](done/067_MunchausenRL_10prd.md) と同じ形）。TBO は Q と同じく実空間で組んで最後に h を掛ける。分布型では Q の分布の目標値の組み立てをそのまま使う。
- target の forward: Qのbootstrapに用いる観測s_{t+n}と、skipのbootstrapに用いる観測s_{t+k}をprimaryのtargetで評価する。Munchausenのtarget modeで必要な既存の評価も維持し、Double DQNが有効ならs_{t+k}の行動選択にonlineの評価を使う。skipのtargetはLearnerではforwardしない。
- loss: L_skip はサンプルごとの loss（スカラーなら Huber、分布型なら分位 Huber。Q の loss と同じ定義）に IS 重みを掛けて平均したもの。総 loss = L_Q + `learner.skip.loss_weight`·L_skip を 1 回 backward する。
- クリップ: `max_grad_norm` の global norm クリップを、primary の全パラメータと skip Network の全パラメータで別々に行う（閾値は同じ）。skip が無効なら今と同一。
- optimizer: 1 つ。param group は primary（通常、Noisy の σ）に skip Network（通常、σ）を足す。lr・weight decay は primary と同じ値。
- 精度: skip が有効なとき、学習の精度が FP16 の AMP なら構築時に `ANET_SYSTEM_ERROR`（GradScaler が inf を optimizer 全体で判定し、skip の勾配の inf で primary の更新も飛ぶため、遮断の分離が崩れる）。BF16 と AMP 無しは今までどおり。
- GNS（勾配ノイズスケール）は今までどおり Q の loss と primary のパラメータで測り、skip の loss を含めない。
- PER: 優先度は \|δ_Q\| + `learner.skip.priority_lambda`·\|δ_skip\| + eps（既存の clip も同じ）。δ_skip は Q の TD と同じ定義（TD は値の差、分布型は分位平均の差）。更新は今と同じ 1 回。
- 指標: skip の loss・\|δ_skip\| の平均・サンプルした k の平均・k_max の平均・skip Network の勾配ノルム（クリップ前）を `BatchUpdateResult` に載せる（§5.7）。
- skip が無効のとき、Learner の forward の回数・順序・RNG の消費・loss・優先度は今と同一。

### 5.5 Actor: スキップの実行と探索

- Actor は lane ごとに「続けている行動」と「残りの step 数」を私有の状態として持つ。`episode_start` の lane は状態を捨てる（残り 0）。checkpoint には入れない。
- 毎 step、全 lane で primary を forward し、行動の方策が全 lane の行動を選ぶ（今と同じ。skip 中の lane も forward する）。方策は forward の出力のうち z（と IQN の taus）を行動と一緒に Actor へ返し、Actor はそれを skip Network の入力にする（ActionPolicy は Rainbow と共有するが、skip が無効なら返した z は使わない）。
- Actor が使う skip Network は、行動に使う primary と同じ側（online の Actor は online の skip、target の Actor は target の skip。D3）。複製と同期は §5.3。
- 判断（残り 0 の lane）: 選んだ行動 a を条件にスキップ長 j を選び、a を j step 続ける状態にする（残り j−1）。判断でない lane（残り > 0）: 方策が選んだ行動を捨て、続けている行動で上書きし、残りを 1 減らす。
- スキップ長の選択: skip Network を 1 回 forward し（分離型は a で絞る、包含型は a を入力にする）、選んだ行動のスキップ長ごとの値から j を選ぶ。選ぶ規則は D9（既定は行動の方策と同じ規則・同じ現在値で、乱数は独立。`actor.[key].skip.policy` があればそれ）。判断でない lane の分も同じ forward に含まれるが使わない。
- 判断の順は論文の Algorithm 1 と同じ: a を選び、次に a を条件に j を選ぶ（[論文解説 付録 B](../../reports/biedenkapp2021_temporl_paper_2026-10-03.md#付録-b-実装の詳細implementation-details)）。
- `BatchActionInfo` の info に `decision`（判断した lane か。replay が 1 step の記録の判断の旗として保存する）を足し、aux に `skip_decision`、`skip_length`（その lane の今のスキップ長 j。判断でない lane は続けている skip の j）、`skip_q_values`（選んだ行動のスキップ長ごとの値。分布型は平均）を足す。`q_values` などの既存の aux は forward の値のまま。
- Actor Q ヒント（actor_approx）は、実行した行動（上書き後）で q_sa と bonus を取り直す（`WithAction` の再 gather と同じ）。
- 強制行動（WithAction）は Actor の状態を変えない（D10）。
- 評価 slot の Actor も同じ Actor 実装なので、スキップを使う（方策が greedy なら skip も greedy）。
- skip が無効のとき、Actor の forward の回数・行動・aux・RNG は今と同一。

### 5.6 設定キー

すべて `DefaultDQNAgent.` の下。不正値はすべて Agent の構築時に、キーと値と期待値を示して `ANET_SYSTEM_ERROR` で止める（値を丸めない、既定へ置き換えない）。

| キー | 型・値域 | 既定 | 意味 |
|---|---|---|---|
| `skip.enabled` | bool | false | スキップを有効にする。false では以下のキーを検証だけして使わず、動作は今と同一 |
| `skip.max_length` | 整数、1 以上、かつ replay の sampleable が構造的に 0 件にならない値 | 無し（enabled なら必須。各 env が既定として置く。実装の既定も「未指定」で、1 のような有効な値を既定にしない） | 最大スキップ長 J。1 は判断が毎 step になる退化形で、明示なら許す |
| `skip.network_mode` | `action_out` ／ `action_in` | `action_out` | `action_out` は分離型（全行動分の値を出し実行行動で絞る）、`action_in` は包含型（行動を入力に取る）。`action_in` なのに `action` を bind する branch が無い、`action_out` なのに bind している、は構築時にエラー |
| `skip.head.use_dueling` | bool | false | skip Head で Dueling を使うか。true のとき V は primary の Q(s,a)（§5.3） |
| `skip.net.*` | primary の `net.*` と同じ記法 | agent.txt の `@baseline` が既定の配線を持つ（§6 の 21） | skip Network の構造。bind の鍵が解決できなければエラー |
| `skip.head.projection.*` | `head.projection.*` と同じ項目 | primary と同じ値 | skip Head の射影の構築設定（Noisy を含む） |
| `skip.head.init.*` | `head.init.*` と同じ項目 | primary と同じ値 | skip Head の射影の初期化 |
| `learner.skip.feature_gradient` | `stop` ／ `merge` | `stop` | skip の loss の勾配を primary へ流すか（遮断／合算） |
| `learner.skip.loss_weight` | 有限、0 以上 | 1.0 | 総 loss の skip の重み w |
| `learner.skip.priority_lambda` | 有限、0 以上 | 0 | PER の優先度の skip の誤差の重み λ |
| `actor.[key].skip.policy.*` | `actor.[key].policy.*` と同じ項目 | 無し（省略時は行動の方策と同じ規則・値） | スキップ長の専用の方策 |

replay に新しいキーは無い（窓の上限は `skip.max_length` から Learner が渡す）。skip が有効で学習の精度が FP16 の AMP の構成は、構築時に両方のキーと値を示して `ANET_SYSTEM_ERROR`（§5.4）。

設定例（コロン記法）:

```
# agent.txt: 無効のまま、おすすめの既定をべた書き（?=）
DefaultDQNAgent.@baseline : skip.enabled ?= false
DefaultDQNAgent.@baseline : net.body.output.[skip_context] ?= main_feature
DefaultDQNAgent.@baseline : skip.net.branch.[skip_feature].bind ?= skip_context
DefaultDQNAgent.@baseline : skip.net.branch.[skip_feature].structure ?= SkipFC1 > SkipReLU > SkipFC2 > SkipReLU
DefaultDQNAgent.@baseline : skip.net.body.output.[features] ?= skip_feature
DefaultDQNAgent.@baseline : skip.network_mode ?= action_out

# GridMaze.txt: env の既定と、skip を使う Run の profile
DefaultDQNAgent.skip.max_length ?= 7
run.@skip : DefaultDQNAgent.skip.enabled = true
run.@skip : metrics.scalar.$ = metrics.scalar.@baseline > metrics.scalar.@skip

# IQN の env（LunarLander.txt）: 融合済みの特徴を skip_context にする
DefaultDQNAgent.net.body.output.[skip_context] = fusion

# CLI・bat: 変種は葉で
DefaultDQNAgent.skip.network_mode=action_in "DefaultDQNAgent.skip.net.branch.[skip_feature].bind=skip_context, action"
```

Agent側の`DefaultDQNAgent.@skip*`や`DefaultDQNAgent.skip.net.@*`のプロファイルは作らない（2026-10-24の設定見直し。ベースはagent.txt、envとCLIは部分上書き）。Run側の`run.@skip`と指標側の`metrics.scalar.@skip`は、有効化と表示項目を選ぶために用意する。LunarLanderやDropMergeの連鎖にある`M1`は、そのenvファイルの`M1.[tag]`で定義する追加metricsの設定群であり、env固有の指標を連鎖に残すために重ねる。上のGridMaze例ではM1の葉がないため含めない。チェーンへ項を足す記法は無く、`X.$ += …`と書くと`X.$+`という読まれない葉になる。

IQN の既定は、primary の融合済み branch（LunarLander は `fusion`、Atari・DropMerge は `iqn_fusion`）を `net.body.output.[skip_context]` で公開し、skip 側の `skip_feature` が `skip_context` を bind する。skip 側で taus を融合する構成を選ぶ場合は、primary の融合前の特徴を公開した鍵と `taus` を bind し、IQN Head が要求する [B,K,F] の特徴を作る（§5.3）。

### 5.7 メトリクスと trace

群は `50_agent_skip`。tag は 40 文字以内。Actor 側は `$action_info` の aux から lane 平均で作り、Learner 側は `$update_result` の鍵。

| tag | 元の鍵 | 意味 |
|---|---|---|
| `50_agent_skip/01_decision_rate` | `$action_info skip_decision_rate` | その step で判断した lane の割合。1／平均スキップ長の近似。ゴールの指標 |
| `50_agent_skip/02_skip_length_mean` | `$action_info skip_length_mean` | 判断した lane が選んだ j の平均 |
| `50_agent_skip/03_skip_length_rate.[j]` | `$action_info skip_length_rate.[j]` | 判断した lane のうち j を選んだ割合（添字付き。J を選び続ける病理の監視） |
| `50_agent_skip/04_skip_q_gap` | `$action_info skip_q_gap` | 選んだ j のスキップ価値と j=1 のスキップ価値の差の平均 |
| `50_agent_skip/11_loss` | `$update_result skip_loss` | skip の loss（IS 重み込みの平均） |
| `50_agent_skip/12_td_abs_mean` | `$update_result skip_td_abs_mean` | \|δ_skip\| の平均 |
| `50_agent_skip/13_sample_k_mean` | `$update_result skip_sample_k_mean` | サンプルした k の平均 |
| `50_agent_skip/14_kmax_mean` | `$update_result skip_kmax_mean` | 起点の k_max の平均（replay 上で同じ行動がどれだけ続いているか） |
| `50_agent_skip/15_grad_norm` | `$update_result skip_grad_norm` | skip Network の勾配ノルム（クリップ前） |

Evaluation Trace の列は `$action_info skip_decision`、`$action_info skip_length`、`$action_info skip_q.[j]`（その lane の j 番目のスキップ価値。j は 1..J で、範囲外の j は列の値を読むとき（Trace の行を作るとき）に `ANET_SYSTEM_ERROR`、skip 無効は NaN）を宣言できる（[094](094_evalpanel_step_history_10prd.md) の列の宣言）。既定の列には入れない。学習側の per-sample の dump は足さない。

### 5.8 checkpoint

- skip が有効な Run の Save は、primary の online・target に skip Network の online と target を加えて書き（4 object）、archive に skip の有無が分かる印を持つ。optimizer の状態は param group の役割（primary の通常・σ、skip の通常・σ）ごとに書く。
- Load は設定が要求するものを読む。設定が有効で archive に skip があれば全部読む。設定が有効で archive に skip が無ければ（無効の Run の checkpoint）、primary と optimizer の primary の役割の group を読み、skip Network と skip の group は初期化して WARN を 1 回出す（基準の Run から skip を有効にして分岐する流儀を §7 の段 4 で使う）。設定が無効で archive に skip があれば `ANET_SYSTEM_ERROR`（黙って読み飛ばさない）。
- optimizer の options の置換（[ADR 0053](../adr/0053-checkpoint-load-applies-current-config-to-optimizer-options.md)）は group の役割ごとに行う（添字ではなく役割で対応付ける）。
- skip が無効の Run の Save/Load は今と同一の形。
- Actor の skip 状態と replay は checkpoint に入らない（replay は今も入らない。[150 §7.6](../design/150_replay_buffer.jp.md#76-checkpoint)）。

### 5.9 移行対象と等価性

契約を変えるもの（クリーンブレーク。互換層は置かない）:

- `Network::Forward` の出力に Body の公開出力が加わる（既存の読み手は固定の鍵で読むので変更不要。テストで出力の鍵を数えるものがあれば直す）。
- `DefaultReplayBuffer` の slot の配列に 1 step の記録が加わる。`ExperienceSamples` に窓の欄が加わる。sampleable の列挙の遅れと容量検査の式が窓の上限を含む（ready range は変えない）。replay のテストはこの契約で書き直す。
- `NetworkModel` の Save/Load の形（skip 有効時の 4 object と印、optimizer の group の役割）。無効時は今と同一。
- `BatchUpdateResult` と `DQNActionInfo::GetScalar` に skip の鍵が加わる。
- 設計文書: [130](../design/130_neural_networks.jp.md)（Network の出力）、[150](../design/150_replay_buffer.jp.md)（1 step の記録、窓の規則、sampleable の列挙の遅れ）、[200](../design/200_dqn_agents.jp.md)（NetworkModel、skip Head、Learner、Actor、設定、metric）、[110](../design/110_agents_and_learning.jp.md)（Actor の私有状態）、[140](../design/140_observability.jp.md)（trace の列）。
- 過去の Run artifact・ADR・実験記録は書き換えない。無効の Run の checkpoint は今までどおり読める。

等価性: skip が無効のとき、forward の回数・順序、RNG の消費、replay の sampling 分布、loss、優先度、checkpoint の形は今と同一で、metrics checksum が一致する（§7 段 0）。1 step の記録は無効でも書くが、窓の規則が渡されないので窓の欄は空で、サンプルの値には使わない。

### 5.10 性能の見積もり（ホットパス）

| 経路 | 頻度 | 単価の増分 | 見込み |
|---|---|---|---|
| Learner の 1 回の更新 | 1 回／更新 | target の forward が B 行増える（+1 単位。current 1、target 1〜2、backward 2 の 4〜5 単位に対し）。skip Network の forward と backward は MLP 分 | 学習 1 step で約 2 割増。Learn が compute-bound の構成では実時間に出る |
| 抽出（Sample） | B 件／更新 | 起点ごとに最大 J 回の走査と加算 | B=256・J=10 で数千演算。forward に比べ無視できる |
| 抽出（k 先の観測の stack） | B 件／更新 | obs・next_obs に続く 3 つ目の stack の組み立てと転送（Atari で 1 件 4×7 KB） | ホスト側の組み立てと転送が今の約 1.5 倍。prefetch で隠れるかは段 0 の性能の計測で確かめる |
| Learner の 1 回の更新（Double DQN 有効時） | 1 回／更新 | s_{t+k} の online の forward が B 行増える | GridMaze の基準構成で効く。Munchausen の本番構成では Double DQN が無効 |
| Actor の 1 step | 1 回／step | skip Network の forward 1 回（MLP、全 lane） | primary の forward に比べ小さい |
| replay のメモリ | slot ごと | +5 B | Atari +0.07%、DropMerge +0.18% |

受入は §7（学習 1 step の実時間が無効の 1.3 倍以内）。

### 5.11 クラス一覧

従えない事情があれば、実装する側が理由を実装計画に書いて変えてよい（質問しない）。

| 名前 | 区分 | 責務 | 置き場（機能グループ） |
|---|---|---|---|
| `SkipConfig` | 新規 | `skip.*` の設定の読み取りと検証（enabled、max_length、network_mode、head、net、head.projection） | DefaultDQN（default_dqn_agent） |
| `SkipHead` ／ `SkipHeadFactory` | 新規 | skip Network の Head。分離型／包含型の射影の幅、型（TD/QR/IQN）、Dueling（`use_dueling`）の advantage の出力 | DQN Head（dqn_based_heads） |
| `NetworkModel` | 変更 | skip Network（online と target）の保持・構築順・入力 spec の導出・同期・Save/Load（印、役割ごとの group）・パラメータの列挙 | DQN 共通 component（dqn_based_agent） |
| `Network` | 変更 | Forward の出力に `net.body.output` の鍵を残す | NN（nn_impl） |
| `SkipValueReader`（実装: `ReadSkipValues`） | 新規 | skip Network の出力を「選んだ行動のスキップ長ごとの値」に統一して返す（分離型は絞る、包含型はそのまま。Dueling（`use_dueling`）の合成を含む） | DQN 共通 component（dqn_based_agent） |
| `SkipLengthSelector`（実装: `ActionPolicy::SelectFromValues / MakeValueQueryTaus`） | 新規 | 行動の方策の規則と現在値（または専用方策）で j を選ぶ。乱数は独立 | DQN 共通 component（dqn_based_agent） |
| `ActionPolicy`（ε-greedy・UQE・Thompson） | 変更 | forward の出力の z と taus を行動と一緒に返す（Rainbow と共有。skip が無効なら使われない） | DQN 共通 component（dqn_based_agent） |
| `dqn::Actor` | 変更 | lane ごとの skip 状態、上書き、skip Network の forward・複製・同期（primary と同時）、info と aux の追加、ヒントの再 gather | DQN 共通 component（dqn_based_agent） |
| `SkipTargetBuilder`（実装: `Learner::MakeLearningTargets`） | 新規 | サンプルの k・割引報酬・終わり方・k 先の観測と target の出力から、Q と同じ target 方策・Munchausen・TBO の規則でスキップの目標値を組む（3 Learner の目標値の関数を流用） | DQN 共通 component（dqn_based_agent） |
| `dqn::Learner`、`TDLearner`、`QRLearner`、`IQNLearner` | 変更 | skip Network の forward と seam の切り離し、skip loss の合算、Network 別クリップ、PER の λ、指標、optimizer の group | DQN 共通 component（dqn_based_agent） |
| `DQNActionInfo` | 変更 | aux の `skip_*` と GetScalar の skip の鍵 | DQN 共通 component（dqn_based_agent） |
| `BatchUpdateResult` | 変更 | skip の指標 5 つ | RL 共通（rl） |
| `ExperienceSamples` | 変更 | 窓の欄（k、割引報酬、終わり方、k 先の観測。名前は window） | RL 共通（rl） |
| `SampleWindowRule` | 新規 | 窓の規則の interface（起点と 1 step の記録から k_max を返す。上限を持つ） | ReplayBuffer（replay_buffer_impl） |
| `DecisionWindowRule` | 新規 | TempoRL の規則（判断の旗で区切る窓。上限 J、終わり方で止まる、区間の中の action の一致を検査） | ReplayBuffer（replay_buffer_impl） |
| `DefaultReplayBuffer` | 変更 | slot の 1 step の記録（reward・終わり方・判断の旗）の保存、sampleable の遅れと容量検査、窓の上限の受け取り | ReplayBuffer（replay_buffer_impl） |
| `ExperienceSampleExtractor` | 変更 | 窓の規則で k を決め、割引報酬・終わり方・k 先の観測を組む | ReplayBuffer（replay_buffer_impl） |
| `DefaultDQNAgent` | 変更 | `skip.*` の読み取り、skip Network の構築、Actor・Learner・replay への受け渡し、`action` の入力 spec | DefaultDQN（default_dqn_agent） |

## 6. 細部（推奨の既定）

1. `skip.enabled` の既定は false（無効で今と同一の受入）。
2. 最大スキップ長 J は env ごとの設定ファイルで `skip.max_length` に与え、既定は無い。設定例は GridMaze 7（論文の表形式と同じ）、LunarLander 4（論文の最良）、Atari 10（論文の Atari と同じ）、DropMerge 10（論文に無いので Atari に合わせる）。
3. 勾配の既定は遮断（`feature_gradient = stop`）。Q の学習を変えず、無効との差を skip の有無だけにする。合算は §7 の腕で試す。
4. loss の重みの既定は 1.0。合算では共有パラメータへの勾配の混ざり方に効き、遮断ではクリップの閾値と Adam の eps を通してだけ効く。対等な出発点として 1.0。
5. PER の λ の既定は 0。
6. Network 別クリップの閾値は primary の `max_grad_norm` を skip にも使う。別の閾値は §8 の先送り。
7. skip の optimizer の options は primary と同じ lr・weight decay。param group は別。別の lr は §8 の先送り。
8. skip Network の `nn_forward_key` は Actor・Learner の各用途で primary と同じキーを参照する。
9. skip の Dueling（`use_dueling`）の基準引きは j=1 引きだけ。Q(s,a) を「1 step 続ける価値」と読む定義に合う。平均引きは作らない。
10. k の抽選の乱数は replay の sampler の RNG。seed の系統を増やさない。
11. sampleable の列挙の遅れは有効で max(1, n_step, J)、無効で max(1, n_step)。PER の初期優先度の完成と eviction 統計は常に max(1, n_step)。
12. 1 step の記録（reward と終わり方）は無効でも常に書く。分岐を減らし、他の 2 本の土台にする。
13. Sample の窓の欄は、窓の規則が渡されていないとき（DefaultDQN では skip が無効のとき）空。
14. checkpoint の扱いは §5.8（有効で 4 object と印、無効の checkpoint を有効で読むときは primary だけ読んで WARN、有効の checkpoint を無効で読むのは fail-fast）。
15. Evaluation Trace の `skip_decision`・`skip_length` は既定の列に入れず、設定例に書く（無効の Run で空の列を出さない）。
16. 可視化の TensorDictFunction の `skip-net.` prefix は §8 の先送り。
17. GraphViz は skip Network を別のグラフとして出す。
18. 群番号は skip が 50、[999_replay_eviction_sample_count](999_replay_eviction_sample_count_10prd.md) は 59。
19. 不正値の扱いは §5.6（構築時に `ANET_SYSTEM_ERROR`）。
20. 分離型／包含型の切替は `skip.network_mode`（`action_out` ／ `action_in`）。
21. 設定の置き場（2026-10-24 の設定見直し）: agent.txt の `DefaultDQNAgent.@baseline` に無効のままおすすめの既定（配線 `SkipFC1 > SkipReLU > SkipFC2 > SkipReLU`、`skip_context`、network_mode・use_dueling・learner.skip の既定）を `?=` で置く。各 env は `skip.max_length ?= J` の既定と `run.@skip`（enabled、metrics のチェーンに `@skip`、`03_skip_length_rate.[2..J]`、Trace の 2 列）を持つ。IQN の env は `net.body.output.[skip_context]` を融合済みの branch（`iqn_fusion` ／ `fusion`）に上書きし、skip 側では taus を融合しない。merge・use_dueling・action_in・層幅は CLI か bat の葉。Agent側の`DefaultDQNAgent.@skip*`と`DefaultDQNAgent.skip.net.@*`のプロファイルは作らない（Run/metrics側のprofileとは区別する）。無効の Run は skip Network を構築しない。§7 の腕はこの葉から派生させる。例外として LunarLander は、2026-10-26 から既定で有効（`run.$ = run.@munch > run.@skip`）で、合算（`learner.skip.feature_gradient = merge`）を env の葉で持つ。§7 の段 2 で、分離型が無効を 3 seed のブレ幅を超えて上回り、合算が最も速く 200 に届いたためである。LunarLander で無効の Run は CLI で `run.$=run.@munch` を渡す。Atari も、2026-10-27 から実時間優先の既定の `run.$`（Atari.txt）の末尾に `run.@skip` を置いて有効にし、合算を env の葉で持つ。無効の連鎖は比較用のコメントとして残す。§7 の段 3 のふるいで、合算が 2 seed とも同じ seed の無効を上回ったためである。bat は `run.$` を CLI で渡すので、この既定の影響を受けない。
22. 包含型の行動入力の鍵は `action`（固定キー。one-hot は自動）。
23. 段 1 の GridMaze は論文の格子と同じ 6×10（設定の width・height）で、壁と穴の密度は今の既定。長い直線が出る大きさで差を見る。
24. skip が有効なときの FP16 の AMP は構築時に止める（§5.4）。
25. 段 2 の基準は現行の [LunarLander.txt](../../apps/runner/config/LunarLander.txt) の無効の腕（暫定版が挙げた `run_20260612-015116` はリポジトリに無い）。
26. DropMerge で続けてよい行動は全行動（D1 の非ゴールにより DropMerge 向けの制限は置かない）。段 4 で DROP の連続が害になると分かったら、行動ごとの許可を設定で足す。

## 7. 検証と受入

| 段 | env と構成 | 腕 | seed・長さ | 判定（結果ごとの分かれ方） |
|---|---|---|---|---|
| 0 | GridMaze・LunarLander・Atari BTR 基準・DropMerge nzrep 基準の短い Run | 無効 | 1 seed | metrics checksum が今と一致する。不一致なら実装の不備 |
| 1 | GridMaze 6×10（論文の格子の大きさ。§6 の 23）、J=7（論文の表形式と同じ） | 無効、分離型×合算、分離型×遮断、包含型×遮断 | 3 seed、500k step（評価 return が飽和していなければ延ばす） | 評価 slot の `01_decision_rate` が 1.0 から下がり、評価 return の立ち上がり（AUC）が無効以上（3 seed のブレ幅を超えて悪化しない）。悪化なら skip の尺度・探索を調べ、段 2 へ進まない |
| 2 | LunarLander、J=4 | 無効、分離型×遮断、分離型×合算 | 3 seed、1〜2M step | AUC が無効と同等以上、評価 slot の decision_rate < 1。論文でも改善は小さいので「落ちない」が主眼。落ちるなら遮断だけ残して原因を調べる |
| 3 | Atari qbert（BTR 基準）、J=10 | 無効、分離型×遮断、包含型×遮断 | 1 seed で 10M（先へ進むかのふるい。ブレ幅は [BTR と qbert の campaign](../experiments/default-dqn/atari/2026-09-27_btr-gap-qbert-bz.md) にある `a5_qbert_btr50m_eps005_end0` の seed 1〜3 の値を使う）。効けば 2 seed で 50M | 最初の 1M で遅れても 5M までに無効に追いつく（論文の型 ii）。評価 slot の判断の回数が減ることは観察する（合否ではない。論文が「約 33% 少ない」と示したのは MsPacman・Pong の型 (i) で、qbert の値は示していない）。追いつかなければ J を 4 に下げて 1 回だけ再試行 |
| 4 | DropMerge（nzrep 基準の checkpoint から有効にして分岐。§5.8） | 無効の継続、分離型×遮断 | 1 seed、50M 以上（ふるい。ブレ幅は [Atari からのフィードバックの campaign](../experiments/default-dqn/dropmerge/2026-10-01_atari-feedback.md) にある nzrep の seed 1・2 の値） | eval_target の maxrank・return が、nzrep の seed 1・2 の値のブレ幅を超えて落ちない。NEET は非ゴールなので見るだけ |

判定に使う値はすべて評価 slot（`$eval.[tag]` で読む）のもの。DropMerge は eval_target と eval_target_greedy、Atari は eval、GridMaze と LunarLander は既定の eval。train 側の `01_decision_rate` は探索で引いた j を含むので監視にだけ使う。`03_skip_length_rate` が J に寄ったら（UQE の risk score は分布の広い長い j に寄る見込み）、`actor.[key].skip.policy` を平均に対する ε-greedy にした腕を足す。作るが段 1〜4 では確かめないもの: `skip.head.use_dueling = true`、`actor.[key].skip.policy`、`learner.skip.priority_lambda`。これらは単体テストで形だけ確かめる。

段 1 の途中経過（2026-10-24。agent seed 1 本ずつ。設定見直しの第 1 依頼後のビルドで 10/23 の Run と全指標が一致することを確かめた上での値）: 6×10（既定の迷路 seed）では分離×遮断が評価 target 報酬 +0.93 に 251k step（policy 報酬は 53k）、分離×合算が 264k（同 60k）で届き、無効は 325k で届いたあと −1.09 に戻った。10/23 の包含×遮断（287k）は `+=` の誤りで action を bind しない退化形だったので、設定見直し第 2 依頼後のビルドで取り直した: 包含×遮断（action を bind）は +0.93 に届くが正になるのは 311k で、Run 全体の平均は −3.78（分離×遮断は −0.30 で 10/23 と同値。Codex の検証 Run `run_20261024-075127_prd098_r2_learning1_GridMaze_default`・`run_20261024-075515_prd098_r2_learning1_GridMaze_action_in`、1 seed）。包含型が分離型より遅いのは 1 seed の観察で、段 2 以降で判断する。5×5 の既定 seed（GridMaze.txt の `GridMazeEnv.seed`）では分離×遮断と包含×遮断が 400k で届かず（評価は毎回 100 step で −1.00、学習 lane の成功率 0、Q がすべて負、終盤は j=1 が 97%）、無効は 83k で届いた。5×5 の `GridMazeEnv.seed = 3` では分離×遮断も届いた（target 94k。無効は 68k）。`GridMazeEnv.seed = 2` は解けない迷路で両腕とも不達。結論: GridMaze では効果は迷路の形に依存して不明。既定 seed の不達は分散ではなく、ゴールに届かないまま終わる形で（2026-10-24 の時点では「崩壊」と書いていたが、10/26 に記録を読み直して言い直した）、原因は §9 の 2。Run は GridMaze-01 の `run_20261023-170715_t98_g1_*`、`run_20261024-0632*_t98_b3_*`、`run_20261024-064441_t98_b3_6x10_s1`、`run_20261024-0653*_t98_b3_m{2,3}_*`。

段 1 の 3 seed（2026-10-26。PRD101 まで入ったビルド、6×10、500k、seed 1〜3。Run は GridMaze-01 の `run_20261026-04*_t98_gm_{off,sep}_s{1,2,3}`）:
- 届く速さ（評価 target の EMA が初めて 0.5 以上になった step）は、分離×遮断 274k／323k／369k、無効 345k／191k／392k で、同じくらいだった。
- 差が出たのは終盤の安定さである。分離×遮断は 3 seed とも、届いたあと最後まで +0.93 を保った（最後の 100k の平均 +0.93）。無効は 3 seed とも届いたあと終盤に不安定になり、最後の EMA は −1.01〜−1.09、最後の 100k の平均は −0.32／−0.81／+0.14 だった。
- Run 全体の評価 target 報酬の平均（AUC）は、分離×遮断 −0.30／−1.04／−2.84、無効 −3.15／−1.73／−5.10 で、分離×遮断の平均（−1.39）は無効の最小（−5.10）を下回らない（段 1 の判定の AUC を満たす）。AUC の差は、主に終盤の安定さの差から来ている。
- GridMaze 全体では、TempoRL は届く速さで有利にならなかった。5×5 では無効のほうが速い（既定 seed で無効 124k 対 ε-greedy の skip 305k、seed 3 で無効 105k 対 分離×遮断 133k。どれも 1 seed）。有利に働いたのは 6×10 の終盤の安定さだけで、それも原因の分からない無効側の不安定さとの比較である。

分離×遮断 s1 の −0.30 は 10/23 の同じ腕と一致した。学習側の判断率は、Run の前 2 割で 0.35、後ろ 4 分の 1 で 0.76 に上がり、j=4 の選択率は 7% から 1% に下がった。評価 slot の判断率は、skip の指標が学習側にしか無いので測っていない（§9 の 3）。無効が終盤に不安定になる原因は確かめていない。GridMaze の設定の UQE の楽観（τ 0.95→0.85）と、楽観的な目標値（`use_optimistic_target = true`）が候補である。

段 2 の結果（2026-10-26。LunarLander、現行の LunarLander.txt の既定（IQN＋Munchausen）、J=4、2M、seed 1〜3。Run は LunarLander-01 の `run_2026102{5,6}-*_t98_ll_{off,sep,mrg,in}_s{1,2,3}`）: 評価 policy 報酬の Run 全体の平均（3 seed の平均 [最小〜最大]）は、無効 21 [−7〜68]、分離×遮断 122 [109〜135]、分離×合算 137 [131〜143]、包含×遮断 96 [21〜137]。終盤（1.5M〜）は、無効 159 [151〜165]、分離×遮断 209 [176〜231]、分離×合算 209 [189〜228]、包含×遮断 193 [172〜203]。評価 target 報酬の EMA が 200 に届いた step は、無効が 3 seed 中 1 seed だけ（1.76M）、分離×遮断 0.85M／1.17M／1.41M、分離×合算 1.09M／1.02M／1.09M、包含×遮断 1.19M／1.18M／1.90M。分離型の 2 腕は、最小の seed が無効の最大の seed を上回り、ブレ幅を超えて良い（段 2 の判定の AUC を満たす）。包含型は平均では上回るが、最小の seed が無効と重なる（段 1 の 1 seed の観察と同じく、分離型より遅い）。学習側の判断率は、分離型で終盤まで 0.42〜0.48（平均のスキップ長 2.1〜2.5）のまま続いた。終盤は j=1 が 28% から 39〜42% に増え、j=4 が 25% から 17〜18% に減った。論文の LunarLander のような「ほぼ毎 step 判断する方策」にはならなかった。評価 slot の判断率は測っていない（§9 の 3）。Evaluation Trace では、高度が高いときや NOOP を選んだときにスキップ長が長くなるのが見えた。無効の腕の終盤（policy 159）は、以前の IQN の 3 seed（終盤 186。[2026-08-08_iqn](../experiments/default-dqn/lunarlander/2026-08-08_iqn.md)）より低いが、Munchausen などの構成が違うので、同じ回の腕どうしでだけ比べる。この結果を受けて、LunarLander の既定を有効（分離×合算）にした（§6 の 21）。あわせて、`run.@skip` が `metrics.scalar.$` を Munchausen の指標を含まない連鎖で上書きしていたのを直した（この段 2 の skip 有効の腕は、Munchausen の指標が記録されていない）。既定を変える前に作った検証資材（`testdata/prd098/run_validation.py` の段 0 など）は LunarLander の既定を無効として扱っているので、再実行するときは `run.$=run.@munch` を渡す。

段 3 のふるい（2026-10-27。段 3 の行の「BTR 基準の qbert、1 seed で 10M」の前に、速い実時間優先の構成で 4 腕 × 2 seed をふるった。構成は Atari.txt の実時間優先の連鎖（RR1、128 env、replay 2M、ラダー 0.4〜0.01、行動ネットの同期 400 train step）に `run.@skip`、J=10、`skip_context = iqn_fusion`、10M、seed 1・2、2026-10-26 19:32 の Release。Run は Atari5-01 の `run_20261026-*_a5_qb_rt_t98{off,mrg,sep,in}_s{1,2}`）:
- 評価（ε 0.01、10 本 / 50 万 step）の Run 全体の平均（s1／s2）は、無効 3,156／2,315、分離×合算 3,174／2,846、分離×遮断 3,058／2,312、包含×遮断 2,262／2,588。7.5〜10M の窓では、無効 4,976／4,510、分離×合算 5,733／4,974、分離×遮断 5,591／3,432、包含×遮断 4,506／4,278。
- ε=0 の評価（5.25M、100 本）は、無効 4,188／3,116、分離×合算 3,571／4,630、分離×遮断 4,236／3,796、包含×遮断 2,189／4,176。10M までにこの 1 点しか無い。
- 学習側のゲームの平均点の 7.5〜10M は、無効 4,009／3,780、分離×合算 4,675／4,229、分離×遮断 3,855／3,077、包含×遮断 3,444／3,492。どの腕も同じラダーなので、腕どうしで比べられる。
- 分離×合算は、同じ seed の無効を、Run 全体の平均（+1%／+23%）、7.5〜10M（+15%／+10%）、学習側の 7.5〜10M（+17%／+12%）のどれでも 2 seed とも上回った。seed 1 は 2.5〜5M に無効より 13% 遅れてから追い越し（論文の型 ii）、seed 2 は遅れなかった。ただし「最小の seed が無効の最大の seed を上回る」は、7.5〜10M で分離×合算の s2（4,974）と無効の s1（4,976）がほぼ同値で、満たさない。
- 分離×遮断と包含×遮断は、seed で無効との上下が入れ替わった（包含×遮断の Run 全体の平均は −28%／+12%）。
- 学習側の判断率は、0〜2.5M の 0.15〜0.19 から 7.5〜10M の 0.23〜0.31 へ上がった。7.5〜10M の平均のスキップ長は、分離型の 2 腕が 4.2〜4.5、包含型が 3.3〜3.8。包含型だけ j=1 の選択率が 0.03〜0.08 から 0.29〜0.32 へ上がり続けた。j=1 に戻る形（段 1 の 5×5 の既定 seed）は、どの腕にも出なかった。評価 slot の判断率は測っていない（§9 の 3）。
- 10M の実時間（s1／s2）は、無効 25.0／27.6 分、分離×合算 29.1／31.6 分、分離×遮断 35.0／32.7 分、包含×遮断 34.6／37.8 分。
- 同じ seed 1・同じ構成の 2026-09-21 の Run（`run_20260921-155634_a5_qbert_rr1_50m`、9 月のビルド）は 7.5〜10M が 7,306 で、今回の無効の s1（4,976）と 47% 違う。ビルドの違いと非決定論だけで、この幅がある。
- この結果を受けて、Atari の実時間優先の既定を有効（分離×合算）にした（§6 の 21）。分離×遮断と包含×遮断は Atari では先へ進めず、段 3 の行の BTR 基準の qbert（サンプル効率優先）は分離×合算で続ける。

受入（外から観測できるもの）:

1. 段 0 の checksum 一致（4 構成）。
2. 単体テスト: replay の走査（done・truncated・dummy・ring 折返し・J の上限・判断の旗での区切り・k の一様性・sampleable の遅れ・完成の遅れの不変）、スキップの目標値の数値例（Munchausen・TBO・分布型・done での bootstrap 無し）、skip Head の形（分離型／包含型、`use_dueling` の j=1 一致）、Actor の skip 状態（episode_start で捨てる、上書き、ヒントの再 gather、判断の旗と aux、複製と同期）、Network 別クリップ、checkpoint の 4 object と印、無効の checkpoint を有効で読む経路、有効を無効で読む fail-fast、§5.6 の不正値と FP16 の fail-fast。
3. smoke Run の `metrics.jsonl` と `metrics.scalar.defs` に §5.7 の 9 つの tag が出て、Evaluation Trace の 2 列が読める。
4. 性能: skip が有効な学習 1 step の実時間が無効の 1.3 倍以内（Atari BTR 構成で測る。見積もりは約 1.2 倍。§5.10）。

## 8. 作らないもの・先送り

| 項目 | 裁定 | 理由と再開条件 |
|---|---|---|
| スカラーのスキップ価値（Q が分布型でも skip はスカラー） | 先送り | 型を 1 つにして Learner の目標値を流用する（D7）。再開条件: 分布型の skip で j の選択が不安定、または QR の分離型で射影が重いと分かったら |
| skip Head の独立 Dueling（V_J と A_J の 2 射影） | 先送り | `use_dueling` の Dueling が同じ役を primary の Q(s,a) で担う（D8）。再開条件: Q(s,a) の基準が skip の学習を不安定にしたら。足すときに V の取り方のキーを足す |
| 全 k を mask 付きで一括するサンプル | 先送り | target forward が J 倍（D13）。再開条件: 1 点 k で skip の学習が遅く、target forward の費用を払えるとき。後続は [999_temporl_skip_followups](999_temporl_skip_followups_10prd.md) の (K)（全 k の一括。作るかは同書の段 0 で決める） |
| skip 専用の lr・クリップ閾値 | 先送り | キーを増やさない（§6 の 6・7）。再開条件: 遮断で skip の学習率を変えたい実測が出たら |
| j の分布のヒストグラム画像 | 先送り | `03_skip_length_rate.[j]` で代用できる。再開条件: j の分布を頻繁に見る必要が出たら |
| 評価 slot でスキップを切る口（常に j=1） | 先送り | 腕の比較に要るかは段 1〜2 で分かる。再開条件: 学習した skip を使う評価と使わない評価の比較が要ると分かったら |
| 特徴空間を別の Network（F）に分ける | 先送り | 切れ目は DAG で表せる（D3）。再開条件: F を単体で差し替える、または事前学習を載せる要件が pin されたら |
| k_max に比例する重み（論文の「接続ごとに等しい重み」に合わせる） | 先送り | 一様な k で始める（D13）。再開条件: 短い k の偏りが skip の学習に効くと分かったら。後続は [999_temporl_skip_followups](999_temporl_skip_followups_10prd.md) の (K) が、全接続を等しく数える形でこの重みも扱う（同書 D7） |
| 可視化の TensorDictFunction の `skip-net.` prefix | 先送り | 入力（obs を primary に通し、包含型では greedy の行動を使う）を決める読み手がまだ無い。再開条件: skip Network の可視化が要ると分かったら |
| skip 中の lane の forward を省く | 範囲外 | §3 の非ゴール。再開条件: TempoRL が同じ step 数で効くと分かり、forward が律速の env があるとき。後続は [999_temporl_skip_followups](999_temporl_skip_followups_10prd.md) の D9・§8（学習では作らない。1 env のリアルタイム実行の PRD が起きたときに、判断する lane が無い step の forward を丸ごと飛ばす形で作る） |
| Env 側のマクロ行動、経過時間の割引、次の判断まで遡る n-step、n-step の目標値の補正 | 範囲外 | それぞれ [086](086_dropmerge_macro_actions_10prd.md)、[999_smdp](999_smdp_transition_duration_10prd.md)、[999_decision_aligned_nstep](999_decision_aligned_nstep_10prd.md)、[999_nstep_target_correction](999_nstep_target_correction_10prd.md) |

## 9. 未決

| # | 事項 | 扱い |
|---|---|---|
| 1 | 強制行動（WithAction）と skip 状態のずれ | D10 で許す。[999_observation_pipeline](999_observation_pipeline_10prd.md) の単位 A（強制行動を MakeAction の入力にする）が入ったら、強制された lane の skip 状態を捨てて新しい判断にする。強制したstep自体を `decision=true` として記録し、そのlaneの新しい判断の開始点にする。現行のReplay窓は区間内の行動不一致をSample時にfail-fastするため、train側へ強制行動を接続する際は状態リセットと判断の旗を同時に満たす。それまでは Evaluation Trace の `skip_decision` と `skip_length` で見分ける。課題は [999_observation_pipeline](999_observation_pipeline_10prd.md) の単位 A に追記した（2026-10-23） |
| 2 | 5×5 の既定 seed で有効の 2 型がゴールに届かない原因（2026-10-26 まで「序盤に見つけたゴールを保持できず崩壊する機構」と書いていた。実際には評価で一度も届かず、学習側もエピソードの報酬の最大が序盤に一度 +0.03 になっただけで、見つけたゴールを失ったのではなかった） | 原因は分かった（2026-10-26。[999_temporl_skip_followups](999_temporl_skip_followups_10prd.md) §7 の段 0）。スキップ長の方策を平均に対する ε-greedy（`actor.[train].skip.policy.policy_type = EpsilonGreedy`、ε は行動の UQE と同じ）にすると、同じ迷路で 305k に届き、最後まで保った。既定（行動の方策と同じ UQE）では、今のビルドでも同じく届かず、指標も 10/24 の Run と一致した（どちらも分離×遮断、1 seed）。スキップ長を UQE の楽観で選ぶとこの迷路ではゴールに届かない、までは言える。届かなくなる仕組み（楽観が長い j と J に寄ることがどう探索や価値を損なうか）は確かめていない。5×5 の seed 3 と 6×10 では、同じ UQE のままでも届いた。段 2（LunarLander）でも問題は出なかった。skip の方策の既定を見直すかは後続の暫定 PRD の §9 の 9 |
| 3 | 評価 slot の判断率 | §7 の段 1・段 2 の判定は評価 slot の `01_decision_rate` を使うと決めたが、`metrics.scalar.@skip` の skip の指標は学習側（`@train`）にしか無く、2026-10-26 の段 1・段 2 では測れなかった。評価 slot（`$eval.[tag]`）の判断率の tag を `@skip` に足すかを、段 3 の前に決める。それまでは Evaluation Trace の `skip_decision` で見る |

## 10. 依存と順序

- 単独で作れる。実装は replay（§5.1・5.2）、Network と Learner（§5.3・5.4）、Actor と指標（§5.5・5.7）の順に段階を分けられ、段階ごとに段 0 の等価性を確かめられる。
- [999_decision_aligned_nstep](999_decision_aligned_nstep_10prd.md) と [999_nstep_target_correction](999_nstep_target_correction_10prd.md) は §5.1 の土台に乗る（それぞれ 1 step の記録に欄を足し、窓の規則を差し替える）。判断に揃える n-step を入れるときは、Q の n-step もサンプル時の窓へ移し、Push 時の n-step の確定を退役させて割引和の機構を 1 つにする。本 PRD では移さない（等価性の確認と、actor_approx の完成が使う target return の作り直しが要る）。
- [999_smdp](999_smdp_transition_duration_10prd.md) は §5.1 の記録に経過時間の欄を足せば、窓の割引を γ^τ の積に変えられる。
- [999_observation_pipeline](999_observation_pipeline_10prd.md) の単位 A は §9 の 1 を解く。
- [086](086_dropmerge_macro_actions_10prd.md) とは独立。

## 11. 参照

- [Biedenkapp et al., 2021] André Biedenkapp, Raghu Rajan, Frank Hutter, Marius Lindauer. "TempoRL: Learning When to Act." ICML 2021. https://arxiv.org/abs/2106.05262（詳細解説: [biedenkapp2021_temporl_paper_2026-10-03.md](../../reports/biedenkapp2021_temporl_paper_2026-10-03.md)）
- [Sharma et al., 2017] Sahil Sharma, Aravind S. Lakshminarayanan, Balaraman Ravindran. "Learning to Repeat: Fine Grained Action Repetition for Deep Reinforcement Learning." ICLR 2017. https://arxiv.org/abs/1702.06054（詳細解説: [sharma2017_figar_paper_2026-10-03.md](../../reports/sharma2017_figar_paper_2026-10-03.md)）
- [Vieillard et al., 2020] Nino Vieillard, Olivier Pietquin, Matthieu Geist. "Munchausen Reinforcement Learning." NeurIPS 2020. https://arxiv.org/abs/2007.14430（n-step での bonus の積み方は [PRD067](done/067_MunchausenRL_10prd.md) D3）
- 設計文書: [130 ニューラルネットワーク](../design/130_neural_networks.jp.md)、[150 ReplayBuffer](../design/150_replay_buffer.jp.md)、[200 DQN 系 Agent](../design/200_dqn_agents.jp.md)、[110 Agent と学習](../design/110_agents_and_learning.jp.md)。
- ADR: [0011](../adr/0011-generation-aware-replay-item-key.md)、[0018](../adr/0018-iqn-via-bind-product-dag.md)、[0024](../adr/0024-replay-sampleable-range-excludes-overwritten-stack-history.md)、[0035](../adr/0035-munchausen-target-learner-local-real-space.md)、[0044](../adr/0044-replay-frame-history-start-from-episode-start-at-push.md)、[0048](../adr/0048-noisynet-epsilon-in-caller-execution-state-and-nn-forward-key.md)、[0051](../adr/0051-nn-forward-prepares-concrete-calls-and-purpose-owned-execution.md)、[0053](../adr/0053-checkpoint-load-applies-current-config-to-optimizer-options.md)、[0062](../adr/0062-networkmodel-holds-auxiliary-network-fed-by-primary-outputs-with-target-used-only-for-acting.md)、[0063](../adr/0063-replay-slot-keeps-per-step-reward-and-end-and-samples-variable-windows.md)。
- wiki: [行動の時間尺度](../wiki/mechanisms/temporal_abstraction.md)。
