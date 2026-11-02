# TempoRL の補助学習のみモード（毎 step 判断し、スキップの出力は補助タスクとしてだけ学ぶ）PRD

> 2026-10-28 起票（暫定。質疑を省き、既定を置いて、決めることを §9 に残した）。行番号は起票時点のもので、実装時は近傍のシンボル名で探し直す。
> 起点: [PRD103](103_temporl_reference_alignment_10prd.md)（TempoRL 第二弾）の段 1（qbert、サンプル効率優先）の結果の読み（2026-10-28 の会話）。遮断・合算・重み 0.3 の 3 本がそろって中盤に無効より遅れ、序盤の速さだけが skip の loss の重みの順に並んだことから、「得は補助学習の側、損は行動を固定する側」という仮説を立て、それを確かめる形を起こした。
> 関連: [PRD098](098_temporl_skip_policy_10prd.md)（TempoRL 対応）、[PRD103](103_temporl_reference_alignment_10prd.md)（第二弾）、[999_temporl_skip_followups](999_temporl_skip_followups_10prd.md)（第三弾。閉ループの割り込み (I) は本書とは別の答え）、[行動の時間尺度（wiki）](../wiki/mechanisms/temporal_abstraction.md)、[ADR 0063](../adr/0063-replay-slot-keeps-per-step-reward-and-end-and-samples-variable-windows.md)（replay の窓の規則）、[ADR 0038](../adr/0038-actor-config-catalog-without-runmode.md)（Agent は Actor の用途を知らない）。
> 改訂: 2026-10-29（グリル。`/anet-create-prd`、Claude Code）。goal anchor と scope screen に合意し、契約 5 件と細部 17 件を決めた。補助タスクの教材（窓）を 2 つにし（連続区間の窓と、エピソードの終わりだけで閉じる窓。D3）、設定キーを Actor 側 `actor.[key].skip.max_length_mode`（`single`／`full`／`variable`）＋`max_length`（ProfiledValue）、Learner 側 `learner.skip.target_window`（`decision`／`action_change`／`episode_end`）に改めた（D1。暫定版の `skip.execution = commit／single` と `learner.skip.window = action_run` は使わない。理由は D1）。組み合わせの検査は (1) だけを止め、(2) は WARN にし、(3) は Agent では判定できないので外した（D4、[ADR 0038](../adr/0038-actor-config-catalog-without-runmode.md)）。固定を徐々に解くスケジュールは `variable` の ProfiledValue で書けるが、段には入れない（§8）。§2.2 に重み 0.3 の Run の 50M までの値を足した。§9 の 1・2・5・6 を閉じた。起草後の別モデル（Opus）の検討で、段 0 の指標を突破率の差から「平均点の差を無効の checkpoint との差で割った g」に改め（45.7M では評価側の突破率の差が閉じているため）、1 本の Run に `[greedy]`／`[greedy_single]` を並べる形にし、段 1 に k=1 だけの補助と終わりだけ・分割の対照 2 腕を足し（7 腕）、段 2 の教材の閾値を一様ランダムの水準 |A|/(|A|−1) から置き直し、D1 の理由（Actor ごとのキーが要る場面）、D8 と CONTEXT.md の文言（k step の収益、上限 J）、`[train]` の既定の置き場、D4 (2) の WARN 文と `priority_lambda`、`single` で省くのは j の選択だけ、§2.1 の事実 2 件、`variable` の検証を ProfiledValue の共通処理で表すこと、skip 無効で J が無いときの検証、丸め（§9 の 9）を直した。`variable` は残した。レビュー（`an-review`、2 往復）で、補助学習のみモードの定義（§6 の 7・D5）と `split` の腕の引き金（§6 の 13・§9 の 7）の文書内の食い違いを直し、`variable` を今作る理由（D1）と段 0 の g の分母が 0 以下のときの読み（§7）を足した。別ツールのレビュー（Claude、2 件）で、受入 3 の `variable` の smoke の構成と合格の条件、`full`・`single` で `max_length` を書いたときの WARN が出る条件（`skip.enabled = true` のときだけ。§5.1・§5.2・§6 の 4・受入 2）を直した。
> 改訂: 2026-10-30（ユーザーの判断）。段 1（LunarLander）の結果は段 2（qbert）の参考にとどめ、段 2 の腕を段 1 の結果で絞らないことにした。LunarLander と qbert で傾向が大きく違い（段 0 の qbert の g は 3 本とも「固定が主因」、段 1 の LunarLander では補助学習のみモードの 2 腕とも無効の最大以下）、段 1 の結果で qbert の実施範囲を決める根拠にならないため。§6 の 10・13、§7 の段 2、§9 の 7 を直した。
> 改訂: 2026-10-31。§7 の段 2 の結果（連続区間は見込みなし、終わりだけは 9.2M で停止、終わりだけ・分割は中止）を記録した（[実験記録](../experiments/default-dqn/atari/2026-10-31_temporl-qbert-50m-learning.md)）。

## 1. 目的と問題

PRD098 の TempoRL は、判断で選んだ行動をスキップ長 j の間そのまま続ける（固定する）。スキップ価値の学習は、skip の loss として共有の特徴量にも勾配を送る（合算・分割のとき）。qbert のサンプル効率優先の構成では、次の 2 つが同時に見えた（§2.2）。

1. **序盤の得は skip の loss の重みの順に並ぶ。** 最初の 5M の ε=0 の評価と最初の 10M の学習側の平均点は、合算 > 重み 0.3 > 遮断 ≈ 無効だった。遮断は行動の固定と探索が合算と同じで、skip の loss が共有部に届かない点だけが違う。
2. **中盤の遅れは重みによらない。** 遮断・合算・重み 0.3 の 3 本とも、学習側の第 8 面の突破率と ε=0 の評価が 30M 以降に無効より遅れ、50M まで戻らなかった。遮断では Q の学習が無効と違うのは「行動を固定して集めたデータ」だけである。

このことから、TempoRL の得は skip の loss が共有部を形づくる補助タスクの効果で、損は行動を固定すること（着地の瞬間の方向を、その前の状態で決めてしまうこと）から来る、という仮説が立つ。仮説が正しければ、行動は毎 step 選び直し（スキップ長を常に 1 にし）、skip の loss だけを補助タスクとして学ぶ形で、得を残して損を消せる。

今の実装には、その形を作る口が無い。

- skip の loss の教材（窓）は「判断から次の判断まで」で区切る（[PRD098](098_temporl_skip_policy_10prd.md) D13、[ADR 0063](../adr/0063-replay-slot-keeps-per-step-reward-and-end-and-samples-variable-windows.md)）。毎 step 判断すると窓はすべて長さ 1 になり、補助タスクは j=1 の値（Q と同じもの）しか学ばない。判断で区切らない窓の規則が要る。
- Actor ごとにスキップを使わずに動かす口も無い。学習済みの TempoRL を固定なしで評価して、Q 自体が育っているかを切り分けることもできない。この口は PRD098 §8 で「評価 slot でスキップを切る口」として先送りされていた（再開条件: 学習した skip を使う評価と使わない評価の比較が要ると分かったら。[324 DQN 系列（wiki）](../wiki/categories/324_dqn.md) の「選ばなかった形」）。§2.2 の切り分けが要るので、再開条件は揃った。

## 2. 現状（2026-10-28 に確認したこと。2026-10-29 に 4 点を足した）

### 2.1 実装

- Actor は lane ごとに「続けている行動」と「残りの step 数」を持つ。`episode_start` か残り 0 の lane だけが判断し、続けている lane は方策の提案を捨てて続けている行動で上書きする。j は skip の方策が skip Network の出力から選ぶ。判断の旗は replay の 1 step の記録に残る（[dqn_based_agent.cpp:2475-2541](../../core/anet-core/src/dqn_based_agent.cpp) の `Actor::ApplySkip`）。skip の方策の乱数は、行動の方策と別の名前付き seed から作る（同 2338 行、[200 DQN 系列 §TempoRL](../design/200_dqn_agents.jp.md) の 315 行）。
- Actor の設定は `DQNActorConfig` が持ち、skip について持つのは skip の方策（`actor.[key].skip.policy.*`）だけである（[agent.hpp:160-168](../../core/anet-core/include/anet/agent.hpp)）。Agent の設定は `actor.[key]` の全エントリをカタログとして構築時に持つ（[default_dqn_agent.hpp:153](../../core/anet-core/include/anet/default_dqn_agent.hpp)）。
- Agent は Actor の用途を解釈しない。Train runner が使う Actor は Runner の設定 `run.train.actor`（既定 `train`）で決まり、Agent には名前の参照だけが渡る（[ADR 0038](../adr/0038-actor-config-catalog-without-runmode.md)、[trainer.cpp:762](../../core/anet-core/src/trainer.cpp)・[:1004](../../core/anet-core/src/trainer.cpp)）。例外として `use_optimistic_target` は `actor.[train].policy` を名前で写す（[default_dqn_agent.hpp:179-186](../../core/anet-core/include/anet/default_dqn_agent.hpp)）が、Train runner の Actor が `[train]` とは限らないので、名前で止めると推測になる。Learner に届く経験にも、どの Actor が作ったかの印は無い。
- Actor カタログの選択は参照先の最終値（Run プロファイルと CLI の葉を含む）を読む（[ADR 0040](../adr/0040-config-selection-final-value-and-run-profile-tier.md)）。`[eval_target].$ = [eval] > @target`、`[eval_panel].$ = [eval_target] > @greedy`（[agent.txt:371-427](../../apps/runner/config/agent.txt)）、`[greedy].$ = [eval] > @greedy`（LunarLander・GridMaze の `run.@skip`。Atari は `A3.actor.[greedy].$ = [eval]` に ε=0 を明示、DropMerge は `[greedy].$ = [eval_target] > @greedy`）で、`[train]` 以外はどれも `[eval]` から派生するので、`[eval]` に書いた葉はこれらに伝わる。
- replay の窓の規則は差し替えられる部品で、今は 1 つだけある。`DecisionWindowRule` は、起点から次の判断の直前まで（終わり方と上限 J でも閉じる）を窓の上限 k_max にし、窓の中で行動が起点と違えばエラーにする（[replay_buffer_impl.cpp:29-89](../../core/anet-core/src/replay_buffer_impl.cpp)）。Learner は skip が有効なら必ずこの規則を使う（[dqn_based_agent.cpp:3339](../../core/anet-core/src/dqn_based_agent.cpp)）。学習を止めた Run（`learner.enabled = false`）では replay も窓の規則も作らない（同 3331 行）。k は起点の k_max 以下から一様に引く。
- 判断で区切る理由は、判断の境界を越えて同じ行動を数えると、次の判断で同じ行動を選び直すかが次の状態に依存し、長い窓に「選び直したくなる状態」が続いた場合だけが入って、長い j の値が過大になるためである（[PRD098](098_temporl_skip_policy_10prd.md) D13、[ADR 0063](../adr/0063-replay-slot-keeps-per-step-reward-and-end-and-samples-variable-windows.md) の Consequences）。
- skip の loss が共有部に届くかは `learner.skip.feature_gradient`（`stop`／`merge`／`split`）で決まる。`stop` では skip の入力を切り離す（[dqn_based_agent.cpp:4646](../../core/anet-core/src/dqn_based_agent.cpp)）。
- 時間で変わる値の器は `ProfiledValueConfig<T>` で、型は `constant`／`exponential`／`linear`／`cosine`／`cosine_restart`／`phased`、下限と上限は `min_value`・`max_value` で `value`／`start`／`end`／各 phase に掛かる（[schedule.hpp:35-211](../../core/anet-core/include/anet/schedule.hpp)）。定数は bare の 1 行で書く（[ADR 0064](../adr/0064-profiledvalue-bare-scalar-reads-as-value-field.md)）。Actor は `UpdateSchedule` の exp_step で評価し、評価 Actor は直近の Sync 時点の学習側の step で進む（[ADR 0038](../adr/0038-actor-config-catalog-without-runmode.md)、PRD101 D7）。
- 学習済みの checkpoint を読んで評価だけを回す口は既にある。`DefaultDQNAgent.auto_load_file` で network を読み（[default_dqn_agent.cpp:255-257](../../core/anet-core/src/default_dqn_agent.cpp)）、`learner.enabled=false` で学習を止める（[agent.hpp:239](../../core/anet-core/include/anet/agent.hpp)）。12 の bat は、`run.train.num_envs=1`・`app.save_agent_on_close=false`・`app.batchrun.exp_exit_step=40` と評価 session の本数を CLI で渡して評価専用 Run を作っている（[12_batch_run.bat:44](../../apps/12_batch_run.bat)）。
- `[greedy]` の評価 session の Actor（Atari）は、ε=0、NoisyNet の摂動なし（`nn_forward_key = mu_only`）、τ 固定で動く（`run_20261027-004945_a5_qb_aln_t98mrg_s1` の config dump。改名前の名前は `[greedy_dist]`）。学習の Actor は NoisyNet の摂動つき（`act_sample`）で動く。
- skip の方策は、専用の葉を書かなければ行動の方策と同じ ε を使う（[200 DQN 系列](../design/200_dqn_agents.jp.md) の 315 行）。qbert のサンプル効率優先では ε が 25M で 0 になる（同 Run の config dump の `policy.epsilon.phase.[off]`）。
- checkpoint は Network 4 object と optimizer の役割を持ち、窓の規則や Actor の skip の状態は持たない（[PRD098 §5.8](098_temporl_skip_policy_10prd.md)）。

### 2.2 実測

qbert、サンプル効率優先（BTR に揃えた構成。PRD103 §7 の段 1 と同じ）、seed 1、1 本ずつ。Run はビルドが違いうる（各 Run の config dump が正）。比べる 4 本（4 本とも 50M まで完走）:

- 無効: `run_20261025-231207_a5_qb_aln_btreps_s1`
- 合算: `run_20261027-004945_a5_qb_aln_t98mrg_s1`
- 遮断: `run_20261027-175541_a5_qb_aln_t98stp_s1`
- 重み 0.3（合算、`learner.skip.loss_weight=0.3`）: `run_20261028-071900_a5_qb_aln_t98lw03_s1`

ε=0 の評価（`[greedy_dist]` session、100 ゲームの平均点。括弧は第 8 面の突破率。突破は 29,000 点以上かつ round_clear ≥ 8）:

| exp step | 無効 | 合算 | 遮断 | 重み 0.3 |
|---|---|---|---|---|
| 5.05M | 7,611 | 10,705 | 7,815 | 9,797 |
| 10.1M | 14,995 | 14,876 | 15,275 | 15,173 |
| 15.2M | 20,922 | 18,351 | 16,777 | 19,682 |
| 20.2M | 22,137 | 23,235 | 25,215 | 24,732 |
| 25.3M | 24,341 | 22,417 | 26,340 | 25,331 |
| 30.5M | 27,330（40%） | 25,614（7%） | 24,625（1%） | 24,720（0%） |
| 35.6M | 34,812（99%） | 22,171（6%） | 23,711（9%） | 25,050（36%） |
| 40.6M | 42,814（100%） | 28,466（42%） | 26,959（46%） | 29,303（62%） |
| 45.7M | 42,496（74%） | 29,818（68%） | 29,638（75%） | 30,784（81%） |

学習側（2.5M の窓ごとの、学習 lane のゲーム。到達は round_clear ≥ 7、突破は上と同じ）:

| 指標 | 無効 | 合算 | 遮断 | 重み 0.3 |
|---|---|---|---|---|
| 最初の 10M の平均点（2.5〜5M・5〜7.5M・7.5〜10M の 3 窓の平均。PRD103 §7 の段 3 と同じ窓） | 8,207 | 9,074 | 7,974 | 8,904 |
| 到達率 15〜17.5M | 20.2% | 2.2% | 1.1% | 7.3% |
| 到達率 20〜22.5M | 47.6% | 24.9% | 24.1% | 28.6% |
| 到達率 22.5M → 25M → 27.5M の窓 | 50.4 → 64.6 → 88.6% | 28.3 → 39.0 → 44.0% | 30.9 → 40.5 → 40.8% | 26.8 → 38.9 → 44.1% |
| 突破率が初めて 50% を超える窓 | 32.5〜35M（62.8%） | 50M まで超えない | 50M まで超えない | 50M まで超えない |
| 突破率 45〜47.5M → 47.5〜50M の窓 | 77.4 → 86.8% | 34.6 → 39.8% | 32.4 → 38.1% | 32.2 → 35.3% |

読み（本書の推定）:

- 序盤（5.05M の評価と最初の 10M の学習側）は、skip の loss が共有部に届く量の順（合算 1.0 > 重み 0.3 > 遮断 0）に並び、遮断は無効と同じだった。行動を固定することと、固定による探索の形の違いは、序盤の得を生んでいない。
- ε=0 の評価では、25M までは TempoRL の 3 本と無効の差が点ごとに上下し（15.2M は 3 本とも 6〜20% 下、20.2M は 3 本とも 5〜14% 上）、一方的には開かない。差が開いて戻らなくなるのは 30M 以降である。無効は ε が 0 になる 25M の直後に学習側の到達率が跳ねた（88.6%）が、TempoRL の 3 本は 44% 前後にとどまった。学習側の遅れは 12.5M ごろから出ている。50M の終わりでも 3 本の突破率は 35〜40% で、無効の 87% に戻らなかった。
- 中盤の遅れは重みによらず 3 本で同じなので、共有部での勾配の混ざり方では説明できない。残る経路は行動側（固定した行動が、着地の瞬間の状況を見ずに決まる）と、その行動で集めたデータの偏り（後半の面まで行くゲームが半分しかなく、第 7・8 面のデータが少ない）である。学習の Actor は NoisyNet の摂動で選んだ行動も固定するので、行動側の誤りは学習側でより大きく出る。
- どれも 1 seed である。BTR の公開実装でも第 8 面の突破の時期は seed で 15M 以上ずれる（PRD103 §2.3）。無効の seed 2（`run_20261028-132638_a5_qb_aln_btreps_s2`、サブ機で実行中）は 10M まで seed 1 とほぼ同じ（ε=0 の評価 10.3M で 15,401 対 14,995）。

LunarLander（IQN＋Munchausen、J=4、2M、3 seed）: 評価 policy 報酬の Run 全体の平均は、無効 21、分離×合算 137（[PRD098](098_temporl_skip_policy_10prd.md) §7 の段 2）。同じ構成で合算に対して分割・k_max の重み・スキップ長の ε-greedy は、どれも基準以上だが上回らなかった（[2026-10-28 の実験記録](../experiments/default-dqn/lunarlander/2026-10-28_temporl-split-kmax-epsilon.md)）。LunarLander の得が補助学習と行動の固定のどちらから来るかは、まだ分けていない。

### 2.3 近い先行例（検索の要旨。原文は確かめていない）

「行動を続ける長さの価値を、行動に使わず補助タスクとしてだけ学ぶ」研究は、2026-10-28 の検索では見つからなかった。近いもの:

- 割引率の違う複数の地平の価値を同時に学ぶことを補助タスクにすると、Atari で Rainbow をしばしば上回ったという報告（Fedus et al., ICLR 2020、[arXiv 1902.06865](https://arxiv.org/abs/1902.06865)。著者は本題の双曲割引とは別の副産物と位置づける）。何 step か先までの報酬を、いくつかの尺度で予測させる点が本書の補助タスクに近い。本書の終わりだけの窓（D3）は、割引率ではなく地平 k を 1〜J に振った n-step の Q を当てる形で、この系統に当たる。
- 補助タスクが表現をどう形づくるかの理論（Lyle et al., AISTATS 2021、[PMLR v130](https://proceedings.mlr.press/v130/lyle21a.html)）。
- 行動の繰り返しの上で successor representation を作る形（t-SR、[arXiv 2209.12331](https://arxiv.org/abs/2209.12331)）。
- frame skip を、ブートストラップの量と課題の地平を調整するものとして分析した報告（[arXiv 2102.03718](https://arxiv.org/abs/2102.03718)）。
- TempoRL の論文は重み共有を「少ない重みで学べる」構成として述べ、skip を補助の loss としては位置づけていない（[論文解説](../../reports/biedenkapp2021_temporl_paper_2026-10-03.md) 3.3）。

## 3. ゴール・非ゴール・仮説と成功の測り方

解く問題: 今の実装では、行動の固定と skip の loss の補助学習が一体で、qbert の序盤の得と中盤の損がどちらから来るかを切り分けられない。

ゴール: 次の 2 つを、設定で選べ、既定では今と同一になる形で足し、§7 の段で仮説を確かめる。

1. Actor ごとに、選べるスキップ長の幅を決められる。常に 1（毎 step 判断）、1〜J（今と同一）、1〜`max_length`（ProfiledValue。時間で変えられる）の 3 つ。学習済みの TempoRL を固定なしで評価する切り分け（段 0）にも使う。
2. skip の loss の目標値を作る区間（窓）を、判断ではなく、同じ行動が続く連続区間か、エピソードの終わりだけで閉じる区間で切れる。すべての Actor がスキップ長を常に 1 にするときだけ使え、skip の出力は行動に使わない補助タスクになる（補助学習のみモード）。

仮説: qbert での TempoRL の序盤の得は skip の loss の補助タスクの効果で、中盤の損は行動の固定から来る。補助学習のみモードは序盤の得を残し、中盤の損を消す。

非ゴール（消える案と、それが本当の制約かどうか）:

- TempoRL（固定する形）の既定を変えること: 段の結果が出る前に既定を動かさない。結果によっては別の改訂で変える。制約ではなく、実験の結果が要ることによる非ゴール。
- 閉ループの割り込み（第三弾の (I)）: 固定に歯止めを掛ける別の答えで、[999_temporl_skip_followups](999_temporl_skip_followups_10prd.md) が持つ。本書の Actor の口と組み合わせられるが、本書では作らない。別の関心による非ゴール。
- 途中まで固定し、途中から毎 step 判断にする切り替え（スケジュール）を段に入れること: `variable` の ProfiledValue で書けるようになるが、仮説の判定に要らない（§8）。制約ではなく、ゴールに要らないことによる非ゴール。
- 補助学習のみモードの skip の出力を、行動の選択や評価の指標に使うこと: 判断で区切らない窓で学んだ出力は「a を続けてから Q に従う」価値ではない（連続区間の窓は選び直しの偏りを含み、終わりだけの窓は a_t から k step の補正の無い収益になる。D3・D8）。制約（意味の違い）による非ゴール。D4 (1) で止める。
- 行動を選ぶ Q の学習を変えること: 本書は Actor のスキップ長の幅と skip の loss の窓だけで閉じる。ゴールに要らないことによる非ゴール。
- 補助タスクの予測先（a と k）を増やす形（複数の割引率など）: 本書は skip Head の予測先を変えず、窓の切り方だけを足す（§8）。別の関心による非ゴール。

成功の測り方（判定の定義は §7）:

- 段 0: 同じ checkpoint の固定あり・なしの ε=0 の評価の平均点の差を、無効の checkpoint との差で割った g。
- 段 1: LunarLander で、補助学習のみモードの 2 腕（連続区間、終わりだけ）の評価の AUC が、無効・合算に対してどこに入るか。対照（スキップ長 1 ＋ `stop`）が無効と同じ帯に入るか。k=1 だけの腕と終わりだけ・分割の腕で、得の出どころ（多地平の予測か、別の Head の TD の追加か、クリップとの結合か）を分ける。
- 段 2: qbert で、補助学習のみモードの最初の 10M の学習側の平均点と、突破率が初めて 50% を超える窓。教材の量は `50_agent_skip/14_kmax_mean` で見る。

## 4. 確定事項

各決定に理由を書き、ほかの決定や事実に依存するときはそれを名指す。

| # | 決定 | 理由と依存 |
|---|---|---|
| D1 | 足す設定キーは 3 つ。Actor ごとの `actor.[key].skip.max_length_mode`（`single`／`full`／`variable`）と `actor.[key].skip.max_length`（整数の ProfiledValue、1〜J。`variable` のときだけ読む）、Learner の `learner.skip.target_window`（`decision`／`action_change`／`episode_end`）。既定は `full`・`decision` で、今と同一。名前の理由: 「選べるスキップ長の幅をどう決めるか」をモード値で明示し、`max_length` の値そのものに切り替えの意味を持たせない（未指定や −1・0 に意味を持たせる形は [AGENTS.md](../../AGENTS.md) §設定値の扱いの sentinel の禁止に当たる）。値に「Agent」を使わない（Actor の意味で使われることが多く、マルチエージェントの概念とも紛れる。フレーム全体の方針）。`_source` のような一ひねり要る語を避ける。「窓」は何の窓かを示す 2 語の組（`target_window`: skip の目標値を作る区間）にし、`window` 単独は使わない。値は「何で閉じるか」で揃える（次の判断／行動の変化／エピソードの終わり。どれも上限 J と終わり方でも閉じる）。暫定版の `skip.execution = commit／single` と `learner.skip.window = action_run` は、`commit` が像を結びにくく、2 値では `max_length` を変えずに切り替えられないので使わない | 行動の固定の幅（Actor の関心）と、skip の loss の窓をどう切るか（replay の窓の規則の関心）は別の機構で、1 つの関心に 1 つのキーを当てる。Actor ごとのキーが要るのは、§7 の段 0 のように 1 本の Run の中で固定あり・なしの評価 session を並べるとき（PRD098 §8 の「評価 slot でスキップを切る口」）と、これからの TempoRL の Run に `single` の評価 session を足して固定あり・なしの差を時系列で取るときである（Agent 全体の 1 キーでは同じ Run で並べられない）。`variable` は仮説の判定には要らないが、固定を徐々に解く実験（§8 の 1 行目）を段 2 の結果の後に設定だけで流せるように今作る（グリルの規約「後で要るものは同時に作る」。要望は 2026-10-29 のグリル）。費用は列挙の値 1 つ、選択の絞り 1 か所、ProfiledValue の既存の検証の流用、`skip_max_length` と `07_max_length` の指標、その単体テストと smoke（§5.3・§5.6・§7 の受入 2・3）で、どれも小さい。名前の基準は [impl.md](../quality/impl.md) 名前の規約 1〜4・9。Agent の J（`skip.max_length`）は skip Head の幅と replay の窓の上限、Actor の `max_length` はその Actor が選べる幅で、同じ意味（上限）を持ち主ごとに持つ |
| D2 | `full` は今と同一。`single` は、すべての lane のすべての step を判断にし、行動は方策の提案をそのまま使い、スキップ長を 1 にする。skip の方策は j の選択だけを省き、スケジュールの更新と τ の問い合わせは今と同じにする（aux の `skip_q_values` の問い合わせが `full` と変わらない）。skip Network の forward と、aux の `skip_decision`（すべて真）・`skip_length`（すべて 1）・`skip_q_values`（J 本）は今と同じく出す。判断の旗はすべての step で立つ。`variable` は、`max_length` を exp_step で評価した上限 c の範囲（j = 1〜c）だけから j を選び、aux は J 本のまま出す。判断で決めた残りの区間は c が下がっても続ける | 読み手（Trace の列、`50_agent_skip/01`〜`04`・`21`〜`23` の指標）を変えずに済む。skip の方策の乱数は行動の方策と別の seed なので（§2.1）、`single` で選択を省いても行動の方策の乱数の系列は変わらない。`variable` は上限を時間で変える器で、c=1 の時点は `single` と同じ動きになる。D1 に依存 |
| D3 | 窓の規則を 2 つ足す。`action_change`（連続区間の窓）は、起点の行動と同じ行動が続く間だけ伸ばし、違う行動の slot の手前で閉じる。`episode_end`（終わりだけの窓）は、行動も判断の旗も見ず、終わり方の slot を含めて閉じる。どちらも上限 J で閉じ、終わり方（done・truncated）の slot は含めて閉じ、判断の旗は見ない。k は今と同じく起点の k_max 以下から一様に引く | 補助学習のみモードの教材は、毎 step 判断する Actor の軌跡しか無い。連続区間の窓は TempoRL の skip の loss（同じ行動の区間）に近いが、教材の量が方策の行動を変える頻度に依存し、qbert では分からない。終わりだけの窓は k_max がほぼ常に J で教材が常にあり、目標値は k を 1〜J に振った n-step の Q（k=1 は Q と同じ、Atari の n_step=3 なら k=3 は Q の目標値と同じ式）になる。前者が序盤の得を再現すれば同じ行動の区間という構造が効き、後者だけが再現すれば得は多地平の収益を当てることそのもので、どちらも再現しなければ得は固定や探索の側から来ていたことになる（§7 の段 1・段 2 の読み）。終わり方と上限の扱いは `decision` と同じにし、違いを閉じる条件の 1 点に絞る。[ADR 0063](../adr/0063-replay-slot-keeps-per-step-reward-and-end-and-samples-variable-windows.md) が「判断の境界を越えて同じ行動を数える窓」を不採用にしたのは skip の出力で固定する前提での話で、本書は出力を行動に使わない構成に限って足す（D4 (1)）。D4 に依存 |
| D4 | 組み合わせを Agent の構築時に検査する。(1) `target_window` が `decision` 以外なのに、スキップ長の幅が 1 を超えうる Actor（`full`、または `variable` で `max_length` の型が使う欄のどれかが 1 を超える。ProfiledValue の共通の検証を `max_value = 1` で掛けて判定する）が宣言された actor カタログに 1 つでもあれば、当たった Actor をすべて示して `ANET_SYSTEM_ERROR` で止める。(2) `target_window` が `decision` 以外で `feature_gradient = stop` なら、止めずに WARN を 1 回出す。(3) 暫定版の「`decision` なのに学習の Actor が `single`」の検査は作らない。`learner.enabled = false` のときは組み合わせを見ず、`skip.enabled = false` のときは値の検証だけをする | (1) 判断で区切らない窓で学んだ skip の出力は「a を続けてから Q に従う」価値ではなく（D3）、それで固定すると意味の無い行動になるので、意味が破綻する設定として止める（[AGENTS.md](../../AGENTS.md) §設定値の扱い）。Agent は Actor の用途を知らない（§2.1）ので、宣言された全 Actor を見るしかない。(2) `stop` は利用者が選んだ明示のモードで「意図しない状態」ではない（[AGENTS.md](../../AGENTS.md) §Fail-Fast 原則の表）。skip の loss の勾配が共有部に届かず出力も行動に使わないので、`priority_lambda` が 0（既定）なら Q の学習に効かない（0 より大きければ PER の優先度を通じて効く）。`single` ＋ `stop` は行動が毎 step 方策の提案で Q の学習の経路も無効と同じなので、「`single` の経路と窓の抽出が Q の学習を変えていないこと」を無効と比べる対照の腕になる（§7 の段 1）。書き忘れは WARN と `14_kmax_mean`・`15_grad_norm` で気づく。(3) 学習に使う Actor を Agent は判定できない（[ADR 0038](../adr/0038-actor-config-catalog-without-runmode.md) は用途ラベルを request に残す案を却下した）。この構成は窓が全部長さ 1 になる退化形で、PRD098 §5.6 が `skip.max_length = 1` を明示なら許す退化形としたのと同じ扱いで本文に書く（§5.2）。段 1 ではこの構成を「k=1 だけの補助」の対照の腕として使う（§7）。`learner.enabled = false` では窓の規則が作られない（§2.1）ので、`target_window` は意図して止めた機能を参照する設定に当たり、エラーにしない。D1〜D3 に依存 |
| D5 | 補助学習のみモードは、新しい値を足すのではなく、`learner.skip.target_window` が `decision` 以外で、全 Actor のスキップ長が常に 1（`skip.max_length_mode = single`、または `variable` で常に 1）の組み合わせで表す。腕のプロファイルは `single` で書く。腕の設定は検証資材のプロファイル `run.@skip_aux`（`core/anet-core/testdata/prd104/arms.txt`）に置き、env の設定ファイルには置かない | モードの値を別に持つと、D1 のキーと意味が重なる（二重化）。組み合わせは D4 (1) の検査で守れる。`[eval]` に書いた葉は `[eval_target]`・`[eval_panel]`・`[greedy]` に伝わる（§2.1）ので、プロファイルは `[train]` と `[eval]` の 2 行で済み、漏れは D4 (1) が起動時に示す。既定にしない実験の腕は env の設定に置かない（PRD103 の T3・T4 の腕と同じ扱い） |
| D6 | 新しいメトリクスは `50_agent_skip/07_max_length`（`variable` の上限の今の値。`full`・`single` では NaN）だけ。教材の量は既存の `50_agent_skip/14_kmax_mean`（サンプルした窓の k_max の平均）と `13_sample_k_mean` で読む。`single` の Actor では `01_decision_rate` が 1、`02_skip_length_mean` が 1 になる | 窓の規則を変えると、既存の 2 つの指標がそのまま窓の長さを表す。`07` はスケジュールを Viewer で確かめるためで、既定の連鎖には入れず腕の葉で足す（§6 の 3）。長さの分布の指標は、段 2 で平均だけでは読めなかったら決める（§9 の 4） |
| D7 | 新しい ADR は起こさず、[ADR 0063](../adr/0063-replay-slot-keeps-per-step-reward-and-end-and-samples-variable-windows.md) の先頭に注記を足す。注記は「Consequences の『窓は判断の旗と終わり方で区切る』は判断で区切る規則の話で、本書は skip の出力を行動に使わない構成に限り、行動の変化で閉じる規則と、上限 J と終わり方だけで閉じる規則を足した。1 step の記録・sampleable の遅れ・sampling 分布は変わらない」の 1〜2 文 | 設定の値で切り替えられ、戻すのが難しい判断ではない（ADR の 3 条件の「戻しにくい」を満たさない）。ADR 0063 自身が窓の規則を差し替えられる部品として決めている。既存 ADR は書き換えず注記を足す規約と、[PRD103 §5.3](103_temporl_reference_alignment_10prd.md) が ADR 0062 に注記だけ足した前例に合わせる。§9 の 5 を閉じる |
| D8 | 新しい語は 2 つで、[CONTEXT.md](../../CONTEXT.md) に入れる。**連続区間**は、同じ行動が続いた step の並び（判断の有無を問わない）。**補助学習のみモード**は、すべての Actor がスキップ長を常に 1 にし、skip の loss を判断で区切らない窓で学ぶ組み合わせ。補助学習のみモードの skip の出力は、「スキップ価値」（その長さだけ続けてから行動価値に従う価値）とは違い、連続区間の窓では「今の方策が同じ行動を続けている間の価値」に偏り、終わりだけの窓では「a_t と、続く k−1 step のデータを集めた Actor の行動の収益に、k 先の行動価値を足したもの」（補正の無い k step の収益。k step の Q）になる | 既存の「スキップ」は判断で選んだ区間で、判断をまたぐ同じ行動の並びを言えない。「スキップ」「判断」「スキップ価値」はどれも、選んだスキップ長で行動を続ける前提の語で、スキップ価値を学ぶが行動には使わない組み合わせを言えない。設定値と文書が使う語なので用語集に置く。§9 の 6 を閉じる |
| D9 | 細部は §6 の推奨の既定で埋める | グリルで「推奨で埋める」を選んだ（2026-10-29） |

## 5. 実装契約

確定事項と本節には実装の語（内部の型、入れ物の型、添字の決め方）を書かない。公開 API 名・設定キー・メトリクス名・クラス一覧の名前は書く。

### 5.1 設定キー

すべて `DefaultDQNAgent.` の下。不正値（列挙値の外、範囲の外）は Agent の構築時に、キーと値と期待値を示して `ANET_SYSTEM_ERROR` で止める（値を丸めない、既定へ置き換えない）。`skip.enabled = false` のときは値の検証だけをして使わない（PRD098 §5.6 と同じ）。

| キー | 型・値域 | 既定 | 意味 |
|---|---|---|---|
| `actor.[key].skip.max_length_mode` | `single` ／ `full` ／ `variable` | `full` | その Actor が選べるスキップ長の幅。`single` は常に 1（毎 step 判断）、`full` は 1〜J（今と同一）、`variable` は 1〜`max_length`（D2） |
| `actor.[key].skip.max_length` | 整数の ProfiledValue。型が使う欄（`constant` は `value`、`linear` などは `start` と `end`、`phased` は各 phase の使う欄）の値は 1 以上 J 以下（ProfiledValue の共通の検証に `min_value = 1`・`max_value = J` を渡す。[schedule.hpp:80-104](../../core/anet-core/include/anet/schedule.hpp)）。bare は定数、スケジュールは `.type`・`.start`・`.end`・`.steps`（PRD101 と同じ記法）。整数への丸めは既存の整数の ProfiledValue と同じ切り捨て | 無し（`skip.enabled = true` で `variable` なら必須。無ければ構築時にエラー） | `variable` のときの幅（上限）。`full`・`single` で書かれていれば値の検証だけをして使わず、WARN を 1 回出す（WARN は `skip.enabled = true` のときだけ。`learner.enabled` の値には依らない）。0・負・J 超はエラー。`skip.enabled = false` のときは J が無いので下限 1 だけを検証し、必須の検査も WARN もしない |
| `learner.skip.target_window` | `decision` ／ `action_change` ／ `episode_end` | `decision` | skip の目標値を作る区間を何で閉じるか。`decision` は今の規則、`action_change` は連続区間、`episode_end` は終わりだけ（D3） |

設定例（コロン記法）:

```
# agent.txt: 既定をべた書き（?=）。既定では今と同一
DefaultDQNAgent.@baseline : learner.skip.target_window ?= decision
DefaultDQNAgent.actor.@eval_base : skip.max_length_mode ?= full
DefaultDQNAgent.@baseline : actor.[train].skip.max_length_mode ?= full   # 既存の @baseline.actor.[train].* の並びに置く

# 段 0 の切り分け: 1 本の評価専用 Run に、固定あり（[greedy]）となし（[greedy_single]）の session を並べる（記法の細部は実装計画で確かめる）
DefaultDQNAgent.actor.[greedy_single].$=DefaultDQNAgent.actor.[greedy] DefaultDQNAgent.actor.[greedy_single].skip.max_length_mode=single
run.eval.[greedy_single].$=run.eval.[greedy] run.eval.[greedy_single].actor=greedy_single run.eval_schedule.[greedy_single].$=run.eval_schedule.[greedy]
DefaultDQNAgent.learner.enabled=false DefaultDQNAgent.auto_load_file=<checkpoint> run.eval.[greedy].eval_episodes=200 run.eval.[greedy_single].eval_episodes=200

# 固定を徐々に解く（decision の窓でだけ許す。段には入れない）
DefaultDQNAgent.actor.[train].skip.max_length_mode=variable
DefaultDQNAgent.actor.[train].skip.max_length.type=linear
DefaultDQNAgent.actor.[train].skip.max_length.start=10
DefaultDQNAgent.actor.[train].skip.max_length.end=1
DefaultDQNAgent.actor.[train].skip.max_length.steps=20000000

# 補助学習のみモード（testdata/prd104/arms.txt のプロファイル。[eval] 由来の Actor には伝わる）
run.@skip_aux : DefaultDQNAgent.learner.skip.target_window = action_change
run.@skip_aux : DefaultDQNAgent.actor.[train].skip.max_length_mode = single
run.@skip_aux : DefaultDQNAgent.actor.[eval].skip.max_length_mode = single
DefaultDQNAgent.learner.skip.target_window=episode_end      # 終わりだけの腕は葉 1 つ
DefaultDQNAgent.learner.skip.feature_gradient=stop          # 対照の腕（WARN が出る）
```

### 5.2 組み合わせの判定

D4 の判定を、判定に効く設定の組み合わせで並べる。「Actor の幅」は宣言された全 Actor について見る。どの組み合わせでも答えは 1 つに決まる。

| `skip.enabled` | `learner.enabled` | `target_window` | Actor の幅（全 Actor） | `feature_gradient` | 答え |
|---|---|---|---|---|---|
| false | どれでも | どれでも | どれでも | どれでも | 値の検証だけ。今と同一 |
| true | false | どれでも | どれでも | どれでも | 値の検証だけ（組み合わせは見ない）。段 0 の形 |
| true | true | `decision` | 全部 `full` | どれでも | 今と同一 |
| true | true | `decision` | `single` か `variable` がある | どれでも | 動く。学習に使う Actor が `single`（または上限 1）なら窓が全部長さ 1 になり、skip Head は 1 step の Q を補助として学ぶ（D4 (3)。段 1 の k=1 だけの腕） |
| true | true | `action_change`・`episode_end` | 全部 `single`（`variable` で常に 1 を含む） | `merge`・`split` | 動く（補助学習のみモード） |
| true | true | `action_change`・`episode_end` | 同上 | `stop` | 動く。WARN 1 回（対照の腕。D4 (2)） |
| true | true | `action_change`・`episode_end` | `full` か、1 を超えうる `variable` が 1 つでもある | どれでも | エラー（D4 (1)。当たった Actor をすべて示す） |

`max_length` の値の検証は、どの行でも行う。上限 J は `skip.enabled = true` のときだけ掛ける（無効では J が未指定なので下限 1 だけ）。`variable` で `max_length` が無いエラーも `skip.enabled = true` のときだけ出す。`full`・`single` で `max_length` が書かれているときの WARN は、`skip.enabled = true` の行では `learner.enabled` の値によらず出し（評価専用 Run でも Actor は動く）、`skip.enabled = false` の行では出さない。

### 5.3 Actor

- `full` は今と同一（§2.1 の 1 項目め）。
- `single` は D2 のとおり。`episode_start` の扱いは要らない（すべての step が判断なので、続けている状態を持たない）。`[greedy]` の評価 session の判断率（`21_greedy_decision_rate`）は 1、長さの平均（`22`）は 1 になる。
- `variable` は、`max_length` を `UpdateSchedule` の exp_step で評価して上限 c を得る（評価 Actor は直近の Sync 時点の学習側の step。§2.1）。判断の lane では、選んだ行動のスキップ価値の j = 1〜c の部分だけから j を選ぶ（分布型も同じ範囲）。aux の `skip_q_values` は J 本のまま（Trace の `skip_q.[j]` の範囲検査を変えない）。c = 1 の時点でも経路は同じ（絞ってから選ぶ）。判断で決めた残りの区間は、c が下がっても続ける（判断は c で絞るが、続けている区間は切らない）。`GetScalar` の `skip_max_length` で c を返す（`full`・`single` では NaN）。
- Actor の skip の状態は今までどおり checkpoint に入らない。

### 5.4 replay の窓の規則

- 窓の規則に `action_change` と `episode_end` を足す。今の規則と同じ部品の形（上限 J、起点ごとの k_max、サンプルの出力）で、違いは閉じる条件だけである（D3）。
  - `action_change`: k_max = max { k ≤ J : t+1..t+k−1 の行動が t の行動と同じで、t..t+k−2 の終わり方がすべて「なし」}。行動が違う slot の手前で閉じる（今の規則が行動の不一致をエラーにする箇所が、閉じる条件になる）。判断の旗は見ない。
  - `episode_end`: k_max = max { k ≤ J : t..t+k−2 の終わり方がすべて「なし」}。行動も判断の旗も見ない。
- dummy の slot が窓の中に入ったときのエラーは今と同じにする（終わり方で閉じるので跨がない。ADR 0063 の Consequences）。
- sampleable の列挙の遅れと容量の下限検査は、上限 J を含む今の決め方のまま（ADR 0063）。k の抽選、`sample_weight = kmax` の重み（k_max ÷ バッチの平均）、PER は変えない。

### 5.5 Learner

skip の loss の目標値（窓の割引報酬と、k 先の状態の行動価値）、`feature_gradient`、`sample_weight`、`loss_weight`、`priority_lambda`、PER は変えない。変わるのは、窓の規則を設定から選ぶことと、D4 の検査と WARN だけである。`skip.head.use_dueling` は既定の false のまま使う（true だと j=1 の値が切り離した Q そのものになり、連続区間の窓の補助の信号は k ≥ 2 の窓からしか来ない）。

### 5.6 メトリクスと Trace

群は `50_agent_skip`。tag は 40 文字以内。

| tag | 元の鍵（取得元） | 意味 |
|---|---|---|
| `50_agent_skip/07_max_length` | `skip_max_length`（学習の Actor の `$actor`。`32_agent_base/04_uqe_tau` と同じ取り方） | `variable` の上限 c の今の値。`full`・`single` では NaN（出力されない）。既定の連鎖には入れず、`variable` の腕の葉で足す |

ほかは足さない（D6）。`single` の Actor の skip の指標は D2・§5.3 の値になる。Trace の `skip_decision`・`skip_length`・`skip_q.[j]` の列はそのまま読める。

### 5.7 移行対象と等価性

既定（`full`・`decision`）では今と同一。移す現用のものは次のとおり。

- コード: `core/anet-core/include/anet/agent.hpp`（3 キー）、`core/anet-core/src/dqn_based_agent.cpp`・`.hpp`（Actor の 3 モードと `skip_max_length`、Learner の窓の規則の選択）、DefaultDQNAgent の設定の解釈と D4 の検査と WARN、`core/anet-core/src/replay_buffer_impl.cpp` と replay の公開ヘッダ（窓の規則 2 つ）。
- テスト: `replay_buffer_test.cpp`（窓の規則）、`dqn_based_agent_test.cpp`（Actor、設定、組み合わせ）、`config_baseline_test.cpp`（既定の葉）。
- 設定: `apps/runner/config/agent.txt`（既定の `?=`）。腕は `core/anet-core/testdata/prd104/arms.txt`（新規）。
- 文書: `docs/design/200_dqn_agents.jp.md` の TempoRL の節（Actor の幅のモードと `max_length`、組み合わせの検査と WARN）、`docs/design/150_replay_buffer.jp.md` の窓の規則の段落（208 行目。`CreateDecisionWindowRule(J)` と並べて 2 つの規則を書く）、`docs/design/110_agents_and_learning.jp.md` の Actor の skip の節（幅のモード）、`docs/design/140_observability.jp.md`（`07_max_length`）。英訳（`.en.md`）は翻訳の手順で追随する。
- 用語集: [CONTEXT.md](../../CONTEXT.md) に「連続区間」「補助学習のみモード」を足す（D8。本書の改訂で足した）。
- ADR: [ADR 0063](../adr/0063-replay-slot-keeps-per-step-reward-and-end-and-samples-variable-windows.md) の先頭に注記（D7。本書の改訂で足した）。

振る舞いが変わる既存の設定・テスト・サンプルは無い（新しいキーの既定が今の振る舞いである）。互換のための仕組みは足さない。過去の Run と記録は書き換えない。暫定版のキー名（`skip.execution`、`learner.skip.window`）は実装されていないので移行対象は無い。

checkpoint は窓の規則も Actor の幅も持たない（§2.1）ので、補助学習のみモードで学んだ checkpoint を `full`・`decision` で読む誤りは止められない。設定値を実験の目的に対して選ぶのは利用側の責任である（[AGENTS.md](../../AGENTS.md) §汎用機構と利用側の責任境界）。

### 5.8 性能

性能の受入は置かない。`action_change`・`episode_end` の規則は、`decision` と同じくサンプル 1 行あたり最大 J 個の slot を読むだけで、1 回の更新の費用の桁は変わらない（B=256、J=10 で 1 更新あたり最大 2,560 slot の読み）。`episode_end` では k 先の観測の stack がほぼ常に J 先になるが、今の最大と同じで桁は変わらない。`single` の Actor は skip の方策の選択を省くだけで、forward の回数は変えない（D2）。`variable` の絞りは値の切り出し 1 回で無視できる。

### 5.9 クラス一覧

> 従えない事情があれば、実装する側が理由を実装計画に書いて変えてよい（質問しない）。

| 名前 | 区分 | 責務 | 置き場（機能グループ） |
|---|---|---|---|
| `ActionChangeWindowRule` | 新規 | 起点と同じ行動が続く間だけ窓を伸ばし、行動の変化・終わり方・上限で閉じる窓の規則 | 強化学習 共通基盤（replay_buffer_impl） |
| `EpisodeEndWindowRule` | 新規 | 行動も判断の旗も見ず、終わり方・上限で閉じる窓の規則 | 強化学習 共通基盤（replay_buffer_impl） |
| `DecisionWindowRule` | 変更なし | 判断で区切る今の窓の規則 | 強化学習 共通基盤（replay_buffer_impl） |
| `DQNActorConfig` | 変更 | Actor ごとの幅のモード（`single`／`full`／`variable`）と `max_length`（ProfiledValue）を持つ | Agent 共通の設定（agent.hpp） |
| `LearnerConfig` | 変更 | skip の目標値の窓（`target_window`）の選択を持つ | Agent 共通の設定（agent.hpp） |
| `Actor`（DQN） | 変更 | `single` のとき毎 step を判断にし j=1 で返す。`variable` のとき上限を評価して選択を絞り、`skip_max_length` を返す | DQN 共通 component（dqn_based_agent） |
| `Learner`（DQN） | 変更 | 窓の規則を設定から選ぶ | DQN 共通 component（dqn_based_agent） |
| `DefaultDQNAgentConfig` | 変更 | 3 キーの解釈、`max_length` の範囲の検証（ProfiledValue の共通の検証に 1〜J の bounds を渡す）、D4 の組み合わせの検査と WARN | DQN 具象（default_dqn_agent） |

## 6. 細部（グリルで推奨の既定で埋めた。2026-10-29）

1. `single` の Actor の動き: skip の方策の j の選択だけを省き（スケジュールの更新と τ の問い合わせは今と同じ）、skip Network の forward と aux（`skip_decision` 全部真、`skip_length` 全部 1、`skip_q_values` J 本）は出す（理由: Trace と指標の読み手を変えない。forward は MLP 1 回。§9 の 2 を閉じる）。
2. `variable` の絞り: 方策の選択だけを上限までの j に絞り、aux の `skip_q_values` は J 本のまま（理由: Trace の `skip_q.[j]` の範囲検査を変えない）。整数への丸めは既存の整数の ProfiledValue と同じ切り捨て。
3. `variable` の上限を指標に出す: Actor の `GetScalar` に `skip_max_length` を足し、tag `50_agent_skip/07_max_length` を腕の葉で足す。`full`・`single` では NaN（理由: スケジュールを Viewer で確かめる。既定の連鎖には入れない）。
4. 読まないモード（`full`・`single`）で `max_length` が書かれていたら: 値の検証（1〜J）だけして使わず、`skip.enabled = true` なら `learner.enabled` の値によらず WARN を 1 回出す。`skip.enabled = false` なら出さない（理由: `variable` の書き忘れに気づける。[999_temporl_skip_followups](999_temporl_skip_followups_10prd.md) §5.1 の λ の扱いと同じ）。
5. D4 (2) の WARN の文: キーと値と「skip の loss の勾配が共有部に届かず、出力も行動に使わない。`priority_lambda` が 0 なら Q の学習に効かない（対照の腕）。0 より大きければ優先度を通じて効く」を 1 回（理由: 2026-10-29 の決定）。
6. D4 (1) のエラー文: 当たった Actor のキー、`max_length_mode`（`variable` なら上限の範囲）、`target_window` の値、「偏った skip 出力で固定する」を示す（理由: 構築時に直し方が分かる）。
7. 用語: 「補助学習のみモード」は「全 Actor のスキップ長が常に 1（`single`、または `variable` で常に 1）で、`target_window` が `decision` 以外」と定義し（D8・§5.2・CONTEXT.md と同じ定義）、「連続区間」とともに CONTEXT.md に入れる（理由: 設定値と文書が使う語で、既存の語で言い換えられない。D8）。
8. プロファイル: `run.@skip_aux`（`target_window = action_change`、`[train]`・`[eval]` を `single`）を `core/anet-core/testdata/prd104/arms.txt` に置く。終わりだけの腕は CLI の葉 `learner.skip.target_window=episode_end`、対照は葉 `feature_gradient=stop`（理由: 腕の違いは葉 1 つ。env の設定ファイルには置かない。`[eval]` 由来の Actor には伝わる）。
9. 段 1 の腕: 無効・合算・連続区間・終わりだけ・対照（`single` ＋ `stop`）・k=1 だけ（`single` ＋ `decision` ＋ `merge`）・終わりだけ・分割（`split`）の 7 腕 × 3 seed × 2M（理由: 対照は `single` の経路の無害さの確認、k=1 だけは「多地平の予測か、別の Head の TD の追加か」の切り分け、分割はクリップとの結合の切り分け。どれもコード不要で、1 本約 20 分）。
10. 段 2 の腕の順: 段 1 の結果によらず、連続区間 → 終わりだけ → 終わりだけ・分割（`split`）の順に流す。連続区間が教材の確認で止まったら、終わりだけへ進む。対照（`stop`）と k=1 だけの腕は、前の腕のどれかが見込みありを満たし、得の出どころを分ける必要が出たときに足す（理由: LunarLander と qbert は傾向が大きく違い、段 1 の結果で qbert の腕を絞る根拠にならないので、段 1 は参考にとどめる（2026-10-30 のユーザーの判断）。仮説に近い腕から流す。対照の 2 腕は、得が出たときの切り分けにだけ要る）。
11. 段 2 の教材の閾値: 連続区間の腕だけ、7.5〜10M の窓（ε がほぼ 0 になったあと）に当たる learn step の `14_kmax_mean` の平均が、一様ランダムに動いたときの水準 |A|/(|A|−1)（qbert は 6 行動で 1.2）に 0.3 を足した 1.5 未満なら止める（理由: 最初の 2.5M は ε≈1 のデータを含むので水準 1.2 に張り付き、方策の性質で決まらない。上積みの 0.3 は暫定。終わりだけの腕には当てない）。
12. 段 0 の形: checkpoint ごとに 1 本の評価専用 Run に `[greedy]`（`full`）と `[greedy_single]`（`single`）の 2 session を置き、各 200 ゲーム（`eval_episodes` を CLI で 200）。無効の checkpoint は `[greedy]` だけ。`[greedy_single]` の平均点とエピソード長の tag は群 `53_evalg` の既存の並びに続く番号で、段 0 の葉で足す（理由: 同じ重み・同じ process で対になり、Run の数が半分になる。200 ゲームで g の標準誤差は約 0.1。閾値 0.5・0.2・−0.2 はその 2 倍以上）。
13. `feature_gradient`: 段 1 の基準（合算）と連続区間・終わりだけ・k=1 だけの腕、および段 2 の腕は `merge`（理由: 比べる相手の §2.2 の合算と揃える）。段 1 だけ、終わりだけ・分割（`split`）の腕を別に持つ（§6 の 9。クリップとの結合の切り分け）。段 2 の分割の腕は、段 1 の結果によらず、終わりだけ（合算）の次に流す（§6 の 10、§9 の 7）。
14. 固定を徐々に解くスケジュール（`variable` ＋ `decision`）: 段には入れない。§8 の先送りを「`decision` の窓では書ける。腕に入れるのは段 2 の結果の後」に書き換えた（理由: 仮説の判定に要らない）。
15. 検証資材: `core/anet-core/testdata/prd104/` に置き、着手前に前の実行体と設定を snapshot（理由: PRD103 D14 と同じ。PRD103 の実装が未コミットで、その時点のビルドが「前」になる）。
16. `target_window` の値とクラス名: `decision`／`action_change`／`episode_end`、`ActionChangeWindowRule`・`EpisodeEndWindowRule`（理由: 何で閉じるかで揃え、既存の `DecisionWindowRule` に合わせる）。
17. 新しい指標: 3 の `07_max_length` 以外は足さない（理由: D6。窓の長さは `14_kmax_mean`・`13_sample_k_mean` で読む）。

## 7. 検証と受入

用語の定義（[999_temporl_skip_followups](999_temporl_skip_followups_10prd.md) §7 と同じ）:

- **AUC**: 評価 slot の評価報酬の、Run 全体での平均。
- **A が B 以上**: 腕 A の 3 seed の AUC の平均が、腕 B の 3 seed の AUC の最小値を下回らない。
- **A が B を上回る**: 腕 A の 3 seed の AUC の最小値が、腕 B の 3 seed の AUC の最大値を上回る。
- **A が B と同じ帯**: 腕 A の 3 seed の AUC の範囲（最小〜最大）が、腕 B の 3 seed の範囲と重なる。
- **突破率**: qbert の学習側の 2.5M の窓ごとの、29,000 点以上かつ round_clear ≥ 8 のゲームの割合（PRD103 §7 と同じ）。ε=0 の評価では、1 session のゲームでの同じ割合。

段 0（切り分け。学習しない。qbert）: §2.2 の合算・遮断・重み 0.3 の Run の終わりの checkpoint（`agent_close.anet`）と、基準として無効の Run の終わりの checkpoint（`run_20261025-231207_a5_qb_aln_btreps_s1/agent_close.anet`）を、それぞれ 1 本の評価専用 Run で読む。Run の作り方は 12 の bat の評価専用 Run と同じ（§2.1）で、今の Atari.txt の `run.$` に、元の Run の CLI の葉（config dump の `config_data.txt` から取る）と、`auto_load_file`・`learner.enabled=false`・`app.save_agent_on_close=false`・`run.train.num_envs=1`・`app.batchrun.exp_exit_step=40` を足す（元の Run の評価 session の名前は `[greedy_dist]` で、今の設定では `[greedy]` なので、元の設定ではなく今の連鎖を使う。`inspect_run.py config --diff` で元の dump との差を確かめる）。TempoRL の 3 本の Run には、`[greedy]`（`full`）と、それから派生させた `[greedy_single]`（`skip.max_length_mode = single`。§5.1 の設定例）の 2 つの評価 session を置き、どちらも 200 ゲーム（`eval_episodes=200`。learn step 0 の初回の発火が 1 回ずつ起きる）。同じ重み・同じ process で対になり、Run の数は checkpoint の数と同じ。無効の checkpoint は skip が無効なので `[greedy]` の 200 ゲームだけ。読む値は各 session の平均点（`[greedy]` は `53_evalg/10_game_score_mean`、`[greedy_single]` は同じ群に段 0 の葉で足す tag。§6 の 12）で、突破率（評価のエピソードのうち突破したものの割合）は副指標にする（45.7M では評価側の突破率が 3 本とも無効と同じ帯に入っていて差を測れない。§2.2）。

checkpoint ごとに g = （`[greedy_single]` の平均点 − `[greedy]` の平均点）÷（無効の `[greedy]` の平均点 − `[greedy]` の平均点）を取り、下の「g の表」の行を決める。分母が 0 以下（その checkpoint の `[greedy]` が無効以上）なら g を作らず、「評価では無効に遅れていない」として記録し、段 1・段 2 は予定どおりに流す（§2.2 の 3 本では 45.7M の分母が約 12K の正）。同じ checkpoint の 2 つの評価の差なので学習の時点のばらつきは入らず、評価のばらつきだけが入る（200 ゲームの平均点の標準誤差は、点の標準偏差を 10〜15K として約 0.7〜1.1K。分母を 12K とすると g の標準誤差は約 0.1）。

使う checkpoint と、行の読み方は次のとおり。前のビルドの 3 本と PRD103 の段 3 の分割の Run（ビルドと構成が違う）は混ぜない。

| 前のビルドの 3 本のうち読めた本数 | 使う checkpoint | 読み方 |
|---|---|---|
| 3 本 | 3 本 | 2 本以上が同じ行に当たれば、その行。当たらなければ「読みが定まらない」 |
| 2 本 | 読めた 2 本 | 2 本が同じ行なら、その行。違えば「読みが定まらない」 |
| 1 本 | 読めた 1 本 | その行を「1 本だけの読み」として記録する |
| 0 本 | PRD103 の段 3 の分割の Run の終わりの checkpoint 1 本（その Run が終わるまで段 0 を待つ） | その行を「1 本だけの読み」として記録する |

g の表:

| g | 読み | 段 1・段 2 への効き方 |
|---|---|---|
| 0.5 以上 | Q は無効の帯に近く育っていて、損の主因は行動の固定 | 仮説の後半を支える。段 2 は予定どおり |
| 0.2 以上 0.5 未満 | 固定の損はあるが、Q の遅れも残る | 段 2 で、補助学習のみモードが無効の帯まで戻らない可能性を見込む |
| −0.2 より大きく 0.2 未満 | 固定の損は評価では見えず、Q 自体が遅れている（データの偏り側） | 段 2 の読みで「行動の固定が集めるデータ」の側を重く見る |
| −0.2 以下 | 固定は評価では得 | 補助学習のみモードは固定の得も捨てる。段 1・段 2 は予定どおりに流し、結果をこの読みと並べる |

「1 本だけの読み」にも、g の表の「段 1・段 2 への効き方」をそのまま当てる。「読みが定まらない」ときは、使った checkpoint ごとの g をそのまま記録する。学習側の遅れ（突破率 35〜40% 対 87%）は NoisyNet の摂動つきの行動で出ているので、摂動つきの評価 Actor（`nn_forward_key = act_sample`、ε=0）で `full` と `single` を比べる session を足すかは §9 の 10。

段 1（LunarLander。現行の LunarLander.txt の既定。IQN、J=4、分離×合算）: 無効（`run.$=run.@munch`）、合算（既定）、連続区間（合算に `run.@skip_aux`）、終わりだけ（連続区間に `learner.skip.target_window=episode_end`）、対照（連続区間に `feature_gradient=stop`）、k=1 だけ（合算に `[train]`・`[eval]` の `skip.max_length_mode=single` だけを足し、`target_window` は `decision` のまま。窓が全部長さ 1 になり、skip Head は 1 step の Q を補助として学ぶ。§5.2 の 4 行目の構成）、終わりだけ・分割（終わりだけに `feature_gradient=split`）の 7 腕 × 3 seed × 2M を同じビルドで流し、評価 policy 報酬の AUC で読む。どの腕もコードの変更は要らない。

| 結果 | 読み |
|---|---|
| 連続区間か終わりだけが合算以上で、無効を上回る | LunarLander の TempoRL の得は、固定なしでも出る（補助学習の側）。連続区間だけならば同じ行動の区間という構造が効き、終わりだけでも出れば得は多地平の収益を当てることそのもの |
| 2 腕とも平均が、無効の最大以下 | LunarLander の得は固定から来る |
| どちらでもない | 両方が効いている。AUC と、評価の EMA が初めて 200 に届く step を腕ごとに記録する |

得の出どころを分ける読み（上の判定に添える）:

| 結果 | 読み |
|---|---|
| k=1 だけの腕も無効を上回る | 得は多地平の収益を当てることではなく、別の Head の 1 step の TD を共有部に足すこと（またはクリップとの結合）から来る |
| 終わりだけ・分割が終わりだけ（合算）と同じ帯 | クリップとの結合は効いていない。差があれば、その差がクリップとの結合の寄与 |

対照（`priority_lambda` は既定の 0、`skip.head.use_dueling` は既定の false）は、無効と同じ帯に入るかを見る観察で、合否にはしない。同じ帯に入らなければ、`single` の経路か窓の抽出が Q の学習に効いている疑いとして記録し、段 2 の前に原因を調べる。

段 2（qbert、サンプル効率優先、§2.2 の合算と同じ構成に `run.@skip_aux`、seed 1、50M。腕は §6 の 10 の順）:

- 教材の確認（連続区間の腕だけ）: `50_agent_skip/14_kmax_mean` は learn step の軸に載るので、exp step の 7.5〜10M に当たる learn step の範囲（exp step を、学習 1 回あたりの exp step で割った値）の点の平均を取る。一様ランダムに動いたときの `14_kmax_mean` の水準は |A|/(|A|−1)（qbert は 6 行動で 1.2）で、最初の 2.5M は ε≈1 のデータを含むのでこの水準に張り付く。7.5〜10M の平均が水準に 0.3 を足した 1.5 未満なら、連続区間がほとんど無く補助タスクに教材が無いので、Run を止めて記録し、終わりだけの腕へ切り替える（§6 の 10・11）。
- 読む値: 最初の 10M の学習側の平均点（§2.2 と同じ 3 窓）、5.05M の ε=0 の評価、突破率が初めて 50% を超える窓、30〜45M の ε=0 の評価、Munchausen の log π の clip 率（無効と同じ帯に戻るか）、眠ったユニットの率（PRD103 D13 の観察と同じ）。比べる相手は §2.2 の無効と合算（ビルドが違う）。
- 見込みありの判定: 最初の 10M の学習側の平均点が無効（8,207）以上で、かつ突破率が初めて 50% を超える 2.5M の窓の始まりが 37.5M 以前（37.5〜40M の窓まで含む。無効の窓の始まり 32.5M から 5M 以内）。満たせば、同じビルドでその腕と無効の seed 2 を流し、既定を変えるかの材料にする（既定の判断は同じビルドの 2 seed で行う。PRD103 §9 の 1 と同じ）。満たさなければ結果を記録し、§6 の 10 の次の腕へ進む。1 seed なので、どの場合も合否ではない。
- qbert の Actor のカタログには `[train]`・`[eval]` のほかの Actor（`[greedy]`・`[eval_target]` など）もあるので、段 2 の `run.@skip_aux` は、宣言された全 Actor を `single` にする（D4 (1) は、`full` の Actor が 1 つでも残ると構築時に止める）。

段 2 の結果（2026-10-31、[実験記録](../experiments/default-dqn/atari/2026-10-31_temporl-qbert-50m-learning.md)）:

- 連続区間（`run_20261030-013612_a5_qb_aln_t104ac_s1`）: 教材の確認の `14_kmax_mean`（7.5〜10M）は 4.66 で、閾値 1.5 を超えた。最初の 10M の学習側の平均点は 8,895 で無効（8,207）以上だが、突破率が初めて 50% を超えた窓の始まりは 47.5M で、37.5M 以前の条件を満たさない（見込みなし）。30〜45M の ε=0 の評価（`53_evalg`。この腕は全 Actor が `single`）は 35.55・40.60・45.65M で 26,070・26,098・29,180（3 回平均 27,116。無効 40,999、合算 27,058）。5.05M（5.25M の session）は 7,855。Munchausen の log π の clip 率（40〜50M）は 0.66%（無効 0.15〜0.38%、合算 2.89〜3.00%）、眠ったユニットの率（40〜50M）は 19.42%（無効 9.00〜9.66%、合算 12.48〜13.66%）。
- 終わりだけ（`run_20261030-192929_a5_qb_aln_t104ee_s1`）は 9.2M で止めた（2026-10-30、ユーザーの判断）。判定しない。9.2M までの値は実験記録にある。
- 終わりだけ・分割は流していない（2026-10-30、ユーザーの判断で中止）。

受入（外から観測できるもの）:

1. 既定（`full`・`decision`）で、PRD103 の検証資材（[README](../../core/anet-core/testdata/prd103/README.md)）の 6 構成（skip 無効の GridMaze・LunarLander・Atari BTR・DropMerge nzrep、skip 有効の GridMaze・LunarLander）の metrics checksum が、本書の前のビルド（着手前の snapshot。§6 の 15）と一致する（`90_perf/`・`91_resource/`・`92_resource/` を除く全 scalar 点の完全一致。比べ方と短縮条件は同 README と同じ）。
2. 単体テスト:
   - `action_change` の窓: 起点が連続区間の途中でも、残りの長さが k_max になる。行動が変わる slot の手前で閉じる。判断の旗が立っていても行動が同じなら閉じない。done・truncated の slot を含めて閉じる。上限 J で閉じる。窓の中の dummy はエラー。
   - `episode_end` の窓: 行動が変わっても判断の旗が立っていても閉じない。done・truncated の slot を含めて閉じる。上限 J で閉じる。窓の中の dummy はエラー。
   - `single` の Actor: 2 lane 以上・数 step で、返す行動が方策の提案と一致し、判断の旗と `skip_decision` がすべて真、`skip_length` がすべて 1、`skip_q_values` が J 本。
   - `variable` の Actor: 上限 c のとき選ばれる j が c 以下で、スケジュールで c が変わると選ばれる範囲が変わる。aux の `skip_q_values` は J 本。`skip_max_length` が c を返し、`full`・`single` では NaN。続けている区間は c が下がっても続く。
   - D4 (1) が §5.2 の表のとおりに止まり（`full` の Actor、`variable` の型が使う欄のどれかが 1 を超える Actor）、エラー文に当たった Actor のキーが入る。§5.2 の動く行は構築できる。D4 (2) で WARN が 1 回出る。`learner.enabled = false` では組み合わせで止まらない。
   - 3 キーの列挙値の外、`max_length` の 0・負・J 超、`variable` で `max_length` が無い（`skip.enabled = true` のとき）、が構築時のエラーになる。`full`・`single` で `max_length` が書かれていると WARN が 1 回出る（`skip.enabled = true` のとき。`learner.enabled = false` でも出る）。`skip.enabled = false` では列挙値と `max_length` の下限だけが検証され、J 超と必須では止まらず、WARN も出ない。
3. smoke Run（LunarLander の短縮条件、`run.@skip_aux`、1 seed）の `metrics.jsonl` で、`50_agent_skip/01_decision_rate` のすべての点が 1、`50_agent_skip/14_kmax_mean` のすべての点が 1 以上で、少なくとも 1 点が 1 を超え、`50_agent_skip/11_loss` が有限である。同じ短縮条件に `learner.skip.target_window=episode_end` を足した smoke Run で、`14_kmax_mean` のすべての点が J − 0.5 以上である。`variable` の smoke Run は `run.@skip_aux` を使わず（`target_window` は既定の `decision`。§5.1 の「固定を徐々に解く」の例と同じ形で、D4 (1) には当たらない）、LunarLander の短縮条件に `[train]` の `max_length_mode=variable`・`max_length.type=linear`・`start=4`・`end=1`・`steps` を smoke の exp step 数の半分、と `07_max_length` の tag を足す葉（§5.6）を足して流す。合格は、`50_agent_skip/07_max_length` を exp step の軸で Run 全体で読み、最初の点が 3（値は 4 から線形に下がり、丸めは切り捨て（§5.1）なので、exp step が 0 より大きい最初の learn step では 3 になる。warmup の終わりは `steps` の 1/3 より前にする）、最後の点が 1、途中の点が前の点より増えず、3・2・1 の点がそれぞれ 1 つ以上あること。
4. 段 0 の評価専用の Run（段 0 で使う checkpoint のうち 1 本。合算・重み 0.3・遮断の順で、最初に読めたもの。どれも読めなければ PRD103 の段 3 の分割の Run のもの）が終了コード 0 で終わり、`[greedy]` の得点の指標（`53_evalg/10_game_score_mean`）と `[greedy_single]` の得点の指標がそれぞれ 1 点以上記録され、`config/config_data.txt` に `[greedy_single]` の actor と session がある。

## 8. 作らないもの・先送り

| 項目 | 裁定 | 理由と再開条件 |
|---|---|---|
| 途中まで固定し、途中から毎 step 判断にする切り替え（スケジュール）を段に入れること | 先送り | `decision` の窓では `variable` の ProfiledValue で書ける（§5.1 の設定例。窓は判断で区切るままなので特別な扱いは要らない）。判断で区切らない窓では D4 (1) が止める（偏った skip の出力で固定する期間を作らない）。再開条件: 段 2 で補助学習のみモードが序盤の得を出さなかったとき、または固定の期間の長さが成績に効くと分かったとき |
| 補助学習のみモードを 1 つのモードの値で持つ形 | 不採用 | D1 のキーと意味が重なる（D5） |
| `single` の Actor で skip Network の forward を省く | 不採用 | 読み手（Trace の列、指標）を変えない（§6 の 1）。再開条件: 1 env のリアルタイム実行の PRD が起きたとき（[999_temporl_skip_followups](999_temporl_skip_followups_10prd.md) §8 と同じ） |
| 閉ループの割り込み | 範囲外 | [999_temporl_skip_followups](999_temporl_skip_followups_10prd.md) の (I)。本書の `single` と同時には使わない（`single` は続けている状態を持たない） |
| 補助タスクの予測先（a と k）を増やす形（複数の割引率など、§2.3 の先行例の形） | 範囲外 | 本書は skip Head の予測先を変えず、窓の切り方だけを足す（終わりだけの窓も、予測するのは「a を打って k step 進む価値」で、予測先は同じ a と k）。別の予測先は別の PRD。再開条件: 段 1 か段 2 で補助学習のみモードが無効を上回る得を出し、補助タスクそのものが効くと分かったとき（どの予測先が効くかを比べる意味が出る） |
| 「`decision` なのに学習に使う Actor が `single`」を止める検査（暫定版の D4 (3)） | 不採用 | Agent は Actor の用途を知らない（ADR 0038）。退化形として本文に書く（§5.2） |
| 判断で区切らない窓なのに `stop` を止める検査（暫定版の D4 (2)） | 不採用 | `stop` は明示のモードで、`single` ＋ `stop` は対照の腕になる（D4）。WARN で受ける |
| Actor の幅を未指定や sentinel（−1・0）で「Agent の J に従う」と読む形、`max_length_source` のような出どころの語 | 不採用 | 未指定に意味を持たせず、モード値で分岐する（D1。[AGENTS.md](../../AGENTS.md) §設定値の扱い） |
| `single` を「skip を持たない Actor」として作る形（skip Network の forward も aux も出さない。Actor 側が無効と同じ経路になり対照が分かりやすい） | 不採用 | Trace の skip の列と `50_agent_skip/01`・`02`・`21`〜`23` が空か NaN になり、読み手が変わる（§6 の 1）。対照の腕は `stop` で組める（D4 (2)） |
| 終わりだけの窓の値の名前を `horizon` などにする形 | 不採用 | 3 値を「何で（だけ）閉じるか」の語で揃える（D1）。`episode_end` は「終わり方と上限 J でだけ閉じる」の意味で、§5.1 に書いた |

## 9. 未決

| # | 事項 | 扱い |
|---|---|---|
| 1 | D4 (3) の「学習の Actor」の識別 | 2026-10-29 に閉じた。Agent は用途を知らないので検査を作らない（D4。ADR 0038）。退化形として §5.2 に書いた |
| 2 | `single` の Actor で skip Network の forward を省くか | 2026-10-29 に閉じた。省かない（§6 の 1、§8） |
| 3 | 段 2 の「教材が無い」の閾値 | 暫定（連続区間の腕だけ）。閾値は一様ランダムの水準 1.2（6 行動）に 0.3 を足した 1.5 で、7.5〜10M の窓で測る（§6 の 11）。上積みの 0.3 は、段 1 の LunarLander（4 行動、水準 1.33）の `14_kmax_mean` が水準をどれだけ超えるかを見て決め直す |
| 4 | 連続区間の長さの分布の指標を足すか | 既定は足さない（D6）。段 2 で平均だけでは読めなかったら足す |
| 5 | ADR 0063 の先頭に注記を足すか、注記に加えて新しい ADR を起こすか | 2026-10-29 に閉じた。注記だけ（D7。本書の改訂で足した） |
| 6 | 用語（連続区間、補助学習のみモード。D8）を CONTEXT.md に入れるか | 2026-10-29 に閉じた。入れる（本書の改訂で足した） |
| 7 | 段 1・段 2 の補助学習のみモードを `merge` で流すか `split` で流すか | 段 1 は `merge` を基準にし、終わりだけ・分割（`split`）の腕も流す（クリップとの結合の切り分け。2026-10-29）。段 2 は `merge` の 2 腕（比べる相手の §2.2 の合算と揃える）のあとに、分割の腕も流す（段 1 の結果によらない。§6 の 10・13。2026-10-30） |
| 8 | 段 0 の g の閾値（0.5・0.2・−0.2） | 暫定。200 ゲームの平均点から作る g の標準誤差（約 0.1）から置いた。段 0 の結果を見て決め直す |
| 9 | `variable` の上限を整数へ丸める規則 | 2026-10-29 に閉じた。既存の整数の ProfiledValue と同じ切り捨て（[schedule.hpp](../../core/anet-core/include/anet/schedule.hpp) の評価が `static_cast` で切り捨てる）。§5.1 に書いた |
| 10 | 段 0 に、NoisyNet の摂動つきの評価 Actor（`nn_forward_key = act_sample`、ε=0）で `full` と `single` を比べる session を足すか | 既定は足さない。学習側の遅れは摂動つきの行動で出ているので、評価専用 Run で `act_sample` の Actor が動くことを試走で確かめてから、段 0 の葉に足すかを決める |

## 10. 依存と順序

- [PRD098](098_temporl_skip_policy_10prd.md) と [PRD103](103_temporl_reference_alignment_10prd.md) の実装に乗る。PRD103 の実装は未コミットなので、着手前の snapshot（§6 の 15）がその時点のビルドになる。
- 段 0 は学習しないので、前のビルドの checkpoint が 1 本以上読めれば実装のあとすぐにできる（§2.2 の 3 本は 2026-10-28 までに 50M まで完走した）。1 本も読めなければ、PRD103 の段 3 の分割の Run（qbert 50M。`run_20261028-215839_a5_qb_aln_t103spl_s1`、実行中）が終わるまで待つ（§7 の段 0 の表の「0 本」の行）。段 1 は LunarLander の 21 本（1 本約 20 分、約 7 時間）。段 2 は qbert の 50M（約 10 時間）で、メイン機の PRD103 の段 3・段 4 との順は流す段取りで決める。
- 第三弾の割り込み (I) とは独立（同時には使わない。§8）。

## 11. 参照

- [PRD098 TempoRL 対応](098_temporl_skip_policy_10prd.md)、[PRD103 TempoRL 第二弾](103_temporl_reference_alignment_10prd.md)、[999 TempoRL の後続](999_temporl_skip_followups_10prd.md)、[PRD101 探索と学習のスケジュール](101_exploration_schedule_profiled_value_10prd.md)（ProfiledValue の記法）
- [ADR 0038](../adr/0038-actor-config-catalog-without-runmode.md)、[ADR 0040](../adr/0040-config-selection-final-value-and-run-profile-tier.md)、[ADR 0062](../adr/0062-networkmodel-holds-auxiliary-network-fed-by-primary-outputs-with-target-used-only-for-acting.md)、[ADR 0063](../adr/0063-replay-slot-keeps-per-step-reward-and-end-and-samples-variable-windows.md)、[ADR 0064](../adr/0064-profiledvalue-bare-scalar-reads-as-value-field.md)
- [200 DQN 系列（設計文書）](../design/200_dqn_agents.jp.md)、[150 ReplayBuffer（設計文書）](../design/150_replay_buffer.jp.md)、[行動の時間尺度（wiki）](../wiki/mechanisms/temporal_abstraction.md)、[324 DQN 系列（wiki）](../wiki/categories/324_dqn.md)、[318 強化学習 共通基盤（wiki）](../wiki/categories/318_rl_core.md)
- [TempoRL 論文解説](../../reports/biedenkapp2021_temporl_paper_2026-10-03.md)、[2026-10-28 LunarLander の実験記録](../experiments/default-dqn/lunarlander/2026-10-28_temporl-split-kmax-epsilon.md)
- [PRD103 の検証資材](../../core/anet-core/testdata/prd103/README.md)、[12 の bat の評価専用 Run の形](../../apps/12_batch_run.bat)
