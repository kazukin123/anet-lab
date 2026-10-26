# 行動の時間尺度

## 今の結論

TempoRLは行動へのコミット長を学ぶ設定で素のQ学習より速い学習を、FiGARはA3C・TRPO・DDPGへの行動反復追加で改善を報告した。小さい時間刻みでQ学習が失敗する条件も報告されている。anet の TempoRL は行動を選ぶ Q を変えないので、NEET の機序には効かない見込みである（DropMerge での実測はまだ無い）。anet の実測（3 seed）では、LunarLander で分離型の 2 腕が無効をブレ幅を超えて上回り、速く届いた。GridMaze では届く速さは無効と同じか遅く、差が出たのは 6×10 の終盤の安定さ（無効が終盤に不安定になる）だけだった。GridMaze 5×5 の既定 seed では、スキップ長を UQE の楽観で選ぶとゴールに届かなかった（1 seed）。長い j の不確かさ・閉ループの割り込み・全 k の一括学習は後続の暫定 PRD で扱い、スキップ中の forward の省略は作らない理由と形をそこに残した。

## 分かっていること

- TempoRLは行動へのコミット長を学ぶ設定で素のQ学習より速い学習を報告するが、anetでの実測ではない。（出典: [biedenkapp2021_temporl_paper_2026-10-03.md](../../../reports/biedenkapp2021_temporl_paper_2026-10-03.md)、アブストラクト）。
- FiGARはA3C・TRPO・DDPGへ行動反復を組み合わせた性能改善を報告している。DQNやDropMergeへそのまま移植した結果ではない。（出典: [sharma2017_figar_paper_2026-10-03.md](../../../reports/sharma2017_figar_paper_2026-10-03.md)、要旨）。
- 小さな時間刻みでQ学習が失敗する条件と、時間刻み変更に頑健な方法の実験が報告されている。単なるframe skip変更の同値性を保証する結果ではない。（出典: [tallec2019_time_discretization_paper_2026-10-03.md](../../../reports/tallec2019_time_discretization_paper_2026-10-03.md)、7. 結論）。
- action gapを広げる作用素について、最適性の十分条件と自転車・Atari実験が報告された。現行のMunchausenの効果量と同じ証拠にはしない。（出典: [bellemare2016_increasing_action_gap_paper_2026-10-03.md](../../../reports/bellemare2016_increasing_action_gap_paper_2026-10-03.md)、概要）。
- TempoRL の論文の 3 構成（連結・文脈化・重み共有）はどれも行動をスキップ側の入力に取り、スキップの目標値の終点の価値は行動価値で測る。分布型や PER との組み合わせは論文に無い。（出典: [biedenkapp2021_temporl_paper_2026-10-03.md](../../../reports/biedenkapp2021_temporl_paper_2026-10-03.md)、3.3・付録 B・キューの論点 4）。
- DropMerge では 1 step の NOOP は盤面をほとんど変えないので、その価値は次の step で最善手を打つ価値を γ で 1 回割り引いたものとほぼ同じになり、最善の DROP との本来の差は (1−γ)·V ≈ 0.3% しかない。推定の揺れで順位が入れ替わる（出典: [999_smdp_transition_duration_10prd.md](../../memo/999_smdp_transition_duration_10prd.md)、§1）。
- anet の TempoRL の設計では行動を選ぶ Q は 1 step の TD のままなので、静止した盤面での NOOP と最善 DROP の差 (1−γ)V は変わらず、NEET の抑制は非ゴールにした。（出典: [098_temporl_skip_policy_10prd.md](../../memo/098_temporl_skip_policy_10prd.md)、§3・§4 D1）。
- 論文の重み共有の構成は、行動出力の argmax をスキップ出力の入力にする形で、全行動分を出して行動で絞る形（anet の分離型）は論文に無い。論文は構成どうしを MountainCar と LunarLander で比べ（重み共有は前者で最良、後者で最悪）、Atari では重み共有だけを使った。（出典: [999_temporl_skip_followups_10prd.md](../../memo/999_temporl_skip_followups_10prd.md)、§2.2）。
- anet の学習（GPU、N=128 前後）では、Actor の forward の費用は行の数より呼び出しの回数に強く依ると推定され（Atari の lane 数の実測からの推定）、判断する lane が 1 本でもあれば forward は 1 回残るので、スキップ中の lane の forward を省いても速くならない見込みである。計算の節約が効くのは lane が 1〜数本のとき。（出典: [999_temporl_skip_followups_10prd.md](../../memo/999_temporl_skip_followups_10prd.md)、§2.3・§4 D9）。
- anet の GridMaze での実測（agent seed 1 本ずつ）: 6×10 では skip 有効の分離型 2 腕（遮断・合算）が無効より早く評価報酬 +0.93 に届き、包含型（action を入力に取る形）も届くが分離型より遅かった（正になる step が 60k 遅く、Run 全体の平均も悪い）。5×5 の既定 seed では有効の 2 型（分離・包含）が 400k step で届かず（評価は毎回タイムアウト、学習 lane の成功率 0、Q がすべて負）、5×5 の別の seed では有効も届くが無効より遅かった。効果は迷路の形に依存して不明で、既定 seed の不達は分散ではなく、ゴールに届かないまま終わる形だった。（出典: [098_temporl_skip_policy_10prd.md](../../memo/098_temporl_skip_policy_10prd.md)、§7「段 1 の途中経過」）。
- anet の GridMaze 6×10（3 seed、500k）: 届く速さ（評価の EMA が初めて 0.5 以上）は、分離×遮断 274k〜369k、無効 191k〜392k で同じくらいだった。差が出たのは終盤の安定さで、分離×遮断は 3 seed とも最後まで +0.93 を保ち、無効は 3 seed とも届いたあと終盤に不安定になった（最後の EMA −1.01〜−1.09）。無効が不安定になる原因は確かめておらず、GridMaze の設定の UQE の楽観と楽観的な目標値が候補。5×5 では無効のほうが速く届き（1 seed ずつ）、GridMaze では TempoRL は届く速さで有利にならなかった。学習側の判断率は 0.35 から 0.76 に上がり、長い j はほとんど使われなくなった。（出典: [098_temporl_skip_policy_10prd.md](../../memo/098_temporl_skip_policy_10prd.md)、§7「段 1 の 3 seed」）。
- anet の LunarLander（IQN＋Munchausen、J=4、2M、3 seed）: 評価 policy 報酬の Run 全体の平均は、無効 21、分離×遮断 122、分離×合算 137、包含×遮断 96 で、分離型の 2 腕は最小の seed が無効の最大の seed を上回った。評価の EMA が 200 に届いたのは、分離×合算が 3 seed とも約 1.0〜1.1M、無効は 3 seed 中 1 seed だけ。包含型は平均では上回るが、最小の seed が無効と重なる。学習側の判断率は終盤まで 0.42〜0.48 で、論文の LunarLander のような「ほぼ毎 step 判断する方策」にはならなかった。（出典: [098_temporl_skip_policy_10prd.md](../../memo/098_temporl_skip_policy_10prd.md)、§7「段 2 の結果」）。
- GridMaze 5×5 の既定 seed で skip 有効の学習がゴールに届かなかったのは、スキップ長を行動と同じ UQE の楽観で選んだときだった。評価では一度も届かず、学習側もほぼ届いていない。スキップ長だけを平均に対する ε-greedy で選ぶと、同じ迷路で 305k に届き（無効は 124k）、最後まで保った。UQE のままの対照は今のビルドでも同じく届かず、指標も前の Run と一致した（どれも 1 seed）。5×5 の seed 3 と 6×10 では、UQE のままでも届いた。（出典: [999_temporl_skip_followups_10prd.md](../../memo/999_temporl_skip_followups_10prd.md)、§7「段 0 の結果」、[098_temporl_skip_policy_10prd.md](../../memo/098_temporl_skip_policy_10prd.md)、§9 の 2）。

## 分かっていないこと

- anet の DQN（IQN・Munchausen・PER・UQE）で TempoRL が判断の回数と学習の速さにどう効くか。（出典: [098_temporl_skip_policy_10prd.md](../../memo/098_temporl_skip_policy_10prd.md)、§7）。
- スキップ長を UQE の楽観で選ぶと、なぜ 5×5 の既定 seed ではゴールに届かなくなるのか（楽観が長い j に寄ることが、どう探索や価値を損なうかは未確認）。GridMaze 6×10 で無効の腕が終盤に不安定になるのが、GridMaze の UQE の設定そのものによるのか。（出典: [098_temporl_skip_policy_10prd.md](../../memo/098_temporl_skip_policy_10prd.md)、§9 の 2・§7「段 1 の 3 seed」）。
- skip 価値のアンサンブルで長い j の不確かさを測り、悲観的に選ぶ形（UTE）や、行動を変える価値の差が閾値を超えたときだけ切り替える形（Lazy-MDP）が、anet で長い j の選び誤りや開ループの硬さに効くか。どちらも原文は未確認。（出典: [999_temporl_skip_followups_10prd.md](../../memo/999_temporl_skip_followups_10prd.md)、§2.2・§9 の 1）。

## 撤回・更新

- 2026-10-26: 「5×5 の既定 seed で崩壊する機構は未解明」を更新した。原因はスキップ長を UQE の楽観で選ぶことと分かった（仕組みは未確認のまま）。「GridMaze での効果は迷路の形に依存して不明」も、6×10 の 3 seed の結果で今の結論を改めた。
- 2026-10-26（2 回目）: 5×5 の既定 seed の件を「序盤に見つけたゴールを保持できず崩壊」と書いていたのを撤回した。評価では一度も届かず、学習側もエピソードの報酬の最大が序盤に一度 +0.03 になっただけで、見つけたゴールを失ったのではなかった。正しくは「スキップ長を UQE で選ぶとゴールに届かない」。あわせて GridMaze の結論を、届く速さ（同じか遅い）と終盤の安定さ（6×10 で差）に書き分けた。（出典: [999_temporl_skip_followups_10prd.md](../../memo/999_temporl_skip_followups_10prd.md)、§7「段 0 の結果」、[098_temporl_skip_policy_10prd.md](../../memo/098_temporl_skip_policy_10prd.md)、§7「段 1 の 3 seed」）。

## 関連

- [atari](../subjects/atari.md)。
- [evaluation](../mechanisms/evaluation.md)。
- [仕組みと契約](../../design/200_dqn_agents.jp.md)。
- [TempoRL 対応 PRD](../../memo/098_temporl_skip_policy_10prd.md)。
- [TempoRL の後続の暫定 PRD](../../memo/999_temporl_skip_followups_10prd.md)。
- [CONTEXT.md](../../../CONTEXT.md) の「スキップ」「スキップ長」「スキップ価値」「判断」。
