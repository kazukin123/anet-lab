# 行動の時間尺度

## 今の結論

TempoRLは行動へのコミット長を学ぶ設定で素のQ学習より速い学習を、FiGARはA3C・TRPO・DDPGへの行動反復追加で改善を報告した。小さい時間刻みでQ学習が失敗する条件も報告されている。anet の TempoRL は行動を選ぶ Q を変えないので、NEET の機序には効かない見込みである（DropMerge での実測はまだ無い）。anet の GridMaze での実測（agent seed 1 本ずつ）では、効果は迷路の形に依存して不明。長い j の不確かさ・閉ループの割り込み・全 k の一括学習は後続の暫定 PRD で扱い、スキップ中の forward の省略は作らない理由と形をそこに残した。

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
- anet の GridMaze での実測（agent seed 1 本ずつ）: 6×10 では skip 有効の分離型 2 腕（遮断・合算）が無効より早く評価報酬 +0.93 に届き、包含型（action を入力に取る形）も届くが分離型より遅かった（正になる step が 60k 遅く、Run 全体の平均も悪い）。5×5 の既定 seed では有効の 2 型（分離・包含）が 400k step で届かず（評価は毎回タイムアウト、学習 lane の成功率 0、Q がすべて負）、5×5 の別の seed では有効も届くが無効より遅かった。効果は迷路の形に依存して不明で、既定 seed の不達は分散ではなく崩壊である。（出典: [098_temporl_skip_policy_10prd.md](../../memo/098_temporl_skip_policy_10prd.md)、§7「段 1 の途中経過」）。

## 分かっていないこと

- anet の DQN（IQN・Munchausen・PER・UQE）で TempoRL が判断の回数と学習の速さにどう効くか。（出典: [098_temporl_skip_policy_10prd.md](../../memo/098_temporl_skip_policy_10prd.md)、§7）。
- 5×5 の既定 seed で skip 有効の学習が、序盤に見つけたゴールを保持できずに崩壊する機構。（出典: [098_temporl_skip_policy_10prd.md](../../memo/098_temporl_skip_policy_10prd.md)、§9 の 2）。
- skip 価値のアンサンブルで長い j の不確かさを測り、悲観的に選ぶ形（UTE）や、行動を変える価値の差が閾値を超えたときだけ切り替える形（Lazy-MDP）が、anet で崩壊や開ループの硬さに効くか。どちらも原文は未確認。（出典: [999_temporl_skip_followups_10prd.md](../../memo/999_temporl_skip_followups_10prd.md)、§2.2・§9 の 1）。

## 撤回・更新

なし。

## 関連

- [atari](../subjects/atari.md)。
- [evaluation](../mechanisms/evaluation.md)。
- [仕組みと契約](../../design/200_dqn_agents.jp.md)。
- [TempoRL 対応 PRD](../../memo/098_temporl_skip_policy_10prd.md)。
- [TempoRL の後続の暫定 PRD](../../memo/999_temporl_skip_followups_10prd.md)。
- [CONTEXT.md](../../../CONTEXT.md) の「スキップ」「スキップ長」「スキップ価値」「判断」。
