# DropMerge の NEET 診断（trace で足りるか確かめてから指標化）暫定 PRD

> 2026-10-03 起票。詳細設計は別途グリルで行う前提で、決めることを §5 に残した。
> 2026-10-23 改訂: §2 の Run 名 4 つを今のフォルダ名に直した（2026-10-15 の改名。値は変えていない）。
> 起点: DropMerge フィードバック実験（2026-10-01〜03）で、NEET の分析を一時スクリプトとログの解析で行ったこと。NOOP の指標の軸、lane ごとの集計、終了理由ごとの集計が、Viewer では見られない。
> 関連: [done/069_metrics_trace_channel_10prd.md](done/069_metrics_trace_channel_10prd.md)（trace チャネル）、[081_evalpanel_episode_trace_10prd.md](081_evalpanel_episode_trace_10prd.md)（EvalPanel の trace）、[932_episode_forensics_10prd.md](932_episode_forensics_10prd.md)（特殊なエピソードを丸ごと捕まえる仕組み。本 PRD とは重ねない）、[CONTEXT.md](../../CONTEXT.md) の「投了」「NEET」「NoDropTimeout」。

## 1. 目的と問題

NEET（合法 DROP があるのに NOOP を選び続ける病理）を、Run のたびに同じ見方で確かめられるようにする。まず trace チャネルで必要な情報が取れるかを確かめ、足りない集計だけをスカラーの指標にする。

2026-10-03 の分析で使った見方は次の 3 つで、どれも Viewer では見られない。

1. NOOP と最善 DROP の差、NOOP の選択率を、学習の進み（exp step）に合わせて見る。
2. lane（spatial exploration の ε の段）ごとに、学習側のエピソード長と終了理由を見る。
3. 評価のエピソードを終了理由（NoDropTimeout、投了、SpawnBlocked、ほか）で分け、スコア・盤面の果物数・エピソード長を比べる。

## 2. 現状（2026-10-03 の分析で確認したこと）

- `51_eval1/41_noop_uqe_win_rate` と `51_eval1/42_noop_uqe_margin`（`52_eval2` も同じ）は、評価 Runner 自身の step 軸で記録されている。QR51 100M（`run_20261002-031314_dmfb_qr`）では 1,426,150 点あり、学習の exp step の窓と重ねられない。分析では、評価セッションの session end 行が持つ `exp_step` と `mean.episode_steps` の累積で、評価の step を学習の exp step に写した。
- 写した結果、QR51 100M の NEET 期（8〜30M）は、差が −0.020〜−0.026 から −0.001〜−0.002 に潰れ、NOOP の選択率は 74〜84% だった。基準 IQN 100M（`run_20261001-191232_dmfb_base`）は −0.021〜−0.033 のままで、選択率は 0〜19% だった。
- lane ごとの指標は無い。分析では、学習ログの verbose 行（`train[<lane>]: Game Over … step_count=`）を lane 番号で 32 本ずつにまとめた。QR51 の NEET 期は、ほぼ貪欲な lane（ε 1e-4〜3e-4）だけエピソード長が 172 から 337 step に伸び、ε 0.2〜0.6 の lane は 148 から 164 step でほぼ変わらなかった。
- 終了理由ごとの集計は、1 エピソードの評価セッションの指標（`51_eval1/73,74,75` の終了理由、`51_eval1/12_ep_frct_mean`、`21_eval/01_target_reward`）を step で突き合わせて行った。IQN + Munchausen τ0.03 100M（`run_20261003-000810_dmfb_munch`）の 70〜82M では、NoDropTimeout で終わった回が 45%（スコア 1,840・果物 31 個）、SpawnBlocked が 34%（1,674・32 個）、投了が 12%（1,765・33 個）で、NoDropTimeout は盤面が埋まった終盤で起きていた。QR51 の 5〜30M の NoDropTimeout は 2% だけだが、果物 8 個・スコア 30 で止まる序盤の NEET だった。
- 時間切れの瞬間に置ける列が無かった割合（`51_eval1/78_timeout_cand_mean`）は、どの Run でもほぼ 0 だった。NoDropTimeout は投了ではなく NEET で終わっている。
- 同じ写し方で、QR51 + Munchausen τ0.03 100M（`run_20261003-081944_dmfb_qr_munch`）も集計した。差は健全な時期に −0.10〜−0.16 で、NEET 期（10〜22M）には 12〜14M に +0.024 まで振れ、NOOP の選択率は 86% に達した。
- 行動ごとの分位点の広がりは、EvalPanel の Q 値表（Std 列）でしか見られず、指標には無い。上の Run の 14M 付近の録画では、NOOP の広がりが 0.005〜0.007 と、DROP の 0.017〜0.045 より細かった。
- Munchausen の `36_agent_munchausen/07_soft_gap` は、既定の購読から外してある（`apps/runner/config/metrics_scalar.txt:313` のコメント行）。上の Run では CLI で M2 に足して記録した。
- trace チャネル（PRD069）と EvalPanel の trace（PRD081）がある。step ごとの行に必要な欄（行動、NOOP の差、ショットクロックの残り、lane、終了理由、スコア、果物数）を載せられれば、`inspect_run.py trace-csv` で同じ分析ができる可能性がある。

## 3. 提案の骨子

1. 段階 1: trace チャネルで §1 の 3 つの見方に必要な欄が取れるかを確かめる。取れるなら、指標は足さずに trace と `inspect_run.py` で分析する手順を整える。
2. 段階 2: trace では足りない集計（学習の exp 軸に載せた NOOP の差と選択率、lane の段ごとのエピソード長と終了理由、終了理由ごとのスコアと果物数）だけを、スカラーの指標にする。
3. スコアの軸は [085](085_dropmerge_eval_score_axes_10prd.md) の定義に合わせる。

## 4. User Stories

1. 研究者として、NEET の始まりと終わりを、学習曲線と同じ横軸で見たい。
2. 研究者として、NEET が貪欲な lane に偏っているかを、ログを解析せずに確かめたい。
3. 研究者として、NoDropTimeout が序盤の NEET か終盤の NEET かを、Run のたびに見分けたい。

## 5. 決めること（詳細設計で詰める）

1. trace の行の量（全 lane・全 step を載せるか、間引くか）。
2. trace に lane 番号と終了理由を載せられるか。
3. 段階 2 で指標にする集計の範囲と tag。
4. PRD932（エピソードを丸ごと捕まえる仕組み）との分担。本 PRD はスカラーと trace の側に限る案を既定にする。

## 6. 対象外

- NEET への対策そのもの（[086](086_dropmerge_macro_actions_10prd.md)、[999_smdp_transition_duration](999_smdp_transition_duration_10prd.md)、[999_decision_aligned_nstep](999_decision_aligned_nstep_10prd.md)）。

## 7. 依存と順序

依存は無い。対策の効き目を見る段になったら、段階 1 から始める。

## 8. 参照

- [CONTEXT.md](../../CONTEXT.md) の「投了」「NEET」「NoDropTimeout」の定義。
