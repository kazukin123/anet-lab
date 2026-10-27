# 評価と比較の条件

## 今の結論

Breakoutではε=0.01で見えない腕の差がε=0に現れ、SNの効果の符号も20Mと50Mで逆転した。DropMergeでは同設定複製のtarget評価が一致した一方、online評価にはずれが残った。ImageClsでは、LRのcosineの長さが違うRunの順位が途中のstepと最終で入れ替わった。TempoRL の評価 slot では、スキップ長にも評価の方策（ε・UQE）が掛かり、行動の方策が `Greedy` 型の slot だけがスキップ長も greedy になる。`run.eval_schedule` で組んだ評価は learn step 0 で必ず 1 回発火し、未学習の方策の点が入る。

## 分かっていること

- ε=0.01で見えない腕の差がε=0で現れ、20Mと50MでSNの符号も変わった。checkpoint時点と評価方策を両方揃える。（出典: [2026-09-15_replay-lane-window.md](../../experiments/default-dqn/atari/2026-09-15_replay-lane-window.md)、現時点のまとめ3〜5）。
- 中間時点のseed差は3〜5割に達し、単一seedの山と頭打ちを一般化できなかった。（出典: [2026-09-23_realtime-profile-100m.md](../../experiments/default-dqn/atari/2026-09-23_realtime-profile-100m.md)、現時点のまとめ1〜4。Run: run_20260923-020945_atari_kung_fu_master / run_20260923-095708_atari_kung_fu_master）。
- DropMergeではtargetの評価専用Runの複製が一致した一方、onlineは同設定複製でずれた。学習中の1局評価と多数局評価で検出できる差も異なる。（出典: [2026-10-15_gradient-noise-batch.md](../../experiments/default-dqn/dropmerge/2026-10-15_gradient-noise-batch.md)、現時点のまとめ4・10）。
- DropMerge の学習中の評価の値はセッションごとの報酬の合計の平均で、終局の罰（−10、NoDropTimeout は −20）を含む。評価専用 Run の平均点はログのゲームの点で、罰を含まない。罰をそろえると、学習中の評価 ×8（Eval1・Eval3 を 1 セッション 8 局）の最後の 10M と最終 checkpoint の評価専用 Run の差は 8 本で平均 −0.1%（−1.5〜+0.8%）で、×8 は系統的には低く出ない。1 本ずつでは ±1.5% ほどずれ、最も低かった 46a は最後の 2 つの 10M 窓が横ばいだった（出典: [2026-10-15_gradient-noise-batch.md](../../experiments/default-dqn/dropmerge/2026-10-15_gradient-noise-batch.md)、この文書の読み方・探索ブロック11 判断更新。Run: run_20261019-130757_dmfb_42g_34a_ctl2 / run_20261019-184443_dmfb_42b_34a_lr7.1e-5 / run_20261021-001853_dmfb_45a_42j_ctl / run_20261022-132905_dmfb_46a_45a_ctl ほか）。
- DropMerge の Run ごとのぶれは、IQN+M の 2 seed で 1〜2%、QR+M の 2 seed で 10〜15%（どちらも評価専用 Run）、nzrep と後の系列の seed 2 で約 6%（学習側の平均点）だった。QR+M の 2 Run の設定 dump の差は run.seed と Run 名だけで、評価専用 Run の標準誤差は 20 前後なので、測定の揺れではなく学習結果のばらつきだった（出典: [2026-10-01_atari-feedback.md](../../experiments/default-dqn/dropmerge/2026-10-01_atari-feedback.md)、現時点のまとめ13・探索ブロック04。Run: run_20261003-081944_dmfb_qr_munch / run_20261003-160717_dmfb_qr_munch_s2）。
- qbert の ε=0 断面の第 8 面到達は、断面ごとに大きく上下した（例: ε 腕 40.6M 93/100 → 45.7M 32/103、64 env・1M の seed 1 20.4M 68/102 → 25.5M 4/109）。同じ落ち込みは 128 env・2M の seed 2・3 と UQE にも出ており、1 断面の比較では判定しない（出典: [2026-09-27_btr-gap-qbert-bz.md](../../experiments/default-dqn/atari/2026-09-27_btr-gap-qbert-bz.md)、現時点のまとめ12・探索ブロック03。Run: run_20260927-035604_a5_qbert_btr50m_eps005_end0 / run_20260928-065041_a5_qbert_btr50m_eps005_env64）。
- ImageCls の ResNet18ish_hr＋Mixup/CutMix では、LR の cosine の長さが違う Run の順位が途中の step と最終で入れ替わった。15M の時点は、学習予算 30M の Run が 74.5%、20M の Run が 82.0%、cosine を 15M で終える Mixup なしの Run が 83.7% で、最終は 86.4・86.0・83.7% だった。cosine の長さが違う Run は最終同士で比べる（出典: [2026-06-25_food101-scratch.md](../../experiments/image-cls/imagecls/2026-06-25_food101-scratch.md)、現時点のまとめ7・探索ブロック05。Run: run_20260707-084617_imgcls_ResNet18HR32_30 / run_20260706-124351_imgcls_ResNet18HRmix / run_20260704-214337_imgcls_ResNet18ish_hr）。
- ImageCls では、同じ実効設定・同じ seed の 2 本が、cudnn deterministic の設定でも同じ exp step で 0.3〜1.0pt ずれた（出典: [2026-06-25_food101-scratch.md](../../experiments/image-cls/imagecls/2026-06-25_food101-scratch.md)、探索ブロック02。Run: run_20260627-234723_imgcls_Res18_do1 / run_20260630-004827_imgcls_ResNet18ish_hr）。
- GridMaze で、安定してゴールできるようになった後も、EvalPanel の手操作で点数の方策がほとんど通らないセルへ誘導してから再開させると、ゴールに着かなかった（ユーザーの観察。Run 名の記録は無い）。今の設定では、開始位置は (0, 0) に固定、観測は位置の 2 次元だけで、手操作の遷移は学習に入らない。固定の開始位置からの評価の成功は、通らない状態から届くことを示さない（出典: [999_intrinsic_reward_novelty_10prd.md](../../memo/999_intrinsic_reward_novelty_10prd.md)、§2.7「観察」「今の構成」）。
- anet の TempoRL では、評価 slot のスキップ長にも評価の方策が掛かる（Atari の `eval` は ε 0.01、LunarLander の `eval` は UQE の τ 0.25）。行動の方策が `Greedy` 型の slot（`[eval_panel]`）だけは、skip の葉が無くてもスキップ長が同じ Greedy になる。ε-greedy の ε 0（Atari の `[greedy]`）は Greedy 型ではないが、スキップ長も ε 0 の ε-greedy で選ぶ。学習したスキップ価値そのものを測るには、行動もスキップ長も greedy の評価が要る（出典: [103_temporl_reference_alignment_10prd.md](../../memo/103_temporl_reference_alignment_10prd.md)、§2.2 の T12・§4 D7）。
- 評価 slot の action_info を `@train` の event で scalar にすると、その値は評価 runner 自身の step 軸に載り、`@session_end` の値（学習側の step 軸）とは換算できない（比は Run の中で 0.000039 から 0.0075 へ動く）。評価 session の中の割合や平均を学習の進みと並べるには、session の終わりに学習側の座標で出す必要がある（出典: [103_temporl_reference_alignment_10prd.md](../../memo/103_temporl_reference_alignment_10prd.md)、§2.2 の T13・§4 D8、[140 可観測性 §6.x](../../design/140_observability.jp.md)）。
- `run.eval_schedule.[tag].interval` で組んだ評価は、step の値によらず初回に必ず 1 回発火する（learn step 0 の未学習の方策の点が入る）。背景の評価は、次の発火のときに前回が終わっていなければ学習が待つ。Atari-5 の 400 エピソードの ε=0 の評価では、未学習の方策が打ち切り上限に張り付いた 2 ゲームで初回の評価が 1 時間級になり、次の発火で学習が 692.6s・885.5s 止まった（出典: [999_eval_schedule_first_fire_10prd.md](../../memo/999_eval_schedule_first_fire_10prd.md)、Context・実測）。

## 分かっていないこと

- GridMaze で通らないセルから再開してゴールに着かないとき、点数の方策は壁へ向かう行動を繰り返して位置が変わらないまま打ち切られているのか（出典: [999_intrinsic_reward_novelty_10prd.md](../../memo/999_intrinsic_reward_novelty_10prd.md)、§2.7「見立て」。Run では未確認）。

## 撤回・更新

なし。

## 関連

- [遷移の結果は Step の結果で運び、速報値か確定値かは読むイベントで分ける](../../adr/0060-transition-results-travel-in-step-result-and-event-decides-provisional-or-final.md)。
- [BatchEnv の Step は終端の lane をリセットせず、Runner が通知のあとに終わったエピソードをリセットする](../../adr/0059-batchenv-step-does-not-reset-runner-resets-ended-episodes-after-notification.md)。
- [step trace の近似優先度は Agent が出す step 部品が計算し、Actor の設定キーで eval_panel にだけヒントを作らせることはしない](../../adr/0058-approx-priority-column-computed-by-agent-step-diagnostics.md)。
- [Evaluation Trace の列はメトリクスの定義と分けて番号付きで宣言し、遷移の時点に合う target だけを読む](../../adr/0057-eval-trace-columns-declared-apart-from-metrics-and-read-at-transition-time.md)。
- [遷移の結果を Step の結果で運ぶ — 通知の中で読む遷移の結果を、終端の lane でもリセット前の値にする PRD](../../memo/097_env_auto_reset_after_notification_10prd.md)。
- [Evaluation Trace — EvalPanel の遷移を step trace として表で見て、その時点を再生する PRD](../../memo/094_evalpanel_step_history_10prd.md)。
- [内的報酬と方策の家族 暫定 PRD（GridMaze の学習の分布の外、§2.7）](../../memo/999_intrinsic_reward_novelty_10prd.md)。
- [replay_ratio_lane_window](../mechanisms/replay_ratio_lane_window.md)。
- [temporal_abstraction](../mechanisms/temporal_abstraction.md)。
- [TempoRL 第二弾（著者の実装との差）の PRD](../../memo/103_temporl_reference_alignment_10prd.md)。
- [eval スケジュールの初回発火の抑止 暫定 PRD](../../memo/999_eval_schedule_first_fire_10prd.md)。
- [仕組みと契約](../../design/200_dqn_agents.jp.md)。
- [Envをまたぐ比較の独自分析](../../experiments/default-dqn/README.md#env-をまたぐ際の扱い)。
