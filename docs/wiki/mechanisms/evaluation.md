# 評価と比較の条件

## 今の結論

Breakoutではε=0.01で見えない腕の差がε=0に現れ、SNの効果の符号も20Mと50Mで逆転した。DropMergeでは同設定複製のtarget評価が一致した一方、online評価にはずれが残った。

## 分かっていること

- ε=0.01で見えない腕の差がε=0で現れ、20Mと50MでSNの符号も変わった。checkpoint時点と評価方策を両方揃える。（出典: [2026-09-15_replay-lane-window.md](../../experiments/default-dqn/atari/2026-09-15_replay-lane-window.md)、現時点のまとめ3〜5）。
- 中間時点のseed差は3〜5割に達し、単一seedの山と頭打ちを一般化できなかった。（出典: [2026-09-23_realtime-profile-100m.md](../../experiments/default-dqn/atari/2026-09-23_realtime-profile-100m.md)、現時点のまとめ1〜4。Run: run_20260923-020945_atari_kung_fu_master / run_20260923-095708_atari_kung_fu_master）。
- DropMergeではtargetの評価専用Runの複製が一致した一方、onlineは同設定複製でずれた。学習中の1局評価と多数局評価で検出できる差も異なる。（出典: [2026-10-15_gradient-noise-batch.md](../../experiments/default-dqn/dropmerge/2026-10-15_gradient-noise-batch.md)、現時点のまとめ4・10）。

## 分かっていないこと

なし

## 撤回・更新

なし。

## 関連

- [遷移の結果は Step の結果で運び、速報値か確定値かは読むイベントで分ける](../../adr/0060-transition-results-travel-in-step-result-and-event-decides-provisional-or-final.md)。
- [BatchEnv の Step は終端の lane をリセットせず、Runner が通知のあとに終わったエピソードをリセットする](../../adr/0059-batchenv-step-does-not-reset-runner-resets-ended-episodes-after-notification.md)。
- [step trace の近似優先度は Agent が出す step 部品が計算し、Actor の設定キーで eval_panel にだけヒントを作らせることはしない](../../adr/0058-approx-priority-column-computed-by-agent-step-diagnostics.md)。
- [Evaluation Trace の列はメトリクスの定義と分けて番号付きで宣言し、遷移の時点に合う target だけを読む](../../adr/0057-eval-trace-columns-declared-apart-from-metrics-and-read-at-transition-time.md)。
- [遷移の結果を Step の結果で運ぶ — 通知の中で読む遷移の結果を、終端の lane でもリセット前の値にする PRD](../../memo/097_env_auto_reset_after_notification_10prd.md)。
- [Evaluation Trace — EvalPanel の遷移を step trace として表で見て、その時点を再生する PRD](../../memo/094_evalpanel_step_history_10prd.md)。
- [replay_ratio_lane_window](../mechanisms/replay_ratio_lane_window.md)。
- [仕組みと契約](../../design/200_dqn_agents.jp.md)。
- [Envをまたぐ比較の独自分析](../../experiments/default-dqn/README.md#env-をまたぐ際の扱い)。
