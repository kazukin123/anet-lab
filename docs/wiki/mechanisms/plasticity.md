# 可塑性指標と成績

## 今の結論

同じ勾配step付近の摩耗はreplay ratio間で似ていたが、付随する新規データ量と成績の推移は異なった。SiLUやLN512で機序指標が改善しても、RR8短期screeningの成績は改善しなかった。

## 分かっていること

- 同勾配step付近の損傷はRR間で似たが、付随する新規データ量は異なった。dead ratioは設定内の転換点と、設定間の水準比較で意味が違う。（出典: [2026-08-28_plasticity.md](../../experiments/default-dqn/atari/2026-08-28_plasticity.md)、現時点のまとめ。Run: run_20260828-062103_plasticity_rr8_breakout）。
- 設定修正後も同勾配座標の摩耗と、RRごとの崩壊・天井・上昇の違いが見られた。（出典: [2026-08-29_bundle-fix-rebaseline.md](../../experiments/default-dqn/atari/2026-08-29_bundle-fix-rebaseline.md)、現時点のまとめ。Run: run_20260829-190614_a5_breakout_rr4）。
- SiLUやLN512で機序指標を改善できても、RR8短期screeningでscore改善は得られなかった。q_gapの定常値が動かないことから因果を導く主張は撤回済みである。（出典: [2026-08-30_protection-screening.md](../../experiments/default-dqn/atari/2026-08-30_protection-screening.md)、現時点のまとめ1〜5）。
- 全層spectral_capは重みの実効ノルムを抑えて発散しなかったが、RR4の成績は動かなかった。（出典: [2026-09-08_replay-ratio-mechanism.md](../../experiments/default-dqn/atari/2026-09-08_replay-ratio-mechanism.md)、現時点のまとめ6・7。Run: run_20260908-073326_rr4_capall_munch）。

## 分かっていないこと

なし

## 撤回・更新

- 2026-09-08: 全層でノルムを抑えると発散するため両立点が無い → spectral_capの当該構成は抑制と非発散を両立したがscoreは改善しなかった（出典: [RR機序](../../experiments/default-dqn/atari/2026-09-08_replay-ratio-mechanism.md)、現時点のまとめ6・7）。

## 関連

- [atari](../subjects/atari.md)。
- [evaluation](../mechanisms/evaluation.md)。
- [仕組みと契約](../../design/200_dqn_agents.jp.md)。
