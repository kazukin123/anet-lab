# 計測の負荷と比較条件

## 今の結論

当時のTimeHistogramでは22.37秒中16.26秒を全生値の再構築が占め、出力intervalを下げても取得・再構築の量は減らなかった。Breakoutの可塑性probeはinterval100で約2.1%のthroughput低下、500ではほぼ差がなかった。

## 分かっていること

- 2026-07のTimeHistogram調査では、画像の出力intervalを下げても全valid値の取得と全生値の再構築は減らず、TrainThreadを同期的に止めていた。Run `run_20260720-141314` の22.37秒中16.26秒が再構築だった。この記録は当時の経路の実測であり、現行の全診断へ同じ比率を適用しない（出典: [PRD056](../../memo/056_timehistogram_fine_grid_10prd.md)、問題・前提）。
- Breakoutの可塑性probeは、interval 100では約2.1%のthroughput低下、500ではほぼ差なしだった。probe batchを変えるとdead ratioの見え方も変わるため、測定条件を腕の差から分ける（出典: [可塑性較正](../../experiments/default-dqn/atari/2026-08-28_plasticity.md)、現時点のまとめ7・8。Run: `run_20260829-094914_ab_s2_probe_b4096_5m` ほか）。

## 分かっていないこと

なし

## 撤回・更新

なし。

## 関連

- [可観測性](../../design/140_observability.jp.md)、§2・§7。
- [PRD092の計測契約](../../memo/092_gradient_noise_scale_10prd.md)。
- [学習経路内のGNS計測の判断](../../adr/0054-gradient-noise-scale-in-band-microbatch-split-includes-bf16.md)。
- [勾配ノイズの知見](gradient_noise_batch.md)。
