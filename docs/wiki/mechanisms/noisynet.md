# NoisyNetの観測

## 今の結論

LunarLanderの3 seedではNoisyNetの成績差はseed幅内で、行動変化率は終盤に2〜4%へ縮んだ。DropMergeではεラダーへの上乗せではNEETが残り、NoisyNetへの置換で抑えられた。

## 分かっていること

- LunarLanderでは初期の行動変化率が大きく、終盤は2〜4%へ縮んだ。成績差は3-seedの幅内で、有効・無効を確定していない。（出典: [2026-10-06_noisynet.md](../../experiments/default-dqn/lunarlander/2026-10-06_noisynet.md)、現時点のまとめ）。
- DropMergeではεラダーへの上乗せと置換の結果が異なり、置換した構成でNEETが抑えられた。（出典: [2026-10-01_atari-feedback.md](../../experiments/default-dqn/dropmerge/2026-10-01_atari-feedback.md)、現時点のまとめ6。Run: run_20261008-015339_dmfb_21_nzrep）。
- batch増加腕は同じσでもQの変化と行動の食い違いが小さく、学習側の上乗せにはノイズ感度の違いが混ざった。（出典: [2026-10-15_gradient-noise-batch.md](../../experiments/default-dqn/dropmerge/2026-10-15_gradient-noise-batch.md)、現時点のまとめ5）。

## 分かっていないこと

なし

## 撤回・更新

なし。

## 関連

- [ニューラルネットワーク](../../design/130_neural_networks.jp.md)。
- [NN Forward の設定解釈を事前準備へ移し、用途別実行オブジェクトが State と Network の寿命を持つ](../../adr/0051-nn-forward-prepares-concrete-calls-and-purpose-owned-execution.md)。
- [NoisyNet の ε は module に持たせず、呼び出し側所有の NN 実行状態と NN 実行設定キーで駆動する](../../adr/0048-noisynet-epsilon-in-caller-execution-state-and-nn-forward-key.md)。
- [atari](../subjects/atari.md)。
- [evaluation](../mechanisms/evaluation.md)。
- [仕組みと契約](../../design/200_dqn_agents.jp.md)。
