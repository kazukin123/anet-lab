# 勾配ノイズスケールは更新自身の batch を小バッチに分けて学習経路の中で測り、BF16 も対象にする

勾配ノイズスケール（[PRD 092](../memo/092_gradient_noise_scale_10prd.md)）は、replay から別に取る probe ではなく、`Optimize` に重み付きの要素損失を渡し、同じ損失グラフから小バッチ k 個の勾配を `torch::autograd::grad`（retain_graph）で取って測る。測りたいのは PER・IS 重み・τ・Munchausen を含む実際の更新のノイズであり、この形なら forward を増やさず、乱数・`.grad`・optimizer の state にも触れないので、測定の ON/OFF で学習系列が変わらない。本線の Learner は BF16 autocast（GradScaler 無し、backward は autocast の外）なので、BF16 も同じ経路で測る。

## Considered Options

- **ReplayBuffer から一様に取る probe（可塑性・churn と同じ型）**: 抽選分布と IS 重みが実際の更新と違い、forward が増える。棄却。
- **ε を小バッチごとに引き直して総ノイズを測る**: forward が増え、RNG を消費する。分岐 Run の結果が条件付きの値の読みと食い違い、かつ Noisy probe の Q 差が大きいときに作る（gate 付きで先送り）。
- **FP16（GradScaler）も対象にする**: scale / unscale を小バッチ勾配にも挟む必要がある。利用する構成が無いので、購読したら学習開始前に fail-fast。
- **AMP の Run を対象外にする（暫定 PRD）**: 本線が BF16 で、この PRD が答えたい Run を測れなくなる。棄却。
