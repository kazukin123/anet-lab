# 全 Agent の AMP 指定を NN 実行設定に置く案

## 元の材料

[RFD0164　全 Agent の AMP 指定を NN 実行設定に置く案](../../../memo/0164_nn_forward_amp_consolidation_10rfd.md) blob:631dc5b64aa7b110aacc8adad998f0bbce7071bc

## 要点

- AMP の設定を NN 実行設定へ集約する暫定案である。どの Agent に適用するか、設定・公開契約をどうするかは未裁定である。
- DefaultDQN の Actor 方策などに分かれていた autocast の有無と FP16・BF16 の指定を整理する。
- 呼び出しごとの NN 実行設定で精度を表し、同じ目的の指定が Agent ごとに重複するのを減らす案である。
- target の行動選択と価値評価では必要な精度が異なる場合があり、個別の呼び出し条件を残す必要がある。
- ノイズの選択とは適用範囲が異なるため、既存の forward 呼び出しと autocast の位置を照合する。
- 既存 Agent の設定移行と値の等価性を確かめてから、共通化の範囲を決める。

## 内容

### 問題と目的

084 では精度を集約しない判断をした。理由は、ノイズが target 構築全体で 1 つの契約なのに精度は target 行動選択と価値評価で別であること、`Network::Forward` が精度を適用する形にすると DefaultDQN 以外の Agent と診断の経路へ波及すること、NoisyNet の目的に必要でないことである。本 RFD は、精度を NN 実行設定へ集約するなら全 Agent を同じ契約で移行することを前提に、必要性と方式を別途検討する。

### 起草時点の現行契約で確認したこと

DefaultDQN の hard target では、target 行動選択は target policy 側の `Autocast` が外側の Learner の autocast を切るため、`@bf16` では FP32 になる。target 価値評価と Munchausen の追加 forward は Learner 側の設定に従い BF16 になる。この非対称は現行の実効精度であり、移行で保存するか揃えるかは裁定事項である。

## 主要な概念

- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
- [321 Agent 共通基盤](../../concepts/framework/321_agent_core.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [計測の負荷と比較条件](../../concepts/measurement/diagnostic_cost.md)
