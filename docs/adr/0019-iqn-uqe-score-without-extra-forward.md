# IQN+UQE の Q 系値は同一 forward の risk-biased action score を使う

IQN+UQE は、行動選択用の1回のforwardへrisk-biased tausを入力する。`uqe_use_tail_mean=true`では実効下限から1までを配置してupper-tail meanを計算し、`false`では実効下限の1点から`Zτ`を得る。任意の`full_distribution_query`を有効にした場合は、full `[0,1]` tausをrisk tausへ連結し、同じforwardの`q_dist`をrisk/fullへ分割する。

`q_values`、Actor Qヒント、`episode_start_action_q_margin`などのQ系値にはrisk側だけから得たaction scoreを使う。full側は`full_q_values`/`full_q_quantiles`として可視化・診断へ公開し、Q系scoreへ混ぜない。`E[Z]`専用の追加forwardは、Actor推論を二重化し[ADR 0010](0010-actor-priority-mean-q-approx.md)の判断にも反するため採用しない。

## Consequences

- IQN+UQEでは`q_values`と`uqe_values`が同じscoreになる。`uqe_use_tail_mean=true`ならupper-tail mean、`false`なら`Zτ`であり、両者を一括してtail meanとは呼ばない。
- Actor QヒントとQ系metricも同じrisk-biased action scoreを使う。これらをfull-distributionの`E[Z]`として解釈しない。
- full queryは既定disabledでIQN+UQEだけが実行時に利用する。非IQN modeではenabled設定を休眠状態として無視し、profile切替時の連動変更を要求しない。IQNでenabledにしたままUQE以外を選ぶ構成はfail-fastする。有効時もNetwork forwardは1回だが、fusion以降の演算量と中間Tensorはrisk K＋full Kへ増える。
- `q_quantiles`はrisk側を維持し、`full_q_quantiles`をQValuePanelが優先する。full分布を使うscalar metricは本決定では追加しない。
- ReplayBuffer共通層は引き続きActor Qヒントをopaque payloadとして運び、IQN/UQEの意味を持ち込まない。
- 再訪条件: full queryの実測costが許容でき、診断価値が確認できた場合にscalar metricやtrain policyでの利用を検討する。
