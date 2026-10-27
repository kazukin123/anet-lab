# DefaultDQNAgent ハイパーパラメータ探索

DefaultDQNAgent 系の探索記録です。ReplayBuffer、PER、batch、replay ratio、target 更新など、Agent 側の共通機構を軸に整理します。


## 知見のまとめ

- [wikiの知見](../../wiki/subjects/atari.md)から題材・機構と正本をたどる。

## Env 別記録

| Env | 記録 | 概要 |
|---|---|---|
| Atari | [Atari 探索記録](atari/README.md) | ALE 直結。ベースライン定義（Breakout / Pong）、Atari-5 横断、可塑性メトリクスの較正と保護機構、BTR 実装との差分移植、replay の lane 窓と eval コスト |
| DropMerge | [DropMerge 探索記録](dropmerge/README.md) | 長期 Run における batch size と replay ratio、IQN 導入、Atari からのフィードバック（Munchausen・NoisyNet など）、勾配ノイズの機序と評価専用 Run による判定 |
| LunarLander | [LunarLander 探索記録](lunarlander/README.md) | IQN 導入、QR/IQN 比較、ReplayBuffer capacity、warmup、UQE、quantile sample 数の探索、NoisyNet の導入確認 |
| GridMaze | [GridMaze 探索記録](gridmaze/README.md) | TempoRL のスキップ長の方策（UQE 対 ε-greedy）の 6×10 比較 |


## 独自の分析

以下は複数campaignを比較するときの手順と、Envをまたぐ判断の方針を定める。Runの数値・比較結果・その結論はwikiから各campaignへたどる。

#### アルゴリズム比較ではseedより先に実効設定を揃える

同じseedでも、ネットワーク構造や乱数消費が違えばQRとIQNの軌道は直ちに分岐する。それでも同じseed集合を使うことは、環境初期化条件を揃えるblockとして有用である。
比較時はseed数を増やす前に、各Run artifactの`config/config_data.txt`で少なくとも次を一致させる。

- ReplayBuffer capacityとlearner warmup
- batch size、replay ratio、学習率、PER、n-step、target更新
- train/eval policyと探索減衰
- exp-step budgetとeval cadence
- backendの決定論設定

単一seedの最良Runではなく、同一budgetでの到達時間、終盤window、失敗Run数、seed間の範囲を比較する。


### Env をまたぐ際の扱い

Agent 側の設定名や更新量の式は共有できますが、最適値は Env に依存します。
特に次の要素が異なる結果は、別 Env へそのまま一般化しません。

- 報酬密度と遅延
- エピソード長
- 行動の時間的意味
- ReplayBuffer 内の経験分布
- 局所解と探索方策

実装契約からEnv共通と判断できる機構や比較上の注意は、単一Envで発見した段階でも共通知見として記録します。
一方、その機構が成績を改善・悪化させる方向まで一般化する場合は、複数Envで再現した事実とEnv固有の推測を分けて記録します。

### Munchausenの温度と別Envへの適用

τ は Q 値の差に対する温度なので、報酬の尺度が違う Env では効き方が変わりうる。効き具合は `36_agent_munchausen` の `03_clip_ratio` と `04_bonus_mean` で見る。Atariで選んだ値の採否は、そのEnvのRunで決める。
