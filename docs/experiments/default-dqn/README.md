# DefaultDQNAgent ハイパーパラメータ探索

DefaultDQNAgent 系の探索記録です。ReplayBuffer、PER、batch、replay ratio、target 更新など、Agent 側の共通機構を軸に整理します。

## Env 別記録

| Env | 記録 | 概要 |
|---|---|---|
| Atari | [Atari 探索記録](atari/README.md) | ALE 直結。ベースライン定義（Breakout / Pong）、Atari-5 横断、可塑性メトリクスの較正と保護機構 |
| DropMerge | [DropMerge 探索記録](dropmerge/README.md) | 長期 Run における batch size と replay ratio を中心とした探索 |
| LunarLander | [LunarLander 探索記録](lunarlander/README.md) | IQN 導入、QR/IQN 比較、ReplayBuffer capacity、warmup、UQE、quantile sample 数の探索 |

## DQN 系列共通の知見

### ReplayBuffer capacity は満杯後の経験分布切替時点でもある

`replay_capacity` は保持できる履歴長だけでなく、ReplayBuffer が grow-only 状態から定常的な overwrite 状態へ移る境界を決める。
`num_envs` 個の経験を1 train stepで追加する構成では、概算の満杯時点は次になる。

```text
fill exp-step   ≈ replay_capacity
fill train-step ≈ replay_capacity / num_envs
```

満杯前は初期方策や learner warmup 中の経験が残り続ける。満杯後は古い経験が新しい方策の経験へ順次置換されるため、sampleされる経験分布が変化する。
初期経験が局所解からの離脱に寄与している Env では、この境界で報酬、Q値、loss、grad norm、episode長などが同時に変化し得る。

LunarLander では、`replay_capacity=512,000` の満杯時点付近で見えていた変化が、capacityを `256,000` へ縮小したprobeで前方へ移動した。
これにより、当初 IQN 固有に見えた約0.6M付近の変化は、少なくとも主因の一つがReplayBufferの満杯・overwrite開始境界である可能性が高まった。
ただし、成績への影響方向は Env、初期方策、PER、warmup、探索減衰に依存するため、capacityを大きくすれば常に良いとは扱わない。

capacityを変更する実験では、次を同時に確認する。

- `replay_capacity` と実際に満杯になる exp-step
- 満杯前後のepisode長、報酬、Q値、loss、grad norm
- PERを使う場合のpriority分布と初期経験のsample比率
- learner warmup終了時点とReplayBuffer満杯時点の距離

### Learner warmup は初期経験の量と生存期間を変える

`update_warmup_steps` は learner update の開始を遅らせるだけではない。
warmup中に蓄積した学習前方策由来の経験が、更新開始後もしばらくReplayBufferに残るため、初期学習へ供給される経験分布を変える。

`update_warmup_steps` は `Learner::CanUpdate(exp_step)` が参照する exp-step であり、capacityとの比率を併せて扱う。

```text
warmup fraction = update_warmup_steps / replay_capacity
```

この比率が大きいほど、更新開始時点でReplayBufferに存在する学習前経験は多くなる。一方、capacityが同じなら最初の経験がoverwriteされ始める絶対時点は変わらない。
したがって、warmupとcapacityは独立な「待ち時間」「履歴長」ではなく、初期経験の構成と入れ替わりを共同で決める設定として比較する。

LunarLanderでは、短いwarmupでホバリング局所解に留まりやすいRunがあり、長めのwarmupで着陸への遷移が改善した例が得られた。
ただしseed差も大きく、最適値はEnv固有である。QR/IQNなどアルゴリズムを比較する際は、capacityとwarmupを必ず一致させる。

### IQNのcurrent/target sample数はraw loss・gradの尺度を変える

IQN lossはcurrent側sample数 `N` をsumし、target側sample数 `M` をmeanする契約になっている。
そのため、`N`を変更するとraw lossとgrad normの尺度も変わり、`M`の変更は主にtarget Monte Carlo推定の分散と計算量を変える。

- 異なる`N`のRunでraw lossやgrad normをそのまま大小比較しない。
- `N`を減らしてloss、grad norm、clip ratioが下がること自体は契約上の期待でもあり、過学習軽減の証拠とは限らない。
- `M`を減らした不安定化はtarget sampling分散の増加と整合するが、単一seedでは断定しない。
- QRと計算規模を揃える比較では、QRのquantile数とIQNの`N/M`によるpair数も明記する。

### 表現の摩耗は勾配 step で決まり、replay ratio は摩耗あたりの新規データ量を決める

可塑性メトリクス（`34_agent_plasticity` 群）で表現側を直接測ると、**摩耗の総量は `replay_ratio` にほぼ依存せず、勾配 step の関数**である。
Breakout の RR8 と RR4 を同一 grad step（約 150k）で並べると、パラメータノルム `weight_norm_feature` が 63.9 対 67.4（差 5.5% = 反復のブレ幅内）、
`dead_ratio` の谷からの倍率が 5.9 対 5.8 で一致する。

`replay_ratio` が決めているのは摩耗量ではなく、**同じ摩耗を買うのに何件の新規データが付いてきたか**である。

```text
摩耗量        ≈ f(grad step)
新規データ量  = grad step × batch size / replay_ratio
```

同じ 150k grad step の時点で RR8 は 5M exp 分、RR4 は 10M exp 分しか見ておらず、前者は崩壊し後者は凹んで回復した。
Kumar 2021 / Sokar 2023 の "updates per datum" がそのまま観測された形である。

この分離は、価値側の代理指標と組で扱うと読みやすい。**`q_gap` の過渡は `replay_ratio` の関数、表現側の摩耗は勾配 step の関数**という役割分担になる。

実務上の扱い。

- `replay_ratio` を動かすときは「摩耗が増える/減る」ではなく「摩耗あたりのデータ量が変わる」と考える。摩耗そのものは予算（勾配 step）で決まる。
- 損傷の署名は `dead_ratio` の谷（そこから上昇へ転じる）と `weight_norm_feature` の底が同じ窓に来る形で現れ、その窓は性能ピークの直後にある。
- **署名が出ないことは最適の証明にならない。** 予算内で摩耗律速に達していないだけの可能性がある。実際 RR1 は 20M では署名が出ず、優位が確認できたのは 100M 側だった。

機序自体は学習機構の性質なので Env 共通として扱うが、**摩耗が成績を崩す閾値と、そこへ到達する予算は Env 依存**である。現時点の実測は Breakout 単一 Env（[可塑性 campaign](atari/2026-08-28_plasticity.md)）。

### 重み減衰は重み成長を抑えるのではなく均衡させる

減衰項 λ·w は w が小さくなるほど弱まるため、勾配側の押し上げと釣り合う点で止まる。
`weight_decay` を入れた Run のパラメータノルムは単調減少ではなく **V 字**を描く。

Breakout / RR8 / 10M の実測では `weight_norm_feature` が 45.8 → 30.5（3.0M で底）→ 43.4（8.0M）→ 41.4 と推移した。
水準は抑えられる（無保護は同条件 5M 時点で 62.7 まで伸びる）が、**早期に稼いだ分の大半は押し戻される**。

したがって「`weight_decay` で重みを寝かせる」という読みは短い予算でしか成立しない。
重みノルムを制御対象にする実験では、終端値だけでなく底とその後の回復まで見る。

### アルゴリズム比較ではseedより先に実効設定を揃える

同じseedでも、ネットワーク構造や乱数消費が違えばQRとIQNの軌道は直ちに分岐する。それでも同じseed集合を使うことは、環境初期化条件を揃えるblockとして有用である。
比較時はseed数を増やす前に、各Run artifactの`config/config_data.txt`で少なくとも次を一致させる。

- ReplayBuffer capacityとlearner warmup
- batch size、replay ratio、学習率、PER、n-step、target更新
- train/eval policyと探索減衰
- exp-step budgetとeval cadence
- backendの決定論設定

単一seedの最良Runではなく、同一budgetでの到達時間、終盤window、失敗Run数、seed間の範囲を比較する。

## Env をまたぐ際の扱い

Agent 側の設定名や更新量の式は共有できますが、最適値は Env に依存します。
特に次の要素が異なる結果は、別 Env へそのまま一般化しません。

- 報酬密度と遅延
- エピソード長
- 行動の時間的意味
- ReplayBuffer 内の経験分布
- 局所解と探索方策

実装契約からEnv共通と判断できる機構や比較上の注意は、単一Envで発見した段階でも共通知見として記録します。
一方、その機構が成績を改善・悪化させる方向まで一般化する場合は、複数Envで再現した事実とEnv固有の推測を分けて記録します。
