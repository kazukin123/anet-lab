# DefaultDQNAgent ハイパーパラメータ探索

DefaultDQNAgent 系の探索記録です。ReplayBuffer、PER、batch、replay ratio、target 更新など、Agent 側の共通機構を軸に整理します。

## Env 別記録

| Env | 記録 | 概要 |
|---|---|---|
| Atari | [Atari 探索記録](atari/README.md) | ALE 直結。ベースライン定義（Breakout / Pong）、Atari-5 横断、可塑性メトリクスの較正と保護機構、BTR 実装との差分移植、replay の lane 窓と eval コスト |
| DropMerge | [DropMerge 探索記録](dropmerge/README.md) | 長期 Run における batch size と replay ratio を中心とした探索 |
| LunarLander | [LunarLander 探索記録](lunarlander/README.md) | IQN 導入、QR/IQN 比較、ReplayBuffer capacity、warmup、UQE、quantile sample 数の探索 |

## DQN 系列共通の知見

### Atari で確立した構成要素（他の Env では未確認）

Atari（主に Breakout）の探索で採用まで進んだ構成要素を、別の Env へ持ち込むときに見落とさないよう 1 か所に並べる。
**成績が上がる向きを測ったのは Atari だけ**で、「機序・理由」の欄には実装契約から Env 共通と言えることを書く。
表の「壁突破率」は、Breakout の train で 432 点（1 枚目の壁）を越えたゲームの割合である。詳しい条件は各行の根拠を参照する。

| 構成要素 | Atari での結果 | 機序・理由 | 別の Env へ持ち込むとき | 根拠 |
|---|---|---|---|---|
| Munchausen RL | 25-50M の壁突破率が 6.00%（N=4）→ 11.35 / 11.52%。単独の変更で最大の効果で、100M では最終窓が最高 | 相対 action gap が 3.6 倍に開き、TD 誤差と勾配ノルムは変わらない。`@munchausen` は Double DQN を OFF にするが、Double OFF 単独は無効果 | α 0.9 / τ 0.03 / clip −1 は、報酬を ±1 にクリップしスケーラを切った Q の尺度で使った値。τ は Q 値の差に対する温度なので、報酬の尺度が違う Env では効き方が変わりうる。効き具合は `36_agent_munchausen` の `03_clip_ratio` と `04_bonus_mean` で見る | [2026-09-01](atari/2026-09-01_btr-feedback-arms.md) 探索ブロック 28・29 |
| target 更新の時定数 | 時定数を env 時間 32,000 exp step にそろえた soft τ=0.008 の 2 本（6.16% / 4.51%）は、hard C=125 の 4 本（5.58〜6.29%）を挟んだ。8 倍遅い soft τ=0.001（256,000 exp step）は 3.48% で、τ の掃引の中で最低 | 時定数は grad step ではなく env 時間で決まる。1 更新あたりの exp step は `replay_batch_size / replay_ratio` で、hard はその C 倍、soft はその 1/τ 倍が時定数になる | `replay_ratio` や batch を変えたら C や τ を換算し直す。32,000 は Atari の代表値で、内点最適として確かめた値ではない | [2026-09-01](atari/2026-09-01_btr-feedback-arms.md) 探索ブロック 21・27 |
| target 更新の方式（hard / soft） | hard C=125 の複製 4 本は範囲 0.71pt に収まった。soft τ=0.008 の複製 2 本は 25M まで一致したあと 1.65pt 開き、予算を延ばしても縮まなかった | — | 1 変数ずつ比べるときの基準には、複製の散らばりが小さい hard を使う | [2026-09-01](atari/2026-09-01_btr-feedback-arms.md) 探索ブロック 27・28 |
| dueling の A ストリームの活性 | V/A の隠れ層を SiLU → ReLU にして壁突破率が 1.35% → 3.48%（2.6 倍）。A だけで 3.31%、V だけでは 1.89%。100M でも崩れない | dueling では argmax Q = argmax A。区分線形の A が行動の入れ替わりを約 1/3 にする。LeakyReLU でも入れ替わりは同じだけ減り、成績の伸びは半分。V 側は dormant を下げるが成績はほとんど動かない | 変えるのは A ストリームの活性だけで足りる | [2026-09-01](atari/2026-09-01_btr-feedback-arms.md) 探索ブロック 05・12・19・20 |
| τ 埋め込みの活性 | 融合 512 次元で ReLU にすると dormant が 2.9 倍になり、成績は基準並み。融合 2304 次元では dormant が下がり、成績が上がった（n=1） | ReLU は半分のチャネルを止める。2304 次元なら 1,152 チャネルが残る | 融合の次元が小さいなら SiLU のままにする | [2026-09-01](atari/2026-09-01_btr-feedback-arms.md) 探索ブロック 11・24 |
| spatial exploration（ε ラダー） | ラダー無しと比べて eval の平均が 257.7 → 358.3、100 点未満の割合が 15.5% → 1.4%。上側はほとんど動かず、下振れだけが消えた。上端 0.01 / 0.2 / 0.4 / 1.0 の 4 点で 0.4 が内点最適。throughput は 2,288 → 2,130 step/s | env（lane）ごとに ε を固定する Ape-X 型のラダー。下振れの原因は margin ではなく被覆だった。上端が低すぎると被覆が足りず、高すぎると探索のコストを終盤まで払う | ラダーの比率分布は `num_envs` に依らない。上端の最適値は Env の被覆の難しさで変わりうる | [2026-08-17](atari/2026-08-17_baseline.md) 探索ブロック 19・20 |
| `per_beta` の 0.2 固定 | RR1・RR4 とも成績は変わらなかった | アニールをやめると予算に依存する学習ハイパラが無くなり、打ち切った Run が同じ設定の長い Run の前半と一致する | 切り替えの前後で勾配系のメトリクスを比べない（Atari では `21_grad_norm` と `01_loss` が不連続に変わった） | [2026-09-01](atari/2026-09-01_btr-feedback-arms.md) 探索ブロック 02・03 |

持ち込んだ Env での採否は、その Env の Run で決める（本文末の「Env をまたぐ際の扱い」）。

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

### capacity は lane あたりの履歴窓としても効く（`capacity / num_envs`）

前節の `replay_capacity / num_envs` は満杯時点を決めるだけでなく、**定常状態で 1 つの env（lane）の履歴が
何 step ぶんバッファに載るか**でもある。`num_envs` 個を 1 step で追加する構成では次が同時に成り立つ。

```text
遷移の総寿命（更新数）= replay_capacity × replay_ratio / batch   ← capacity だけで決まる
lane 窓（lane step）  = replay_capacity / num_envs               ← 比で決まる
```

**総寿命と lane 窓は独立に動く。** `num_envs` を倍にすると総寿命は不変のまま lane 窓だけ半分になる。
`capacity` を倍にすると両方が倍になる。

Breakout（RR4 / batch 256）の 3 点で、**成績を決めていたのは lane 窓だけだった**。

| `num_envs` | `capacity` | lane 窓 | 総寿命 | greedy 評価 |
|---|---|---|---|---|
| 64 | 1,048,576 | **16,384** | 16,384 | **685.12** |
| 128 | 1,048,576 | **8,192** | 16,384 | 615.73 |
| 128 | 2,097,152 | **16,384** | 32,768 | **685.30** |

lane 窓が等しい 2 点は `num_envs` も総寿命も 2 倍違うのに **685.12 対 685.30**（学習 Run 複製幅 18.7 の 1/100）で、
lane 窓が半分の 1 点だけが **−69.4**（同 3.7 倍）。**`num_envs` 単独でも `capacity` 単独でもなく、比が効く。**

**要求量はエピソード長で決まり、効き方は閾値的である。** 上の 3 点で greedy のエピソード長は 7,750〜8,327 で、
lane 窓 8,192 は約 1 本ぶん、16,384 は約 2 本ぶん。**1 本ぶんでは壊れ 2 本ぶんで直った。**

**制御変数は lane step で測った窓であって、更新数で測った量ではない。** 3 点は多くの量で違うので、
成績のラベル（good / bad / good）を追う量だけを残すと絞れる。

| 量 | G1 | B | G2 | 追従 |
|---|---|---|---|---|
| 遷移の総寿命（更新） | 16,384 | 16,384 | 32,768 | × |
| 完全なエピソードの同居時間（更新） | 8,634 | 884 | **17,268** | × |
| **lane 窓（lane step）** | **16,384** | **8,192** | **16,384** | **○** |
| **同居時間（lane step）** | **8,634** | **442** | **8,634** | **○** |

**更新数で測った量はどれも追わない。** G1 と G2 は同居時間が更新数で 2 倍違うのに成績は一致し、
G1 と B は総寿命が等しいのに大きく離れる。

これは **`n_step` 由来の筋を否定する** — 長さ `L` のエピソードの尻尾から頭へ価値を伝えるには
`n_step` あたり 1 段ずつ逆行するので `L / n_step` 回ぶんの機会が要る、という読みなら、
**機会が 2 倍ある G2 が G1 を上回るはず**だが一致する。

言えるのは **「lane 窓 / エピソード長が同じなら、更新を何回かけても着地点は同じ」**までである。
これは「良い側 2 点が別の ε>0 到達点（602.9 と 632.0）から同じ ε=0 の 685 へ着く」という観測とも整合する。

**その比が何を通じて成績を決めるかは未測。** 「W の窓に丸ごと載っているエピソードの割合」（= `1 − L/W`）は
この比の単調関数のひとつにすぎず、「lane 自身の履歴をどこまで遡れるか」なども同じく追従する。
とくに **「完全なエピソードが載っていること」自体が要件だとは言いにくい** —
`n_step` のブートストラップは 1 回の更新に完全なエピソードを要求しないし、
実際に更新数で測った同居時間は追従しなかった。

実務上の扱い。

- **`num_envs` を変えるときは `capacity` も同じ比率で動かす。** 片方だけ動かすと replay の構成が変わる。
- 診断は既存メトリクス 2 本で足りる。`46_agent_replay_fit/24_sampled_age_mean` が lane 窓のほぼ定数倍を返す
  （Breakout の実測で lane 窓 / 1.82、4 Run すべてで一致）ので、エピソード長と割れば Run 中に比が見える。
- **エピソード長は Env 内でも題材で大きく違う。** Atari-5 の 5 ゲームは train 側 p50 で 662〜5,440 と 8.2 倍離れ、
  同じ `capacity` / `num_envs` でも安全な題材と危険な題材が混在する。

機構自体（2 つの量が独立に動くこと、比が replay 構成を決めること）は実装契約から従うので Env 共通として扱う。
**閾値がエピソード 2 本ぶんに来ることは Breakout 単一 Env の実測**である
（[replay lane 窓 campaign](atari/2026-09-15_replay-lane-window.md)）。

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

### greedy 評価でしか見えない故障がある

学習中 eval に探索 ε を残すと、**ε を外したときに伸びるかどうか**が測れない。
この「貪欲変換」（ε>0 → ε=0 の得）は腕によって桁違いに差が出る。

Breakout / 50M の実測で、同じ ε=0.01 水準に到達した腕が ε=0 では大きく分かれた。

| 腕 | ε=0.01 | ε=0 | 貪欲変換 |
|---|---|---|---|
| 基準 | 602.9 | **685.1** | **+82.2** |
| spectral norm 除去 | 576.4 | 586.4 | **+10.0** |
| lane 窓 半減 | 598.1 | 615.7 | **+17.6** |

**ε>0 の差が複製幅の内側でも、ε=0 では複製幅の 3〜5 倍離れることがある。**
「ε を揃えた比較なら公平」という扱いは、**ε=0 で初めて出る故障を見逃す**。

扱いの指針。

- **貪欲変換が腕依存であることは交絡ではなく信号である。** 方策が ε 摂動なしでどこまで伸びるかは、
  行動価値の順位がどれだけ安定しているかの測定であって、比較から排除する対象ではない。
- **予算依存が強い。** 上の spectral norm 軸は 50M で −98.7 だが **20M では +5.0 と符号すら逆**になる。
  貪欲変換は訓練とともに片方で育ち片方で潰れるので、**短い予算の greedy 評価で腕を判定しない**。
- **ブレ幅は 2 種類あり、ε=0 のほうが ε>0 より広い。** eval プロトコル自体（同一スナップショットを 2 回）は
  水準によって変わり一般の物差しにならない（Breakout で 2.12〜16.2 点）。
  判定に使うのは学習 Run どうしの複製幅で、Breakout では **ε=0 が 18.7 点・ε=0.01 が 6.7 点**だった。
- **方策内部の指標は当てにならない。** 上の lane 窓の故障に対し、`action_churn` / `q_gap_rel` / `loss` /
  `dormant_ratio` / `srank_ratio` はどれもラベルを追わなかった。検出できたのは
  **train 側の閾値越え率**（Breakout の `≥432` 率で複製幅の 3.8 倍）で、**閾値を上げるほど相対的に鋭い**。
  故障が「長いエピソードを完走できない」という裾に出るため、平均では埋もれる。

### eval のコストは 本数 × エピソード長 / 間隔 で決まる

学習 Run に評価を同居させる場合、eval が消費する env step は次で決まる。

```text
eval / train の env step 比 = eval_episodes × eval のエピソード長 / 間隔(env step)
```

**この比は設定の書き方によって訓練とともに膨らむ。** `interval` を learn step で書くと env step 換算では固定になるが、
**エピソード長は学習が進むほど伸びる**ので、比は勝手に上がる。

Breakout / RR1 の実測では、eval を 20 本/1000 learn step から 100 本/977 learn step へ変えた結果、
比が **0.27 → 1.76** になり、**eval デューティが 99.7%**（セッションが背中合わせで途切れない）まで飽和した。

**env stepping が律速の構成では、eval の 1 step は train の 1 step を 1:1 で奪う。**
eval 設定が大きく違う 2 本で総 env step スループットが 6,568 対 6,402（2.6% 差）と一致し、
train 側だけが 5,027 → 2,385 に落ちていた。

実務上の扱い。

- **eval 設定が違う Run の wall-clock は比較できない。** 同じ構成でも eval 比率が 21% と 99.7% では 2 倍以上ずれる。
  スループットを比べるなら **eval を切った Run 同士**で測る。
- **同じ env step 間隔でも実時間の重さは `replay_ratio` で変わる。** 低 RR ほど env step が速く進むので、
  同じ間隔が実時間では短くなる。Breakout では RR4 でデューティ 60%、RR1 で飽和した。
- **時系列と分布は必要な本数が違う。** 窓平均で読む時系列は 1 セッションの本数をほとんど要求しない
  （Breakout で 10 本と 100 本の窓平均が複製幅 6.7 の内側）。本数が要るのはセッション内の分布統計
  （max、閾値越え率）だけなので、**2 チャネルに分ければ総コストを数分の 1 にできる。**

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
