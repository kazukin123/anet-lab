# PRD 073: replay 抽選履歴別の TD・損失メトリクス

起点: 2026-09-09。仕様合意: 2026-09-10。最終グリル: 2026-09-10（決定 D1〜D7 を本文へ反映。理由と棄却案は [ADR 0039](../adr/0039-replay-fit-sampling-history-groups-not-holdout.md)）。
本書は DefaultDQN 共通の任意診断である `replay_fit` の実装契約を定める。
今回の承認範囲は本書、`CONTEXT.md`、ADR 0039 の文書更新までであり、コード・設定の実装、ビルド、実験は後続とする。
測定対象は厳密な held-out 集合ではなく、既存の抽選履歴で分けた群である。

## 1. 背景とゴール

### 1.1 実験上の不足

[Breakout の replay ratio 実験記録](../experiments/default-dqn/atari/2026-09-08_replay-ratio-mechanism.md)では、
高 RR の腕で学習バッチの loss / TD が小さい一方、成績は RR2 より低かった。
以下は同記録の 25–50M exp step 区間からの抜粋であり、本 PRD による新規実測ではない。

| 腕 | `38_agent_loss/01_loss` | `37_agent_qtd/04_td_std` | `42_env/40_game_score_ge432` |
|---|---:|---:|---:|
| RR4 | 0.360 | 0.367 | 8.85% |
| RR4 + B512 | 0.391 | 0.397 | 8.84% |
| RR1（2 Run） | 0.557 / 0.642 | 0.542 / 0.585 | 11.40% / 11.64% |
| RR2 | 0.609 | 0.560 | 14.33% |

RR4 + B512 は勾配更新頻度を RR2 と揃えても成績が RR4 に近く、再利用と劣化の関係を調べる動機になった。
ただし、学習バッチの誤差と新しいエピソードの成績は異なる量であり、この比較だけで過学習の因果は確定しない。
target の動き、生成方策やデータの新しさ、PER の選択、分布回帰の当てはまりなども区別する必要がある。
表現統計だけでは同キャンペーンの成績差を説明できていないため、別の角度から当てはまりを観測する。

### 1.2 ゴール

- 学習系列を変えず、replay の未抽選群と抽選済み群を同一条件の TD・損失で比較する。
- 両群の平均・比・母数・平均年齢を記録し、比の変化の内訳（群の構成の違いを含む）と測定不成立の理由を追えるようにする。
- 実際に学習へ渡った PER バッチと replay 全体の平均 TD を比較し、選択効果を観測する。
- DefaultDQN の通常 TD・QR・IQN で共通に使える任意診断とする。
- メトリクスの購読とその時点の測定要求から、必要な処理だけを実行する。

### 1.3 非ゴール

厳密な train / validation split、学習からのデータ除外、直近窓、抽選回数別の追跡、年齢を揃えた対照群、
過学習対策、eval 側の計器、plasticity 指標の拡張、Rainbow・ImageCls・MuZero・NoisyNet への展開は含めない。
計器の実装完了と、長期実験による仮説検討・実データでの比較精度の較正は別段階とする。

## 2. 現行実装から確認した前提

確認日は 2026-09-10。以下は現行実装の事実であり、後続節の追加仕様と区別する。

| 事実 | 根拠・設計への含意 |
|---|---|
| `sampled_once_` は通常 `Sample` の抽選時に立ち、slot の再利用時に戻る | [ReplayBuffer 実装](../../core/anet-core/src/replay_buffer_impl.cpp)の `MarkSampledOnce`、`Push`。optimizer 完了や観測の既知・未知を表さない |
| `07_evicted_unsampled_ratio` は退去時の集計である | `RecordEvictionIfSampleable` は上書き対象の ready 判定を使う。現在の sampleable 集合内の未抽選率ではなく、RR4 の未抽選群が不足するという判断には使えない |
| PER は各抽選で優先度総量に対する乱数を引く | `SampleIndices` の現行処理を層化抽出と説明しない |
| 保存済み RR4 構成は `per_initial_priority_mode = max` | [実効設定の保存記録](../experiments/default-dqn/atari/config/run_20260908-073326_rr4_capall_munch.txt)。一様抽選を仮定した「直近1万件の約96%が未学習」という算術は保証にならない |
| `SampleUniqueUniform` は caller-owned RNG を使い、通常抽選の履歴・優先度を変更しない | 既存の抽出・snapshot・prefetch の非干渉契約を新 API に引き継ぐ |
| plasticity は特徴までの部分 forward、policy churn は専用バッチの FP32 評価である | [DQN 実装](../../core/anet-core/src/dqn_based_agent.cpp)の `CapturePlasticityProbe`、`ForwardPolicyChurnExpectedQ`。TD 計算済みの経路があるとはみなさない |
| QR / IQN の期待値 TD と分位点回帰損失は別の量である | `UpdateFromSamples` と方式別の loss helper。期待値 TD は PER 更新に使われ、分布回帰の目的関数ではない |
| 群番号41は使用済みである | [現行 scalar 定義](../../apps/runner/config/metrics_scalar.txt)の `41_agent_on`。本機能は `46_agent_replay_fit` を使う |

`replay_ratio` は環境遷移に対する学習サンプルの使用量を調整する設定である。
`earned_credit = num_envs * replay_ratio / replay_batch_size` という更新制御から、
個々の遷移の抽選回数、現在の未抽選率、有限区間の厳密な寿命平均を直接断定しない。

## 3. 対象集合と測定時点

### 3.1 抽選履歴群

測定時点の sampleable range から dummy を除いた集合を、既存の抽選履歴で二分する。

- **未抽選群 U**: 現在の replay item として通常の学習用抽選を一度も経験していない遷移。
- **抽選済み群 S**: 現在の replay item として通常の学習用抽選を一度以上経験した遷移。

両群の母数を `n_U`、`n_S` とする。これらは抽出バッチの件数ではなく、測定時点の候補数である。
frame stack / n-step で他のサンプルと観測・報酬が重なることはあり、U も情報として未知であるとは限らない。
S は現在処理中・prefetch 済みのバッチの遷移を含み、学習更新が適用済みであることを条件にしない。
測定用の抽出では U から S へ移さない。

遷移の年齢を、その遷移が属する lane の write cursor（Push 済み件数）から当該遷移の logical index を引いた値と定義する。
単位は lane の Push 回数（= train step）であり、lane をまたいだ exp step ではない。
両群の平均年齢を `age_U`、`age_S` とする。母数と同じ snapshot の 1 回走査で加算して求め、dummy は sampleable 外なので年齢にも含めない。
年齢は群の構成記述子であり、年齢を揃えた対照群を作ることはしない（§1.3）。

### 3.2 一回の測定

通常の学習バッチを受け取った後、optimizer 更新と当該 update の target 同期より前に測定する。
現在の online / target parameter、正規化統計、target 方策の現在の risk 基準を、必要な全バッチの評価で共有する。
学習更新や target 同期を評価の途中に挟まない。

必要な群の候補・母数・抽出は同じ replay snapshot から取得する。
各群から要求件数を一様・非復元抽出し、候補が要求件数未満ならその群のバッチは返さない。
重複や他群からの補充、暗黙の件数縮小は行わない。必要な群だけを要求できる契約は §6 に定める。

prefetch wrapper では既存の読取 probe と同様、呼び出し時点までの worker FIFO を整合させる。
通常の prefetched batch を消費・破棄・再抽選せず、その履歴も巻き戻さない。
実 PER バッチは先読み時点に抽選されたものでもよく、その抽選確率を現在の snapshot の確率と同一視しない。

## 4. 評価する誤差と解釈

### 4.1 共通の評価条件

- 全方式で NoGrad、online / target とも eval mode、autocast 無効の FP32 とする。
- IQN の current / target はそれぞれ `(i + 0.5) / K` の固定 midpoint を使い、既定 `K = 32`。各群・実 PER バッチで同じ規則を使う。
- QR は network の固定分位点を使い、通常 TD には分位点を導入しない。
- n-step return、実 n-step 数、terminal による bootstrap mask、gamma、Double DQN、TBO、Munchausen の式は現行の学習契約に従う。
- 通常 TD の loss 用 `td_clip`、分位点回帰の Huber 閾値、PER priority clip を混同しない。絶対 TD 自体には loss / priority 用 clip を掛けない。
- Munchausen は OFF と `target | online | online_reuse` を対象にする。bonus は先頭 return へ一度加え、終端でも残す。TBO の実空間化と完成 target の変換を含め、[ADR 0035](../adr/0035-munchausen-target-learner-local-real-space.md)を維持する。
- target 方策は Greedy と UQE に対応する。hard IQN UQE（tail_mean）では、診断が現在の risk 区間 `[uqe_tau, 1]` に固定 midpoint `K` 本を配置した risk taus を生成し、policy の `SelectAction` へ注入して選ばせる。注入時は policy の `tau_rule` による tau 生成を行わず、RNG を消費しない。スコア計算は policy 実装をそのまま使い、診断側に式を写さない。point UQE と QR の UQE は指定分位点をそのまま使い、注入は不要である。soft IQN UQE は現行どおり全範囲の分位点から経験分位の score を作る。この違いを統合しない。
- 学習が有効な状態で本計器を購読し、解決済み target が ThompsonSampling の場合は購読設定時に fail-fast とする。Greedy への代替は行わない。
- DropPath / Dropout は測定時には発火させない。BatchNorm、Spectral Normalization、正規化の学習統計、方策の schedule、学習 RNG を測定で更新しない。既存 capture・診断値も汚染しない。

「同じ式」は評価条件も通常の学習 forward と同一という意味ではない。
例えば IQN の学習側が random ×8、BF16、train mode でも、診断は固定 ×32、FP32、eval mode とする。
通常の学習ログとの絶対値一致は要求しない。
決定的な診断では Munchausen の `online` と `online_reuse` の値が一致する場合もあるが、学習側の契約は統合しない。

### 4.2 群別 TD と損失

サンプル x の誤差を次のように定義する。

- `d(x) = abs(q_sa(x) - target_mean(x))`。通常 TD では scalar target を使う。QR / IQN では現在と target の分位点平均の差を使い、TBO 時の計算空間も現行の TD 定義に従う。
- `l(x)` は方式別の IS 重み適用前のサンプル損失。通常 TD は現行の Smooth L1、QR は `ComputeQuantileHuberLoss`、IQN は `ComputeIqnQuantileHuberLoss` の定義を使う。分位点方向の sum / mean と kappa の規約を変えず、診断専用の分位点数正規化も追加しない。

U / S から抽出したバッチの平均を `D_U, D_S, L_U, L_S` とする。

```text
unsampled_td_ratio   = D_U / D_S
unsampled_loss_ratio = L_U / L_S
```

比は「個別誤差の比の平均」ではなく「群平均の比」である。
平均 Q が同じでも分布は一致しないため、期待値 TD と損失を併記する。
比が1より大きい場合に言えるのは、未抽選群の平均誤差が大きいことまでである。
PER による選択、データの新しさ、生成方策などが両群の構成に影響するため、純粋な記憶ギャップとは呼ばない。

比は無次元だが、異なる評価条件の腕を無条件に比較できるわけではない。
比較時は評価件数・分位点数・target / loss の設定と共通の exp step 窓を確認する。
U は若い遷移に偏り、RR が低い腕ほど古い未抽選遷移も含むため、腕をまたぐ比の差は群の若さの違いを伴いうる。
両群の平均年齢（§3.1）を併記し、その有無を確認できるようにする。年齢を揃えることはしない。

### 4.3 全体平均と PER 選択比

両群の平均を母数で重み付けし、全体平均を推定する。両群を同数抽出しても単純な二群平均にはしない。

```text
uniform_td_mean = (n_U * D_U + n_S * D_S) / (n_U + n_S)
per_td_mean     = mean(d(x) for x in the actual PER batch)
per_selectivity = per_td_mean / uniform_td_mean
```

母数ゼロの群は加重和に寄与させない。非空の群に必要な平均が得られなければ全体平均は NaN とする。
実 PER バッチの重複行は抽選回数どおりに含め、IS 重みは掛けない。
比が1付近なら「この評価条件で平均 TD に差がない」と読み、一様抽選への退化と断定しない。

### 4.4 値が成立しない場合

- 件数不足の群の平均と、それに依存する比は NaN。他方の要求された平均と母数は、成立する範囲で出力する。
- 分母ゼロの比は NaN。epsilon 加算、clamp、前回値による穴埋めは行わない。
- PER 無効時も U / S の比較と `uniform_td_mean` は利用可能。`per_td_mean` と `per_selectivity` は NaN とし、そのための評価処理を行わない。
- 既知だが未購読・非測定回・入力不足の key は NaN、未知 key だけ `std::nullopt` とする。
- 不正な型・値域・内部状態は通常の fail-fast の対象であり、件数不足と同じ欠測として扱わない。

## 5. メトリクスと設定

### 5.1 13指標

表示グループは `46_agent_replay_fit` とする。全行は `@learn $learn_step $update_result interval:503`。
ソースキーはタグの番号部分を除いた名前に `replay_fit_` を付ける。

| タグ末尾 | ソースキー | 値 |
|---|---|---|
| `01_unsampled_td_mean` | `replay_fit_unsampled_td_mean` | D_U |
| `02_sampled_td_mean` | `replay_fit_sampled_td_mean` | D_S |
| `03_unsampled_td_ratio` | `replay_fit_unsampled_td_ratio` | D_U / D_S |
| `11_unsampled_loss_mean` | `replay_fit_unsampled_loss_mean` | L_U |
| `12_sampled_loss_mean` | `replay_fit_sampled_loss_mean` | L_S |
| `13_unsampled_loss_ratio` | `replay_fit_unsampled_loss_ratio` | L_U / L_S |
| `21_unsampled_count` | `replay_fit_unsampled_count` | n_U |
| `22_sampled_count` | `replay_fit_sampled_count` | n_S |
| `23_unsampled_age_mean` | `replay_fit_unsampled_age_mean` | age_U（lane の Push 回数単位） |
| `24_sampled_age_mean` | `replay_fit_sampled_age_mean` | age_S（lane の Push 回数単位） |
| `31_uniform_td_mean` | `replay_fit_uniform_td_mean` | 母数で加重した全体平均 TD の推定 |
| `32_per_td_mean` | `replay_fit_per_td_mean` | 実 PER バッチの平均絶対 TD |
| `33_per_selectivity` | `replay_fit_per_selectivity` | 実 PER バッチ / 全体の TD 比 |

分子・分母の平均6本は、比の変化の内訳を追うために残す。母数2本と平均年齢2本は群の構成記述子であり、同じ走査から得る。既定の EMA 行は追加しない。
本機能は新規契約であり、旧案の `td_holdout_ratio` などの alias・互換読取は作らない。

### 5.2 設定の正本

| キー | 既定 | 契約 |
|---|---:|---|
| `DefaultDQNAgent.learner.replay_fit.probe.batch_size` | 1024 | 各群の固定抽出件数。正整数 |
| `DefaultDQNAgent.learner.replay_fit.iqn.num_taus` | 32 | IQN 診断の固定分位点数。正整数。通常 TD / QR では使用しない |

頻度は metrics 行の `interval` だけで指定し、learner 側に enabled / interval キーを重複させない。
独立した `metrics.scalar.@replay_fit` profile に13行を定義する。
既定の profile 選択へは追加せず、利用側が明示的に選択して有効化する。
profile の定義だけが残っていても、実際に attach されなければ測定しない。
コメントアウトによる OFF は、他の選択 profile に同じソースへの購読が残っていないことも含め、解決後の購読集合で判定する。

既定は503 learner updates ごとである。同じ exp step まで進めると高 RR の腕ほど測定点が多くなるため、
長期実験では共通の exp step 窓で比較する。
実効設定、対象 snapshot の意味、分位点数を変えた場合は、その条件を実験記録へ残す。

## 6. 購読依存の実行制御

### 6.1 原則

**購読された出力に必要な処理だけを、その出力の測定タイミングで実行する。**
「13指標のどれかが ON なら全指標を計算する」という固定処理にはしない。
一部の行をコメントアウトしたときも、他の出力から必要とされない処理を停止する。

初期化時に解決済み購読を受け取り、各 update では interval に到達した key の依存処理を合成する。
異なる interval があるときは、その回に必要な処理の和集合を一度だけ実行し、重複評価を避ける。
最小 interval で他の全指標も固定的に計算する方式は採らない。
計算のために内部で得た平均は再利用してよいが、未購読・非測定回の出力を最新値として公開しない。

### 6.2 必要な処理の表

| その回の要求 | 必要な追加処理 | 実行しない処理 |
|---|---|---|
| 全行 OFF / 未 attach | 購読有無の軽量な分岐のみ | replay 走査、母数集計、履歴管理の追加、抽出、Tensor 構築、転送、正規化、forward、損失計算、測定結果 pack、測定用同期 |
| 購読あり・全行 interval 到達前 | cadence の判定のみ | 測定用の走査・抽出・転送・forward・損失計算 |
| 母数・平均年齢だけ | sampleable 集合の履歴別件数と年齢和を同じ 1 回の走査で集計 | 抽選 RNG、経験バッチ構築、転送、正規化、GPU / network 評価 |
| 片方の群の平均だけ | 必要な群の候補抽出と該当誤差の評価 | 他群のバッチ抽出・評価、実 PER バッチの再評価 |
| TD 系だけ | 必要なバッチの current / target 評価と期待値 TD | QR / IQN の回帰損失用 N×M 全ペア Tensor・Huber 計算 |
| 損失系だけ | 必要なバッチの current / target 評価と方式別損失 | 診断用 TD 統計の追加集約、不要な実 PER バッチ再評価 |
| `uniform_td_mean` | 同一 snapshot の母数と、寄与する各群の平均 TD | 実 PER バッチの再評価、回帰損失計算 |
| `per_td_mean` だけ | 受領済みの実 PER バッチを同条件で評価 | 診断用 replay snapshot・群別抽出・母数集計、回帰損失計算 |
| `per_selectivity` | 全体平均 TD の推定と実 PER バッチ評価 | 回帰損失計算 |

比だけを購読した場合も分子・分母の平均計算は必要である。
例えば `unsampled_loss_mean` と `sampled_loss_mean` の表示を外しても、
`unsampled_loss_ratio` が残れば両群の損失計算は止めない。
一方、全損失系を外せば、TD 評価のためだけに回帰損失の全ペア計算を残してはならない。
通常 TD の損失に必要な scalar 残差など、要求された出力自身が必要とする算術はこの禁止に含めない。

### 6.3 資源と通常経路への非干渉

- 診断の抽出が要求される場合だけ、Agent 所有の専用 named RNG `replay_fit_probe` を用意する。母数だけ・実 PER バッチ平均だけなら抽選 RNG は不要。全 OFF では専用 RNG・作業領域・worker を作らない。
- learner は update ごとの要求・一時入力・結果を所有する。資源の配置は [所有権ガイド](../ownership_guideline.md)に従う。
- plasticity / policy churn と実バッチ、乱数、cadence を共有しない。既存の RNG stream の seed や消費順を変えない。
- 未要求の群のための候補保持・抽出を行わない。必要な群や母数を識別するための snapshot 内の走査は許す。
- 母数を出すための常時更新カウンタや per-entry 抽選回数を追加しない。年齢も走査内で write cursor と logical index から求め、per-entry の書込時刻キャッシュを追加しない。既存の `sampled_once_` 更新は従来の処理として維持する。
- 全 OFF では毎 update の測定用配列クリアや空 Tensor の再構築へ入る前に skip する。「コストゼロ」は追加の測定処理・資源を持たない意味で、CPU 命令数が厳密にゼロという主張ではない。
- キャッシュ済み統計・正規化の診断値・network の training mode / buffer・既存 capture を測定で書き換えたまま残さない。通常処理順による保持も含め、既存出力の不変性を受入で検証する。
- 実行可能な既定値へ勝手に件数・頻度・精度を下げない。契約違反や明示要求の不履行は通常の fail-fast に従う。

## 7. 実装時の境界

### 7.1 ReplayBuffer 読取 API

`ReplayBuffer` の公開 interface に、履歴別の母数・平均年齢と指定した群の一様非復元バッチを 1 回の呼び出しで返す操作を追加する。
要求は「母数・平均年齢」「U の抽出」「S の抽出」を組み合わせられる形とし、
**常に両群を抽出する API にはしない**。抽出を伴う場合の RNG は caller-owned とする。

```cpp
struct SamplingHistoryProbeRequest {
    bool counts = false;                          ///< 母数 n_U / n_S と平均年齢 age_U / age_S を返す
    std::optional<int64_t> unsampled_batch_size;  ///< U から一様・非復元で抽出する件数
    std::optional<int64_t> sampled_batch_size;    ///< S から一様・非復元で抽出する件数
};
struct SamplingHistoryProbeResult {
    int64_t unsampled_count = 0;                  ///< counts 要求時に有効
    int64_t sampled_count = 0;
    float unsampled_age_mean = std::numeric_limits<float>::quiet_NaN(); ///< 単位は lane の Push 回数。群が空なら NaN
    float sampled_age_mean = std::numeric_limits<float>::quiet_NaN();
    std::optional<ExperienceSamples> unsampled;   ///< 要求なし・件数不足は nullopt
    std::optional<ExperienceSamples> sampled;
};
virtual SamplingHistoryProbeResult ProbeSamplingHistory(
    const SamplingHistoryProbeRequest& request, anet::RandomGenerator* random) const = 0;
```

`random` は抽出を伴わない要求では `nullptr` でよく、抽出要求があるのに `nullptr` なら fail-fast とする。
群の判定は既存の `sampled_once_` を flat slot 単位で読む。母数・年齢・抽出は `SampleUniqueUniform` と同じ storage / metadata の排他区間で同じ sampleable 集合から取得し、
U / S が共に要求された回では両群を同じ snapshot から抽出する。年齢は走査中に lane の write cursor と logical index から求める。
要求された群だけを、件数が足りる場合に固定件数で返す。不足した群のために他群の取得を失敗させない。
返却サンプルは既存と同じ frame stack / n-step / terminal / generation-aware item key 契約を持ち、IS 重みは 1 とする。
`PrefetchingReplayBuffer` は `SampleUniqueUniform` と同じく、呼び出し時点までの queued Push と in-flight prefetch を settle してから inner へ委譲する。

pure virtual として追加し、`DefaultReplayBuffer`、`PrefetchingReplayBuffer`、interface を実装する現用 test double
（`dqn_based_agent_test.cpp` の `RecordingReplayBuffer`、`replay_buffer_test.cpp` の `BlockingReplayBuffer`、`trainer_test.cpp` の `HintRecordingReplayBuffer`）を同じ変更内で更新する。
黙って throw する既定実装は置かない（[ADR 0031](../adr/0031-plasticity-metrics-out-of-band-partial-forward.md) と同じ方針）。
既存 `SampleUniqueUniform` とその利用側の契約は維持し、汎用 predicate DSL や永続的な別 buffer は導入しない。

### 7.2 Learner の評価と出力

target 組立とサンプル別誤差の計算を、learner（TD / QR / IQN）ごとの 2 関数へ抽出し、**学習経路と測定経路の両方が同じ関数を呼ぶ**。

- `BuildTarget`: NoGrad の target 組立。入力は samples（target return・実 n-step・terminal・action）、正規化済み obs / next_obs、
  Munchausen 用の detach 済み current 出力、IQN では target taus。target 行動選択（Greedy / UQE、Double DQN の network 選択）、
  Munchausen の 3 mode、TBO の実空間化と再変換、n-step 割引と terminal mask を現行式のまま内包する。
  IQN の taus と Munchausen 用 current 出力は呼び出し側が渡す。学習側は現在と同じ位置・順序で RNG を消費して taus を生成し、診断側は固定 midpoint を渡す。
  hard IQN UQE の target 行動選択は §4.1 の risk taus 注入で行う。
- `ComputeElementError`: current 出力と target からサンプル別の `abs_td`（§4.2 の `d(x)`）と `element_loss`（同 `l(x)`、IS 重み適用前）を返す純粋関数。
  既存の `ComputeQuantileHuberLoss` / `ComputeIqnQuantileHuberLoss` / Smooth L1 をこの中で使う。

現行の TD の inline target 組立と QR / IQN の `MakeMunchausenTarget`（内部で taus を生成し target forward を行う）はこの 2 関数へ再編する。
学習全体の `UpdateFromSamples` を診断から呼んで optimizer・priority 更新・通常統計まで動かす方式にはしない。
学習側の forward / RNG 消費 / capture の順序は変えない。PER 更新用の診断・勾配統計など、購読結果に不要な学習付随処理を診断経路へ持ち込まない。

共有関数は batch 次元をテンソルから取り、`config_.replay_batch_size` を参照しない。
測定は総抽出件数を維持し、全件のサンプル平均を返す。メモリ都合で分割して forward する場合は件数で加重し、
分割の有無・幅は契約にしない。複数 forward の間で parameter / target / 正規化統計 / risk 基準を変更しない。

診断の実行位置は Learn ループの既存 probe（plasticity / policy churn）の後、`UpdateFromSamples` の前とする。
実 PER バッチは受領済みの device 上サンプルをそのまま同条件で評価し、再抽選しない。

結果は対応する `BatchUpdateResult` に保持する。
一つの LearnEvent に複数 update が含まれても、後の update が前の結果を上書きしない。
`GetScalar` から抽出・forward を起動しない。
主要な抽出・転送・forward・誤差計算・集約には既存規約に沿った ProfileRange を設け、
不要な処理の実行有無と負荷を追えるようにする。

### 7.3 ドキュメントの責任境界

用語の意味は [CONTEXT.md](../../CONTEXT.md)、本機能の測定・購読契約は本 PRD を正本とする。
[ReplayBuffer 設計](../design/150_replay_buffer.jp.md)と [DQN 設計](../design/200_dqn_agents.jp.md)は
後続のコード実装と同じ変更内で、新 API、所有権、実行順、購読依存、評価条件を現行仕様として更新する。
文書化段階では未実装の API を現行実装として記述しない。

## 8. 受入条件

以下は後続の実装で実施する検証であり、この文書更新で実施済みとは扱わない。

### 8.1 数式と出力

- 通常 TD・QR・IQN の既知の小規模例で、サンプル損失、群別平均、平均の比が期待値に一致する。
- 母数が異なる二群で全体平均の重み付けを検証し、同数抽出を理由に単純平均へ置き換えていないことを確認する。
- 群別の平均年齢が fixture の Push 回数から期待どおりになる。ring 折り返し・history margin・dummy 除外を含め、lane ごとの write cursor 基準で求める。
- 分母が正で同じ群平均なら比は1になり、平均 Q が同じでも分布が異なる例では回帰損失の違いを検出する。
- gamma、実 n-step、terminal / truncated、TBO、Munchausen の各 mode、Greedy / UQE、許容される Double DQN 構成で target の意味が現行式と一致する。
- 件数不足、空の群、全体が空、分母ゼロ、PER 無効、既知 key の欠測、未知 key を区別する。欠測が他の成立する出力を不必要に消さない。
- 一つの LearnEvent に複数 update がある場合も、それぞれの測定結果と step が対応する。

### 8.2 抽出と snapshot

- 初回抽選、slot の上書き・世代交代、ring 折り返し、history margin、n-step 未確定、dummy・terminal / truncated を含む fixture で、群分けと母数を検証する。
- 抽出が sampleable な対象だけを返すこと、必要な群だけを返すこと、要求件数と一様・非復元の選択規則を確認する。
- 一方の群が不足しても、他方の必要なバッチと母数を取得できる。
- `ProbeSamplingHistory` が counts のみ / U のみ / S のみ / 両方の各要求で、要求した項目だけを返し、未要求の群を抽出しない。
- 抽出要求があるのに `random` が `nullptr` なら fail-fast し、抽出なしの要求では `nullptr` で成立する。
- 診断呼び出しが sampled-once、優先度、generation key、通常 RNG、通常の prefetched batch を変更しないことを確認する。
- prefetch 有無で snapshot の定義が保たれ、通常のサンプル消費・queued push の順序を壊さない。

### 8.3 購読設定で不要になる処理のゼロ確認

§6.2 の各行をテストケースにし、結果が NaN になるだけでなく、**不要な処理の呼び出し回数がゼロ**であることを確認する。
対象は replay snapshot / 母数集計、各群抽出、実 PER バッチ再評価、device 転送、network forward、
QR / IQN の回帰損失全ペア計算、診断 RNG・作業領域の生成、測定用同期とする。
呼び出し回数は既存の boundary counter（`RecordingReplayBuffer` の API 呼数、テスト network の forward counter）で観測し、
本体へ test-only API や内部カウンタを足さない。

特に次を必須にする。

- 全行をコメントアウトした構成、profile 定義があるが未選択の構成では追加の測定処理・専用資源がない。
- 母数・平均年齢のみでは経験バッチ構築・抽選 RNG・GPU / network 評価がゼロ。
- 片方の群の平均のみでは他方の群の抽出・評価がゼロ。
- TD のみでは分位点回帰損失 helper の呼び出しがゼロ。逆に損失比を残した構成では、平均表示行を外しても必要な両群損失が計算される。
- 実 PER バッチ平均のみでは診断用 replay 読取・群別抽出・母数集計がゼロ。
- 全体平均のみでは実 PER バッチ再評価がゼロ。
- 異なる interval の TD・損失を購読し、損失の非測定回にその全ペア計算が残らない。同一回の共有 forward は重複しない。
- 他の選択 profile に同じソース購読が残る場合は必要な処理が残り、最後の依存購読が消えた場合に停止する。
- `learner.enabled=false` は意図された無効化として測定しない。学習有効時の Thompson target と本計器の購読併用は起動時に検出する。

### 8.4 学習系列と既存メトリクスの非干渉

実装前に比較用の固定 seed・実効設定・ビルド条件と基準結果を保存する。
旧実装 / 新実装 OFF、および同一ビルド OFF / ON を比較する。
§7.2 で学習経路の target 組立が共有関数へ書き換わるため、旧実装 / 新実装 OFF の同 seed 比較は共有化そのものの受入でもある。
時刻・性能値・今回追加する出力を除き、通常の抽選系列、学習結果、既存の deterministic なメトリクスと
plasticity / policy churn の診断系列を維持する。
NN parameter / buffer、RNG、正規化統計、priority 更新と抽選履歴も対象にする。

基準は再現性を確保した構成で取得し、差異を非決定論という説明だけで未調査のまま除外しない。
事前基準が欠けた場合は未達として明記し、代替検証を checksum 一致と呼ばない。
Debug ビルドと関連する DQN / ReplayBuffer / subscription の回帰テストを通す。
TD・QR・IQN、Munchausen、BF16 と DropPath を含む構成で配線を確認し、
既存の probe を同時に購読しても互いの系列を変えないことを確認する。

### 8.5 性能

`x64-RelWithDebInfo` の Breakout RR4 を用い、既定の各群1024件・IQN固定32分位点・interval503・13指標を測る。
他の購読と実効設定、seed、device は ON / OFF で揃える。

- warmup を除き、両群のバッチが成立して実際に全指標の評価が20回以上作動する区間を使う。
- ON / OFF を3組比較し、実行順を交互にする。各組の比較区間は同じ exp step 範囲に揃える。
- 各組で `1 - throughput_ON / throughput_OFF` を求め、その中央値を **0.05 以下**とする。
- 実効設定、ビルド条件、測定区間、成立した測定回数、throughput、実時間、主要 ProfileRange を証拠として残す。

5% を超えた場合は計測箇所を調べて改善する。件数・頻度・精度・閾値を黙って緩めない。
必要な測定が欠測・未購読で実行されていない区間を使って性能合格とはしない。

### 8.6 長期実験との分離

RR1 / RR2 / RR4 / RR4 + DropPath の長期比較と、件数・分位点数を増やした際の比較精度の較正は後続実験とする。
「RR4 が最大・RR2 が最小になること」は計器の実装合格条件ではない。
逆順・差なしでも測定結果として扱い、それだけで特定の対立仮説を採択しない。

## 9. 複雑さの監査と採否

2026-09-10 の最終簡素化6項目、購読依存の追加確認、および同日のグリル決定（D1〜D7）を記録する。

| 観点 | 判断 | 理由・残す価値 |
|---|---|---|
| 全体の過剰さ | keep | 二群の TD・損失・母数・平均年齢と PER 補助比較の13指標を残す。PER の比較を削ると選択効果を直接見られない不足が戻る |
| 要件の実在性 | keep | TD・QR・IQN 対応は DefaultDQN 共通の任意診断として使う明示要件。損失と母数も今回の観測不足に対応する |
| 前提が変わった決定 | cut | 学習済み / 未学習という呼称、過学習の断定、直近窓、二つの比の差による分布シフト分離、既存学習 loss の直接流用を外す |
| 最小構成との差 | keep | 比3本と母数2本に対して追加となる平均6本を残す。計算は比にも必要であり、平均を省くと変化の内訳が分からない |
| 段階の独立性 | defer-behind-gate | 文書化、計器の実装・検証、長期実験を分離する。長期実験は計器の受入を満たした後に実施する |
| 成功の測定可能性 | keep / defer-behind-gate | 数式・抽出・非干渉・配線・5%上限を実装受入とする。実データ上の比較精度の較正は計器完成後の実験に置く |
| 一部 OFF 時の処理 | shrink | 出力ごとの依存で必要群・誤差計算を絞る。常時の母数カウンタ、不要群の抽出、不要な全ペア損失計算、未購読時の専用資源を残さない |
| 群の平均年齢（グリル D1） | add | 群の構成記述子。母数と同じ走査内の加算だけで、forward・抽出・RNG は増えない。腕をまたぐ比の差に若さの違いが伴うかを追える |
| target 組立の共有（グリル D2 / D3） | keep | learner ごとの `BuildTarget` / `ComputeElementError` を学習経路も呼ぶ。式の正本を 1 箇所にし、同 seed 等価性ゲートで学習側不変を担保する。hard IQN UQE は risk taus 注入で policy 実装を使う |
| chunk 分割の固定（グリル D4） | cut | 共有関数を batch 次元非依存にすれば不要。契約は件数維持と全件平均だけにする |

厳密 held-out は、この観測だけでは必要な因果の切り分けができないと判明した場合に別の実験設計として再検討する。
Thompson target は利用が具体化した時点で確率的方策の評価方法を定義する。
Rainbow・ImageCls・MuZero・NoisyNet の対応は、利用要求と各データ・評価契約が具体化するまで追加しない。
今回の用語は `CONTEXT.md`、仕様は本書、決定の理由と棄却案は [ADR 0039](../adr/0039-replay-fit-sampling-history-groups-not-holdout.md) へ記録する。

## 10. 関連する既存契約

- [PRD 050: ready / sampleable と履歴保護](done/050_replay_ring_stack_margin_10prd.md)
- [PRD 062: plasticity の購読駆動計測](done/062_plasticity_metrics_10prd.md)
- [PRD 066: 独立 RNG と固定条件の policy churn](done/066_policy_churn_metrics_10prd.md)
- [ADR 0031: 学習経路から分離した測定](../adr/0031-plasticity-metrics-out-of-band-partial-forward.md)
- [ADR 0033: policy churn の固定 probe](../adr/0033-policy-churn-fixed-probe-and-target-lag.md)
- [ADR 0035: Munchausen target の責任境界](../adr/0035-munchausen-target-learner-local-real-space.md)
- [ADR 0039: 抽選履歴群による当てはまり診断（held-out 分割の不採用）](../adr/0039-replay-fit-sampling-history-groups-not-holdout.md)
