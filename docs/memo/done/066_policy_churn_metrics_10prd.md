# Policy churn メトリクス（35_agent_churn）PRD

> 状態: 正式 PRD。2026-09-01 のグリルで未決事項 0 件、文書契約上は実装着手可能。
> 本書は実装着手を意味しない。実装は別の明示依頼と実装計画で行う。
> 起点: 2026-08-31 の BTR 公開実装の読解（[調査記録](../../../reports/btr_code_reading_2026-08-31.md)）。
> 関連: [ADR 0033](../../adr/0033-policy-churn-fixed-probe-and-target-lag.md)、
> [PRD 062](062_plasticity_metrics_10prd.md)、
> [PRD 063](063_plasticity_weight_norm_10prd.md)、
> [PRD 065](065_nn_spectral_norm_10prd.md)、
> [Atari 探索記録](../../experiments/default-dqn/atari/2026-08-31_rr4-ceiling-ln512.md)。

## Context / Problem Statement

**表現の健全性を測り切っても、Breakout の成績や壁突破を説明できない。**

2026-08-31 の Atari 探索では、RR1 + LayerNorm が dormant / dead ratio と srank を大きく改善した一方、eval1 の終端窓平均とエピソード分布は対照とほぼ同じだった。τ 整合 + LayerNorm も健全な表現指標を保ちながら、15M step 以降の壁突破率が LN512 単独より大きく低下した。

`34_agent_plasticity` は「表現が痩せていないか」を測る群であり、「表現が健全なのに方策が安定しない、または target とのずれ方が不適切」という状態は切り分けられない。

本 PRD は、ReplayBuffer の一様な状態集合に対して、**1 回の learner update が online DQN の expected Q と greedy 行動をどれだけ変えたか**を `policy churn` として観測する。あわせて、同じ測定時点で online と target の greedy 方策がどれだけ食い違うかを `target policy disagreement` として観測し、target の遅れを方策空間で読めるようにする。

## Goal

- 1 learner update による online の greedy 行動変化率と Q 変化量を、学習へ干渉せず観測する。
- online update と target update の後に、online / target の方策不一致と Q 差を同じ probe 状態上で観測する。
- hard target update の位相を `target_sync_age` で記録し、target 指標の解釈を可能にする。
- DefaultDQN の baseline で常時利用でき、購読が無い場合は測定処理を完全に不活性にする。

## Non-goals

- churn を低下させる介入、正則化、target 更新方式の変更。
- BTR が保存する action transition 表の再現、または BTR の churn 絶対値との一致。
- `34_agent_plasticity` の測定契約や実バッチの変更。
- ImageCls、Rainbow、NoisyNet への公開・対応。
- 本 PRD だけで科学的な因果結論を出す Run 比較。

## 用語と数理契約

正規用語は [CONTEXT.md](../../../CONTEXT.md) の `policy churn` と `target policy disagreement` に従う。

### Expected Q と greedy 行動

churn の action は ActionPolicy の探索、UQE、Thompson Sampling から独立した、network の expected Q に対する `argmax` とする。

- TD / QR: network が返す Q 値を用いる。
- IQN: churn 専用の固定 midpoint taus 32 本を用い、quantile 出力を tau 次元で平均して expected Q とする。tau は `τ_i = (i + 0.5) / 32`（`i = 0..31`）で、online-before / online-after / target-after の最大 3 forward で同一 Tensor を共有する。
- TBO: Q 差は inverse transform 後の価値尺度ではなく、network 出力空間で定義する。異なる TBO 構成間で Q 差の絶対値を直接比較しない。

churn の online-before / online-after / target-after の最大 3 forward は、外側の learner AMP 設定にかかわらず **autocast を明示的に無効化し、FP32 で行う**。greedy action の境界に近い Q gap を BF16 の丸めで反転させないためである。現行 `34_agent_plasticity` probe は learner と同じ autocast 構成を使うが、activation 統計を測る既存契約なので本 PRD では変更しない。churn は離散的な `argmax` 境界を測るため、意図的に異なる精度契約を持つ。

forward は NoGrad + eval mode で行う。現行実装に NoisyNet は存在しないため、本 PRD では noise snapshot API を導入しない。将来 NoisyNet を導入する場合は、before / after / target で同一ノイズを保証する契約を別途確定するまで churn 対応を有効化しない。

## メトリクス契約

### タグと source key

`35_agent_churn` に次の 13 tag を置く。source key は 7 種で、Q 由来 6 種は raw と EMA の 2 tag を持つ。すべて `@learn $learn_step $update_result interval:503` とし、全 DefaultDQN Run の baseline で既定 ON にする。EMA は `$ema ema_alpha:0.1` を付けた別 tag として同一 source key から算出するため、追加の source key も追加計算も発生しない。

tag 名は Metrics Viewer の一覧表示に `_ema` 付きでも収まるよう詰めてある。実測した表示上限は接頭辞込み 44 文字で、本群の最長は `35_agent_churn/12_target_disagreement_ema` の 41 文字である。source key は tag 名と独立なので、CONTEXT.md の正規用語（`target policy disagreement`）は source key `policy_churn_target_policy_disagreement` 側が保持する。

| tag | source key | 定義 |
|---|---|---|
| `35_agent_churn/01_action_churn_ratio` | `policy_churn_action_ratio` | `mean(argmax Q_online_before(s) != argmax Q_online_after(s))` |
| `35_agent_churn/02_action_churn_ratio_ema` | 同上 | `01` の EMA |
| `35_agent_churn/03_q_delta_abs` | `policy_churn_q_delta_abs_mean` | 状態 × 行動で平均した `abs(Q_online_after - Q_online_before)` |
| `35_agent_churn/04_q_delta_abs_ema` | 同上 | `03` の EMA |
| `35_agent_churn/05_q_delta_signed_max` | `policy_churn_q_delta_signed_max` | 行動ごとに状態平均した signed ΔQ の最大値 |
| `35_agent_churn/06_q_delta_signed_max_ema` | 同上 | `05` の EMA |
| `35_agent_churn/07_q_delta_signed_min` | `policy_churn_q_delta_signed_min` | 行動ごとに状態平均した signed ΔQ の最小値 |
| `35_agent_churn/08_q_delta_signed_min_ema` | 同上 | `07` の EMA |
| `35_agent_churn/11_target_disagreement` | `policy_churn_target_policy_disagreement` | `mean(argmax Q_online_after(s) != argmax Q_target_after(s))` |
| `35_agent_churn/12_target_disagreement_ema` | 同上 | `11` の EMA |
| `35_agent_churn/13_target_q_delta_abs` | `policy_churn_target_q_delta_abs_mean` | 状態 × 行動で平均した `abs(Q_online_after - Q_target_after)` |
| `35_agent_churn/14_target_q_delta_abs_ema` | 同上 | `13` の EMA |
| `35_agent_churn/15_target_sync_age` | `policy_churn_target_sync_age` | hard update で前回 target 同期からの learn_step 数。soft update では NaN。**位相カウンタなので EMA を持たない** |

`05` / `07` は、BTR の行動別 signed Q 変化をスカラ基盤へ載せるため最大・最小へ畳む。現時点では成績説明力の実測根拠が弱いが、BTR 対応の探索計器であり、`03` と同じ差分 Tensor から低い限界費用で算出できるため残す。

EMA は Q 由来 6 種すべてに漏れなく用意する。利用予定の有無で取捨せず、後から番号を振り直さずに済ませるためである。非有限値は EMA 更新の前段で除外されるため（`observers.cpp` の `!std::isfinite` ガード）、NaN が EMA 状態を汚染しない。alpha は疎な系列の慣行に合わせて 0.1 とする。

### 値未成立と未知 key

- LearnEvent が存在しない warmup 中は、メトリクス点自体を出さない。
- update が成立していても、完全な probe batch を取得できない場合は Q 由来の 6 source key（tag では `01`〜`08`、`11`〜`14`）を NaN とする。batch size を縮小して代替しない。
- `15_target_sync_age` は probe の成否から独立して算出する。hard update では有限値、soft update では既知 key の値未成立として NaN とする。
- 既知 source key で値が成立しない場合は NaN を返す。`std::nullopt` は未知 source key にだけ返す。

## Probe 契約

### 状態集合

churn は ReplayBuffer から**一様・非復元**で 1024 件を 1 回抽出する。plasticity と共有するのは sampler の仕組みだけであり、plasticity の実バッチや既定 4096 件は共有・流用しない。

1 回の churn 測定で取得した状態 Tensor を Q 由来 6 source key すべてで共有する。これにより online-before / online-after / target-after の比較対象を同一に保つ。PER sample は「学習が触りやすい状態」に分布が偏るため用いない。

専用設定は次の 2 つとする。

```text
learner.policy_churn.probe.batch_size = 1024
learner.policy_churn.iqn.num_taus = 32
```

両値とも 1 以上を現行契約とし、1 未満は設定読み込み時に fail-fast する。実装時に別の暗黙既定値、sentinel、縮小 fallback を追加しない。

### RNG 所有権

ReplayBuffer の一様非復元 API は、将来実装で次の caller-owned RNG 契約へクリーンブレークする。

```cpp
SampleUniqueUniform(..., RandomGenerator& random)
```

- plasticity は Agent 所有の named seed `"plasticity_probe"` から作った RNG を渡す。
- churn は Agent 所有の named seed `"policy_churn_probe"` から作った RNG を渡す。
- ReplayBuffer は probe 用 RNG を所有しない。
- churn の ON / OFF や cadence が、通常 Replay sample と plasticity probe の乱数系列を変えない。

旧 overload、ReplayBuffer 所有 RNG、互換分岐は残さず、現用呼び出し元を同一変更内で移行する。

このクリーンブレークでは、plasticity probe の乱数源が現行の ReplayBuffer 所有 `unique_uniform_sampler_` から Agent 所有の named seed `"plasticity_probe"` へ変わるため、**plasticity probe の抽選系列を意図的に断絶する**。同じ seed であっても、新旧ビルド間では `34_agent_plasticity/4x_probe_*` の metrics checksum 一致を要求しない。移行時の等価性アッセイは当該 probe 系メトリクスを除外するか、本変更を基準断絶点として以後の新ビルド同士を比較する。受入条件の churn ON / OFF 等価性は、この移行後の同一ビルド内だけを対象とする。

## 測定シーケンス

1 回の churn 測定は、対象 learner update に対して次の順序に固定する。

1. 一様非復元で churn probe 状態を 1 回取得し、update を跨いで保持する。
2. 通常の学習 forward / backward / grad clip を完了する。
3. optimizer step 直前に `online-before` を eval + NoGrad で forward する。
4. optimizer step を実行する。
5. optimizer step 直後に `online-after` を同じ状態・同じ fixed taus で forward する。
6. 通常の hard copy または soft update を実行する。
7. target update 後に `target-after` を同じ状態・同じ fixed taus で forward する。
8. UpdateResult を確定し、購読された source key だけを公開する。

target 指標は必ず **sync / soft update 後の `online-after` と `target-after`** を比較する。hard update の sync 点では `11` と `13` が厳密に 0 になることをゼロ点として持つ。soft update でも指標の定義は変えない。

購読に応じて必要最小限の処理だけを行う。

- `01`〜`08`: probe sample + online-before + online-after。
- `11`〜`14`: probe sample + online-after + target-after。
- `15` のみ: probe sample も forward も行わず、target update mode と learn_step だけで算出する。
- 複数群を同時購読した場合: probe と online-after を共有し、最大 3 forward とする。
- 関連購読が無い場合: sample、forward、統計、UpdateResult の churn payload をすべて不活性にする。

測定本体は DQN 系で共有できる `dqn_based_agent` の共通内部へ置く。ただし初回実装で購読、config、metrics を公開するのは DefaultDQN だけとし、Rainbow への公開は deferred gate とする。

## Target update と cadence

### target_sync_age

- hard update: target update 後の測定時点で `learn_step % hard_update_interval`。
- soft update: sync の離散時点が無いため NaN。
- probe 不足や Q 指標の購読有無に依存しない。

### 503 cadence と位相 WARN

baseline の全 7 指標は `interval:503` とする。503 は現行の代表的な hard update interval 500 / 10000 と互いに素であり、測定位相が 1 点へ固定されるのを避ける。

custom interval は許可する。hard update interval を `C`、metrics interval を `I` とすると、観測する位相数は `C / gcd(C, I)` である。互いに素性や位相数はアルゴリズムの正当性ではなく利用目的に関わるため、fail-fast 条件にはしない。**位相数 1 だけを構造的な縮退として WARN 対象**とし、起動時に集約して 1 回だけ出す。WARN には固定される `target_sync_age` の位相値、metrics interval、hard update interval を含める。位相数 2 以上は少数でも許容し、被覆の十分性は記録された `target_sync_age` を見て実験設定側が判断する。閾値を追加して WARN 条件を目的依存にしない。

## BTR 参照との差分

BTR は churn 専用 batch を update 前後で比較する着想の参照元だが、実装を直写ししない。

- BTR は毎 update の PER sample、本 PRDは interval 503 の一様非復元 probe。
- BTR の IQN tau は forward ごとに変わりうるが、本 PRD は 32 本の fixed midpoint taus を共有する。
- BTR の行動別配列は保存せず、本 PRD は scalar 13 tag（7 source key）だけを出す。
- BTR 絶対値との parity より、パラメータ update に帰属できる非干渉測定を優先する。
- target は sync 前後の churn ではなく、sync 後の online / target 不一致を測る。

## 将来実装の受入条件

1. **ゼロ点**: 同一 Q、学習率 0、または optimizer skip では、action churn と全 online Q 差が厳密に 0 になる。
2. **FP32 境界**: 外側で BF16 autocast を有効にしたテストでも、churn の最大 3 forward 内では autocast が無効で出力が FP32 になる。BF16 の同一丸め区間へ入る近接 Q を合成し、FP32 の `argmax` と差分が保持されることを検証する。
3. **合成 online update**: 決定論的な合成更新で action flip、abs 差、signed max / min の期待値を厳密に検証できる。
4. **hard target**: 非 sync update で構成した既知差分、sync 後の `11` / `13` の厳密な 0、`15` の modulo 位相を決定論的に検証できる。
5. **soft target**: `11` / `13` は同じ定義で有限値を返し、`15` は NaN になる。
6. **購読ゲート**: 購読なしでは sample、forward、統計が完全に不活性である。online のみ、target のみ、age のみの購読でも上記の必要最小限の処理だけが走る。
7. **同一ビルド内の非干渉**: RNG 移行後の同じビルドで churn ON / OFF を比較し、学習 parameter、通常 Replay sample 系列、plasticity probe 系列、既存 metrics checksum が一致する。RNG クリーンブレーク前後の `4x_probe_*` checksum は比較対象外とする。
8. **cadence と診断**: interval 503 の発火、位相数 1 の custom interval での起動時 1 回 WARN、位相数 2 以上の custom interval の許可を検証する。
9. **値未成立**: update 後の probe 不足で Q 由来 6 source key が NaN、age が独立して成立し、未知 key だけが `std::nullopt` になる。
10. **設定検証**: `probe.batch_size < 1` と `iqn.num_taus < 1` はキーと指定値を含むエラーで fail-fast する。
11. **性能**: Release の代表 Run で ON / OFF の throughput 中央値差を ±3% 以内とする。超過時は既定値を自動変更せず、baseline 既定 ON を再審議する。

通常学習で churn が正値になることや、target disagreement が sync 間で単調増加することは受入条件にしない。どちらも正しい実装でも学習状態に依存して成立しないため、合成入力による決定論的検証へ置き換える。

## Complexity audit

### Keep

- 13 tag / 7 source key（online 4 + EMA 4、target 2 + EMA 2、位相 1）。EMA は同一 source key の別 tag なので追加計算は無い。
- churn 専用の一様非復元 probe batch 1024 と独立 RNG。比率の最小刻みは `1 / 1024 ≈ 0.098%` で、探索計器として想定する churn 1〜10% の変化を読むには十分な解像度を持つ。
- IQN fixed midpoint taus 32 本。
- baseline の cadence 503 と、hard update の 1 位相だけを測る場合の 1 回 WARN。
- `dqn_based_agent` 共通内部 + DefaultDQN 限定公開。
- `05` / `07`（と対の EMA）。実測根拠は弱いが、BTR 対応の探索計器であり追加計算の限界費用が小さいため残す。

### Defer behind an explicit gate

- ImageCls での class prediction churn。
- Rainbow の config、購読、metrics 公開。
- NoisyNet の固定 noise snapshot API と churn 対応。
- action transition 表、histogram、image 系出力。
- churn を抑える介入。
- 科学的な Run 比較と既定値の再評価。

### Cut

- plasticity が取得した実バッチの共有。
- plasticity の probe batch 4096 件の流用。
- custom interval と hard update interval の strict coprime fail-fast。
- LearnEvent の無い warmup 中に NaN 点を生成する経路。
- 通常学習で churn が必ず正値になるというテスト。
- target disagreement の単調鋸歯を要求するテスト。

## 文書レビュー結果

- 未決事項: **0 件**。
- 旧案との矛盾: **0 件**。
- 実装着手可否: **可能**。ただし、別の明示依頼と実装計画が必要。
- 本 PRD の正式化変更は PRD 本体、CONTEXT.md、ADR 0033 の docs-only 3 点に限定する。
