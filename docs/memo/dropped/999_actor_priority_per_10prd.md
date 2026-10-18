# Actor-side PER initial priority PRD 草案

> 中止(理由: PRD 035 として正式採番・実装済み(docs/memo/done/035_approx_actor_priority_per_10prd.md)。本書は旧草案)

> 番号 999。検討草案。
> テーマ: `per_initial_priority` / `max_prio` による仮の初期優先度を、Ape-X 風の actor 推定 priority に置き換えるかを検討する。

## Problem Statement

DropMerge 系 run で PER 初期優先度の妥当性が分かりづらい。現状は新規 transition が learner による TD error 更新を受けるまで、固定 `per_initial_priority` または `max_prio` 相当の仮 priority で SumTree に入る。この仮 priority が高すぎると新規 transition が replay 予算を支配し、低すぎると一度も sample されずに eviction されやすい。

直近で追加した観測では、定常状態の速報値が概ね次の状態になっている。

- `per_sample_initial_ratio`: 約 0.73
- `replaybuffer.per.initial_mass_ratio`: 約 0.74
- `replaybuffer.per.last_evicted_never_sampled_ratio`: 約 0.26

この状態は「固定初期 priority が低すぎて新規 transition が無視されている」わけではない。一方で、ReplayRatio=1 の学習予算の多くが learner 未更新の初期 priority transition に使われており、PER 本来の TD error 選別が効く余地は 2〜3 割程度に見える。

問題は、`per_initial_priority` を固定値として統計から調整する運用が、環境・報酬スケール・TBO・network 変化・学習段階に強く依存することである。固定値を下げると PER 再利用は増えるが coverage が悪化し、`max_prio` に寄せると新規優先がさらに強くなりやすい。

Ape-X 風に actor 側で transition の priority を推定できれば、固定初期値チューニング問題を「actor 推定 TD error の精度・staleness・coverage 管理」の問題へ置き換えられる可能性がある。本 PRD は、その導入に必要な設計判断と段階実装方針を整理する。

## Solution

ReplayBuffer に入る新規 transition へ、actor 側で計算した optional priority hint を付与できるようにする。ReplayBuffer は n-step sequence が sampleable になったタイミングで、priority hint があればそれを初期 priority として使い、無ければ従来の `per_initial_priority` / `max_prio` 系 fallback を使う。

重要なのは、「初期 priority という概念を完全に無くす」のではなく、**固定または max による仮 priority を、actor 推定 priority に置き換える**ことである。learner がその transition を sample した後は、従来通り learner 側 TD error で priority を上書きする。

最初の実装は Ape-X 完全形ではなく、既存 single-process DQN/Rainbow 系に合う hybrid とする。

- Actor または action/experience assembly 境界で、priority hint を optional に保持する。
- ReplayBuffer の `Push` signature は変えず、既存の experience metadata 経路に priority hint を載せる。
- ReplayBuffer は priority hint が valid かつ finite のときだけ使用する。
- invalid / missing / unsupported の場合は従来の初期 priority に fallback する。
- learner 更新後の `UpdatePriorities` は現行通り最終権威として残す。

導入後、既存 `initial_mass_ratio` の意味は「固定初期値 mass」ではなく「learner 未更新 priority mass」へ変わる。そのため、metric 名と説明は `unlearned_priority_mass_ratio` 方向へ整理するか、既存名を互換 alias として残す。

## User Stories

1. As an RL experimenter, I want new replay entries to receive actor-estimated priority, so that fixed `per_initial_priority` does not dominate sampling behavior.
2. As an RL experimenter, I want low actor-priority transitions to still have a minimum chance of being sampled, so that useful but underestimated experiences are not silently discarded.
3. As an RL experimenter, I want actor priority to fall back to the existing fixed initial priority when unavailable, so that unsupported agents keep working.
4. As an RL experimenter, I want learner TD error to overwrite actor priority after sampling, so that stale actor estimates do not remain authoritative.
5. As an RL experimenter, I want to compare fixed initial priority, max priority, and actor priority under the same metrics, so that the new behavior can be evaluated without changing the experiment workflow.
6. As a performance investigator, I want to know the extra actor-side forward cost, so that actor priority does not hide a training speed regression.
7. As a performance investigator, I want actor priority staleness to be observable, so that bad results can be separated into priority quality vs network lag.
8. As a maintainer, I want the ReplayBuffer push interface to remain stable, so that PrefetchingReplayBuffer and existing tests do not need a broad API rewrite.
9. As a maintainer, I want priority source metadata to be explicit, so that metrics can distinguish fixed, actor-estimated, and learner-updated priority.
10. As a maintainer, I want DQN and QR-DQN priority formulas to share one conceptual path, so that distributional and non-distributional agents do not drift.
11. As a maintainer, I want MuZero and ImageCls paths to be unaffected, so that this feature remains scoped to DQN-style PER.
12. As a test author, I want behavior tests around priority source and fallback, so that tests do not depend on private SumTree implementation details.
13. As a future optimizer, I want actor-priority metrics to expose actor-vs-learner priority agreement, so that we can decide whether actor priority is worth keeping.
14. As a user tuning DropMerge, I want a practical knob that starts with new-sample coverage and later allows flatter evaluation, so that early exploration and late PER selection can both be controlled.

## Implementation Decisions

### 1. 導入単位は optional priority hint

- 新規 transition は optional `initial_priority_hint` を持てる。
- hint が存在し、finite で、正の値である場合だけ ReplayBuffer の初期 priority として使う。
- hint が無い場合、または invalid の場合は従来の `per_initial_priority` / `max_prio` 初期化へ fallback する。
- ReplayBuffer の public `Push` signature は変更しない。priority hint は experience metadata として流す。
- ReplayBuffer 側は priority hint の source を知らなくてもよいが、metrics のために priority source は最低限区別できるようにする。

### 2. actor priority は learner priority の前段推定に留める

- actor priority は「初期値の代替」であり、learner priority の代替ではない。
- learner が sample した transition は、従来通り learner TD error から priority を再計算して上書きする。
- actor priority は network staleness、探索行動、reward scaler、TBO、n-step terminal 処理のズレを含むため、最終権威にしない。
- learner 更新済みかどうかの状態は引き続き追跡する。既存の `per_is_initial_priority` は意味を「learner 未更新 priority」へ読み替える。

### 3. priority source を状態として持つ

priority mass と sample ratio を正しく読むため、各 slot の priority source を少なくとも次の状態で区別する。

- `none` / invalidated
- `fixed_initial`
- `actor_initial`
- `learner_updated`

この分類により、次のような metric を出せる。

- learner 未更新 priority mass ratio
- actor initial priority mass ratio
- fixed fallback priority mass ratio
- sampled actor-initial ratio
- sampled fixed-initial ratio
- learner-updated sample ratio

既存の `initial_mass_ratio` は、互換性を考えるなら当面 `unlearned_priority_mass_ratio` の alias として扱う。将来的には metric 名を整理する。

### 4. priority floor と fallback は必須

actor 推定 TD error をそのまま priority にすると、低く見積もられた新規 transition が一度も sample されずに eviction されやすくなる。

そのため、actor priority は最低でも次のように補正する。

- `priority = max(actor_priority, actor_priority_floor)`
- NaN / Inf / 非正値は fallback
- 必要なら上限 clip または percentile clip を入れる

floor の初期値は、現行 fixed priority より十分小さくしつつ、`last_evicted_never_sampled_ratio` が悪化しすぎない範囲で調整する。最初の評価では actor priority 本体よりも、この floor の影響が大きくなる可能性がある。

### 5. actor priority の計算位置は action selection そのものではなく experience 完成境界を優先する

行動選択時点では reward と next state がまだ無いため、完全な TD error は計算できない。したがって、実装候補は次のどちらかになる。

1. 行動選択時に `Q(s,a)` や `Q(s,*)` を保存し、n-step transition が完成した時点で target 側だけを追加計算して priority を作る。
2. n-step transition 完成時点で、actor/runner 側の network を使って `Q(s,a)` と target をまとめて計算する。

最初の草案では 1 を優先する。理由は、行動選択 forward の結果を活かせるため追加 cost が小さく、現在の action metadata に Q 値を持たせる設計と親和性があるためである。ただし、action metadata のうち ReplayBuffer へ永続化される情報と aux-only 情報の境界は整理が必要である。

### 6. DQN / QR-DQN / TBO の式を明示的に揃える

actor priority は learner TD error と同じ意味の値を目指す。

- DQN: `abs(Q(s,a) - target)`
- QR-DQN: priority を quantile loss 由来にするか、mean target 差分にするかを決める
- TBO: h-space の TD error を使うのか、real-space に戻した差分を使うのかを learner 実装に合わせる
- reward scaler: learner と actor で target reward scale がズレないようにする
- terminal / truncated / n-step: ReplayExperienceBuilder と同じ境界条件で target を作る

最初の implementation では、DQN / QR-DQN の既存 learner priority と一致する定義を優先し、TBO や分布型の厳密化で迷う場合は fallback または feature flag で無効にする。

### 7. staleness を測る

actor priority は learner より古い network で計算される可能性がある。これを不可視にすると、悪化時に原因が分からない。

priority hint には可能なら次の情報を付ける。

- actor network version
- learner version または sync step
- priority computed step
- model age / staleness step

最初は version 管理が大きすぎる場合、actor sync count だけでもよい。後から actor-vs-learner priority の相関を見るための足場を残す。

### 8. metric の再定義と追加

既存 3 metric は導入後も使うが、意味を明確化する。

- `per_sample_initial_ratio`: learner 未更新 priority のまま sample された割合
- `initial_mass_ratio`: learner 未更新 priority mass ratio。名前は将来変更候補
- `last_evicted_never_sampled_ratio`: source に依存せず継続して重要

追加候補:

- `per_sample_actor_initial_ratio`
- `replaybuffer.per.actor_initial_mass_ratio`
- `replaybuffer.per.fixed_fallback_mass_ratio`
- `per_actor_priority_abs_error_mean`
- `per_actor_priority_rank_corr`
- `per_actor_priority_staleness_mean`

ただし、最初から増やしすぎない。初期実装では `actor_initial_mass_ratio`、`sample_actor_initial_ratio`、`fixed_fallback_mass_ratio` の 3 つを優先候補とする。

### 9. config は mode と safety knobs に分ける

config は固定初期値の代替方式と安全弁を分ける。

- priority initialization mode: `fixed`, `max`, `actor`
- actor priority enabled flag
- actor priority floor
- actor priority clip
- invalid actor priority fallback
- actor priority metrics enabled

既存 run との比較を壊さないため、default は現行挙動のままにする。actor priority は明示 opt-in にする。

### 10. 段階導入にする

一度に Ape-X 完全形へ寄せない。

1. ReplayBuffer が optional priority hint を受け取り、fallback できる。
2. priority source と metrics を追加する。
3. DQN actor path だけで priority hint を生成する。
4. learner 更新後 priority と actor priority の差分 metrics を出す。
5. QR-DQN / TBO / Rainbow 対応を広げる。
6. fixed / actor / floor の実 run 比較で採用判断する。

## Testing Decisions

- テストは private SumTree の内部配列ではなく、外部から見える sampling priority と metrics で確認する。
- priority hint がある transition は fixed initial priority ではなく hint 由来 mass として数えられること。
- priority hint が NaN / Inf / 非正値 / missing の場合、従来 fallback priority になること。
- learner が sample 後に `UpdatePriorities` すると、priority source が learner updated へ変わること。
- `per_sample_initial_ratio` は actor priority 導入後も learner 未更新 sample ratio として成立すること。
- `last_evicted_never_sampled_ratio` は priority source に依存せず、未サンプル eviction を継続して測れること。
- actor priority floor を下げたとき、未サンプル eviction が増えるケースを小容量 buffer で再現できること。
- DQN priority hint の値が、同じ条件で計算した learner TD error 由来 priority と概ね一致すること。
- QR-DQN / TBO が未対応の場合は、明示的に fallback することをテストする。
- PrefetchingReplayBuffer の `Push` / `Sample` / `UpdatePriorities` ordering は既存 regression を維持する。
- full test に加え、少なくとも replay buffer、DQN PER、transfer/prefetch 関連タグを対象にする。

## Out of Scope

- Ape-X の分散 actor / learner プロセス分離そのもの。
- replay service 化。
- GPU resident SumTree。
- ReplayRatio の自動調整。
- actor priority だけで learner priority update を省略すること。
- MuZero replay buffer への同時適用。
- ImageCls agent への適用。
- 既存 `per_initial_priority` の削除。
- 既存 metrics の即時 rename による履歴破壊。
- `max_prio` 初期化の再設計全般。比較対象として残すが、本 PRD の主目的ではない。

## Further Notes

- 現在の実測値では fixed `per_initial_priority=1.0` でも初期 priority mass が 7 割強を占める。actor priority 導入の狙いは、この mass を単に減らすことではなく、未更新 transition の中でも TD error が高そうなものへ順序を付けることである。
- `PER_BETA` は sampling 分布ではなく IS weight 補正側なので、actor priority 導入による `per_sample_initial_ratio` / mass ratio の変化とは別に評価する。
- actor priority は fixed initial priority tuning を不要にする銀の弾丸ではない。新たに、actor network の古さ、target 式のズレ、priority floor、低 priority transition の coverage を管理する必要がある。
- 初期評価では、`last_evicted_never_sampled_ratio` が現行 0.26 から大きく悪化しないことを重視する。actor priority で PER の意味が強くなっても、coverage が壊れるなら採用しない。
- 直感として「最初は未採用優先、だんだん平坦に評価したい」は妥当。actor priority を入れる場合でも、early phase は floor を高め、late phase は floor を下げる schedule を別途検討する価値がある。
