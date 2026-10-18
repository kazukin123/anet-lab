# Codex 引継ぎ: DropMergeEnv の NoLegalDrop 終端追加

## 背景

`DropMergeEnv` の `action_mode=direct_noop` で、eval 中に NOOP が連続して `no_drop_timeout_steps` まで続くケースがあった。
当初は NEET 化・学習側バグの疑いがあったが、実際に盤面と Q 値を確認すると、該当ケースの多くは **盤面がほぼ詰みで、
どの DROP を選んでも SpawnBlocked/GameOver になる状態で NOOP を選んでいる** ものだった。

この場合、NOOP は不合理なサボりではなく、次のような合理的な「投了 / 終局待ち」に近い。

```text
DROP -> SpawnBlocked + game_over_penalty
NOOP -> 状態はほぼ不変、追加報酬なし
```

現状はこの状態を `no_drop_timeout_steps=500` まで待ってから `Timeout` として打ち切るため、次の問題がある。

- eval の実時間効率が悪い。
- metrics 上は `Timeout` が増え、見た目上「NEET / 無操作失敗」に見える。
- 実際には `NoLegalDrop`、つまり「合法的に置ける DROP がない終局状態」と分けて扱うべき。

## 目的

`DropMergeEnv` に **NoLegalDrop 終端**を追加する。

`direct_noop` モードで、エージェントが NOOP を選び、かつ現在の果物をどの DROP 位置に置いても spawn area が塞がっている場合、`no_drop_timeout_steps` まで待たずに即時 episode 終了とする。

今回の方針では **train/eval どちらも同じ処理**にする。

```text
NoLegalDrop 検出時:
  done      = true
  truncated = false
  term_reason = NoLegalDrop
  game_over_ は false のまま
  game_over_penalty は付けない
```

`truncated` ではなく `done=true` にする。理由は、この状態は外部都合の打ち切りではなく、追加報酬が得られない実質的な終端状態として扱いたいため。
Replay 側では `done=true` が terminal bootstrap cut になり、`truncated` は bootstrap 継続扱いになるため、この修正では `done=true` が意図に合う。

## 対象ファイル

主対象:

```text
core/envs/dropmerge1/src/DropMergeEnv.hpp
core/envs/dropmerge1/src/DropMergeEnv.cpp
```

必要に応じて run config / metrics 定義:

```text
DropMerge.txt など、metrics.scalar 定義を持つ設定ファイル
```

## 現状コードの要点

現状の `DropMergeEnv::Step()` は、おおむね次の流れ。

1. `step_count_++`
2. action から `is_drop_action` / `is_noop_action` を判定
3. `steps_since_last_drop_` 更新
4. `processAction(action)`
5. Box2D step / merge / reward 計算
6. `time_penalty`、`game_over_penalty`
7. `done = game_over_`
8. `truncated = max_step` または `no_drop_timeout_steps`
9. `term_reason_` を `MaxStep` / `Timeout` 等に設定
10. episode 終了時統計を記録
11. `state.done` / `state.truncated` を設定
12. `noop_penalty`
13. `episode_reward_` 更新

`processAction()` では `DirectNoop` の `action == 0` が NOOP、`action > 0` が `DROP_(action - 1)`。
DROP 時に `isSpawnAreaClear(actual_x, spawn_y, r_drop)` が false なら `game_over_ = true`、`term_reason_ = SpawnBlocked` になる。

## 実装方針

### 1. TerminationReason に NoLegalDrop を追加

`DropMergeEnv.hpp` の `TerminationReason` に `NoLegalDrop` を追加する。

```cpp
enum class TerminationReason {        ///< メトリクス集計用
    None,
    Timeout,
    SpawnBlocked,
    Overflow,
    MaxStep,
    NoLegalDrop
};
```

順番はどこでもよいが、既存値の意味を気にする必要がなければ末尾追加が安全。

### 2. helper を追加

`DropMergeEnv.hpp` private に helper を追加する。

```cpp
bool IsNoLegalDropState() const;
bool HasAnyLegalDropForCurrentFruit() const;
bool HasClearSpawnXInRange(float x_min, float x_max, float y, float r) const;
```

命名は既存 style に合わせて `isNoLegalDropState` などでもよい。周辺コードは lower camel の private method が多いので、最終的には既存 style 優先。

### 3. spawn overlap margin を共通化

現状 `isSpawnAreaClear()` 内に local constant としてある `kOverlapMargin = 0.95f` を、`DropMergeEnv.cpp` file scope の定数に移す。

```cpp
constexpr int kNumScalarObsDim = 4;
constexpr float kSpawnOverlapMargin = 0.95f;
```

`isSpawnAreaClear()` と新しい interval 判定の両方で同じ値を使う。

### 4. NoLegalDrop 判定条件

`IsNoLegalDropState()` は false positive を避ける。以下をすべて満たす場合だけ `true` を返す。

```text
action_mode_ == ActionMode::DirectNoop
game_over_ == false
dropper_.is_busy == false
dropper_.pending_body == nullptr
dropper_.current_rank が 1..kFruitTypeCount の範囲
merge_requests_ が空
bodies_to_destroy_ が空
isWorldSettled() == true
HasAnyLegalDropForCurrentFruit() == false
```

実装例:

```cpp
bool DropMergeEnv::IsNoLegalDropState() const
{
    anet::ProfileRange r("DropMergeEnv::IsNoLegalDropState");

    if (action_mode_ != ActionMode::DirectNoop) return false;
    if (game_over_) return false;
    if (dropper_.is_busy) return false;
    if (dropper_.pending_body != nullptr) return false;
    if (dropper_.current_rank < 1 || dropper_.current_rank > kFruitTypeCount) return false;
    if (!merge_requests_.empty()) return false;
    if (!bodies_to_destroy_.empty()) return false;
    if (!isWorldSettled()) return false;

    return !HasAnyLegalDropForCurrentFruit();
}
```

### 5. legal drop 判定

`HasAnyLegalDropForCurrentFruit()` は、現在の `processAction()` の direct drop 座標計算と同じルールで `drop_col` を走査する。

重要点:

- `num_drop_actions_` 個の drop column を調べる。
- `base_x` は `processAction()` と同じ式を使う。
- `spawn_y = config_.ground_y + config_.box_height`。
- `r_drop = config_.fruit_radii[dropper_.current_rank - 1]`。
- `drop_noise` を考慮する。
- false positive、つまり「本当は置けるのに NoLegalDrop と判定する」ことを避ける。
- そのため、ある column に **置ける可能性が少しでもある**なら `true` を返す。

実装例:

```cpp
bool DropMergeEnv::HasAnyLegalDropForCurrentFruit() const
{
    anet::ProfileRange r("DropMergeEnv::HasAnyLegalDropForCurrentFruit");

    if (dropper_.current_rank < 1 || dropper_.current_rank > kFruitTypeCount) {
        return false;
    }

    const float spawn_y = config_.ground_y + config_.box_height;
    const float r_drop = config_.fruit_radii[dropper_.current_rank - 1];

    const float min_x = -config_.box_width * 0.5f;
    const float max_x =  config_.box_width * 0.5f;
    const float cell_w = (max_x - min_x) / static_cast<float>(num_drop_actions_);

    const float half_w = config_.box_width * 0.5f;
    const float limit = half_w - r_drop - 0.01f;
    const float noise = std::max(0.0f, config_.drop_noise);

    for (int col = 0; col < num_drop_actions_; ++col) {
        const float base_x = min_x + (static_cast<float>(col) + 0.5f) * cell_w;

        float x_min = std::clamp(base_x - noise, -limit, limit);
        float x_max = std::clamp(base_x + noise, -limit, limit);
        if (x_min > x_max) std::swap(x_min, x_max);

        if (HasClearSpawnXInRange(x_min, x_max, spawn_y, r_drop)) {
            return true;
        }
    }

    return false;
}
```

### 6. drop noise 範囲の厳密判定

`drop_noise` があるため、単純に `base_x` だけを見ると false positive/false negative が出る。
NoLegalDrop は誤終端が一番危険なので、次の方針にする。

```text
ある DROP action の actual_x 範囲内に、spawn area が clear な x が1点でもあれば legal とみなす。
actual_x 範囲全体が既存 fruit の blocked interval で覆われている場合だけ illegal とみなす。
```

`HasClearSpawnXInRange()` は、既存 fruit が固定 `spawn_y` 上で塞ぐ x 区間を計算し、その union が `[x_min, x_max]` 全体を覆っているかを見る。

実装例:

```cpp
bool DropMergeEnv::HasClearSpawnXInRange(float x_min, float x_max, float y, float r) const
{
    anet::ProfileRange pr("DropMergeEnv::HasClearSpawnXInRange");

    if (x_min > x_max) std::swap(x_min, x_max);

    // noise=0 等で実質1点の場合は既存判定を使う。
    if (std::abs(x_max - x_min) <= 1.0e-6f) {
        return isSpawnAreaClear(x_min, y, r);
    }

    std::vector<std::pair<float, float>> blocked_intervals;

    for (b2Body* b = world_->GetBodyList(); b; b = b->GetNext()) {
        if (b->GetType() != b2_dynamicBody) continue;
        if (b == dropper_.pending_body) continue;

        auto data = DecodeUserData(b->GetUserData().pointer);
        if (data.first != BodyType::Fruit) continue;

        const b2Vec2 pos = b->GetPosition();
        const float r_other = config_.fruit_radii[data.second - 1];
        const float radius_sum = (r + r_other) * kSpawnOverlapMargin;
        const float dy = pos.y - y;
        const float rem = radius_sum * radius_sum - dy * dy;
        if (rem <= 0.0f) continue;

        const float dx = std::sqrt(rem);
        const float left = std::max(x_min, pos.x - dx);
        const float right = std::min(x_max, pos.x + dx);
        if (left <= right) {
            blocked_intervals.emplace_back(left, right);
        }
    }

    if (blocked_intervals.empty()) {
        return true;
    }

    std::sort(blocked_intervals.begin(), blocked_intervals.end(),
        [](const auto& lhs, const auto& rhs) {
            return lhs.first < rhs.first;
        });

    float covered_until = x_min;
    for (const auto& interval : blocked_intervals) {
        if (interval.first > covered_until) {
            return true; // gap がある = clear な x がある
        }
        covered_until = std::max(covered_until, interval.second);
        if (covered_until >= x_max) {
            return false; // 範囲全体が塞がっている
        }
    }

    return covered_until < x_max;
}
```

注意:

- `isSpawnAreaClear()` と完全に同じ body filter / margin を使うこと。
- `HasClearSpawnXInRange()` が `true` を返す意味は「legal drop が存在する可能性がある」。
- `false` を返す意味は「その column の noise 範囲はすべて塞がっている」。

### 7. Step() への組み込み

`DropMergeEnv::Step()` の終端判定部分を変更する。

現状:

```cpp
bool done = game_over_;
bool truncated = (step_count_ >= config_.max_step);
```

修正後のイメージ:

```cpp
const bool no_legal_drop_terminal =
    !game_over_ &&
    is_noop_action &&
    IsNoLegalDropState();

if (no_legal_drop_terminal) {
    term_reason_ = TerminationReason::NoLegalDrop;
    LOG::verbose() << "Episode done: no legal drop remains. episode_score="
        << episode_score_ << " step_count=" << step_count_ << " x=" << dropper_.x;
}

bool done = game_over_ || no_legal_drop_terminal;
bool truncated = (!done && step_count_ >= config_.max_step);
```

その後の max step / no-drop timeout は、必ず `!done` 条件で処理する。

```cpp
if (!done && truncated) {
    term_reason_ = TerminationReason::MaxStep;
    ...
}

if (!done && config_.no_drop_timeout_steps > 0 &&
    steps_since_last_drop_ >= config_.no_drop_timeout_steps) {
    truncated = true;
    if (term_reason_ != TerminationReason::MaxStep) {
        term_reason_ = TerminationReason::Timeout;
    }
    ...
}
```

重要:

- `NoLegalDrop` は `game_over_` を立てない。
- `game_over_penalty` は付けない。
- `SpawnBlocked` / `Overflow` など、既に `game_over_` になっているケースは NoLegalDrop で上書きしない。
- `NoLegalDrop` は `truncated=false`、`done=true`。
- `no_drop_timeout_steps` は NoLegalDrop の後に上書きしてはいけない。

既存の `time_penalty` / `noop_penalty` は現状設定では 0 のため実質影響なし。
実装では既存の処理順をできるだけ維持してよい。ただし `game_over_penalty` だけは NoLegalDrop に付かないこと。

### 8. metrics scalar を追加

`DropMergeEnv::GetScalar()` に以下を追加する。

```cpp
if (key == "term_reason_no_legal_drop") {
    if (!episode_just_ended_) return nan;
    return (term_reason_ == TerminationReason::NoLegalDrop) ? 1.0f : 0.0f;
}
```

可能なら以下も追加すると分析しやすい。

```cpp
if (key == "ep_steps_since_last_drop") {
    if (!episode_just_ended_) return nan;
    return static_cast<float>(last_episode_steps_since_last_drop_);
}
```

ただし `last_episode_steps_since_last_drop_` を新規 member として持つ必要があるため、最小修正では `term_reason_no_legal_drop` だけでよい。

### 9. metrics 設定の追加案

現在の `DropMerge.txt` には `term_reason_timeout` / `term_reason_spawn_blocked` / `term_reason_overflow` 等がある。
必要なら次を追加する。

```ini
M.[42_env/18_tr_no_legal_mean]      = $env mean.term_reason_no_legal_drop @train
M.[42_env/69_tr_no_legal_mean_ema]  = $env mean.term_reason_no_legal_drop @train $ema ema_alpha:0.001

M.[51_eval1/65_tr_no_legal_mean_ema] = $env mean.term_reason_no_legal_drop @episode_end $eval.[eval1] $ema ema_alpha:0.001
M.[52_eval2/65_tr_no_legal_mean_ema] = $env mean.term_reason_no_legal_drop @episode_end $eval.[eval2] $ema ema_alpha:0.001
```

既存 ID と衝突しないように最終確認すること。

## ProfileRange

CPU 負荷確認用に以下へ `ProfileRange` を入れる。

```cpp
anet::ProfileRange r("DropMergeEnv::IsNoLegalDropState");
anet::ProfileRange r("DropMergeEnv::HasAnyLegalDropForCurrentFruit");
anet::ProfileRange r("DropMergeEnv::HasClearSpawnXInRange");
```

`DropMergeEnv::Step` には既に `ProfileRange` があるので、新規 helper の粒度で計測できるようにする。

初版では quick check は不要。
理由:

- NoLegalDrop 判定は `is_noop_action` かつ `DirectNoop` かつ settled 状態だけで走る。
- 終局 NoLegalDrop なら 1 回の判定で episode が終わるため、500 step 分の無駄な timeout tail を削れる。
- legal drop が存在する通常局面では、多くの場合、早い column で `HasAnyLegalDropForCurrentFruit()` が true になって return する。
- まず ProfileRange で実測し、必要なら後から cache / grid quick check を入れる。

## 受け入れ条件

### 期待動作

1. `direct_noop` で、全 DROP が SpawnBlocked になる詰み盤面で NOOP を選んだ場合:

```text
done == true
truncated == false
term_reason_no_legal_drop == 1
term_reason_timeout == 0
term_reason_spawn_blocked == 0
game_over_penalty は加算されない
```

2. 同じ詰み盤面で DROP を選んだ場合:

```text
done == true
truncated == false
term_reason_spawn_blocked == 1
game_over_penalty が加算される
```

3. 1つでも置ける可能性がある DROP column がある状態で NOOP を選んだ場合:

```text
NoLegalDrop では終了しない
既存どおり no_drop_timeout_steps まで行けば Timeout になり得る
```

4. world がまだ不安定、dropper busy、pending body あり、merge request あり、destroy 待ち body ありの場合:

```text
NoLegalDrop では終了しない
```

5. `action_mode != direct_noop` では、今回の変更による挙動変化を起こさない。

6. `MaxStep` / `Timeout` / `SpawnBlocked` / `Overflow` の既存 metrics は壊さない。

### 性能確認

Profile で以下を確認する。

```text
DropMergeEnv::IsNoLegalDropState
DropMergeEnv::HasAnyLegalDropForCurrentFruit
DropMergeEnv::HasClearSpawnXInRange
```

NoLegalDrop が増えた場合、episode step が 500 近く短縮され、eval の実時間効率が改善するはず。

## 注意点

- NoLegalDrop は「NOOP が悪い」ことを表す metrics ではない。むしろ、詰み局面で SpawnBlocked を避けた合理的終端として扱う。
- `term_reason_timeout` は、今後は「DROP 可能なのに NOOP し続けた疑いがある真の timeout」として見る。
- NoLegalDrop 判定は false positive を避けること。迷った場合は「legal drop がある」と判定して episode を継続する方が安全。
- `drop_noise` を無視しないこと。壁クランプも `processAction()` と同じにすること。
- `NoLegalDrop` は `game_over_` を立てない。GameOver とは別の正常終局扱いにする。
- Replay 側の挙動を変える必要はない。`done=true` にすれば既存の terminal 処理で bootstrap が切れる。

## 変更しないこと

- Agent / DQN / UQE / spatial exploration には手を入れない。
- `noop_penalty` / `time_penalty` / `game_over_penalty` の設定値は変更しない。
- `no_drop_timeout_steps` を単純に短くする対応にはしない。
- eval 専用処理にはしない。train/eval とも同じ NoLegalDrop 判定を使う。
