# ProfiledValue による ImageCls learning_rate 制御

> 設計分担: Claude/Codex=設計/PRD、実装=Codex、Run/commit=ユーザー。
> 本書は self-contained。実装時は行番号ではなく、近傍のシンボル名で再検索する。

## Context（背景・目的）

`ImageClsLearner` は現在、AdamW の learning rate を固定値として扱う。画像分類では
linear warmup、cosine decay、SGDR など、step に応じて learning rate を変動させる運用が
標準的に必要になる。

ただし今回導入する概念は LR scheduler 専用ではない。目的は、任意の値を step や profile
上の位置に応じて変動させる共通部品 `ProfiledValue<T>` を作り、まず ImageCls の
`learning_rate` に適用することである。将来的には `net.config_profile` も同じ概念へ寄せられる。

## 確定した設計判断

1. 共通部品名は `ProfiledValue<T>` とする。
2. 設定構造は `ProfiledValueConfig<T>` と `ProfiledValuePhaseConfig<T>` に分ける。
3. `ProfiledValueConfig<T>` と `ProfiledValuePhaseConfig<T>` は素の struct とし、読み取り処理を struct 本体へ持たせない。
4. `ProfiledValueConfig<T>` の root field と phase field は、継承では共通化しない。メンバ重複は許容し、設定項目と struct メンバ構成の対応を優先する。
5. `ANET_READ_CONFIG(config_data, learning_rate)` の一行で構造化 config を読めるように、`ConfigReader<ProfiledValueConfig<T>>` 特化を追加する。
6. `type` に関係なく全 field を読む。未使用 field は dormant config として許容し、override で `type` だけ差し替える運用を支える。
7. `constant` は `value` を使う。`linear` / `cosine` / `cosine_restart` は `start` / `end` / `steps` を使う。
8. `phased` は `phases` リストに書かれた順序で `phase.[name]` を評価する。`phase.[name]` 定義の出現順は実行順にしない。
9. ImageCls では `ImageClsAgent.learning_rate.*` を使う。`lr.*` や `learning_rate.schedule.*` は使わない。
10. 旧 scalar 形式 `ImageClsAgent.learning_rate = 1e-3` への fallback は実装しない。
11. learning rate metric は `ImageClsUpdateResult` ではなく Agent 自体の scalar として公開し、`$agent learning_rate @learn` で記録する。
12. `ProfiledValue<T>` 自体は thread-safe ではない。排他は呼び側で確保する。
13. `ProfiledValue<T>` は step axis を知らない。`exp_step` か `learn_step` かなど、どの step を渡すかは呼び側の責務とする。

## 共通 Config / Runtime API

### Config struct

```cpp
template <typename T>
struct ProfiledValuePhaseConfig {
    std::string type = "constant";
    T value{};
    T start{};
    T end{};
    uint64_t steps = 0;
    double cycle_mult = 1.0;
};

template <typename T>
struct ProfiledValueConfig {
    std::string type = "constant";
    T value{};
    T start{};
    T end{};
    uint64_t steps = 0;
    double cycle_mult = 1.0;
    std::vector<std::string> phases;
    anet::OrderedMap<std::string, ProfiledValuePhaseConfig<T>> phase;
};
```

`cycle_mult` は `cosine_restart` 用で、restart ごとに cosine cycle の `steps` を何倍するかを表す。
たとえば `steps=500000, cycle_mult=2` の場合、cycle 長は `500000 -> 1000000 -> 2000000 -> ...` となる。
`0 < cycle_mult < 1` も有効な設定値として扱い、cycle 長は restart ごとに縮小する。縮小後の cycle 長は 1 step を下限として飽和する。

### Runtime class

```cpp
template <typename T>
class ProfiledValue {
public:
    explicit ProfiledValue(ProfiledValueConfig<T> config);

    void Update(uint64_t step);
    T Value() const;
    T Evaluate(uint64_t step) const;
    T EvaluateByIndex(size_t index, size_t count) const;
};
```

- `Update(step)` は渡された step に応じて現在値を更新して cache する。
- `Value()` は cache 済みの現在値を返す。再計算、config lookup、lock を行わない。
- `Evaluate(step)` は副作用なしで step に対応する値を計算する。
- `EvaluateByIndex(index, count)` は `net.config_profile` など、step ではなく profile 上の位置から値を求める用途に使う。`constant` / `linear` / `cosine` のみ対応し、`cosine_restart` / `phased` は index-based profile として未定義なので `ANET_SYSTEM_ERROR` で失敗させる。

## 評価ルール

### `constant`

- active field: `value`
- `start` / `end` / `steps` / `phase` は読まれるが評価では使わない。

### `linear`

- active field: `start`, `end`, `steps`
- `steps > 0` を必須とする。
- `step >= steps` では `end` を保持する。

### `cosine`

- active field: `start`, `end`, `steps`
- `steps > 0` を必須とする。
- `step >= steps` では `end` を保持する。

### `cosine_restart`

- active field: `start`, `end`, `steps`, `cycle_mult`
- `steps > 0` と `cycle_mult > 0` を必須とする。
- 各 cycle は `start -> end` の cosine decay を行い、cycle 終了後に `start` へ restart する。
- `cycle_mult != 1` の場合、restart ごとに次の cycle 長へ倍率を掛ける。
- `0 < cycle_mult < 1` の場合は cycle 長が縮小し、1 step を下限として飽和する。
- `EvaluateByIndex(index, count)` には対応しない。

### `phased`

- active field: `phases`, `phase`
- `phases` は空を許可しない。
- 各 phase は `phases` に列挙された順に評価する。
- phase の開始条件は「前 phase が終了したら開始」で固定する。
- phase の長さは各 `phase.[name].steps` で決まる。
- 最終 phase 終了後は、最終 phase の終端値を保持する。
- `EvaluateByIndex(index, count)` には対応しない。

## Config 読み取り仕様

`ProfiledValueConfig<T>` は以下のように既存流儀で読む。

```cpp
ProfiledValueConfig<double> learning_rate;
ANET_READ_CONFIG(config_data, learning_rate);
```

`Config::ReadConfig()` に型別 reader の分岐を追加する。

- `ConfigReader<T>` が有効な型は、`ConfigReader<T>::Read(...)` に委譲する。
- 通常型は既存の `ConfigData::Read()` ベースの読み取りを維持する。
- `ConfigReader<T>` primary template は config infrastructure として `config.hpp` に置く。
- `ConfigReader<ProfiledValueConfig<T>>` 特化と `Read(...)` 本体は、schedule カテゴリとして `schedule.hpp` に置く。`config.hpp` は `ProfiledValueConfig` を参照しない。

`Config` には composite reader 用の protected helper を追加する。

```cpp
template <typename T>
void ReadSubConfig(
    const ConfigData& config_data,
    const std::string& root_key,
    const std::string& sub_key,
    T& value);

std::string MakeTaggedSubConfigKey(
    const std::string& root_key,
    const std::string& sub_key,
    const std::string& tag) const;
```

`ReadSubConfig(config_data, "learning_rate", "type", learning_rate.type)` は、既存の prefix / override 解決に従って
`ImageClsAgent.learning_rate.type` と override prefix 側の `learning_rate.type` を読む。

`ConfigReader<ProfiledValueConfig<T>>` の読み取りイメージ:

```cpp
template <typename T>
struct ConfigReader<ProfiledValueConfig<T>> {
    static void Read(
        Config& owner,
        const ConfigData& config_data,
        const std::string& key,
        ProfiledValueConfig<T>& value)
    {
        owner.ReadSubConfig(config_data, key, "type", value.type);
        owner.ReadSubConfig(config_data, key, "value", value.value);
        owner.ReadSubConfig(config_data, key, "start", value.start);
        owner.ReadSubConfig(config_data, key, "end", value.end);
        owner.ReadSubConfig(config_data, key, "steps", value.steps);
        owner.ReadSubConfig(config_data, key, "cycle_mult", value.cycle_mult);
        owner.ReadSubConfig(config_data, key, "phases", value.phases);

        value.phase.Clear();
        for (const auto& phase_name : value.phases) {
            ProfiledValuePhaseConfig<T> phase;
            const auto phase_key = owner.MakeTaggedSubConfigKey(key, "phase", phase_name);

            owner.ReadSubConfig(config_data, phase_key, "type", phase.type);
            owner.ReadSubConfig(config_data, phase_key, "value", phase.value);
            owner.ReadSubConfig(config_data, phase_key, "start", phase.start);
            owner.ReadSubConfig(config_data, phase_key, "end", phase.end);
            owner.ReadSubConfig(config_data, phase_key, "steps", phase.steps);
            owner.ReadSubConfig(config_data, phase_key, "cycle_mult", phase.cycle_mult);

            value.phase.Set(phase_name, phase);
        }
    }
};
```

`ConfigReader` が `Config` の protected helper を呼べるように、`Config` に以下を追加する。

```cpp
template <typename T>
friend struct ConfigReader;
```

## ImageCls への適用

### Config

`ImageClsAgentConfig::learning_rate` は `double` から `ProfiledValueConfig<double>` に置き換える。

```cpp
ProfiledValueConfig<double> learning_rate;
```

`ImageClsAgentConfig` の constructor では次を維持する。

```cpp
ANET_READ_CONFIG(config_data, learning_rate);
```

`ImageCls.txt` の baseline は構造化 key へ移行する。

```txt
ImageClsAgent.learning_rate.type = phased
ImageClsAgent.learning_rate.phases = warmup main

ImageClsAgent.learning_rate.phase.[warmup].type = linear
ImageClsAgent.learning_rate.phase.[warmup].start = 0
ImageClsAgent.learning_rate.phase.[warmup].end = 1e-3
ImageClsAgent.learning_rate.phase.[warmup].steps = 500000

ImageClsAgent.learning_rate.phase.[main].type = constant
ImageClsAgent.learning_rate.phase.[main].value = 1e-3
ImageClsAgent.learning_rate.phase.[main].start = 1e-3
ImageClsAgent.learning_rate.phase.[main].end = 1e-5
ImageClsAgent.learning_rate.phase.[main].steps = 9500000
```

main のみで動かす場合:

```txt
A.learning_rate.phases = main
```

warmup から cosine decay へ切り替える場合:

```txt
A.learning_rate.phase.[main].type = cosine
```

旧形式はサポートしないため、以下は使わない。

```txt
ImageClsAgent.learning_rate = 1e-3
```

### Learner

`ImageClsLearner` は `ProfiledValue<double> learning_rate_` を保持する。

- constructor で `config_.learning_rate` から `learning_rate_` を構築する。
- AdamW の初期 LR には `learning_rate_.Value()` または `learning_rate_.Evaluate(0)` を使う。
- `UpdateFromBatch()` 内で、optimizer step 前に `learning_rate_.Update(step.exp_step)` を呼ぶ。
- 更新後の `learning_rate_.Value()` を全 param group へ反映してから `optimizer_->step()` を呼ぶ。

`ProfiledValue<T>` は step axis を知らない。ImageCls では呼び側判断として `step.exp_step` を渡す。

### Agent metric

learning rate metric は `ImageClsUpdateResult` ではなく Agent 自体から公開する。

- `ImageClsAgent::GetScalar("learning_rate")` が現在の learning rate を返す。
- `ImageClsLearner` と `ImageClsAgent` が同じ current value を参照できるように、Agent 所有の scalar state または共有 `ProfiledValue<double>` を使う。
- 排他は既存の `ImageClsAgent` / `ImageClsLearner` の mutex 境界で確保する。

metrics config は `$agent` を使う。

```txt
metrics.scalar.[39_agent/09_learning_rate] = $agent learning_rate @learn interval:100
```

## `net.config_profile` との関係

初回実装では既存 `net.config_profile` の挙動を壊さない。

ただし `net.config_profile` は同じ「profile 上の位置に応じて値を変える」概念なので、後続で
`ConfigProfileConfig` を `ProfiledValueConfig<double>` へ寄せる。移行時は、既存の
`net.config_profile.[name].type = linear`, `start`, `end` の設定 surface を維持し、
`EvaluateByIndex(index, count)` を使って現在と同じ結果を出す。初回移行対象は既存 surface に合わせて `linear` を想定し、`constant` / `cosine` も type-aware に評価する。

## 受け入れ基準

1. `ProfiledValue<T>` が `constant` / `linear` / `cosine` / `phased` / `cosine_restart` を評価できる。
2. `ProfiledValue<T>::Value()` は `Update()` 済み値を再計算なしで返す。
3. `ANET_READ_CONFIG(config_data, learning_rate)` で `learning_rate.*` と `learning_rate.phase.[name].*` が読まれる。
4. `Config::ToConfigString()` と `Config::ToJson()` に `learning_rate.type`, `learning_rate.phase.[warmup].start` などが展開される。
5. dormant field を許容し、baseline に `constant` 用 `value` と cosine 用 `start/end/steps` を同居できる。
6. override で `type` だけを `cosine` に変えた場合、baseline 由来の `start/end/steps` が使われる。
7. 未知 `type`、active な `linear` / `cosine` の `steps == 0`、`phased` かつ `phases` 空、`EvaluateByIndex` に渡された `cosine_restart` / `phased` は `ANET_SYSTEM_ERROR` で失敗する。
8. ImageCls の optimizer param group LR が `optimizer_->step()` 前に更新される。
9. `$agent learning_rate @learn` metric で現在 learning rate が記録される。
10. 旧 scalar `ImageClsAgent.learning_rate = 1e-3` に依存した config は更新され、fallback なしで扱われる。

## テスト項目

1. `ProfiledValue<double>` constant: `value` を返す。
2. linear: `step=0` で `start`、`step=steps` で `end`、中間で線形補間。
3. cosine: 端点と中点を検証する。
4. cosine overrun: `step > steps` で `end` を保持する。
5. cosine_restart: cycle restart と `cycle_mult` による cycle 長変更を検証する。`0 < cycle_mult < 1` の縮小 cycle と 1 step 下限も検証する。
6. phased: `warmup -> main` の phase 切り替えと最終値保持を検証する。
7. `EvaluateByIndex(index, count)` が `constant` / `linear` / `cosine` を type-aware に評価し、`cosine_restart` / `phased` を拒否する。
8. `ConfigReader<ProfiledValueConfig<double>>` が root field と phase field を読む。
9. config merge: baseline の dormant field を override の `type` 切り替えで利用できる。
10. ImageCls: `ImageClsAgent.learning_rate.*` の構造化 config が読まれ、`$agent learning_rate @learn` で記録できる。

## 実装対象

- `core/anet-core/include/anet/config.hpp`
  - `ConfigReader<T>` customization point
  - `ReadSubConfig`
  - `MakeTaggedSubConfigKey`
- `core/anet-core/include/anet/schedule.hpp`
  - `ProfiledValuePhaseConfig<T>`
  - `ProfiledValueConfig<T>`
  - `ProfiledValue<T>`
  - `ConfigReader<ProfiledValueConfig<T>>`
- `core/anet-core/src/schedule.cpp`
  - schedule カテゴリの translation unit
- `core/anet-core/include/anet/image_cls_agent.hpp`
  - `ImageClsAgentConfig::learning_rate`
  - Agent metric surface
- `core/anet-core/src/image_cls_agent.cpp`
  - Learner での learning rate update と optimizer 反映
  - Agent metric state
- `core/anet-core/src/schedule_test.cpp`
  - `ProfiledValue` と config reader のテスト
- `apps/runner/config/ImageCls.txt`
  - `ImageClsAgent.learning_rate.*`
  - `$agent learning_rate @learn` metric

## 検証

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test'
core\anet-core\bin\Debug\anet-core-test.exe "[profiled_value],[config],[image_cls]"
core\anet-core\bin\Debug\anet-core-test.exe
git diff --check
```

`anet-core-test.exe` が 120 秒前後で timeout する場合は、stdout/stderr をリダイレクトして長めの timeout で再実行する。
