# LunarLanderEnv obs_include_action（直前 action の観測化・暫定）

> 設計分担: Claude/Codex=設計/PRD、実装=Codex、Run/commit=ユーザー。
> 本書は self-contained。実装時は行番号ではなく、近傍のシンボル名で再検索する。

## Context（背景・目的）

LunarLanderEnv は風（wind/turbulence）が観測に含まれない POMDP であり、エージェントは
frame stack（stack_count=4）の速度履歴の差分から風を推定する必要がある。しかし速度差分には
自機エンジンの推力が混入するため、観測だけからの風推定は「まず行動を物理署名
（Δv_angle のトルク跳ね、Δvy の main エンジン跳ね）から間接デコードし、残差を風に帰属する」
という 2 段階の暗黙学習を要求する。

「自分の方策だから行動は観測から復元できる」という仮定は本構成では成立しない。

1. 学習時の行動は確率的（UQE の eps、spatial exploration）で、観測 stack から行動は決まらない。
2. Replay buffer のデータは過去の方策が生成したもので、現在のネットには復元不能。
3. a_{t-1} は stack_{t-1}（o_{t-4}..o_{t-1}）から選ばれており、o_{t-4} は現在の
   stack（o_{t-3}..o_t）の外にある。窓の切り詰めにより原理的にも復元できない。

この間接デコードの獲得遅れが、seed によってスパイク（報酬急落）の収束が遅れる現象の
仮説の一つである。本 PRD では、直前に実行した action の one-hot を観測ベクトルに追加する
暫定フラグ `obs_include_action` を LunarLanderEnv に追加し、外乱推定を well-posed にする。
R2D2/Agent57（prev action を入力に含める）、RMA（state-action 履歴から環境パラメータを推定）
と同型の標準手段である。

エージェント側の入力パスに action-conditioning を持たせる恒久設計は「State・TD 統一」基盤に
絡む大工事のため、まず Env 側の最小実装で仮説検証する（それゆえ「暫定」）。

## 確定した設計判断

1. config は `LunarLanderEnv.obs_include_action`（bool、default **false**）。false 時は
   従来と完全に同一挙動（obs 次元・値・乱数消費列すべて不変）とする。
2. one-hot(4) は既存 8 次元の **後ろに追加**して 12 次元にする。既存 index（0..7）は不変。
   `computeShaping()` は obs[0..7] を index 直読みしているため、順序保持は必須要件。
3. 意味論は「obs_t には、この obs に至らせた行動 a_{t-1} を入れる」。
   - `Reset()` が返す obs: one-hot 全ゼロ（未行動）。
   - `Step(a)` が返す next_state の obs: a の one-hot。
4. one-hot の並びは既存 action 定数の順（`kActionNoop=0, kActionLeft=1, kActionMain=2,
   kActionRight=3`）。力ベクトルへの変換はしない（action→力は定数マップなので one-hot と等価）。
5. reward は観測に含めない（shaping 由来の情報は obs から導出可能、終端報酬のリーク管理が
   不要なだけ複雑になる）。
6. 新しい obs キーは作らず、既存 `ObsKeys::kVector` に連結する。これにより stacker
   （キー単位で stack）と NN config（`Flatten > MLP_FC1 > ...`、Linear の in_features は
   入力から自動推論）は無変更で通る。
7. `GetSpec()` の obs_spec は flag に応じて shape/labels/min/max を拡張する。
   labels は既存の短縮流儀に合わせ `a_noop, a_left, a_main, a_right`。
8. flag は base scope（`LunarLanderEnv.obs_include_action`）に置いて運用する。train env と
   eval env（test1/test2 含む）で食い違うと obs 次元不一致で破綻するため、variant 個別の
   override はしない運用とする（仕組みとして禁止はしない）。
9. Env コンストラクタの `MetricsLogger::Instance()->Log(...)` は null ガード付きに変更する。
   `Instance()` は `Init()` 前は null を返すため、現状のままでは単体テストで env を直接
   構築できない。026（MetricsLogger null-safe 化）の先行局所対応であり、026 と衝突しない。
10. obs_norm の特別扱いはしない。LunarLander 運用は `obs_norm.pass_through = true`。
    pass_through=false 構成でも one-hot は 0/1 で SymLog はほぼ恒等〜定数スケールであり許容。

## 仕様

### Config

```txt
LunarLanderEnv.obs_include_action = false   # default。直前actionのone-hot(4)をobsに追加
```

- `LunarLanderEnvConfig` に `bool obs_include_action = false;` を追加し、
  `ANET_READ_CONFIG(config_data, obs_include_action);` で読む（config dump に自動で載る）。
- `apps/runner/config/LunarLander.txt` にはコメントアウトで実験用の行を追加しておく:

```txt
#LunarLanderEnv.obs_include_action = true   # 風POMDP対策: 直前actionをobsに含める(032)
```

### Obs レイアウト（flag ON 時）

| index | 内容 | labels | min | max |
|---|---|---|---|---|
| 0..7 | 従来どおり | x, y, vx, vy, angle, v_angle, leg_l, leg_r | 従来どおり | 従来どおり |
| 8..11 | a_{t-1} の one-hot | a_noop, a_left, a_main, a_right | 0 | 1 |

### 状態遷移

- コンストラクタ / `Reset()`: `last_action_ = -1`（one-hot 全ゼロに対応）。
- `Step(action)`: 先頭で `last_action_ = action;` を設定してから物理 step と
  `makeState()` を行う。
- `makeState()` の `!lander_body_` 分岐（dead state）も同じ次元
  （`torch::zeros({obs_dim})`、obs_dim = flag ? 12 : 8）を返す。

### 実装スケッチ

`LunarLanderEnv.hpp`:

```cpp
constexpr int kActionCount = 4;   // kActionNoop..kActionRight の総数（既存定数の近傍に追加）

struct LunarLanderEnvConfig : public anet::Config {
    // ...
    bool obs_include_action = false;
    // ctor 内: ANET_READ_CONFIG(config_data, obs_include_action);
};

class LunarLanderEnv /* ... */ {
    // ...
    int64_t last_action_ = -1;   ///< 直前に実行した action。-1 は未行動（Reset直後）
};
```

`LunarLanderEnv.cpp` の `makeState()`（末尾の追加のみ。既存 8 要素の順序・値は不変）:

```cpp
std::vector<float> v = { x, y, vx, vy, angle, angular_vel, left_contact, right_contact };
if (config_.obs_include_action) {
    for (int64_t a = 0; a < kActionCount; ++a) {
        v.push_back(last_action_ == a ? 1.0f : 0.0f);
    }
}
// torch::tensor(v) で従来と同 dtype/device の obs を生成
```

`GetSpec()`（flag ON 時に拡張）:

```cpp
if (config_.obs_include_action) {
    obs_spec.shape = { 8 + kActionCount };
    // labels   += { "a_noop", "a_left", "a_main", "a_right" }
    // min_values += { 0, 0, 0, 0 }
    // max_values += { 1, 1, 1, 1 }
}
```

コンストラクタの MetricsLogger null ガード:

```cpp
if (auto logger = anet::MetricsLogger::Instance()) {
    logger->Log("LunarLanderEnv", config_.ToJson());
}
```

### テスト基盤（新規）

lunarlander1 には単体テストが無いため、imagecls1 の流儀を移植する。

`core/envs/lunarlander1/CMakeLists.txt`:

```cmake
file(GLOB_RECURSE LUNARLANDER_TEST_SOURCES CONFIGURE_DEPENDS "src/*test.cpp")
list(FILTER ANET_PRIVATE_SOURCES EXCLUDE REGEX ".*test\\.cpp$")

add_executable(LunarLanderEnv-test)
target_sources(LunarLanderEnv-test PRIVATE ${LUNARLANDER_TEST_SOURCES})
target_include_directories(LunarLanderEnv-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(LunarLanderEnv-test PRIVATE LunarLanderEnv anet-core catch2)
add_test(NAME LunarLanderEnv-test COMMAND LunarLanderEnv-test)
```

リンクは imagecls1 の `ImageClsEnv-test` と同構成（static lib の PRIVATE 依存は
`$<LINK_ONLY:...>` で推移する）。box2d / anet-wx でリンクエラーになる場合のみ明示追加する。

## 非対象（Non-goals）

- エージェント側入力パスの action-conditioning（State・TD 統一基盤に絡む恒久設計）。
- reward の観測化。
- Conv1D 時間軸構成の変更（別実験。LunarLander.txt にコメント済み構造があり、本 flag と
  合流検証する場合も config のみで可能）。
- `computeShaping()` の `.item<float>()` 連打の最適化（別件の性能改修）。
- 他 Env への同機構の横展開（効果確認後に汎化を検討）。

## 受け入れ基準

1. `obs_include_action = false`（default）で従来と同一挙動: obs 次元 8、spec 不変、
   乱数消費列不変（one-hot 生成は乱数を使わないため、ON でも乱数列は不変）。
2. `true` で `GetSpec()` の obs_spec が shape {12}・labels 12 個（末尾 a_noop, a_left,
   a_main, a_right）・min/max 拡張済みになる。
3. `true` で `Reset()` の obs は index 8..11 が全ゼロ、`Step(a)` の next_state obs は
   a に対応する index のみ 1.0、他 0.0。
4. `true` でも obs の先頭 8 次元は false 時と同値（同 seed・同 action 列で一致）。
5. train / eval1 / eval2 / test1 / test2 のいずれの経路でも spec と実 obs の次元が一致し、
   `ValidateObservation` を通過する（flag は base scope で全 variant に効く）。
6. NN config 無変更で runner が起動し学習が回る（Linear in_features 自動推論で 32→48）。
7. run の config dump（`runs/<name>/config/config_data.txt` および `json/LunarLanderEnv.json`）
   に `obs_include_action` が出力される。
8. MetricsLogger 未 Init の状態で `LunarLanderEnv` を構築してもクラッシュしない。
9. 新設の `LunarLanderEnv-test` を含む全テストが green。

## テスト項目（LunarLanderEnv_test.cpp）

1. default（false）: spec shape {8}、Reset/Step の obs numel 8（regression）。
2. ON: spec shape {12}、labels/min/max のサイズと末尾 4 要素を検証。
3. ON: Reset 直後 obs[8..11] == {0,0,0,0}。
4. ON: 4 action すべてについて `Step(a)` → one-hot ブロックは該当 index のみ 1.0。
5. ON/OFF 同 seed 並走: 同一 action 列を Step し、先頭 8 次元が全 step で一致する。
6. 終端まわり: done（crash）または truncated（limit_step 小設定）到達 step でも obs 次元が
   12 のまま（dead-state 分岐含む）。
7. MetricsLogger 未 Init で構築してもクラッシュしない（null ガードの検証）。

## 実装対象

- `core/envs/lunarlander1/src/LunarLanderEnv.hpp`
  - `kActionCount`、`LunarLanderEnvConfig::obs_include_action`、`last_action_` メンバ
- `core/envs/lunarlander1/src/LunarLanderEnv.cpp`
  - `ANET_READ_CONFIG`、`GetSpec()` 拡張、`makeState()` 追記、`Reset()`/`Step()` の
    `last_action_` 更新、コンストラクタの MetricsLogger null ガード
- `core/envs/lunarlander1/src/LunarLanderEnv_test.cpp`（新規）
- `core/envs/lunarlander1/CMakeLists.txt`
  - テストターゲット新設（imagecls1 準拠）
- `apps/runner/config/LunarLander.txt`
  - 実験用のコメントアウト行を追加

## 検証

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target LunarLanderEnv-test'
core\envs\lunarlander1\bin\Debug\LunarLanderEnv-test.exe
git diff --check
```

runner smoke（ユーザー実施）: `obs_include_action = true` で起動し、
nn_viz の入力次元（Flatten 後 48）、QValuePanel 等のラベルに `a_*` が出ること、
config dump に flag が載ることを確認。

## 実験計画（暫定運用）

- ベースライン: rbfix 反映後の現行設定（3 seed）。
- 比較: `obs_include_action = true` のみ変更（3 seed）。判定は同等設定・複数 seed の
  終盤 eval EMA の平均ブレ幅基準（ブレ内は有意差なし）。
- 注視点: スパイクの頻度・持続（seed 間ブレ）、特に test2（wind_power=15。風加速が
  side エンジン加速を上回り、行動の誤帰属コストが最大の条件）。
- 当たった場合: Conv1D 時間軸構成（LunarLander.txt コメント済み）との合流版を 1 本検証。
- 外れた場合: flag は default false のまま残置コスト極小。恒久化判断は効果確認後。
