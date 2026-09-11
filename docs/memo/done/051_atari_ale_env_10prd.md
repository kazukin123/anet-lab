# PRD 051: ALE (Atari) Env 統合 — C++ 直結・Single seam・自前前処理

- 起票日: 2026-08-13
- 状態: 実装完了・長時間学習検証待ち
- 対象: 新規モジュール `core/envs/atari1`、ルート/runner CMake、`apps/runner/config/Atari.txt`
- 関連: ADR 0025・ADR 0026（本PRDで新設）、ADR 0009（Env seam。single XOR batch variant registry）、PRD 050 / ADR 0024（RB stack margin。stack 済み経路の前提）
- 設計文書: `docs/design/220_atari_env.jp.md`（本PRDの仕様の恒久版。ALE 設定キー契約・View 詳細はこちらが正）、`docs/design/120_environments.jp.md`（§8 Env 追加チェックリスト準拠）
- 参考: `reports/atari_env_survey_2026-08-13.md`（ALE サーベイ。ライブラリ選定・プロトコル・ベンチマーク動向の根拠）

## Context（背景・目的）

ENV バリエーション拡充の初手として Atari 2600（ALE: Arcade Learning Environment）を導入する。ALE は DQN 以来の事例が最も厚い標準ベンチマークであり、既知スコアとの突き合わせで anet-lab のアルゴリズム実装を外部基準で検証できるようになる。

方式は **ALE v0.12.0 の C++ API 直結**（Python/GymEnv 連携は将来検討）。ALE のビルド（Windows/MSVC、Debug/Release 両建て）と C++ API 疎通は検証済みで、確定ビルド構成は §3 に記す。ALE は GPL-2.0 のため、リポジトリ非同梱・オプショナルビルドとし、本体の Apache-2.0 を維持する（ADR 0025）。

環境実装は SingleDiscreteEnv として登録し、並列化・auto-reset・frame stack はすべて既存機構（ThreadPoolDiscreteEnv / stucker / ReplayBuffer stack_count）に乗る。env 自身の責務は「1 ALE インスタンスの駆動と標準前処理（skip / max-pool / resize）で単フレーム uint8 観測を出す」ことに限定する（ADR 0026）。

## 0. 決定一覧（グリル確定値）

| ID | 決定 |
|---|---|
| D1 | PH1 スコープ = AtariEnv 実装 + config でのゲーム切替 + 複数ゲームでの学習成立確認まで |
| D2 | 対象ゲーム = Pong と Breakout |
| D3 | 受入基準は方向性基準: Pong eval 平均 ≥ +10 / Breakout eval 平均 ≥ 30 を目安に単調改善を確認。予算はゲームあたり〜10M frames（= 2.5M agent steps）。数値は目安であり厳密な閾ではない |
| D4 | 行動空間は `getMinimalActionSet()` 既定。`full_action_space=true` で全 18 行動へ切替可。ALE に per-step 合法手マスクは存在しないため action_mask キーは出さない |
| D5 | プロトコル既定は v5/Machado 準拠（sticky 0.25 / NoOp reset なし / episodic life OFF / fire reset OFF / reward clip ON）。クラシック構成は `AtariEnv.classic` プロファイルとして同梱し `.$` AutoMerge で切替 |
| D6 | seam は SingleDiscreteEnvFactory。1 instance = 1 `ale::ALEInterface`。並列は既存 `env.worker_type/worker_threads`、stack は既存 stucker/RB。ALE 内蔵 AtariVectorEnv は不採用（理由・将来拡張は §10・ADR 0026） |
| D7 | `Step()` の reward は `reward_clip=true` のとき sign clip 済み（学習報酬）。生スコアは env 内部で累積し `GetScalar("episode_score")` で報告（未確定 step は NaN 慣行） |
| D8 | config は `Atari.txt` 1 本 + `AtariEnv.game` キー。ゲーム切替は workspace 側 override 1 行。ハイパラは全ゲーム共通（Atari の作法） |
| D9 | ROM 解決は環境変数 `ATARI_ROM_DIR` 基本、`AtariEnv.rom_dir`（既定空）が非空なら優先。パスは `<rom_dir>/<game>.bin`。いずれも無効なら構築時 fail-fast |
| D10 | ビルド gating は三値 cache 変数 `ANET_ENABLE_ATARI` = `AUTO`（既定）/`ON`/`OFF`。AUTO は `ALE_ROOT` の存在・実在で自動判定。ON で ALE_ROOT 不備は configure エラー |
| D11 | 観測は grid キーのみ `{1, screen_size, screen_size}` kUInt8（連続、NN 境界で自動 /255）。`screen_size` 既定 84。vector キーは出さない |
| D12 | 成果物 = 本PRD + ADR 0025/0026 + `CONTEXT.md` 用語 4 件（sticky actions / flavor / 生スコア / プロトコルプリセット） |

## 1. スコープ

**含む**: AtariEnv（SingleDiscreteEnv）/ AtariEnvFactory / AtariView（grid プレビュー最小）/ 前処理（skip・max-pool・resize）/ config `Atari.txt` + v5・classic プリセット / 三値 gating ビルド統合 / NOTICE / 単体テスト / Pong・Breakout の学習成立確認手順。

**含まない**（§10 将来拡張へ）: AtariVectorEnv（batch-native）接続、RGB・RAM 観測、flavor（mode×difficulty）を使った実験、Atari 100k・Atari-5 プロトコルの整備、マルチエージェント（2P）、Python/GymEnv 連携。

## 2. 用語

`CONTEXT.md`「Atari/ALE」節を正とする: **sticky actions** / **flavor** / **生スコア** / **プロトコルプリセット**。本文では定義を再掲しない。

## 3. 外部依存（ALE）とライセンス

### 3.1 ALE ビルド確定構成（検証済み・2026-08-13）

| 項目 | 値 |
|---|---|
| バージョン | tag `v0.12.0` 固定（`git clone --depth 1 --branch v0.12.0`）。vcpkg 依存も manifest の builtin-baseline で固定される |
| 配置 | リポジトリ外の任意パス（例 `C:/dev/ALE`）。**環境変数 `ALE_ROOT`** で参照 |
| configure | `cmake -S %ALE_ROOT% -B %ALE_ROOT%/build -DBUILD_PYTHON_LIB=OFF -DSDL_SUPPORT=ON -DSDL_DYNLOAD=ON -DVCPKG_TARGET_TRIPLET=x64-windows-static-md` |
| build | `cmake --build %ALE_ROOT%/build --config Release` と `--config Debug` の両方 |
| リンク | `%ALE_ROOT%/build/src/ale/<Config>/ale.lib`（全部入り静的 ~276MB）+ zlib のみ |
| zlib | `%ALE_ROOT%/build/vcpkg_installed/x64-windows-static-md` を `CMAKE_PREFIX_PATH` に足して `find_package(ZLIB)`（Debug は zlibd を config が自動選択） |
| include | `%ALE_ROOT%/src` + `%ALE_ROOT%/src/ale` + `%ALE_ROOT%/build/src/ale`（生成 `version.hpp`） |
| SDL | `SDL_DYNLOAD=ON` が **Windows では必須**（ALE の `SDL2.hpp` が `WIN32 && !SDL_DYNLOAD` で全 SDL 関数を dllimport 宣言するため、静的 SDL ではリンク不能）。DYNLOAD によりリンク時 SDL 依存は消え、`display_screen=true` 時のみ `SDL2.dll` を実行時ロード（DLL は ale-py wheel 同梱物などから調達） |
| install | 使わない。upstream の install ルールは `if(UNIX ...)` ガードで Windows 未対応のため、ビルド木直接参照が標準解 |

### 3.2 ROM

- ROM（.bin）はリポジトリ非同梱・自己調達。ファイル名は ale-py 同梱 ROM の snake_case（`pong.bin` / `breakout.bin`）に揃える。
- 解決順: `AtariEnv.rom_dir`（非空なら優先）→ 環境変数 `ATARI_ROM_DIR`。得られたディレクトリに `<game>.bin` が実在しなければ **Env 構築時に fail-fast**（エラーメッセージに探索した実パスと設定手段を含める）。
- ALE の `loadROM()` は ROM の md5 で対応ゲームを判定するため、壊れた ROM は ALE 側でも検出される。

### 3.3 ライセンス（方針A、ADR 0025）

ALE は GPL-2.0。義務は配布時のみ発火するため、①リポジトリ非同梱（ALE_ROOT 外部参照）②Atari env はオプショナルビルド ③リリースパッケージ（PRD 043 系）には Atari を含めない（リリースビルドは `ANET_ENABLE_ATARI=OFF`）④`core/envs/atari1/NOTICE.md` に「ALE(GPL-2.0) とリンクした成果物の配布は GPL-2.0 に従う」旨を記載、で本体 Apache-2.0 を維持する。

## 4. アーキテクチャ（ADR 0026）

```
BatchEnvBuilder (env.class_id = AtariEnv)
  └─ SingleDiscreteEnvFactory: AtariEnvFactory
       └─ AtariEnv × N lanes（ThreadPoolDiscreteEnv が並列実行・auto-reset）
            ├─ ale::ALEInterface（1 対 1。frame_skip=1 固定、sticky は ALE 側）
            └─ 前処理: 自前 skip ループ → grayscale 2 フレーム max-pool → area resize → uint8 [1,S,S]
frame stack: 出さない（Actor=DictFrameStacker / RB=stack_count の既存両輪。DropMerge 実運用実績）
```

- RB には単フレーム uint8 が保存され、stack はサンプル時に展開される（ストレージ増なし）。wrap 境界は PRD 050 の history margin が守る。
- AtariVectorEnv 不採用の理由: stack 済み出力 `[N,4,84,84]` を RB に入れると同一フレームが 4 遷移に重複保存され RB メモリ 4 倍 / 独自 autoreset と `continue_state` 契約の突き合わせが必要 / opencv4 依存追加 / stucker との二重化。詳細は ADR 0026。

## 5. Env 仕様

### 5.1 モジュール構成（既存標準に準拠）

```
core/envs/atari1/include/anet/env/Atari.hpp   … void InitAtari(); 宣言のみ
core/envs/atari1/src/Atari.cpp                … InitAtari()（RegistEnvFactory + RegisterViewCreator）
core/envs/atari1/src/AtariEnv.hpp/.cpp        … AtariEnvConfig / AtariEnv / AtariEnvFactory
core/envs/atari1/src/AtariPreprocess.hpp/.cpp … 前処理 free 関数（named namespace、ALE 非依存でテスト可能に）
core/envs/atari1/src/AtariView.hpp/.cpp       … grid プレビュー（ImageClsView 前例準拠、class_id 一致）
core/envs/atari1/src/AtariEnv_test.cpp        … Catch2
core/envs/atari1/src/pch.hpp                  … 共通 3 行
core/envs/atari1/NOTICE.md                    … GPL 注意書き
```

- class_id は `AtariEnv`。`AtariEnv : public SingleDiscreteEnvBase, public anet::RandomHolder`。
- Factory は GridMazeEnvFactory と同型の 3 行実装（config 構築 → `make_shared<AtariEnv>`）。

### 5.2 config キー（prefix `AtariEnv`、`anet::Config` 既定値つき）

| キー | 型 | 既定 | 意味 |
|---|---|---|---|
| `game` | string | （必須） | ROM 名（snake_case、拡張子なし）。`pong` / `breakout` |
| `rom_dir` | string | `""` | 非空なら `ATARI_ROM_DIR` より優先 |
| `screen_size` | int | 84 | 出力解像度 S（正方形） |
| `frame_skip` | int | 4 | 1 step あたりのエミュレータフレーム数（自前 skip ループ） |
| `max_pool` | bool | true | skip 窓の最後 2 フレームの pixel-wise max（フリッカー対策） |
| `repeat_action_probability` | float | 0.25 | sticky actions。ALE へそのまま設定 |
| `noop_max` | int | 0 | Reset 時に 1..noop_max 回の NOOP を乱数挿入（0=無効） |
| `fire_reset` | bool | false | Reset 直後に FIRE を 1 回実行（FIRE を持つゲームのみ。無ければ no-op） |
| `episodic_life` | bool | false | ライフ減少を done として学習系へ見せる（§5.6） |
| `reward_clip` | bool | true | Step reward の sign clip |
| `full_action_space` | bool | false | true で legal 18 行動 |
| `mode` | int | -1 | -1=ALE 既定。それ以外は `setMode()` へ |
| `difficulty` | int | -1 | -1=ALE 既定。それ以外は `setDifficulty()` へ |
| `max_episode_frames` | int | 108000 | エミュレータフレーム数上限で truncation（v5 既定値。0=無効） |
| `retain_rgb_frame` | bool | true | Step 毎に RGB 画面を保持し `GetTensor("rgb_frame")` で公開（AtariView 用） |
| `display_screen` | bool | false | SDL 観戦ウィンドウ（SDL2.dll 実行時ロード） |
| `sound` | bool | false | SDL 音声（display と独立） |

プリセット（`Atari.txt` 内に定義、`.$` AutoMerge で選択）:

```ini
# v5/Machado 準拠（既定。上表の既定値そのもの＝差分なしの空プロファイルでも成立するが明示する）
AtariEnv.v5.repeat_action_probability = 0.25
AtariEnv.v5.noop_max = 0
AtariEnv.v5.episodic_life = false
AtariEnv.v5.fire_reset = false

# クラシック（DeepMind 2015 系。CleanRL/SB3 の数字と比較する時用）
AtariEnv.classic.repeat_action_probability = 0.0
AtariEnv.classic.noop_max = 30
AtariEnv.classic.episodic_life = true
AtariEnv.classic.fire_reset = true

AtariEnv.$ = AtariEnv.v5 > E
```

config 契約ノート（詳細は `docs/design/220_atari_env.jp.md` §4.2）:

- `screen_size` の正方形単一値は標準前処理（84×84、アスペクト比破壊が慣行）への準拠。長方形が必要になったら分割は env 内互換変更で行う
- `display_screen` は sandbox・単発デバッグ用。SDL ウィンドウは ALE インスタンス毎に生成されるため N lane では N ウィンドウが開き、ThreadPool の worker スレッドからの SDL 呼び出しは未サポート動作。Runner での観戦は AtariView が正。env は有効時に log.warn を 1 回出す

### 5.3 EnvSpec

- `state_spec.obs_spec` = `{ grid: TensorSpec{ Grid, {1, S, S}, kUInt8, num_classes=0, min=0, max=255 } }`。vector / action_mask キーは出さない（D4/D11）。
- `action_spec`: `is_discrete=true`。`value_labels` は選択した行動集合の ALE 行動名（`ale::action_to_string` 由来、例 NOOP/FIRE/RIGHT/LEFT）。minimal set は ROM ロード後に `getMinimalActionSet()` で取得し、`Step(action)` の `action` は **この集合の index**（ALE の Action enum 値ではない）。
- `reward_range`: `reward_clip=true` なら `{-1, 1}`、false ならゲーム依存のため制限なし（`{-FLT_MAX, FLT_MAX}`）。

### 5.4 ALE インスタンス設定（構築時）

| ALE 設定 | 値 | 理由 |
|---|---|---|
| `random_seed` | lane seed から変換（§5.7） | |
| `repeat_action_probability` | config 値 | sticky は ALE 内部 RNG・フレーム単位判定に委譲（自前 skip でも act() 単位=フレーム単位なので原義と同一の意味論） |
| `frame_skip` | **1 固定** | skip ループは env 自前（max-pool のため中間フレームが必要） |
| `max_num_frames_per_episode` | **0（無効）** | truncation は env 自前カウントで判定し done と厳密に区別する（ALE 側に任せると game_over に truncation が混ざる） |
| `display_screen` / `sound` | config 値 | |

### 5.5 Step の意味論

```
Step(action_index):
  a = action_set[action_index]
  reward_raw = 0
  for i in 0 .. frame_skip-1:
      reward_raw += ale.act(a)                       # sticky は ALE 内で毎フレーム判定
      if max_pool:                                   # 当該Step内のrolling 2-slot
          getScreenGrayscale(latest)
      if ale.game_over(false): break                 # real game overだけ早期終了
  obs_frame = max_pool ? max(最後の最大2実行フレーム) : getScreenGrayscale(最終)
  grid = Resize(obs_frame)                            # §5.8。uint8 [1,S,S]
  episode_score += reward_raw                         # 生スコア（clip 前）
  done = ale.game_over(false)
  if episodic_life and !done and lives < prev_lives:  # ライフ減少
      done = true; life_loss_pending = true
  truncated = (!done and max_episode_frames > 0
               and ale.getEpisodeFrameNumber() >= max_episode_frames)
  reward = reward_clip ? sign(reward_raw) : reward_raw
```

- `frame_skip=1` のとき max_pool は単フレーム（比較対象が無いのでそのまま）。
- max-pool は前 Step のフレームを混ぜない。real game over で1フレームしか実行できなければ、その1枚をそのまま使う。
- `episodic_life` の lives 判定は skip 窓完走後に1回だけ行い、life lossで残りの反復を中断しない。窓内の全報酬は同じ Step に帰属する。
- truncation と `episode_frames` の正本は `ale.getEpisodeFrameNumber()` とし、自前 frame counter は持たない。hard Reset の NOOP/FIRE と soft-reset NOOP も ALE frame に含まれる。
- done と truncated は独立 bool（既存契約）。両立はしない（done 優先）。
- Step 内で ALE 例外（ROM 異常等）は握りつぶさず伝播（fail-fast）。

### 5.6 Reset の意味論（episodic_life の soft-reset 契約）

auto-reset wrapper は `done || truncated` の lane に `Reset()` を呼ぶ（既存契約）。episodic_life ON のライフ減少 done では **ALE をリセットしてはならない**（ゲーム続行が原義）ため:

```
Reset():
  if life_loss_pending and !ale.game_over():
      life_loss_pending = false
      ale.act(NOOP)                                   # 1 フレーム進めて観測を作る（クラシック慣行）
      （episode_score / ALE episode frame は継続。真の episode 境界ではない）
  else:                                               # 実 game over / truncation / 初回
      ale.reset_game()
      if noop_max > 0: NOOP を RandomHolder RNG で 1..noop_max 回
      if fire_reset and FIRE ∈ action_set: ale.act(FIRE)
      episode_score = 0; life_loss_pending = false
  obs を構築して返す（episode_start=true は基底/wrapper 既存機構に従う）
```

- 救済パス（実装で確定、レビュー承認済み）: ①soft-reset 経路に入った時点で既に実 game over に到達していた場合は hard reset へフォールバックする。②hard reset 中の NOOP / FIRE 実行で game over が発生した場合は reset をやり直す。いずれも ALE が terminal 後の `act()` を拒否する仕様に対する必須の分岐。
- 注意: episodic_life ON では「学習系に見える episode」（life 単位）と「metrics の episode_score が確定する区間」（真の game over 単位）が乖離する。`episode_score` / `episode_len` の確定・報告は**実 game over / truncation 時のみ**行う（life-loss done では NaN のまま）。この乖離はクラシックプロトコルの既知性質であり、PRD として許容する。

### 5.7 seed と再現性

- lane seed は既存機構（MasterSeedManager → `SeedMaker::MakeIndexedSeed(i)`）から `seed_t`（uint64）で届く。ALE へは `setInt("random_seed", static_cast<int>(seed & 0x7FFFFFFF))` で下位 31bit を渡す（ALE の設定は int。0 も有効値）。
- env 内乱数（noop_max の回数抽選のみ）は `RandomHolder`（lane seed 初期化）を使い、ALE 内部 RNG（sticky）とは分離する。
- 再現契約: 同 seed + 同行動列 → 同一の観測・報酬・終端列。sticky・NoOp reset を含めて決定的（ALE RNG は random_seed で固定される）。

### 5.8 前処理 free 関数（`AtariPreprocess.hpp`、named namespace）

- `PixelwiseMax(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b, std::vector<uint8_t>& out)` — 210×160 グレー 2 枚の max。
- `ResizeGrayscale(const uint8_t* src, int src_h, int src_w, int dst_size) -> torch::Tensor` — `torch::from_blob`（uint8, `[1,1,H,W]`）→ float 化 → `interpolate(mode=area)` → `round().clamp(0,255).to(kUInt8)` → `[1,S,S]`。ImageData.cpp の既存 resize 実装（interpolate→round→clamp→uint8）と同じ流儀。area 補間は慣行実装（cv2 INTER_AREA）と同系で、事例比較性を保つ。
- ALE 非依存の純関数とし、合成入力の golden テストを可能にする（§8）。

### 5.9 GetScalar（metrics）

NaN 慣行（未確定 step は NaN、バッチ集約は wrapper の mean./max./min. prefix）に従う:

| key | 確定タイミング | 値 |
|---|---|---|
| `episode_score` | 実 game over / truncation | 生スコア（clip 前）の episode 合計 |
| `episode_len` | 同上 | agent step 数 |
| `episode_frames` | 同上 | エミュレータフレーム数 |
| `lives` | 常時 | 現在ライフ（即時値。NaN を返さない） |

`GetTensor("rgb_frame")` は `retain_rgb_frame=true` 時に直近 Step の RGB 画面 uint8 `[3,210,160]`（CHW）を返す（AtariView 用）。`GetConfigData()` は `AtariEnvConfig` の実効値を返す（既存 env と同型。Run の `config/env.*.txt` ダンプに乗る）。

## 6. ビルド統合

### 6.1 三値 gating（ルート CMakeLists.txt）

```cmake
set(ANET_ENABLE_ATARI "AUTO" CACHE STRING "Atari/ALE env module (AUTO/ON/OFF)")
# 解決: ALE_ROOT は -DALE_ROOT= が環境変数より優先。
#   AUTO: ALE_ROOT が定義され、かつ ${ALE_ROOT}/src/ale/ale_interface.hpp が実在 → 有効。それ以外 → 無効(STATUS)
#   ON  : 同条件を満たさなければ FATAL_ERROR（fail-fast）
#   OFF : 常に無効
# 有効時: add_subdirectory(core/envs/atari1) + set(ANET_ATARI_ENABLED TRUE)
# configure ログに STATUS で有効/無効と ALE_ROOT 実パスを必ず表示
```

### 6.2 atari1/CMakeLists.txt（標準雛形 + ALE 参照）

- 標準雛形（STATIC lib / include PUBLIC / pch / test / bin 出力 / source_group）は imagecls1 と同型。
- ALE 参照（sandbox 検証済みの構成をそのまま移植）:

```cmake
target_include_directories(AtariEnv PRIVATE
    "${ALE_ROOT}/src" "${ALE_ROOT}/src/ale" "${ALE_ROOT}/build/src/ale")
list(APPEND CMAKE_PREFIX_PATH "${ALE_ROOT}/build/vcpkg_installed/x64-windows-static-md")
find_package(ZLIB REQUIRED)
target_link_libraries(AtariEnv PRIVATE
    "${ALE_ROOT}/build/src/ale/$<IF:$<CONFIG:Debug>,Debug,Release>/ale.lib"   # RelWithDebInfo/Release → Release
    ZLIB::ZLIB
    ${TORCH_LIBRARIES} anet-wx anet-core)
```

- Debug は `_ITERATOR_DEBUG_LEVEL`/CRT 整合のため必ず Debug の ale.lib（MSVC LNK2038 対策）。RelWithDebInfo は Release 側 lib を使う（両方 /MD・IDL=0 で整合）。

### 6.3 runner / RunnerApp

- `apps/runner/CMakeLists.txt`: `if(ANET_ATARI_ENABLED)` で `AtariEnv` をリンクし `target_compile_definitions(AnetRLRunner PRIVATE ANET_HAS_ATARI)`。
- `RunnerApp.cpp`: `#ifdef ANET_HAS_ATARI` で `#include <anet/env/Atari.hpp>` と `anet::rl::env::InitAtari();` を既存 Init 列（`:289-294` 付近）に追加。

### 6.4 テストの gating

- `AtariEnv-test` は module と同時にビルドされる（gating の内側なので ALE 無し環境ではそもそも生成されない）。
- ROM 依存テストケースは実行時に ROM を解決できなければ `SKIP()`（Catch2）で明示スキップ（CI に ROM を置かない構成を許容）。前処理 golden テストは ALE/ROM 非依存で常時実行。

### 6.5 CI（GitHub Actions、将来）

ALE ビルド木ごとキャッシュする方針。キャッシュキーは `ALE tag (v0.12.0) + triplet (x64-windows-static-md) + ALE configure オプション`。本PRDでは CI 変更は必須としない（ローカルビルドで受入可能）。

## 7. View

`AtariView` の詳細仕様は `docs/design/220_atari_env.jp.md` §5 を正とする。要点: 生 RGB（`GetTensor("rgb_frame")`、lane 0 固定）を主、前処理後 obs（step_result の grid）を副として並置し、テキストオーバーレイ（episode_score 生 / lives / episode 内 step・frame / 直前 action 名 / 直前 reward）を付す。`ViewBase` 構成、`GetTargetClassId() == "AtariEnv"`。描画基盤は **wxGLCanvas**（HeatMapPanel / Conv2dPanel の既存 GL pane パターン流用、GL_NEAREST 整数倍拡大。採用理由は前例整合と将来のスムーズ観戦モードへの布石であり性能ではない）。SDL の観戦ウィンドウ（`display_screen`）とは独立・併用可。

## 8. テスト計画（120_environments §8 チェックリスト準拠）

ALE/ROM 非依存（常時実行）:
1. `PixelwiseMax` / `ResizeGrayscale` の golden テスト（合成入力 → 期待出力固定。area 補間の縮約値を数点ピン留め）
2. config 既定値・プリセット合成（v5/classic）の読み込み検証

ROM 依存（`ATARI_ROM_DIR` 解決可否で SKIP）:
3. EnvSpec と実観測の一致（shape/dtype/値域、`ValidateObservation`）
4. minimal/full 行動集合と `value_labels` の対応（Pong=6 等）
5. seed 再現: 同 seed + 同行動列 → 観測・報酬・終端列が bit 一致（sticky 0.25 のまま）
6. 終端契約: 実 game over → done / `max_episode_frames` 到達 → truncated（done=false）
7. episodic_life ON: ライフ減少 → done、直後の Reset が soft-reset（ALE 継続、スコア累積継続、実 game over で初めて episode_score 確定）
8. reward clip: Step reward ∈ {-1,0,1} かつ `episode_score` は生値
9. Reset 直後 flag（episode_start）/ B=1 と複数 lane / Vectorized と ThreadPool で意味不変（既存チェックリスト項目）

## 9. 受入基準と実験手順（D1-D3）

1. workspace を新設（例 `at-pong`）し、`_main.txt` で `$include <Atari.txt>` + `AtariEnv.game = pong`
2. 既存 DQN 系 baseline 構成（stucker.stack_count=4 有効）で 10M frames（2.5M steps）学習（後続検証）
3. `AtariEnv.game = breakout` の workspace で同様に実行（ハイパラは共通のまま）
4. 後続受入: Pong eval 平均 ≥ +10 / Breakout eval 平均 ≥ 30 を目安に、eval 曲線の単調改善を確認（数値は方向性基準であり厳密な閾ではない）。実験結果は `docs/experiments/` に実測のみ記録する

## 10. 将来拡張（本PRD 対象外）

- **AtariVectorEnv（batch-native）**: env 側が律速になった場合の選択肢。variant registry により `AtariEnv`（single）と別 class_id で並存可能。RB へは最新 1 フレーム切り出しで格納する設計が前提
- **RGB / RAM 観測**: `getScreenRGB` / `getRAM` の観測キー追加（config 切替）
- **flavor 実験**: mode/difficulty config は本PRDで通すが、系統的な汎化実験（HackAtari 的運用）は別途
- **Atari 100k / Atari-5 プロトコル**: sticky なし 26 ゲーム設定・5 ゲーム回帰の整備
- **マルチエージェント Atari（2P）** / **Python/GymEnv 連携**
