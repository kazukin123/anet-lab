# Atari Env（ALE 統合）

> 主たる観点: 具象実装仕様（AtariEnv。外部依存 ALE の契約・設定キー・View を含む）

## 1. はじめに

### 1.1 目的

本書は、ALE（Arcade Learning Environment）を C++ 直結で統合する `AtariEnv` の実装仕様を説明する。Env 共通 contract（[環境](120_environments.jp.md)）の上に成り立つ具象仕様として、ALE 側の設定キー契約、前処理チェーン、終端意味論、AtariView を記録する。

導入判断の経緯は ADR 0025（外部参照ビルドとライセンス）、ADR 0026（Single seam と前処理所有権）、実装計画は `docs/memo/051_atari_ale_env_10prd.md` を参照。用語（sticky actions / flavor / 生スコア / プロトコルプリセット）は `CONTEXT.md` を正とする。

### 1.2 対象読者

- AtariEnv を実装・変更する開発者
- ALE の設定キー・バージョン更新の影響を確認する開発者
- Atari の実験プロトコル（プリセット）を確認する利用者

### 1.3 記載範囲

AtariEnv / AtariEnvFactory / AtariView / 前処理関数群、ALE v0.12.0 の設定キー契約、ビルド gating。ALE 自体のビルド手順の確定値は §3、実験プロトコルの背景は `reports/atari_env_survey_2026-08-13.md`。

## 2. 全体像

```
BatchEnvBuilder (env.class_id = AtariEnv)
  └─ AtariEnvFactory (SingleDiscreteEnvFactory)
       └─ AtariEnv × N lanes（並列実行とStep内の自動リセットは共通 wrapper）
            ├─ ale::ALEInterface（1対1、ALE 側 frame_skip=1 固定）
            └─ 前処理: 自前 skip ループ → grayscale 2フレーム max-pool → selected resize → uint8 [1,S,S]
```

- frame stack は出さない。Actor 側 stucker と ReplayBuffer 側 stack_count の既存両輪が担う。
- RB には単フレーム uint8 が保存され、stack はサンプル時に展開される（ADR 0026）。
- 責務分担: **AtariEnv = 1 ALE の駆動と標準前処理**。並列・stack・Step内の終端lane自動リセット・seed 配布はフレームワーク既存機構。

## 3. 外部依存（ALE v0.12.0）

### 3.1 ビルド確定構成

| 項目 | 値 |
|---|---|
| バージョン | tag `v0.12.0` 固定。vcpkg 依存は manifest の builtin-baseline で固定 |
| 配置 | リポジトリ外（環境変数 `ALE_ROOT` で参照）。`third_party/` は同梱物専用のため使わない |
| configure | `-DBUILD_PYTHON_LIB=OFF -DSDL_SUPPORT=ON -DSDL_DYNLOAD=ON -DVCPKG_TARGET_TRIPLET=x64-windows-static-md` |
| build | Release と Debug の両方（MSVC の IDL/CRT 整合のため混在リンク不可） |
| 消費側リンク | `%ALE_ROOT%/build/src/ale/<Config>/ale.lib`（全部入り静的）+ zlib（`build/vcpkg_installed/x64-windows-static-md` を `CMAKE_PREFIX_PATH` に追加して `find_package(ZLIB)`） |
| include | `%ALE_ROOT%/src` + `%ALE_ROOT%/src/ale` + `%ALE_ROOT%/build/src/ale`（生成 `version.hpp`） |
| SDL | `SDL_DYNLOAD=ON` が Windows では必須（ALE の `common/SDL2.hpp` が `WIN32 && !SDL_DYNLOAD` で SDL 関数を dllimport 宣言するため静的 SDL はリンク不能）。リンク時 SDL 依存はなく、`display_screen`/`sound` 有効時のみ `SDL2.dll` を実行時ロード。DLL は `third_party/runtime_dlls/` に置けばビルド時に runner bin へ配備される（同 README 参照。入手は ale-py wheel 同梱等、zlib ライセンス） |
| install | 使わない（upstream の install ルールは `if(UNIX ...)` ガードで Windows 空振り）。ビルド木直接参照が標準 |

### 3.2 ALE 設定キー契約（v0.12.0 実測）

ALE の設定は `setBool/setInt/setFloat/setString` で行い、**未登録キーは throw で fail-fast する**（`emucore/Settings.cxx` の `verifyVariableExistence`）。settable な全体集合は `Settings::setDefaultSettings()`（external）とコンストラクタ（internal、Stella 由来）で確定する。以下が v0.12.0 の全キーと AtariEnv での扱いである。

**external キー（`setDefaultSettings()` 登録）**

| キー | 型 | ALE 既定 | 意味 | AtariEnv での扱い |
|---|---|---|---|---|
| `random_seed` | int | -1 | Env RNG（sticky 等）。-1 は時刻 auto | **lane seed から設定**（§4.6。-1 sentinel と衝突しない値域に変換） |
| `repeat_action_probability` | float | 0.25 | sticky actions | **config 透過**（`AtariEnv.repeat_action_probability`） |
| `frame_skip` | int | 1 | ALE 内蔵 skip（max-pool なしの単純リピート） | **1 固定**（skip は自前。max-pool に中間フレームが必要） |
| `max_num_frames_per_episode` | int | 0 | ALE 側 episode フレーム上限（game_over に truncation が混入する） | **0 固定**（truncation は自前カウントで done と厳密区別） |
| `max_num_frames` | int | 0 | 全 episode 通算フレーム上限 | 触らない（0 のまま） |
| `truncate_on_loss_of_life` | bool | false | ライフ喪失で episode を truncation 終端 | **false 固定**。`episodic_life` は自前実装（done + soft-reset）であり意味論が異なる（ALE 版は truncation、クラシック慣行は terminal） |
| `color_averaging` | bool | false | 直近 2 フレームの平均合成（max-pool の旧代替） | false 固定（フリッカー対策は自前 max-pool） |
| `reward_min` / `reward_max` | int | INT_MIN / INT_MAX | ALE 側の報酬 clamp | 触らない（sign clip は自前。clamp と sign は別物） |
| `display_screen` | bool | false | SDL 観戦ウィンドウ | **config 透過** |
| `sound_obs` | bool | false | 音声観測（`getAudio()` 有効化、SDL 不要） | 使わない（将来拡張: 音声観測キー） |
| `cpu` | string | "low" | Stella CPU エミュレーション忠実度（low=速度優先） | 既定のまま |
| `system_random_seed` | int | 4753849 | Stella System 内部 RNG。既定で固定値=決定論 | 触らない |
| `paddle_min` / `paddle_max` | int | -1 | パドル物理範囲（連続行動用） | 使わない（連続行動は対象外） |
| `restricted_action_set` | bool | false | legacy（FIFO インターフェース用） | 使わない（minimal set は `getMinimalActionSet()` で取得） |
| `run_length_encoding` | bool | true | FIFO controller 用 | 無関係 |
| `send_rgb` | bool | false | FIFO 用 | 無関係 |
| `rom_file` | string | "" | `loadROM()` が内部で設定 | 直接触らない |
| `record_screen_dir` | string | "" | 画面 PNG 連番の自動記録 | 使わない（将来: 観戦記録） |
| `record_sound_filename` | string | "" | 音声記録 | 使わない |
| `fragsize` | int | 64 | サウンド同期用フラグメント | 既定のまま |

**internal キー（Stella 由来、コンストラクタ登録）**

| キー | 型 | 既定 | 意味 | AtariEnv での扱い |
|---|---|---|---|---|
| `sound` | bool | false | SDL 音声出力 | **config 透過**（`AtariEnv.sound`。`display_screen` と独立） |
| `palette` | string | "standard" | カラーパレット（standard/z26/user） | 既定のまま |
| `freq` / `tiafreq` | int | 31400 | 音声サンプリング | 既定のまま |
| `volume` | int | 100 | 音量 | 既定のまま |
| `clipvol` | bool | true | 音量クリップ | 既定のまま |

契約上の注意:

- 「触らない」キーは AtariEnv からは設定しない。ALE 既定値の変化は tag 更新時の差分レビュー対象（本表が v0.12.0 の基準値）。
- `setInt("random_seed", ...)` 等は ROM ロード**前**に設定する（`loadROM()` が設定を反映して環境を構築する）。
- sticky actions の判定は skip ループの内側で毎フレーム行われる（`environment/stella_environment.cpp` の act ループ）。自前 skip（frame_skip=1 で act() を k 回）でも act() 単位=フレーム単位の判定となり、Machado et al. 2018 の原義と同一の意味論が保たれる。

### 3.3 ROM

- ROM（.bin）はリポジトリ非同梱・自己調達。ファイル名は snake_case（`pong.bin`）。
- 解決順: `AtariEnv.rom_dir`（非空なら優先）→ 環境変数 `ATARI_ROM_DIR`。`<rom_dir>/<game>.bin` が実在しなければ Env 構築時に fail-fast（探索パスと設定手段をエラーメッセージへ）。
- `loadROM()` は ROM md5 で対応ゲームを判定するため、非対応/破損 ROM は ALE 側でも検出される。

### 3.4 ライセンス（ADR 0025）

ALE は GPL-2.0。①非同梱（外部参照）②オプショナルビルド ③リリースパッケージ除外（リリースビルドは `ANET_ENABLE_ATARI=OFF`）④`core/envs/atari1/NOTICE.md` に GPL 注意書き、で本体 Apache-2.0 を維持する。

## 4. AtariEnv 仕様

### 4.1 モジュール構成

```
core/envs/atari1/include/anet/env/Atari.hpp   … void InitAtari();
core/envs/atari1/src/Atari.cpp                … factory + view creator 登録
core/envs/atari1/src/AtariEnv.hpp/.cpp        … AtariEnvConfig / AtariEnv / AtariEnvFactory
core/envs/atari1/src/AtariPreprocess.hpp/.cpp … 前処理関数とGrayscaleResizer（named namespace、ALE非依存）
core/envs/atari1/src/AtariView.hpp/.cpp       … AtariView（§5）
core/envs/atari1/src/AtariEnv_test.cpp
core/envs/atari1/src/pch.hpp
core/envs/atari1/NOTICE.md
```

class_id は `AtariEnv`。`AtariEnv : public SingleDiscreteEnvBase, public anet::RandomHolder`。

### 4.2 config キー（prefix `AtariEnv`）

| キー | 型 | 既定 | 意味 |
|---|---|---|---|
| `game` | string | （必須） | ROM 名（snake_case、拡張子なし） |
| `rom_dir` | string | `""` | 非空なら `ATARI_ROM_DIR` より優先 |
| `screen_size` | int | 84 | 出力解像度 S（正方形） |
| `resize_mode` | string | adaptive_average | adaptive_average（従来）/pixel_area（面積平均＋最後に偶数丸め）。pixel_areaはS=1〜160 |
| `frame_skip` | int | 4 | 1 step のエミュレータフレーム数（自前 skip ループ） |
| `max_pool` | bool | true | skip 窓の最後 2 フレームの pixel-wise max |
| `repeat_action_probability` | float | 0.25 | sticky actions（ALE 透過） |
| `noop_max` | int | 0 | Reset 時 1..noop_max 回の NOOP 乱数挿入（0=無効） |
| `fire_reset` | bool | false | Reset 直後に FIRE 1 回（FIRE を持たないゲームでは no-op） |
| `episodic_life` | bool | false | ライフ減少を done として学習系へ見せる（§4.5） |
| `reward_clip` | bool | true | Step reward の sign clip |
| `full_action_space` | bool | false | true で legal 18 行動 |
| `mode` / `difficulty` | int | -1 | -1=ALE 既定。それ以外は `setMode`/`setDifficulty`（flavor） |
| `max_episode_frames` | int | 108000 | 自前カウントの truncation 閾（0=無効） |
| `retain_rgb_frame` | bool | true | Step毎にRGBを保持し結果のauxへ参照を保存。現在画面は`GetTensor("rgb_frame")`（§5） |
| `display_screen` | bool | false | SDL 観戦ウィンドウ（ALE 透過） |
| `sound` | bool | false | SDL 音声（ALE 透過。display と独立） |
| `ram_metric.[game].[label]` / `ram_metric.[game].metrics` | string | 定義なし | ゲーム別の RAM 定義と番号付け（§4.9） |

プロトコルプリセット（`Atari.txt` に定義、`AtariEnv.$` の選択チェーンで指定。既定 = `v5_noop0`）:

- `AtariEnv.v5_noop0`: sticky 0.25 / noop_max **0** / episodic_life false / fire_reset false（raw ALE v5 準拠。sticky のみが確率源）
- `AtariEnv.v5_noop30`: sticky 0.25 / noop_max **30** / episodic_life false / fire_reset false（v5 + Gymnasium `AtariPreprocessing` 既定）
  - **`v5` という単独のプリセットは意図的に置いていない。** 「v5」は env id の世代であって noop の有無を決めないため、名前だけで選べてしまうと比較先との食い違いが無言で入る。差分は `noop_max` の 1 行のみ（後述の系統依存の項を参照）。
- `AtariEnv.classic`: sticky 0.0 / noop_max 30 / episodic_life true / fire_reset true
- `AtariEnv.100k`: sticky 0.0 / noop_max 30 / episodic_life true / fire_reset false（Atari-100k ベンチ。予算 100k steps = 400k frames は Run 設定側で指定。100k の共通不変は sticky なし+400k frames のみで、torch 系多数派＝SPR/EfficientZero の条件に合わせた。Dopamine 系＝DrQ(ε)/DER(ε)/BBF は noop 0・episodic_life false と分裂しているため、比較先の実装系統を必ず確認する）

標準準拠ノート（2026-08-18 に SB3 `atari_wrappers.py` / Gymnasium `AtariPreprocessing` / ale-py v5 登録 / rlpyt と突き合わせて監査済み）:

- 標準実装は 2 系統ある。**baselines 系**（OpenAI Baselines 2017 → SB3 → CleanRL、および rlpyt＝SPR/EfficientZero の土台。episodic_life=true で運用）と **Dopamine 系**（Dopamine → Gymnasium `AtariPreprocessing` → dqn_zoo。terminal_on_life_loss=False で運用）。本 env は wrapper 挙動（noop の乱数範囲と適用位置、FIRE 系列、max-pool 対象、sign clip、grayscale=ALE 直、episodic_life 判定）をこの2系統と一致させてある。縮小の数値規則は§4.3の方式選択に従う。系統間で割れる箇所は「その分岐を実際に使う側」に合わせる（例: episodic_life の `lives > 0` ガードは baselines 系に一致。§4.5）。
- **v5 は Machado et al. 2018 勧告と 1 点だけ乖離している**: Machado は full action set（18 行動）を勧告するが、ale-py v5 の既定は `full_action_space=False`（minimal）。本 env の v5 プリセットは ale-py 側（false）に合わせる。
- **「v5 で noop 0」は系統依存**: ale-py v5 の raw env に noop start は無く（Dopamine/Machado 系）、一方 Gymnasium `AtariPreprocessing` wrapper の既定は `noop_max=30`。v5 世代の事例でも「+ Gymnasium 標準前処理」の構成は noop 30 で走っているため、比較先の wrapper 構成を確認する。
- classic の `max_episode_frames=108000` は歴史的 v4（gym 登録の TimeLimit=100,000 フレーム）と 8% 違う。SB3 `AtariWrapper` 自体は上限を持たないため、v5 と揃えた 108,000 を採用している。
- 標準側の既知の癖 2 件は追随しない: skip 窓途中終端時の max-pool（標準はゼロ/古フレームと max する。本 env は実行フレームのみで、done 直後の 1 観測しか差が出ない）、soft-reset 中の報酬（SB3 は完全に破棄。本 env は `game_score` へ加算し、生スコアとしてはこちらが正確）。

config 契約ノート:

- `screen_size` が正方形単一値なのは標準前処理（84×84）への準拠であり、技術的制約ではない。210×160→84×84 はアスペクト比を意図的に破壊するのが慣行（維持しようとする letterbox 等はかえって非標準）。長方形が必要になった場合は `screen_height`/`screen_width` への分割を env 内に閉じた互換変更として行う。
- **SDL 音声は Windows では DirectSound バックエンドが必須**。ALE の `SoundSDL` は legacy `SDL_OpenAudio(desired, obtained)` を AUDIO_U8 前提で使うが、`obtained` 非 NULL のこの API は**フォーマット変換をせず**ハードウェア形式を返す。既定の WASAPI バックエンドは常に float32/stereo を返すため、U8 サンプルが float32 として解釈され**ほぼ無音**になる（ALE v0.12.0 の upstream バグ。sandbox の beep テストで実測確認済み: WASAPI → `format=0x8120` で無音、directsound → `format=0x8` で正常）。このため **AtariEnv は `sound=true` のとき、環境変数 `SDL_AUDIODRIVER` が未設定なら SDL 初期化前に `directsound` を自動設定する**（`EnsureSdlAudioDriverDefault`、プロセス唯一実行）。明示設定された環境変数は尊重されるので、検証等で WASAPI を試す場合は環境変数で上書きできる。upstream 修正は `obtained=NULL` 渡し（SDL 側自動変換）が本筋で PR 候補。また **SDL 音声は実機速度（60 エミュレータフレーム/s）で駆動することが前提** — それより速く回すと音声レジスタキューが溜まり `SoundSDL::processFragment` の刈り込みが働いて音割れする（EvalPanel は `app.online.eval_panel.fps = 15` × `frame_skip=4` で実速）。なお TIA 音源自体が harsh なローファイ（例: Space Invaders の行進音はバズ音）であり、これは仕様。診断ツールとして sandbox（C:/dev/ale-sandbox）に `beep`（SDL 出力層の format/レート検証）と `wav`/`wavnoop`（SDL を介さない TIASound 純合成の WAV 化）モードがある。
- `display_screen` は **sandbox・単発デバッグ用**（`num_envs=1` + `env.worker_type=SINGLE_THREAD` 想定）。SDL ウィンドウは ALEInterface インスタンス毎に生成されるため（`OSystem.cxx` の ScreenSDL 生成）、N lane では N ウィンドウが開く。さらに SDL video/event はスレッド安全でなく、ThreadPool の worker スレッドからの SDL 呼び出しは未サポート動作となる。Runner での観戦は AtariView（§5）が正。env は `display_screen=true` のとき log.warn を 1 回出す。lane 0 限定でのウィンドウ表示は、env が lane を知る手段が name 解析しかなく「Env name を挙動決定に使わない」規約（`CONTEXT.md`）に反するため行わない。

### 4.3 観測と前処理チェーン

観測は grid キーのみ: `TensorSpec{ Grid, {1, S, S}, kUInt8, num_classes=0, min=0, max=255 }`。uint8 連続扱いのため NN 境界（`NetworkBoundaryPreprocessor::Format`）で自動的に /255 float 化される。vector / action_mask キーは出さない。

```
Step(action_index):
  a = action_set[action_index]
  reward_raw = Σ ale.act(a)                         # sticky は ALE 内で毎フレーム判定
      （当該Step内でrolling 2-slotに画面を取得し、real game overだけ早期終了）
  frame = max_pool ? 当該Stepで実行できた最後の最大2フレームのmax : 最終フレーム
  grid  = resizer.Resize(frame)                       # area 補間 → round/clamp → uint8 [1,S,S]
```

前処理は `AtariPreprocess` の関数群と `GrayscaleResizer`（named namespace、ALE非依存）で行い、合成入力と固定画像のテストで数値を確認する。

- `PixelwiseMax(a, b, out)` — 210×160グレー2枚の要素max。
- `GrayscaleResizer(src_h, src_w, S, mode).Resize(src)` — 各Envが設定と重みを所有し、Reset/Stepの両方でCPU縮小してから観測をdeviceへ送る。
- `pixel_area`の作業領域は縮小器が構築時に確保し、各Resizeで再利用する。同一縮小器のResizeは同時に呼び出さない。各AtariEnvが独自の縮小器を持ち、同じEnvのReset/Stepは直列に実行する。出力Tensorは呼び出しごとに独立して所有する。
- `adaptive_average`は既存のfloat32 `interpolate(mode=area)`→`round().clamp(0,255).to(kUInt8)`を維持する。非整数比では、整数窓の等重み平均でありOpenCV INTER_AREAと同じ値とは限らない。
- `pixel_area`は出力領域と元画素の重なり面積で平均する。各軸の重みを構築時に準備し、横→縦を整数化せず計算した最後に偶数丸めしてuint8 `[1,S,S]`を返す。正方形のS=1〜160を認め、拡大はエラーとする。OpenCVの全分岐の丸めを再現する契約ではない。

`AtariEnv.resize_mode ?= adaptive_average`を既定とし、`run.@pixel_area : AtariEnv.resize_mode = pixel_area`をRun選択チェーンへ加えると新方式になる。`run.eval.[tag].env.resize_mode`の明示値があれば、その評価と同じtagを使うEvalPanelだけ上書きする。実効設定と各Envの構築時verboseログに方式が残る。既定のinfo表示では構築ログを表示せず、verbose表示とファイルログで個別のEnvを追跡する。未知・空の方式名や不正なサイズは、キー・指定値・期待値を含むエラーで構築時に失敗する。

### 4.4 行動

- 既定は `getMinimalActionSet()`（ROM ロード後に取得）。`full_action_space=true` で `getLegalActionSet()`（常に 18）。
- `Step(action)` の action は**選択済み集合の index**（ALE Action enum 値ではない）。
- `ActionSpec.value_labels` は `ale::action_to_string` 由来の行動名（NOOP/FIRE/RIGHT/LEFT/...）。
- ALE に per-step 合法手は存在しない（全行動が常時受理され、無効行動は no-op として振る舞う）。

### 4.5 報酬と終端

- `reward_clip=true`（既定）のとき Step の reward は `sign(reward_raw)`（学習報酬）。生スコアは env 内部で累積し、実 game over / truncation で `game_score` として確定する（生スコア／学習報酬の区別は `CONTEXT.md`）。キー名が `episode_` ではなく `game_` なのは、`episodic_life=true` のとき RL のエピソード境界（life 単位）と確定境界（ゲーム 1 回）が一致しないため。
- `done = ale.game_over(false)`（ALE 側 truncation は無効化済みなので純粋な terminal）。real game overだけが skip 窓の早期終了理由になる。
- `truncated` = `ale.getEpisodeFrameNumber()` が `max_episode_frames` に到達（done 優先）。自前 frame counter は持たず、Reset中のNOOP/FIREもALE frameへ含める。
- `episodic_life=true` の場合: skip 窓完走後に lives を1回だけ比較し、**ライフ減少かつ残 lives > 0** なら done=true を返す。`lives > 0` ガードは baselines 由来（SB3 / CleanRL / rlpyt と同一の条件式）。Qbert 系は game over の数フレーム前から lives=0 を報告するため、ガードが無いと実終端の直前に偽の life-loss done が 1 回出る（Gymnasium `AtariPreprocessing` はガード無しだが、あちらは `terminal_on_life_loss=False` 既定で分岐が実質使われない。episodic_life を実際に true で運用する系統は全てガード有り）。life loss後の残り反復と報酬も同じStepに含める。Stepは学習用観測・RGB・auxを確定した後、NOOPを1フレーム進める。この報酬はgame_scoreだけに加算し、返す遷移を変えない。このNOOPで実game overになった場合も同じStep内で完了値を確定する。同じBatch Step内の `Reset()` は実 game over でなければ **soft-reset**（`ale.reset_game()`せず、game_score・ALE episode frameは継続）。**`fire_reset=true` のときは続けて FIRE 系列（§4.5.1）を打つ**（FIRE系列は2フレーム。報酬は game_score へ加算）。Breakout 系はライフ喪失でボールが消え FIRE を押すまで再投入されないため、これが無いとエージェントが自力で FIRE を選ぶまで無報酬 step が流れる。標準の wrapper 構成が `FireResetEnv` を `EpisodicLifeEnv` の外側に置き life-loss 後の reset でも FIRE を入れるのに合わせた契約。`noop_max` は soft-reset では打たない（`NoopResetEnv` 相当は実 reset のみに効く）。BatchEnvがStep内で終端laneをResetする契約とこの soft-reset で「学習上の episode = life 単位、メトリクス上の単位 = 実ゲーム 1 回」を両立する。`game_score` / `game_len` / `game_frames` の確定は実 game over / truncation 時のみ。configured `eval1` / `eval2` とEvalPanel用`eval_panel`は`env.episodic_life=false`を明示し、評価セッションのepisode境界を実game over / truncationへ揃える。

#### 4.5.1 FIRE 系列（`fire_reset`）

`fire_reset=true` のとき、Reset で次の 2 アクションを順に打つ。stable-baselines3 の `FireResetEnv`（`step(1)` の後に `step(2)`）と同じ手順である。

1. `PLAYER_A_FIRE`
2. **action set の 3 番目**（index 2）

いずれも実行後に game over していれば `reset_game()` する。2 番目を打つのは FIRE 単独では開始しないゲームへの手当てで、**意味は問わず index で指定する**（Breakout では RIGHT、Seaquest では UP に当たる）。

適用条件は本家の `assert` に対応する 2 つ:

- action set が 3 要素以上
- action set の index 1 が `PLAYER_A_FIRE`

ALE の action set は `Action` enum 順（NOOP=0, FIRE=1, UP=2, RIGHT=3, ...）に並ぶため、FIRE を含むゲームでは index 1 が必ず FIRE になる。条件を満たさないゲーム（Freeway のように FIRE を持たない minimal action set）では**何も打たない**。本家が `if "FIRE" in get_action_meanings()` で wrapper 自体を適用しないのと同じ安全弁である。

**`full_action_space=true` では安全弁が効かない**。18 アクション固定になり全ゲームで FIRE が action set に入るため、本来 FIRE を使わないゲームでも Reset 時に FIRE 系列が打たれる。`full_action_space` と `fire_reset` を同時に有効化する場合はこの副作用を理解した上で行う。

呼び出し箇所は hard-reset（`ApplyResetActions()` の末尾、`noop_max` 抽選の後）と soft-reset（§4.5）の 2 つ。報酬の扱いは異なり、hard-reset は直後に `game_score` を 0 にするため破棄、soft-reset は継続中のゲームなので `game_score` へ加算する。

### 4.6 seed と再現性

- lane seed（`SeedMaker::MakeIndexedSeed(i)` 由来の uint64）を `setInt("random_seed", static_cast<int>(seed & 0x7FFFFFFF))` で ALE へ渡す（-1=auto sentinel と衝突しない非負値域）。ROM ロード前に設定する。
- env 内乱数（noop_max の回数抽選のみ）は `RandomHolder`（lane seed 初期化）を使用し、ALE 内部 RNG（sticky）と分離する。
- 再現契約: 同 seed + 同行動列 → 同一の観測・報酬・終端列（sticky・NoOp reset を含めて決定的）。

### 4.7 Env accessor（Module インターフェース）

| API | キー | 確定タイミング | 値 |
|---|---|---|---|
| Step結果.GetScalar | `game_score` / `game_len` / `game_frames` | 通常StepはNOOP前速報値、life-only終端はNaN、実game over/truncationは確定値 | scoreは生報酬合計、lenはagent step、framesはALE frame |
| Step結果.GetScalar | `game_score.ge.[N]` / `hns57` / `hns49` | game_scoreと同じ成立条件 | 閾値判定・HNS基準表から計算。Nは負値を含むfloat |
| Step結果.GetScalar | `ram_metric.[n]` | 通常Stepはsnapshot、life-only終端/現在ゲームに番号なしはNaN、実終端は確定値 | 全ゲームに番号がなければ未知 |
| Step結果.GetScalar | `lives` | 毎Step、補助NOOP前 | 返す遷移に対応する残機 |
| Env.GetScalar | `lives` | 常時 | 現在ALEの残機。上記ゲーム統計キーは未知 |
| Env.GetTensor | `rgb_frame` | 常時 | 現在画面。ViewはStep結果のaux.rgb_frameを使用 |

Step結果は生成時の値を保持し、Resetや通知後も変わらない。env本体には完了値を持ち越さず、移した統計キーは未知とする。結果の`lives`は補助NOOP前、env本体の`lives`は現在ALEの値である。

バッチ集約（`mean.` 等の prefix）と NaN 慣行は wrapper の共通規約に従う。`GetConfigData()` は実効 config を返す（Run の `config/env.*.txt` ダンプ対象）。

**ゲーム完了の verbose ログ。** 上表の確定と同じ時点・同じ値で 1 行出す（`Game over.` または `Game truncated by max_episode_frames.` に `game_score` / `game_len` / `game_frames` が続く）。番号付き RAM 定義がある場合は末尾へ ` ram_metric: floor_clear=0 boss_kill=0` を番号順で足す。prefix は lane 名（`env.[train].[0]:` / `env.[EvalPanel].[0]:` など）。Run のログファイルには常に残り、`app.log_panel.level` は LogPanel の表示だけを絞る。metrics / trace を持たない EvalPanel のゲームも、ここからミリ秒の時刻付きで追える。

**集約の分母に注意。** ゲーム単位のキーはlife-only終端でNaNを返し、episode_endでは実game over/truncationだけが集約に入るため、バッチ集約は NaN を分母から除外する（`core/anet-core/src/util.cpp`。全 env が NaN なら結果も NaN）。したがって `mean.game_score` の分母は num_envs ではなく **その step で実際にゲームを終えた env の数**であり、複数 env が同時に終えたときだけ複数ゲームの平均になる。同時完了率は λ = num_envs / 平均ゲーム長 で決まり、Breakout（128 env / 約 1,900 step、λ ≈ 0.07）では実測 97% が単独完了なので、`mean.` の系列は事実上「ゲーム 1 回の素点の列」である。ゲーム長が短い題材ほど平均化が効いてピークが潰れるため、ゲーム横断の比較では `max.` を併置して読む。`lives` だけは常時確定なので分母は num_envs であり、これは本物のバッチ平均になる。

`game_score.ge.[N]` はこの分母を `game_score` と共有する 0/1 指標で、`mean.` を取ると「その step でゲームを終えた env のうち N 点以上だった割合」になる。窓平均すれば発火イベント単位ではなく**ゲーム 1 回単位の割合**が得られるため、閾値越えの頻度を見る用途では `mean.` の潰れと `max.` の飽和の両方を避けられる（`mean.` は二山分布の両端を潰し、`max.` は評価セッションの episode 数を増やすほど飽和する）。Breakout の 1 画面 = 432 のようにゲーム依存の閾値は §4.8 の HNS 表と違いキー側のパラメータで持つ。1 Run で複数の閾値を同時に測るためと、閾値がゲームの物理定数ではなく解析上の判定基準だからである。

### 4.8 人間正規化スコア（HNS）

生スコアを「人間プレイヤー比」へ正規化した指標。単一ゲームでも到達度が直感的に読め（100 = 人間）、複数ゲームの集計にも使える。式は参照実装（DeepMind `dqn_zoo`、Dopamine、IQN、Agent57）と同一で、**分母は絶対値化しない**:

```
hns = 100 * (game_score - random) / (human - random)
```

**基準表を 2 系統持ち、両方を常に出す。** 同じゲームでも値が実質的に異なり、取り違えると文献比較が狂うため:

| キー | 表 | 出典 | ゲーム数 |
|---|---|---|---|
| `hns57` | 57 ゲーム表 | Wang et al. 2016（Dueling）系。DeepMind `dqn_zoo/atari_data.py` の `_ATARI_DATA` と同一値 | 57 |
| `hns49` | 49 ゲーム表 | Mnih et al. 2015（Nature）Extended Data Table 2 | 49 |

差の実例: Pong の human は 57 表 14.6 / 49 表 9.3。同じ生スコア +7 でも HNS は 78.5% 対 92.3% になる。Breakout は 30.5 / 31.8。Rainbow・IQN・Agent57・BBF はすべて 57 表を使うため、現代論文との比較には `hns57` を用いる。`hns49` は Nature DQN の数字（Breakout 401.2 = 1327.2%）と並べるときに使う。

契約:

- Step結果の`game_score`と同じ時点と成立条件（通常はNOOP前速報値、life-only終端はNaN、実game over/truncationは確定値）
- **基準表に載らないゲームは NaN**（`std::nullopt` ではない）。`nullopt` は `DiscreteBatchEnvBase` の集約でバッチ全体を打ち切ってしまうため。ALE は 104 ゲームを持つのに対し標準表は 57 / 49 しか覆わず、57 表のみに載る 8 ゲーム（berzerk / defender / phoenix / pitfall / skiing / solaris / surround / yars_revenge）は `hns49` が NaN になる
- 対象は生スコア。reward clip 後の学習報酬ではない
- 49 表の値は Extended Data Table 2 の "Normalized DQN (% Human)" 列で検算済み（`AtariEnv_test.cpp` の `[atari][hns]`）

CHNS（capped HNS、Agent57）と human gap（IQN Table 1、`gap = min(max(1 - HNS, 0), 1)`）は env からは出さない。両者は `gap = 1 - CHNS` の恒等関係にあり、単一ゲームの時系列では値が飽和して動かない（Breakout は常に 100% / 0）。キャップの意義は複数ゲーム集計での外れ値抑制にあるため、HNS からの導出として後処理側で扱う。

### 4.9 RAM メトリクス

`AtariEnv.ram_metric.[game].[label] = <0x80..0xFF> <畳み方>` で RAM バス番地と意味名を宣言し、同じゲームの `.metrics = 1:label 2:label ...` で汎用番号に結ぶ。全ゲームの宣言を ROM 読み込み前に検証する。番号のない定義は RAM 知識として保持するが評価しない。eval env の `run.eval.[tag].env.ram_metric.*` 上書きは受け付けず、設定読み込み時に fail-fast する。

ゲーム名は小文字 snake_case に限定し、空の `.metrics` 行は fail-fast する。受理した定義と `.metrics` 行は、番号のない定義も含めて Module Config の実効設定へ記録する。
eval env の設定ダンプ（`config/env.eval.txt` など）には `run.eval.[tag].env.ram_metric.*` の形で出る。これは実効値の記録であり、同じ形を入力に書くと fail-fast する。

畳み方は `max_seen` / `min_seen`（`reset_game()` 直後の値を含む最大・最小）、`inc_count` / `dec_count`（前フレームからの増加・減少回数）、`reach_count:N`（0～255 の N へ入った回数）の 5 種。比較するのは生の 1 バイトで、毎 `act()` 後に観測する。集計は実ゲーム 1 回を単位とし、episodic-life の soft reset をまたいで継続、`reset_game()` ごとに仕切り直す。

Step結果の`ram_metric.[n]`は通常Stepのsnapshotまたは実ゲーム完了時の確定値を返し、life-only終端ではNaNとする。どこかのゲームが n を使えば既知キーなので、現在ゲームに n がない場合も NaN を返す。番号は全ゲーム共通の大まかな意味を保つ: 1 = 面クリア回数、2 = ボス撃破回数、3 = ボス命中回数。当てはまらないゲームには番号を付けない。4 は共通の意味を持たず、ゲームごとに数えたいものへ使う（ゲームをまたいで 4 の列を並べて読まない）。trace は `@episode_end`、scalar の `mean.` は完了 lane だけから値を取る。番号を持たないゲームでは trace は `null`、scalar は行が出ない。番号の具体的なラベルは Run の `config/config_data.txt` で引く。

| ゲーム | 番号付き定義 | 番号なしの参考定義 |
|---|---|---|
| kung_fu_master | `1:floor_clear` = `0x9F inc_count`、`2:boss_kill` = `0xCC reach_count:0`、`3:boss_hit` = `0xCC dec_count` | `floor_max` = `0x9F max_seen`、`boss_hp_min` = `0xCC min_seen` |
| qbert | `1:round_clear` = `0xE3 inc_count` | `side_color` = `0x82 min_seen` |
| phoenix | `1:wave_clear` = `0xCA inc_count`、`2:boss_kill` = `0xFC inc_count` | なし |
| breakout | `1:wall_clear` = `0x9E inc_count` | `score_ge600` = `0xCC reach_count:6` |
| beam_rider | `1:sector_clear` = `0x80 inc_count`、`2:boss_kill` = `0x94 reach_count:141`、`4:torpedo_fired` = `0xD3 dec_count` | なし |
| freeway | `4:car_hit` = `0xEA reach_count:91` | なし |

番号付き定義の進行中の値は AuxData の `ram_metric.<label>` に int64 で載せる。Step結果の番号キーは同じsnapshotへ問い合わせる。こちらのラベルキーはAtariViewの観戦表示にも使う。詳細な調査根拠と既知の限界は [PRD082](../memo/082_atari_ram_progress_metrics_10prd.md) と [ADR 0046](../adr/0046-atari-ram-metrics-bound-per-game-to-numbered-keys.md) を参照する。

## 5. AtariView

`ViewBase<AtariData, AtariPanel>` 構成（ImageClsView と同型）。`GetTargetClassId() == "AtariEnv"`。

### 5.1 表示内容

- **主表示: 生 RGB 画面**（210×160、`event.step_result->GetAuxDataList(0)`の`rgb_frame`）。人間の観戦とデバッグの主対象。
- **副表示: 前処理後 obs**（S×S グレー、TrainEvent の step_result から lane 0 の grid を取得）。「エージェントが見ているもの」の確認用で、resize / max-pool の不具合を生画面との見比べで検出できる。
- **既定は生 RGB 画面（カラー）のみ**。中クリックで「元画像のみ → Agent画像（前処理後 obs）のみ → 両方 → 元画像のみ」と循環する。描画面・テキストオーバーレイ・Panel のどこでも1クリックで1段階進み、停止中も即座に再描画する。
- 単独表示はキャンバス全体を使って中央配置し、両方表示は RGB を左 2/3、前処理後 obs を右 1/3 の領域に並べる。既存の余白と整数倍拡大（ドット絵の視認性優先、GL_NEAREST）を維持する。画像未取得の領域は背景色のままとする。
- 表示モードは Panel ごとに保持し、データ更新・episode 切り替えでは変えない。Panel 再作成時は既定へ戻る。
- **オーバーレイ（テキスト行）**: game_score（生・累積中の暫定値）/ lives / ゲーム内 step・frame / 直前 action 名（value_label）/ 直前 reward（clip 後）。RAM 定義があるゲームでは末尾へ `RAM: boss_hit=0 boss_kill=0 floor_clear=0` のようにラベル順で進行中の値を加える。

### 5.2 データ経路と制約

- 表示対象は **lane 0 固定**（全 lane 同一ゲーム・同一設定のため代表 1 lane で挙動確認の目的を満たす。lane 選択 UI は将来拡張）。
- RGBはenvがStep毎に`getScreenRGB`で取得し、そのTensor参照をStep結果のauxに保持する（`retain_rgb_frame`、既定 true。保持コストは約 100KB memcpy/step/lane で数十 µs）。ヘッドレス運用や性能検証で切りたい場合は config で false。
- 更新は既存 View 機構の標準契約（`UIDataStore`、既定 `force_update_interval_ms=200` ≈ 5Hz。TrainEvent 毎に描画するわけではない）。
- **描画基盤は wxGLCanvas**（HeatMapPanel / Conv2dPanel の既存 GL pane パターンを流用。テクスチャ 2 枚 + GL_NEAREST で整数倍拡大）。採用理由は既存パターンとの整合と、将来のスムーズ観戦モード（update interval 短縮による 30〜60fps 描画、SDL 窓の実用代替）への布石であり、**性能ではない**——既定 5Hz・210×160 規模では GDI（wxBitmap 流）でも性能問題は構造的に発生しないことを確認済み。テキストオーバーレイは wx 側で重畳描画する。スムーズ観戦モード自体（interval の config 化）は本仕様の対象外（将来拡張）。
- SDL の観戦ウィンドウ（`display_screen`）とは独立・併用可（View = Runner pane 統合、SDL = ALE ネイティブ・音声付き）。

## 6. ビルド統合

- 三値 cache 変数 `ANET_ENABLE_ATARI` = `AUTO`（既定）/`ON`/`OFF`。AUTO は `ALE_ROOT` の存在と `src/ale/ale_interface.hpp` の実在で判定。ON で不備なら configure エラー。判定結果は STATUS ログへ。
- 有効時のみ: `add_subdirectory(core/envs/atari1)`、runner へ `AtariEnv` リンク + `ANET_HAS_ATARI` 定義、`RunnerApp.cpp` の `#ifdef ANET_HAS_ATARI` で `InitAtari()`。
- ale.lib は `$<IF:$<CONFIG:Debug>,Debug,Release>` で選択（RelWithDebInfo は Release 側。IDL/CRT 整合）。
- テストは module と同時ビルド。通常の ROM 依存ケースは `ATARI_ROM_DIR` 未解決なら Catch2 `SKIP()`。PRD097 の必須受入 `[prd097_atari]` は Asterix ROM の利用を `REQUIRE` で検査し、未解決を合格・skip扱いにしない。前処理 golden テストは常時実行。

### 6.1 テストと拡張時の確認事項

縮小方式の変更時は[PRD099検証手順](../../core/anet-core/testdata/prd099/README.md)を参照する。固定24画像は各画像で全画素差1以下・99.9%以上一致を要求し、既定の学習系列のchecksumを保持する。性能受入は84×84・64 envの交互比較で速度比中央値0.98以上、現行同士の対照が1%以内に収まらない場合は判定保留とする。

RAM 定義の番号や畳み方を変更するときは [PRD082 検証手順](../../core/anet-core/testdata/prd082/README.md) の ROM 不要テストと ROM 依存テストを確認する。未番号付け定義は検証のみ行い、AuxData・ログ・`GetScalar` へ出さない。既定 `Atari.txt` の群ごとの RAM 指標の揃いとプロトコルプリセットの値は、anet-core-test の `[shipped_config][atari]`（`config_baseline_test.cpp`）が検査する。

## 7. 関連文書

- [環境](120_environments.jp.md) — Env 共通 contract と追加チェックリスト（§8）
- ADR 0025 / ADR 0026 — 導入判断
- `docs/memo/051_atari_ale_env_10prd.md` — 実装計画（PH1 スコープ・受入基準）
- `reports/atari_env_survey_2026-08-13.md` — ALE サーベイ（プロトコル・ベンチマーク動向）
- `CONTEXT.md` — 用語（sticky actions / flavor / 生スコア / プロトコルプリセット）

遷移の観測・RGB・auxは残機喪失時の補助NOOPより前に作る。NOOPでゲームが終わったとき、Step結果のゲーム確定値にはNOOPの得点とframeを含めるが、返すreward/next_state/RGB/livesは置き換えない。AtariViewは結果のrgb_frameを表示するため、自動Reset後の画面へずれない。結果は後続StepやResetから独立し、envへ完了値を持ち越さない。受入AsterixのRAM条件と公開API試験は[PRD097検証手順](../../core/anet-core/testdata/prd097/README.md)を参照する。
