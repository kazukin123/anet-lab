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
       └─ AtariEnv × N lanes（並列・auto-reset は既存 wrapper）
            ├─ ale::ALEInterface（1対1、ALE 側 frame_skip=1 固定）
            └─ 前処理: 自前 skip ループ → grayscale 2フレーム max-pool → area resize → uint8 [1,S,S]
```

- frame stack は出さない。Actor 側 stucker と ReplayBuffer 側 stack_count の既存両輪が担う。
- RB には単フレーム uint8 が保存され、stack はサンプル時に展開される（ADR 0026）。
- 責務分担: **AtariEnv = 1 ALE の駆動と標準前処理**。並列・stack・autoreset・seed 配布はフレームワーク既存機構。

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
core/envs/atari1/src/AtariPreprocess.hpp/.cpp … 前処理 free 関数（named namespace、ALE 非依存）
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
| `retain_rgb_frame` | bool | true | Step 毎に RGB 画面を保持し `GetTensor("rgb_frame")` で公開（AtariView 用。§5） |
| `display_screen` | bool | false | SDL 観戦ウィンドウ（ALE 透過） |
| `sound` | bool | false | SDL 音声（ALE 透過。display と独立） |

プロトコルプリセット（`Atari.txt` に定義、`AtariEnv.$` AutoMerge で選択。既定 = v5）:

- `AtariEnv.v5`: sticky 0.25 / noop_max 0 / episodic_life false / fire_reset false
- `AtariEnv.classic`: sticky 0.0 / noop_max 30 / episodic_life true / fire_reset true
- `AtariEnv.100k`: sticky 0.0 / noop_max 30 / episodic_life true / fire_reset false（Atari-100k ベンチ。予算 100k steps = 400k frames は Run 設定側で指定。100k の共通不変は sticky なし+400k frames のみで、torch 系多数派＝SPR/EfficientZero の条件に合わせた。Dopamine 系＝DrQ(ε)/DER(ε)/BBF は noop 0・episodic_life false と分裂しているため、比較先の実装系統を必ず確認する）

標準準拠ノート（2026-08-18 に SB3 `atari_wrappers.py` / Gymnasium `AtariPreprocessing` / ale-py v5 登録 / rlpyt と突き合わせて監査済み）:

- 標準実装は 2 系統ある。**baselines 系**（OpenAI Baselines 2017 → SB3 → CleanRL、および rlpyt＝SPR/EfficientZero の土台。episodic_life=true で運用）と **Dopamine 系**（Dopamine → Gymnasium `AtariPreprocessing` → dqn_zoo。terminal_on_life_loss=False で運用）。本 env は wrapper 挙動（noop の乱数範囲と適用位置、FIRE 系列、max-pool 対象、sign clip、grayscale=ALE 直、INTER_AREA、episodic_life 判定）をこの 2 系統と一致させてある。系統間で割れる箇所は「その分岐を実際に使う側」に合わせる（例: episodic_life の `lives > 0` ガードは baselines 系に一致。§4.5）。
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
  grid  = ResizeGrayscale(frame, 210, 160, S)        # area 補間 → round/clamp → uint8 [1,S,S]
```

前処理は `AtariPreprocess` の free 関数（named namespace）とし、ALE 非依存の golden テストで数値を固定する:

- `PixelwiseMax(a, b, out)` — 210×160 グレー 2 枚の要素 max
- `ResizeGrayscale(src, src_h, src_w, dst_size)` — `torch::from_blob`（uint8）→ `interpolate(mode=area)` → `round().clamp(0,255).to(kUInt8)`。area 補間は慣行実装（cv2 INTER_AREA）と同系

### 4.4 行動

- 既定は `getMinimalActionSet()`（ROM ロード後に取得）。`full_action_space=true` で `getLegalActionSet()`（常に 18）。
- `Step(action)` の action は**選択済み集合の index**（ALE Action enum 値ではない）。
- `ActionSpec.value_labels` は `ale::action_to_string` 由来の行動名（NOOP/FIRE/RIGHT/LEFT/...）。
- ALE に per-step 合法手は存在しない（全行動が常時受理され、無効行動は no-op として振る舞う）。

### 4.5 報酬と終端

- `reward_clip=true`（既定）のとき Step の reward は `sign(reward_raw)`（学習報酬）。生スコアは env 内部で累積し、実 game over / truncation で `game_score` として確定する（生スコア／学習報酬の区別は `CONTEXT.md`）。キー名が `episode_` ではなく `game_` なのは、`episodic_life=true` のとき RL のエピソード境界（life 単位）と確定境界（ゲーム 1 回）が一致しないため。
- `done = ale.game_over(false)`（ALE 側 truncation は無効化済みなので純粋な terminal）。real game overだけが skip 窓の早期終了理由になる。
- `truncated` = `ale.getEpisodeFrameNumber()` が `max_episode_frames` に到達（done 優先）。自前 frame counter は持たず、Reset中のNOOP/FIREもALE frameへ含める。
- `episodic_life=true` の場合: skip 窓完走後に lives を1回だけ比較し、**ライフ減少かつ残 lives > 0** なら done=true を返す。`lives > 0` ガードは baselines 由来（SB3 / CleanRL / rlpyt と同一の条件式）。Qbert 系は game over の数フレーム前から lives=0 を報告するため、ガードが無いと実終端の直前に偽の life-loss done が 1 回出る（Gymnasium `AtariPreprocessing` はガード無しだが、あちらは `terminal_on_life_loss=False` 既定で分岐が実質使われない。episodic_life を実際に true で運用する系統は全てガード有り）。life loss後の残り反復と報酬も同じStepに含める。直後の `Reset()` は実 game over でなければ **soft-reset**（`ale.reset_game()` せず NOOP 1 フレームで観測のみ更新。game_score・ALE episode frameは継続）。**`fire_reset=true` のときは続けて FIRE 系列（§4.5.1）を打つ**（計 3 フレーム。報酬は game_score へ加算）。Breakout 系はライフ喪失でボールが消え FIRE を押すまで再投入されないため、これが無いとエージェントが自力で FIRE を選ぶまで無報酬 step が流れる。標準の wrapper 構成が `FireResetEnv` を `EpisodicLifeEnv` の外側に置き life-loss 後の reset でも FIRE を入れるのに合わせた契約。`noop_max` は soft-reset では打たない（`NoopResetEnv` 相当は実 reset のみに効く）。auto-reset wrapper（done||truncated → Reset）の契約とこの soft-reset で「学習上の episode = life 単位、メトリクス上の単位 = 実ゲーム 1 回」を両立する。`game_score` / `game_len` / `game_frames` の確定は実 game over / truncation 時のみ。この結果、`episodic_life=true` では eval の記録周期がライフ数倍に伸びる（値は正しく、周期だけが変わる）。

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
| GetScalar | `game_score` | 実 game over / truncation | 生スコアのゲーム 1 回分の合計（未確定 step は NaN） |
| GetScalar | `game_len` | 同上 | agent step 数 |
| GetScalar | `game_frames` | 同上 | エミュレータフレーム数 |
| GetScalar | `hns57` | 同上 | 人間正規化スコア %（57 ゲーム表。§4.8） |
| GetScalar | `hns49` | 同上 | 人間正規化スコア %（49 ゲーム表。§4.8） |
| GetScalar | `lives` | 常時 | 現在ライフ |
| GetTensor | `rgb_frame` | 常時（`retain_rgb_frame=true` 時） | 直近 Step の RGB 画面 uint8 `[3, 210, 160]`（CHW） |

バッチ集約（`mean.` 等の prefix）と NaN 慣行は wrapper の共通規約に従う。`GetConfigData()` は実効 config を返す（Run の `config/env.*.txt` ダンプ対象）。

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

- 確定タイミングは `game_score` と同じ（実 game over / truncation のみ。未確定は NaN）
- **基準表に載らないゲームは NaN**（`std::nullopt` ではない）。`nullopt` は `DiscreteBatchEnvBase` の集約でバッチ全体を打ち切ってしまうため。ALE は 104 ゲームを持つのに対し標準表は 57 / 49 しか覆わず、57 表のみに載る 8 ゲーム（berzerk / defender / phoenix / pitfall / skiing / solaris / surround / yars_revenge）は `hns49` が NaN になる
- 対象は生スコア。reward clip 後の学習報酬ではない
- 49 表の値は Extended Data Table 2 の "Normalized DQN (% Human)" 列で検算済み（`AtariEnv_test.cpp` の `[atari][hns]`）

CHNS（capped HNS、Agent57）と human gap（IQN Table 1、`gap = min(max(1 - HNS, 0), 1)`）は env からは出さない。両者は `gap = 1 - CHNS` の恒等関係にあり、単一ゲームの時系列では値が飽和して動かない（Breakout は常に 100% / 0）。キャップの意義は複数ゲーム集計での外れ値抑制にあるため、HNS からの導出として後処理側で扱う。

## 5. AtariView

`ViewBase<AtariData, AtariPanel>` 構成（ImageClsView と同型）。`GetTargetClassId() == "AtariEnv"`。

### 5.1 表示内容

- **主表示: 生 RGB 画面**（210×160、`env->GetTensor("rgb_frame", 0)`）。人間の観戦とデバッグの主対象。
- **副表示: 前処理後 obs**（S×S グレー、TrainEvent の step_result から lane 0 の grid を取得）。「エージェントが見ているもの」の確認用で、resize / max-pool の不具合を生画面との見比べで検出できる。
- 両者を並置し、整数倍拡大（ドット絵の視認性優先）で描画する。
- **オーバーレイ（テキスト行）**: game_score（生・累積中の暫定値）/ lives / ゲーム内 step・frame / 直前 action 名（value_label）/ 直前 reward（clip 後）。

### 5.2 データ経路と制約

- 表示対象は **lane 0 固定**（全 lane 同一ゲーム・同一設定のため代表 1 lane で挙動確認の目的を満たす。lane 選択 UI は将来拡張）。
- RGB は env が Step 毎に `getScreenRGB` を保持して公開する（`retain_rgb_frame`、既定 true。保持コストは約 100KB memcpy/step/lane で数十 µs）。ヘッドレス運用や性能検証で切りたい場合は config で false。
- 更新は既存 View 機構の標準契約（`UIDataStore`、既定 `force_update_interval_ms=200` ≈ 5Hz。TrainEvent 毎に描画するわけではない）。
- **描画基盤は wxGLCanvas**（HeatMapPanel / Conv2dPanel の既存 GL pane パターンを流用。テクスチャ 2 枚 + GL_NEAREST で整数倍拡大）。採用理由は既存パターンとの整合と、将来のスムーズ観戦モード（update interval 短縮による 30〜60fps 描画、SDL 窓の実用代替）への布石であり、**性能ではない**——既定 5Hz・210×160 規模では GDI（wxBitmap 流）でも性能問題は構造的に発生しないことを確認済み。テキストオーバーレイは wx 側で重畳描画する。スムーズ観戦モード自体（interval の config 化）は本仕様の対象外（将来拡張）。
- SDL の観戦ウィンドウ（`display_screen`）とは独立・併用可（View = Runner pane 統合、SDL = ALE ネイティブ・音声付き）。

## 6. ビルド統合

- 三値 cache 変数 `ANET_ENABLE_ATARI` = `AUTO`（既定）/`ON`/`OFF`。AUTO は `ALE_ROOT` の存在と `src/ale/ale_interface.hpp` の実在で判定。ON で不備なら configure エラー。判定結果は STATUS ログへ。
- 有効時のみ: `add_subdirectory(core/envs/atari1)`、runner へ `AtariEnv` リンク + `ANET_HAS_ATARI` 定義、`RunnerApp.cpp` の `#ifdef ANET_HAS_ATARI` で `InitAtari()`。
- ale.lib は `$<IF:$<CONFIG:Debug>,Debug,Release>` で選択（RelWithDebInfo は Release 側。IDL/CRT 整合）。
- テストは module と同時ビルド。ROM 依存ケースは `ATARI_ROM_DIR` 未解決なら Catch2 `SKIP()`。前処理 golden テストは常時実行。

## 7. 関連文書

- [環境](120_environments.jp.md) — Env 共通 contract と追加チェックリスト（§8）
- ADR 0025 / ADR 0026 — 導入判断
- `docs/memo/051_atari_ale_env_10prd.md` — 実装計画（PH1 スコープ・受入基準）
- `reports/atari_env_survey_2026-08-13.md` — ALE サーベイ（プロトコル・ベンチマーク動向）
- `CONTEXT.md` — 用語（sticky actions / flavor / 生スコア / プロトコルプリセット）
