<!-- translated-from: 220_atari_env.jp.md blob:6dd78b83dcc6d1cf98e96364e2ffc15128a62037 date:2026-09-19 progress:done -->
# Atari Env (ALE Integration)

> Primary perspective: concrete implementation specification (AtariEnv, including external ALE contracts, configuration keys, and View)

## 1. Introduction

### 1.1 Purpose

This document describes `AtariEnv`, which integrates ALE (Arcade Learning Environment) directly in C++. As a concrete specification built on the common Env contract ([Environments](120_environments.en.md)), it records ALE configuration-key contracts, the preprocessing chain, termination semantics, and AtariView.

See ADR 0025 (external-reference builds and licensing) and ADR 0026 (Single seam and preprocessing ownership) for adoption decisions, and `docs/memo/051_atari_ale_env_10prd.md` for the implementation plan. `CONTEXT.md` is authoritative for terminology (sticky actions / flavor / raw score / protocol preset).

### 1.2 Intended Audience

- Developers implementing or modifying AtariEnv
- Developers checking the impact of ALE configuration-key or version changes
- Users checking Atari experimental protocols (presets)

### 1.3 Scope

AtariEnv / AtariEnvFactory / AtariView / preprocessing functions, ALE v0.12.0 configuration-key contracts, and build gating. Section 3 gives the established ALE build configuration; `reports/atari_env_survey_2026-08-13.md` covers experimental protocol background.

## 2. Overview

```
BatchEnvBuilder (env.class_id = AtariEnv)
  └─ AtariEnvFactory (SingleDiscreteEnvFactory)
       └─ AtariEnv × N lanes（並列・auto-reset は既存 wrapper）
            ├─ ale::ALEInterface（1対1、ALE 側 frame_skip=1 固定）
            └─ 前処理: 自前 skip ループ → grayscale 2フレーム max-pool → area resize → uint8 [1,S,S]
```

- Does not produce frame stacks. The existing Actor-side stacker and ReplayBuffer-side stack_count jointly handle stacking.
- RB stores individual uint8 frames; stacks are expanded during sampling (ADR 0026).
- Responsibility: **AtariEnv drives one ALE instance and performs standard preprocessing**. Existing framework mechanisms handle parallelism, stacking, autoreset, and seed distribution.

## 3. External Dependency (ALE v0.12.0)

### 3.1 Established Build Configuration

| Item | Value |
|---|---|
| Version | Pinned to tag `v0.12.0`; vcpkg dependencies pinned by the manifest's builtin-baseline |
| Location | Outside the repository, referenced through `ALE_ROOT`; do not use `third_party/`, which is reserved for bundled components |
| Configure | `-DBUILD_PYTHON_LIB=OFF -DSDL_SUPPORT=ON -DSDL_DYNLOAD=ON -DVCPKG_TARGET_TRIPLET=x64-windows-static-md` |
| Build | Both Release and Debug; mixed linking is unsupported because MSVC IDL/CRT must match |
| Consumer linking | `%ALE_ROOT%/build/src/ale/<Config>/ale.lib` (all-in-one static library) + zlib (add `build/vcpkg_installed/x64-windows-static-md` to `CMAKE_PREFIX_PATH` and use `find_package(ZLIB)`) |
| Include | `%ALE_ROOT%/src` + `%ALE_ROOT%/src/ale` + `%ALE_ROOT%/build/src/ale` (generated `version.hpp`) |
| SDL | `SDL_DYNLOAD=ON` is required on Windows: ALE's `common/SDL2.hpp` declares SDL functions dllimport under `WIN32 && !SDL_DYNLOAD`, preventing static SDL linking. There is no link-time SDL dependency; `SDL2.dll` loads at runtime only when `display_screen`/`sound` is enabled. Placing the DLL in `third_party/runtime_dlls/` deploys it to runner bin during builds (see that README; available from sources such as ale-py wheels, under the zlib license) |
| Install | Unused: upstream install rules are guarded by `if(UNIX ...)` and do nothing on Windows. Direct build-tree references are standard |

### 3.2 ALE Configuration-Key Contract (Verified on v0.12.0)

ALE is configured through `setBool/setInt/setFloat/setString`; **unregistered keys throw and fail fast** (`verifyVariableExistence` in `emucore/Settings.cxx`). The complete set of configurable keys is defined by `Settings::setDefaultSettings()` (external) and the constructor (internal, inherited from Stella). The following lists all v0.12.0 keys and their treatment in AtariEnv.

**External keys (registered by `setDefaultSettings()`)**

| Key | Type | ALE default | Meaning | AtariEnv handling |
|---|---|---|---|---|
| `random_seed` | int | -1 | Env RNG (sticky actions, etc.); -1 uses time automatically | **Set from lane seed** (§4.6; mapped to a range avoiding the -1 sentinel) |
| `repeat_action_probability` | float | 0.25 | Sticky actions | **Passed through from config** (`AtariEnv.repeat_action_probability`) |
| `frame_skip` | int | 1 | ALE built-in skipping (simple repetition without max-pooling) | **Fixed at 1**; custom skipping needs intermediate frames for max-pooling |
| `max_num_frames_per_episode` | int | 0 | ALE episode frame limit (mixes truncation into game_over) | **Fixed at 0**; custom truncation counting strictly separates it from done |
| `max_num_frames` | int | 0 | Total frame limit across all episodes | Untouched (remains 0) |
| `truncate_on_loss_of_life` | bool | false | Truncate an episode on life loss | **Fixed at false**; custom `episodic_life` uses done + soft-reset, with different semantics: ALE truncation versus classic terminal behavior |
| `color_averaging` | bool | false | Average the latest two frames (older alternative to max-pooling) | Fixed at false; custom max-pooling handles flicker |
| `reward_min` / `reward_max` | int | INT_MIN / INT_MAX | ALE reward clamping | Untouched; sign clipping is custom, and clamp differs from sign |
| `display_screen` | bool | false | SDL spectator window | **Passed through from config** |
| `sound_obs` | bool | false | Audio observations (enables `getAudio()`, no SDL required) | Unused; future extension: audio observation key |
| `cpu` | string | "low" | Stella CPU emulation fidelity (low prioritizes speed) | Default retained |
| `system_random_seed` | int | 4753849 | Internal Stella System RNG; fixed default means deterministic | Untouched |
| `paddle_min` / `paddle_max` | int | -1 | Physical paddle range for continuous actions | Unused; continuous actions are outside scope |
| `restricted_action_set` | bool | false | Legacy FIFO interface setting | Unused; minimal set obtained through `getMinimalActionSet()` |
| `run_length_encoding` | bool | true | FIFO controller setting | Irrelevant |
| `send_rgb` | bool | false | FIFO setting | Irrelevant |
| `rom_file` | string | "" | Set internally by `loadROM()` | Not modified directly |
| `record_screen_dir` | string | "" | Automatic sequential PNG screen recording | Unused; future spectator recording |
| `record_sound_filename` | string | "" | Audio recording | Unused |
| `fragsize` | int | 64 | Fragment size for audio synchronization | Default retained |

**Internal keys (from Stella, registered in the constructor)**

| Key | Type | Default | Meaning | AtariEnv handling |
|---|---|---|---|---|
| `sound` | bool | false | SDL audio output | **Passed through from config** (`AtariEnv.sound`, independent of `display_screen`) |
| `palette` | string | "standard" | Color palette (standard/z26/user) | Default retained |
| `freq` / `tiafreq` | int | 31400 | Audio sampling | Default retained |
| `volume` | int | 100 | Volume | Default retained |
| `clipvol` | bool | true | Volume clipping | Default retained |

Contract notes:

- AtariEnv does not set untouched keys. Changes to ALE defaults must be reviewed when updating the tag; this table is the v0.12.0 baseline.
- Set values such as `setInt("random_seed", ...)` **before** loading the ROM: `loadROM()` applies settings when constructing the environment.
- Sticky actions are decided every frame inside the skip loop (the act loop in `environment/stella_environment.cpp`). Custom skipping—calling act() k times with frame_skip=1—still evaluates per act(), hence per frame, preserving the original semantics of Machado et al. 2018.

### 3.3 ROMs

- ROMs (.bin) are not bundled; users supply them. Filenames use snake_case (`pong.bin`).
- Resolution order: nonempty `AtariEnv.rom_dir`, then environment variable `ATARI_ROM_DIR`. If `<rom_dir>/<game>.bin` does not exist, Env construction fails fast, including search paths and configuration methods in the error.
- `loadROM()` identifies supported games by ROM md5, so ALE also detects unsupported/corrupt ROMs.

### 3.4 Licensing (ADR 0025)

ALE is GPL-2.0. The main project's Apache-2.0 license is maintained through (1) no bundling, using external references; (2) optional builds; (3) exclusion from release packages (`ANET_ENABLE_ATARI=OFF` for release builds); and (4) a GPL notice in `core/envs/atari1/NOTICE.md`.

## 4. AtariEnv Specification

### 4.1 Module Structure

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

The class_id is `AtariEnv`. `AtariEnv : public SingleDiscreteEnvBase, public anet::RandomHolder`.

### 4.2 Configuration Keys (Prefix `AtariEnv`)

| Key | Type | Default | Meaning |
|---|---|---|---|
| `game` | string | Required | ROM name (snake_case, without extension) |
| `rom_dir` | string | `""` | Takes precedence over `ATARI_ROM_DIR` when nonempty |
| `screen_size` | int | 84 | Output resolution S (square) |
| `frame_skip` | int | 4 | Emulator frames per step (custom skip loop) |
| `max_pool` | bool | true | Pixelwise maximum of the last two frames in the skip window |
| `repeat_action_probability` | float | 0.25 | Sticky actions, passed through to ALE |
| `noop_max` | int | 0 | Insert a random 1..noop_max NOOPs on Reset; 0 disables |
| `fire_reset` | bool | false | One FIRE immediately after Reset; no-op for games without FIRE |
| `episodic_life` | bool | false | Expose life loss as done to learning (§4.5) |
| `reward_clip` | bool | true | Sign-clip Step rewards |
| `full_action_space` | bool | false | True selects all 18 legal actions |
| `mode` / `difficulty` | int | -1 | -1 uses ALE defaults; otherwise `setMode`/`setDifficulty` (flavor) |
| `max_episode_frames` | int | 108000 | Custom truncation threshold; 0 disables |
| `retain_rgb_frame` | bool | true | Retain the RGB screen each Step and expose it through `GetTensor("rgb_frame")` for AtariView (§5) |
| `display_screen` | bool | false | SDL spectator window, passed through to ALE |
| `sound` | bool | false | SDL audio, passed through to ALE independently of display |

Protocol presets (defined in `Atari.txt`, selected through the `AtariEnv.$` selection chain; default = `v5_noop0`):

- `AtariEnv.v5_noop0`: sticky 0.25 / noop_max **0** / episodic_life false / fire_reset false (raw ALE v5 behavior; sticky actions are the only source of stochasticity)
- `AtariEnv.v5_noop30`: sticky 0.25 / noop_max **30** / episodic_life false / fire_reset false (v5 + Gymnasium `AtariPreprocessing` defaults)
  - **There is deliberately no standalone `v5` preset.** v5 identifies an env ID generation, not whether NOOP starts are enabled; allowing selection by this name alone could silently mismatch comparison conditions. The only difference is the `noop_max` line; see the implementation-lineage discussion below.
- `AtariEnv.classic`: sticky 0.0 / noop_max 30 / episodic_life true / fire_reset true
- `AtariEnv.100k`: sticky 0.0 / noop_max 30 / episodic_life true / fire_reset false (Atari-100k benchmark; specify the budget of 100k steps = 400k frames in Run configuration. The only common invariants of 100k are no sticky actions and 400k frames; this preset follows the torch-majority SPR/EfficientZero conditions. Dopamine-based DrQ(ε)/DER(ε)/BBF instead use noop 0 and episodic_life false, so always check the comparison implementation's lineage.)

Standard-conformance notes (audited on 2026-08-18 against SB3 `atari_wrappers.py`, Gymnasium `AtariPreprocessing`, ale-py v5 registration, and rlpyt):

- There are two standard lineages: **baselines** (OpenAI Baselines 2017 → SB3 → CleanRL, plus rlpyt, the basis of SPR/EfficientZero; uses episodic_life=true) and **Dopamine** (Dopamine → Gymnasium `AtariPreprocessing` → dqn_zoo; uses terminal_on_life_loss=False). This env matches their wrapper behavior: NOOP random ranges and placement, FIRE sequences, max-pooling inputs, sign clipping, direct ALE grayscale, INTER_AREA, and episodic_life decisions. Where the lineages differ, follow the one that actually uses that branch; for example, the episodic_life `lives > 0` guard follows baselines (§4.5).
- **v5 differs from Machado et al. 2018 recommendations in one respect**: Machado recommends the full 18-action set, while ale-py v5 defaults to `full_action_space=False` (minimal). This env's v5 presets follow ale-py (false).
- **NOOP 0 in v5 depends on lineage**: raw ale-py v5 has no NOOP starts (Dopamine/Machado lineage), whereas Gymnasium's `AtariPreprocessing` wrapper defaults to `noop_max=30`. Even v5 examples using standard Gymnasium preprocessing run with NOOP 30; check the comparison wrapper configuration.
- Classic `max_episode_frames=108000` differs by 8% from historical v4 (gym registration TimeLimit=100,000 frames). SB3 `AtariWrapper` itself has no limit, so 108,000 is used to match v5.
- Two known standard quirks are not followed: max-pooling when termination occurs midway through a skip window (standard implementations pool against zero/old frames; this env uses only executed frames, affecting only the single observation immediately after done), and rewards during soft-reset (SB3 discards them entirely; this env adds them to `game_score`, yielding a more accurate raw score).

Configuration contract notes:

- A single square `screen_size` follows standard preprocessing (84×84), rather than a technical limitation. Intentionally distorting the aspect ratio from 210×160 to 84×84 is conventional; letterboxing to preserve it would be nonstandard. If rectangular output becomes necessary, split into `screen_height`/`screen_width` as a compatible change confined to the env.
- **SDL audio on Windows requires the DirectSound backend.** ALE's `SoundSDL` uses legacy `SDL_OpenAudio(desired, obtained)` assuming AUDIO_U8, but this API with non-NULL `obtained` returns the hardware format **without format conversion**. Default WASAPI always returns float32/stereo, causing U8 samples to be interpreted as float32 and resulting in **near silence** (an upstream ALE v0.12.0 bug, measured with the sandbox beep test: WASAPI → silent at `format=0x8120`; directsound → normal at `format=0x8`). Therefore, **when `sound=true` and `SDL_AUDIODRIVER` is unset, AtariEnv automatically sets it to `directsound` before SDL initialization** (`EnsureSdlAudioDriverDefault`, once per process). Explicit environment settings are respected, allowing WASAPI testing through overrides. The proper upstream fix is passing `obtained=NULL` for SDL automatic conversion, a PR candidate. **SDL audio also assumes real-time operation at 60 emulator frames/s**; faster operation accumulates audio register queues and triggers pruning in `SoundSDL::processFragment`, producing distortion. EvalPanel runs at real speed with `app.online.eval_panel.fps = 15` × `frame_skip=4`. The TIA source itself is harsh lo-fi—for example, Space Invaders marching sounds are buzzes—and this is expected. Diagnostic modes in the sandbox (`C:/dev/ale-sandbox`) include `beep` for SDL output format/rate checks and `wav`/`wavnoop` for pure TIASound synthesis to WAV without SDL.
- `display_screen` is for **sandbox and one-off debugging** (assuming `num_envs=1` + `env.worker_type=SINGLE_THREAD`). SDL windows are created per ALEInterface instance (`ScreenSDL` creation in `OSystem.cxx`), so N lanes open N windows. SDL video/events are also not thread-safe; SDL calls from ThreadPool workers are unsupported. AtariView (§5) is the proper Runner spectator interface. The env emits one log.warn when `display_screen=true`. It does not restrict windows to lane 0, because identifying the lane would require name parsing, violating the `CONTEXT.md` rule against using Env names to determine behavior.

### 4.3 Observations and Preprocessing Chain

Observations contain only the grid key: `TensorSpec{ Grid, {1, S, S}, kUInt8, num_classes=0, min=0, max=255 }`. As continuous uint8 data, they are automatically converted to float with /255 at the NN boundary (`NetworkBoundaryPreprocessor::Format`). No vector / action_mask keys are emitted.

```
Step(action_index):
  a = action_set[action_index]
  reward_raw = Σ ale.act(a)                         # sticky は ALE 内で毎フレーム判定
      （当該Step内でrolling 2-slotに画面を取得し、real game overだけ早期終了）
  frame = max_pool ? 当該Stepで実行できた最後の最大2フレームのmax : 最終フレーム
  grid  = ResizeGrayscale(frame, 210, 160, S)        # area 補間 → round/clamp → uint8 [1,S,S]
```

Preprocessing uses free functions in `AtariPreprocess` (a named namespace), with numerical behavior fixed by ALE-independent golden tests:

- `PixelwiseMax(a, b, out)` — elementwise maximum of two 210×160 grayscale frames
- `ResizeGrayscale(src, src_h, src_w, dst_size)` — `torch::from_blob` (uint8) → `interpolate(mode=area)` → `round().clamp(0,255).to(kUInt8)`. Area interpolation belongs to the same family as conventional cv2 INTER_AREA implementations.

### 4.4 Actions

- Defaults to `getMinimalActionSet()`, obtained after ROM loading. `full_action_space=true` selects `getLegalActionSet()` (always 18).
- The action passed to `Step(action)` is an **index into the selected set**, not an ALE Action enum value.
- `ActionSpec.value_labels` uses action names from `ale::action_to_string` (NOOP/FIRE/RIGHT/LEFT/...).
- ALE has no per-step legal-action set: all actions are always accepted, and ineffective actions behave as no-ops.

### 4.5 Rewards and Termination

- With `reward_clip=true` (default), the Step reward is `sign(reward_raw)` (training reward). Raw scores accumulate inside the env and are finalized as `game_score` at real game over / truncation; see `CONTEXT.md` for raw score versus training reward. The key uses `game_`, not `episode_`, because with `episodic_life=true`, RL episode boundaries (individual lives) differ from score finalization boundaries (one game).
- `done = ale.game_over(false)` is a pure terminal condition because ALE truncation is disabled. Only real game over ends the skip window early.
- `truncated` means `ale.getEpisodeFrameNumber()` has reached `max_episode_frames`, with done taking precedence. There is no separate frame counter; NOOP/FIRE during Reset also count as ALE frames.
- With `episodic_life=true`, compare lives once after completing the skip window and return done=true for **life loss with remaining lives > 0**. The `lives > 0` guard comes from baselines and matches SB3 / CleanRL / rlpyt. Qbert-like games report lives=0 several frames before game over; without this guard, a false life-loss done would occur just before real termination. Gymnasium `AtariPreprocessing` lacks the guard, but defaults to `terminal_on_life_loss=False`, so the branch is effectively unused; every lineage that actually uses episodic_life=true includes the guard. Remaining repetitions and rewards after life loss belong to the same Step. The next `Reset()` performs a **soft-reset** unless real game over occurred: no `ale.reset_game()`, only one NOOP frame to update the observation, retaining game_score and ALE episode frames. **If `fire_reset=true`, follow with the FIRE sequence (§4.5.1)**, for three frames total, adding rewards to game_score. In Breakout-like games, losing a life removes the ball until FIRE is pressed; without this sequence, rewardless steps continue until the agent selects FIRE itself. This matches standard wrappers placing `FireResetEnv` outside `EpisodicLifeEnv`, thus applying FIRE after life-loss resets too. `noop_max` is not applied during soft-reset; its `NoopResetEnv` equivalent affects real resets only. The auto-reset wrapper's done||truncated → Reset contract and this soft-reset jointly support life-based learning episodes and real-game-based metrics. `game_score` / `game_len` / `game_frames` finalize only at real game over / truncation. Configured `eval1` / `eval2` and EvalPanel's `eval_panel` explicitly set `env.episodic_life=false`, aligning evaluation-session episode boundaries with real game over / truncation.

#### 4.5.1 FIRE Sequence (`fire_reset`)

With `fire_reset=true`, Reset performs these two actions in order, matching stable-baselines3 `FireResetEnv` (`step(1)` followed by `step(2)`):

1. `PLAYER_A_FIRE`
2. **The third action in the action set** (index 2)

After each, call `reset_game()` if game over has occurred. The second action accommodates games that do not start with FIRE alone; it is **selected by index regardless of meaning** (RIGHT in Breakout, UP in Seaquest).

Two applicability conditions correspond to upstream assertions:

- The action set contains at least three elements.
- Action-set index 1 is `PLAYER_A_FIRE`.

ALE action sets follow `Action` enum order (NOOP=0, FIRE=1, UP=2, RIGHT=3, ...), so games containing FIRE always have it at index 1. Games failing these conditions, such as Freeway's minimal set without FIRE, **receive no actions**. This is equivalent to upstream applying the wrapper only when `if "FIRE" in get_action_meanings()` holds.

**This safeguard does not work with `full_action_space=true`.** All games then have the fixed 18 actions including FIRE, so Reset applies the FIRE sequence even to games that do not normally use FIRE. Understand this side effect before enabling both `full_action_space` and `fire_reset`.

The sequence is called from hard-reset (at the end of `ApplyResetActions()`, after the `noop_max` draw) and soft-reset (§4.5). Reward handling differs: hard-reset discards it by zeroing `game_score` immediately afterward; soft-reset adds it to `game_score` because the game continues.

### 4.6 Seeds and Reproducibility

- Pass the uint64 lane seed from `SeedMaker::MakeIndexedSeed(i)` to ALE using `setInt("random_seed", static_cast<int>(seed & 0x7FFFFFFF))`, mapping to a nonnegative range that avoids the -1 auto sentinel. Set it before loading the ROM.
- Env randomness (only the noop_max count draw) uses `RandomHolder` initialized with the lane seed, separately from ALE's internal sticky-action RNG.
- Reproducibility contract: same seed + same action sequence → identical observation, reward, and termination sequences, including sticky actions and NoOp resets.

### 4.7 Env Accessors (Module Interface)

| API | Key | Finalization time | Value |
|---|---|---|---|
| GetScalar | `game_score` | Real game over / truncation | Total raw score for one game; NaN on unfinalized steps |
| GetScalar | `game_score.ge.[N]` | Same | 1 if `game_score >= N`, otherwise 0; NaN on unfinalized steps. `N` is parsed as float and may be negative |
| GetScalar | `game_len` | Same | Agent step count; episode boundaries differ from generic `episode_steps` under `episodic_life` |
| GetScalar | `game_frames` | Same | Emulator frame count |
| GetScalar | `hns57` | Same | Human-normalized score percentage, using the 57-game table (§4.8) |
| GetScalar | `hns49` | Same | Human-normalized score percentage, using the 49-game table (§4.8) |
| GetScalar | `lives` | Always | Current lives |
| GetTensor | `rgb_frame` | Always when `retain_rgb_frame=true` | Latest Step RGB screen, uint8 `[3, 210, 160]` (CHW) |

Batch aggregation (prefixes such as `mean.`) and NaN handling follow common wrapper conventions. `GetConfigData()` returns effective configuration, dumped to the Run's `config/env.*.txt`.

**Watch the aggregation denominator.** Keys finalized at real game over / truncation return NaN on unfinalized steps; batch aggregation excludes NaNs from its denominator (`core/anet-core/src/util.cpp`; if all envs are NaN, the result is NaN). Thus, the denominator of `mean.game_score` is **the number of envs that actually finished a game at that step**, not num_envs. It averages multiple games only when multiple envs finish simultaneously. The simultaneous-completion rate is λ = num_envs / mean game length. In Breakout (128 envs / about 1,900 steps, λ ≈ 0.07), measurements showed 97% single completions, so the `mean.` series is effectively a sequence of individual game scores. Shorter games experience more averaging and flattened peaks; compare games with `max.` alongside it. Only `lives` is always finalized, so its denominator is num_envs and it is a true batch mean.

`game_score.ge.[N]` is a 0/1 metric sharing this denominator with `game_score`. Its `mean.` is the fraction of envs finishing at that step whose scores were at least N. Window averaging yields a **per-game proportion**, rather than a per-firing-event proportion, avoiding both the flattening of `mean.` and saturation of `max.` when inspecting threshold-crossing frequency: `mean.` flattens the ends of a bimodal distribution, while `max.` saturates as evaluation-session episode count increases. Game-dependent thresholds, such as one Breakout screen = 432, are key parameters rather than entries in the HNS tables of §4.8. This allows multiple simultaneous thresholds in one Run and reflects that thresholds are analytical criteria, not physical game constants.

### 4.8 Human-Normalized Score (HNS)

This metric normalizes raw score relative to human players. It makes achievement intuitive even for one game (100 = human) and supports aggregation across games. The formula matches reference implementations (DeepMind `dqn_zoo`, Dopamine, IQN, Agent57); **the denominator is not made absolute**:

```
hns = 100 * (game_score - random) / (human - random)
```

**Maintain two reference tables and always expose both.** Values differ materially even for the same game, so mixing them invalidates literature comparisons:

| Key | Table | Source | Game count |
|---|---|---|---|
| `hns57` | 57-game table | Wang et al. 2016 (Dueling) lineage; identical to `_ATARI_DATA` in DeepMind `dqn_zoo/atari_data.py` | 57 |
| `hns49` | 49-game table | Mnih et al. 2015 (Nature), Extended Data Table 2 | 49 |

For example, Pong's human baseline is 14.6 in the 57-game table and 9.3 in the 49-game table. The same raw score of +7 yields HNS 78.5% versus 92.3%. Breakout uses 30.5 / 31.8. Rainbow, IQN, Agent57, and BBF all use the 57-game table, so use `hns57` for modern-paper comparisons. Use `hns49` when comparing with Nature DQN figures, such as Breakout 401.2 = 1327.2%.

Contract:

- Finalization matches `game_score`: only real game over / truncation; otherwise NaN.
- **Games absent from the reference table return NaN**, not `std::nullopt`, because `nullopt` terminates aggregation for the entire batch in `DiscreteBatchEnvBase`. ALE has 104 games, while the standard tables cover only 57 / 49. The eight games appearing only in the 57-game table—berzerk / defender / phoenix / pitfall / skiing / solaris / surround / yars_revenge—return NaN for `hns49`.
- Input is raw score, not reward-clipped training reward.
- The 49-game table was checked against the "Normalized DQN (% Human)" column of Extended Data Table 2 (`[atari][hns]` in `AtariEnv_test.cpp`).

The env does not expose CHNS (capped HNS, Agent57) or human gap (IQN Table 1, `gap = min(max(1 - HNS, 0), 1)`). They satisfy `gap = 1 - CHNS` and saturate in single-game time series, becoming uninformative (Breakout stays at 100% / 0). Capping is useful for suppressing outliers when aggregating games, so these are derived from HNS in postprocessing.

## 5. AtariView

Uses `ViewBase<AtariData, AtariPanel>`, the same structure as ImageClsView. `GetTargetClassId() == "AtariEnv"`.

### 5.1 Display Contents

- **Primary display: raw RGB screen**, 210×160, from `env->GetTensor("rgb_frame", 0)`. Main view for human spectating and debugging.
- **Secondary display: preprocessed observation**, S×S grayscale, using lane 0 grid from TrainEvent's step_result. Shows what the agent sees and helps detect resize/max-pooling errors by comparison with the raw screen.
- Display both side by side with integer scaling, prioritizing pixel-art clarity.
- **Overlay text**: game_score (raw provisional accumulated value) / lives / in-game step and frame / previous action name (value_label) / previous clipped reward.

### 5.2 Data Path and Constraints

- Display is **fixed to lane 0**. All lanes share the same game and configuration, so one representative lane suffices for behavior inspection; a lane-selection UI is a future extension.
- The env retains and exposes `getScreenRGB` each Step through `retain_rgb_frame`, default true. Retention costs about 100 KB memcpy/step/lane, taking tens of microseconds. Set false for headless operation or performance measurements.
- Updates follow the standard View contract: `UIDataStore`, default `force_update_interval_ms=200`, approximately 5 Hz. Rendering does not occur on every TrainEvent.
- **Rendering uses wxGLCanvas**, reusing the existing GL pane pattern from HeatMapPanel / Conv2dPanel: two textures and GL_NEAREST integer scaling. The rationale is consistency with existing patterns and preparation for a future smooth spectator mode (shorter update intervals for 30–60 fps, a practical SDL-window alternative), **not performance**. At the default 5 Hz and 210×160 size, GDI (wxBitmap-style rendering) has been confirmed not to pose a structural performance problem. wx draws the text overlay. Smooth spectator mode itself, including configurable intervals, is outside this specification and remains a future extension.
- Independent of the SDL spectator window (`display_screen`); both may be used together. View integrates into Runner panes; SDL provides native ALE display with audio.

## 6. Build Integration

- Three-valued cache variable `ANET_ENABLE_ATARI` = `AUTO` (default) / `ON` / `OFF`. AUTO checks `ALE_ROOT` and the existence of `src/ale/ale_interface.hpp`. ON with missing prerequisites produces a configure error. Log the decision at STATUS.
- Only when enabled: `add_subdirectory(core/envs/atari1)`, link `AtariEnv` to runner and define `ANET_HAS_ATARI`, then call `InitAtari()` under `#ifdef ANET_HAS_ATARI` in `RunnerApp.cpp`.
- Select ale.lib using `$<IF:$<CONFIG:Debug>,Debug,Release>`; RelWithDebInfo uses Release, matching IDL/CRT.
- Tests build with the module. ROM-dependent cases use Catch2 `SKIP()` if `ATARI_ROM_DIR` cannot be resolved. Preprocessing golden tests always run.

## 7. Related Documents

- [Environments](120_environments.en.md) — Common Env contract and extension checklist (§8)
- ADR 0025 / ADR 0026 — Adoption decisions
- `docs/memo/051_atari_ale_env_10prd.md` — Implementation plan (PH1 scope and acceptance criteria)
- `reports/atari_env_survey_2026-08-13.md` — ALE survey (protocols and benchmark trends)
- `CONTEXT.md` — Terminology (sticky actions / flavor / raw score / protocol preset)
