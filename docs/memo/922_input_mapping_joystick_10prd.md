# 操作入力の Joystick 対応と操作マッピングの設定化（EvalPanel）暫定 PRD

> 2026-11-03 番号を 999 から 922 へ振り直した（PRD114）。

> 暫定 PRD（グリル前のドラフト）。番号 999。§6 の未決を裁定してから正式採番する。本書は実装着手を意味しない。
> 起点: 2026-09-26。EvalPanel の行動上書きがキーボード固定（矢印 4 つ + テンキー 0〜9、対応表はコードに直書き）で、
> Atari の 18 行動や複合行動（`UPFIRE` 等）を打てず、Joystick も使えない。
> 関連: [923_input_target_train_forced_action_10prd.md](923_input_target_train_forced_action_10prd.md)（同日起票の対 PRD。入力先を Train へ切り替える。本 PRD の入力層をそのまま使う）、
> [081](081_evalpanel_episode_trace_10prd.md)（EvalPanel の手動行動ログ D6）、[083](done/083_action_context_retire_10prd.md) と [921](921_agent_prev_action_obs_10prd.md)（forced action を `MakeAction` 入力にする方針）、
> [160 §RunnerFrame](../design/160_applications_and_tools.jp.md)、[020 §6.2](../design/020_user_guide_run.jp.md)、[CONTEXT.md「実行時 UI 操作」](../../CONTEXT.md)。
> 担当: 未定（グリル時に決める。既定の担当は定めない）。

## 1. Context / Problem Statement

### 1.1 現行の経路（2026-09-26 確認）

- キー入力は、`anet::rl::gui::Panel` を継承する全パネル（`TrainPanel`、各 Env の View パネル `AtariPanel` / `LunarLanderPanel` 等）が
  `wxEVT_CHAR_HOOK` を `ForwardedKeyEvent` に包んで Frame へ転送し（[gui.cpp](../../core/anet-core/src/gui.cpp) の `Panel::OnKey`）、
  `RunnerFrame::OnKey`（[RunnerFrame.cpp](../../apps/runner/src/RunnerFrame.cpp)）が 1 つの `switch` で解釈する。
  **どのパネルにフォーカスがあってもキーは EvalPanel 行き**である。
- 対応表はコード直書き。`WXK_UP/DOWN/LEFT/RIGHT` → 0/2/1/3（コメントは LunarLander の NOOP / MAIN / LEFT / RIGHT）、
  `WXK_NUMPAD0`〜`9` → 0〜9。`SHIFT` = Train トグル、`SPACE` = Eval pause トグル、`CTRL` = Eval 1 step は行動ではなくシステム操作。
- 行動キーは `eval_panel_->DoStep(action)` を 1 回呼ぶ。つまり **1 押下 = 1 強制 step** で、EvalPanel が timer で回っていても
  pause 中でも同じ（timer の step は Actor が選び、押下のたびに強制 step が割り込む）。
- `QValuePanel` のグリッド行クリックも同じ `action_handler_` 経由で `EvalPanel::DoStep(action)` に入る（[RunnerFrame.cpp](../../apps/runner/src/RunnerFrame.cpp) の `Initialize`）。
- `EvalPanel::DoStep(action)` → `EvalRunner::DoStep(action)` → `DoStepInternal` が `actor_->MakeAction(source_counts_, state_)` を呼んだ後、
  `action >= 0` なら `action_info_raw->WithAction(tensor{action})` で差し替える（[trainer.cpp](../../core/anet-core/src/trainer.cpp)）。
  DQN の `WithAction` は Actor Q ヒントの行動 Q と Munchausen bonus を強制行動へ再 gather する（[dqn_based_agent.cpp](../../core/anet-core/src/dqn_based_agent.cpp) の `DQNActionInfo::WithAction`）。
  この事後差し替えは 921 の D1（2026-09-26）で「`MakeAction` の入力にする」と方針が決まっており、[083](done/083_action_context_retire_10prd.md) の後続 PRD が引き受ける。
- EvalPanel は 1 lane。timer は `app.eval_panel.fps` × `step_per_frame`（Atari は 14.5 fps × 1 = 実機速度）。

### 1.2 行動集合は Env ごとに違う（`ActionSpec.value_labels`）

| Env | 行動数 | ラベル |
|---|---|---|
| Atari | ゲーム依存（minimal action set。最大 18） | ALE の名前から `PLAYER_A_` を除いたもの（`NOOP` / `FIRE` / `UP` / `UPFIRE` / `DOWNLEFTFIRE` …）。**順序も個数もゲームで変わる** |
| LunarLander | 4 | `do nothing` / `fire left` / `fire main` / `fire right` |
| DropMerge | action mode 依存 | `NOOP` / `LEFT` / `DROP` / `RIGHT` / `F_LEFT` … / `DROP_<i>` |
| GridMaze | 4 | `Up` / `Left` / `Down` / `Right` |
| CartPole | 2 | `right` / `left` |
| ImageCls | クラス数 | クラス名 |

現行の固定表は LunarLander の 4 行動と「番号 0〜9 を直接打つ」しか表現できない。Atari の kung_fu_master は 14 行動で、
`DOWNRIGHTFIRE` のような複合行動を 1 キーに割り当てられない限りまともに遊べない。

### 1.3 Joystick の実装事実

- ビルドが使う wxWidgets 3.3.3（`C:/dev/wxWidgets-3.3.3/install`）は `wxUSE_JOYSTICK 1` で、`wx/joystick.h` が入っている。
  `wxJoystick` は `WXDLLIMPEXP_ADV` 指定だが 3.3 では `WXDLLIMPEXP_ADV` = `WXDLLIMPEXP_CORE`（`wx/dlimpexp.h`）なので、
  現在リンクしている `core` に含まれ、CMake の `find_package(wxWidgets ... core base gl aui)` は変更不要（実装時に再確認）。
- API: `wxJoystick::GetNumberJoysticks()` / `IsOk()` / `GetPosition()` / `GetButtonState()` / `GetPOVPosition()`、
  および `SetCapture(win, pollingFreq)` で `wxEVT_JOY_BUTTON_DOWN` / `_UP` / `wxEVT_JOY_MOVE` / `_ZMOVE` を受けられる。
  wxMSW の実装は WinMM（`joyGetPosEx`）の薄い包みで、XInput 系のパッドも互換層で見える。POV（十字キー）は角度値で返り中心は特別値
  （値域は実装時に確認）。
- 代替: XInput 直叩き（Windows 専用、Xbox 系パッド）、SDL2 GameController（`SDL2.dll` は ALE 用に `third_party/runtime_dlls/` へ置く運用だが、
  ヘッダとインポートライブラリは本ビルドに無い）。**最小は wxJoystick**。ボタン配列名の解決が要らない範囲ではこれで足りる。
- キー名の文字列化には wx の `wxAcceleratorEntry::FromString` / `ParseAccel`（`wx/accel.h`）があり、`Up` / `Down` / `Space` / `KP_0` のような wx 綴りを
  そのまま使えば独自のキー名表を持たずに済む。

### 1.4 Atari の行動の構造（ALE、2026-09-26 確認）

2600 の標準コントローラは 8 方向のデジタルスティックとボタン 1 個で、ALE の行動は「スティック 9 状態（中立 + 8 方向）× ボタン 2 状態」の 18 個。
名前は `UP` / `DOWN` と `RIGHT` / `LEFT` と `FIRE` の組み立てで、ALE も組み立てどおりに方向ごとのイベントと fire イベントを別々に立てる
（`C:/dev/ALE/src/ale/environment/ale_state.cpp` の `ALEState::applyActionJoysticks`）。番号は ALE の `Action` enum。

| スティック | ボタンなし | ボタンあり |
|---|---|---|
| 中立 | 0 `NOOP` | 1 `FIRE` |
| 上 | 2 `UP` | 10 `UPFIRE` |
| 右 | 3 `RIGHT` | 11 `RIGHTFIRE` |
| 左 | 4 `LEFT` | 12 `LEFTFIRE` |
| 下 | 5 `DOWN` | 13 `DOWNFIRE` |
| 右上 | 6 `UPRIGHT` | 14 `UPRIGHTFIRE` |
| 左上 | 7 `UPLEFT` | 15 `UPLEFTFIRE` |
| 右下 | 8 `DOWNRIGHT` | 16 `DOWNRIGHTFIRE` |
| 左下 | 9 `DOWNLEFT` | 17 `DOWNLEFTFIRE` |

- enum にはほかに 2P 用（`PLAYER_B_*`、18〜35）とシステム操作（`RESET` 等、40〜45）があるが、AtariEnv の行動集合には入らない。
- AtariEnv は既定で minimal action set を使う（`AtariEnv.full_action_space ?= false`、[Atari.txt](../../apps/runner/config/Atari.txt)）。
  enum 順から効かない行動を抜いて詰めた順なので、**同じ行動の番号がゲームで変わる**。`full_action_space = true` なら全ゲーム 18 個で enum の番号どおり。
- ラベルは ALE の enum 名から `PLAYER_A_` を除いたもの（[AtariEnv.cpp](../../core/envs/atari1/src/AtariEnv.cpp)）。ゲームにも ALE の版にも依らず綴りは固定。

主なゲームの行動集合（ALE の `getMinimalActionSet` を直接読んだ値）:

| ゲーム | 行動数 | 行動集合（番号順） | 無い組み合わせ |
|---|---|---|---|
| battle_zone / double_dunk | 18 | 上表の全部 | — |
| kung_fu_master | 14 | 上表から `FIRE` / `UPFIRE` / `UPRIGHT` / `UPLEFT` を除いたもの | ボタン単独 |
| phoenix | 8 | `NOOP FIRE RIGHT LEFT DOWN RIGHTFIRE LEFTFIRE DOWNFIRE` | 上、斜め |
| name_this_game | 6 | `NOOP FIRE RIGHT LEFT RIGHTFIRE LEFTFIRE` | 上下、斜め |
| qbert | 6 | `NOOP FIRE UP RIGHT LEFT DOWN` | 斜め、方向 + ボタン |
| breakout | 4 | `NOOP FIRE RIGHT LEFT` | 上下、方向 + ボタン |

- 同じ `RIGHTFIRE` が battle_zone 11 / kung_fu_master 7 / phoenix 5 / name_this_game 4。ms_pacman は FIRE 系が 1 つも無い（8 方向 + `NOOP` の 9 個）。
- **パドルのゲーム**: ROM 属性が `PADDLES` のゲーム（breakout = Breakaway IV、pong = Video Olympics、kaboom）では、ALE が「右を含む行動」「左を含む行動」を
  つまみを一定量回す操作に、「FIRE を含む行動」をパドルのボタンに読み替える（`ALEState::applyActionPaddles`、`PADDLE_DELTA`）。上下の成分は効かない。
  つまみの量を連続値で渡す口（`ALEInterface::act(action, paddle_strength)`）もある。
- EvalPanel の Atari は `app.online.eval_panel.fps = 14.5` × `step_per_frame` 1 × `frame_skip` 4 でほぼ実機速度、1 step = 4 フレーム（約 67 ms）。
  `run.eval.[eval_panel].env.repeat_action_probability = 0`（sticky なし）なので、強制行動はその step の最初のフレームから効く。
- **実機にゲーム別の割当ては無い。** 本体は方向スイッチ 4 本（RIOT の `SWCHA`）とボタン 1 本（TIA の `INPT4`）の状態を読むだけで、
  組み合わせの解釈は各ゲームのプログラムが決める。ゲームごとに変わるのはコントローラの種類（ジョイスティック / パドル等）だけで、
  エミュレータ（Stella）も ROM の属性でコントローラの種類を選ぶだけ。minimal action set は「そのゲームで区別がつく行動」の一覧で、入力装置の側の区別ではない。

### 1.5 行動集合に無い入力をゲームはどう扱うか（実測、2026-09-27）

ALE で状態を複製し、行動集合に無い行動と行動集合の各行動を同じ手順で打ち比べた（乱数方策でばらまいた 60 状態、8 フレーム押して 24 フレーム NOOP、
RAM が完全一致した行動を「ゲームが同じに扱う行動」とする）。§3.8 の汎用規則の答えと突き合わせた結果:

- Atari-57 のうち 30 本は 18 行動すべてが行動集合にあり、解決の問題が起きない。
- 残り 27 本の「行動集合に無い入力」273 個のうち 220 個（81%）で、汎用規則の答えがゲーム自身の解釈と一致した。
  17 本は全一致（kung_fu_master / name_this_game / phoenix / breakout / pong / space_invaders / ms_pacman / freeway / assault / asterix /
  asteroids / atlantis / crazy_climber / demon_attack / skiing / surround / up_n_down）。
- 食い違う 10 本は、ゲーム側の優先順位が違う。tetris（Atari-57 外）は 13 個中 9 個一致。

| ゲーム側の解釈 | ゲーム | 例（汎用規則 → ゲームの実際） |
|---|---|---|
| 斜めでは上下を残す | amidar / wizard_of_wor / tutankham / time_pilot / tetris | tetris `DOWNRIGHT`: `RIGHT` → `DOWN` |
| 斜めを無視する | qbert / video_pinball | qbert `UPRIGHT`: `RIGHT` → `NOOP` |
| 方向より FIRE を残す | beam_rider / enduro | enduro `DOWNFIRE`: `DOWN` → `FIRE` |
| 独特 | bowling / gopher | bowling `RIGHT`: `NOOP` → `DOWN`、gopher `DOWN`: `NOOP` → `UP` |

- 優先順位がゲームごとに違うので、実機と全一致する普遍的な規則は無い。
- 状態で変わる組もある。time_pilot の斜めは 72〜77% の状態でだけ上下と一致する。breakout の「左右 + FIRE」は 89% の状態で左右と一致し、
  残りは発射と移動を同時にする状態で、行動集合のどれとも一致しない（minimal set では表せない動き）。
- `NOOP` は調べた 58 本すべてで行動集合の 0 番にある。qbert の `FIRE` は 60 状態すべてで `NOOP` と同じだった。

## 2. ゴール / 非ゴール

**ゴール**

- **G1**: EvalPanel の行動上書きを Joystick（十字キー / スティック / ボタン）からも行える。
- **G2**: 操作 → 行動の対応を設定ファイルで宣言でき、コードの固定表を消す。Env ごとに違う行動集合へ追従できる。
  Atari は全ゲーム共通の 1 枚で書け、ゲームごとの行動集合の違いは Env 側の別名表で吸収する（§3.8）。
- **G3**: 複合行動（Atari の方向 + FIRE）を同時押しで表現できる。
- **G4**: 入力の解釈（どの操作が押されているか → 行動番号）を Runner から分離し、対 PRD（入力先 Train）が同じ入力層を使える。

**非ゴール**

- **NG1**: Train 側への入力（対 PRD）。
- **NG2**: 連続値行動（スティックの倍率をそのまま行動にする）。本 PRD は離散行動のみ。
- **NG3**: 操作の記録・再生（人間プレイのファイル保存）。
- **NG4**: QValuePanel のクリック経路の変更。既存のまま `EvalPanel::DoStep(action)` を呼ぶ。
- **NG5**: `WithAction` の廃止そのもの。083 後続の観測パイプライン PRD が担う。本 PRD は現行経路のまま動く形で書き、
  `MakeAction` 入力化が入ったらその入力へ乗り換える（§3.4）。

## 3. 提案（暫定）

### 3.1 層に分ける: 装置 → 押下状態 → 行動の指定 → 行動番号

```
[キーボード] ─┐                          ┌─ 操作マッピング(設定) ──┐   ┌─ 行動の解決(Env が出す表) ─┐
              ├─ 押下中の操作の集合 ────►│ 集合 → 行動の指定 or 無し │──►│ 指定 → 行動番号              │──► 強制行動
[Joystick]  ─┘   (key:Up, joy:button:0…) └─────────────────────────┘   │ 行動集合に無いラベルは別名表 │   (EvalPanel::DoStep(action) / 決定点への入力)
                                                                         └──────────────────────────────┘
```

- **装置層**は「いま押されている操作の集合」を作るだけで、行動を知らない。キーボードは key down / up の追跡、Joystick は timer での polling
  （`GetButtonState` / `GetPOVPosition` / 軸のしきい値）。イベント方式（`SetCapture`）は要らない見込み。
- **マッピング層**は集合を「行動の指定」（番号かラベル。Q2）へ引く純粋な表引き。同時押し（chord）は「操作の集合」との一致で表す（§3.2）。
- **解決層**は行動の指定を Env の行動番号へ引く。番号はそのまま、ラベルは `value_labels` を引き、行動集合に無いラベルは Env が出す別名表を引く（§3.8）。
  入力層は Env ごとの綴りの意味を知らない。
- **適用層**は既存の `EvalPanel::DoStep(action)`。押下の意味論は §3.3。

操作の識別子（案）:

| 識別子 | 意味 |
|---|---|
| `key:<wx 名>` | キー。`key:Up` / `key:KP_0` / `key:Z` |
| `joy:button:<n>` | Joystick ボタン n（0 始まり） |
| `joy:pov:<up\|down\|left\|right>` | 十字キー（POV）。斜めは 2 方向の同時押しとして扱う |
| `joy:axis:<x\|y>:<+\|->` | 左スティックの方向。しきい値は固定値から始める |

### 3.2 操作マッピングの設定（案）

行動集合は Env ごとなので、宣言は各 Env 設定ファイル（`Atari.txt` / `LunarLander.txt` …）の `app` 節へ置く。
入力層は対 PRD で Train と共有するので、`app.eval_panel.*` ではなく **`app.input.*`** に置く。

```
# 行動番号 → 同時押しの操作集合（空白区切り = 同時押し）。1 行 1 組み合わせ。
app.input.map.[0].[a] = key:Up          # NOOP (LunarLander)
app.input.map.[1].[a] = key:Left
app.input.map.[2].[a] = key:Down
app.input.map.[3].[a] = key:Right
app.input.map.[2].[b] = joy:button:0    # 同じ行動へ別の操作
```

```
# Atari（全ゲーム共通の 1 枚。スティック 9 状態 × ボタンの 18 ラベルを宣言する。
#        そのゲームの行動集合に無いラベルは §3.8 の別名表で行動集合のどれかへ解決される）
app.input.map.[UP].[a]            = joy:pov:up
app.input.map.[UPRIGHT].[a]       = joy:pov:up joy:pov:right
app.input.map.[FIRE].[a]          = joy:button:0
app.input.map.[UPRIGHTFIRE].[a]   = joy:pov:up joy:pov:right joy:button:0
app.input.map.[UP].[b]            = key:Up
app.input.map.[UPRIGHTFIRE].[b]   = key:Up key:Right key:Z
# … NOOP を除く 17 ラベル × 装置の数だけ並べる（NOOP は宣言しない = 何も押していない状態。Q4）
```

- 外側の `[...]` は行動の指定。番号（ユーザー要件「操作と Action 番号のマッピング」）かラベルか、両方許すかは Q2。
  Atari はゲームで番号が変わるのでラベルで書く（§1.4）。内側の `[a]` `[b]` は同じ行動への代替操作を並べるための任意タグで、
  意味を持たない（`ram_metric.[game].[name]` と同じ `MakeSubConfigData` で列挙できる）。
- 解決規則: 押下中の集合と**完全一致**する組み合わせがあればその行動。無ければ「押下中の集合の部分集合で最大のもの」を採る。
  これで「方向だけ押している間は `RIGHT`、FIRE を足すと `RIGHTFIRE`」が自然に出る。同率の扱いは Q13（§3.5）。
- Atari は 18 ラベルを全部宣言するので、スティック / 十字キーのどの状態も完全一致で引け、同率は起きない（逆方向の同時押しは §3.8）。
  ゲームに無い行動は、ここでは考えずに解決層（§3.8）へ任せる。

### 3.3 押下の意味論（現行との差）

| モード | 挙動 | 使い所 |
|---|---|---|
| **保持型**（新設・主） | timer 駆動の各 step で押下状態をサンプルし、行動が引けたらその step を強制、引けなければ Actor が選ぶ | Joystick で遊ぶ。Train へ向ける対 PRD もこれ |
| **押下型**（現行） | 1 押下 = 1 強制 step。timer と無関係に割り込む | pause 中にコマ送りで打つ |

- 保持型が本命。Joystick は「押している間」が自然で、Train 側（対 PRD）も「決定点でサンプルする」しかない。
- 押下型は **EvalPanel が pause 中のときだけ**残す案（§6 Q3）。走行中にも割り込ませると 1 秒に 2 回動く。
- 何も押していない step の扱い（Actor に任せる / NOOP）は §6 Q4。Atari の人間プレイは NOOP が自然、補助的に遊ぶなら Actor。

### 3.4 適用点

- 当面は既存の `EvalPanel::DoStep(action)` → `EvalRunner::DoStep(action)`（`WithAction`）で動かす。保持型は `EvalPanel::OnTimer` の
  `DoUpdateFrame` の中で step ごとに引く必要があるので、`DoUpdateFrame` の `pre_step_func` から次 step の強制行動を渡す口が要る
  （現行の `EvalRunner::DoStep(int64_t)` は 1 step 単位なので、timer 経路には乗らない）。
- 083 後続の観測パイプライン PRD で forced action が `MakeAction` の入力（lane mask + action）になったら、**入力層の出力をその入力へ渡す**だけにし、
  `WithAction` 経路は消える。本 PRD 単独で先に入る場合も、入力層の境界（「押下集合 → 行動番号 or 無し」）は変えない。

### 3.5 検証と fail-fast（AGENTS.md「Fail-Fast 原則」）

| 状態 | 扱い |
|---|---|
| `app.input.map.[<指定>]` が番号なら `ActionSpec` の範囲外、ラベルなら `value_labels` にも別名表（§3.8）にも無い。未知の操作識別子。同じ操作集合が 2 つの行動に付いている | 起動時に `ANET_SYSTEM_ERROR`（EvalPanel の Env spec が確定した時点） |
| `AtariEnv.action_alias.[<現在のゲーム>].[<ラベル>]` のラベルが 18 個の外、または値が行動集合に無い | AtariEnv の起動時に `ANET_SYSTEM_ERROR` |
| 現在のゲーム以外の `action_alias` 表 | 読まない（検証もしない。`ram_metric` と同じ扱い） |
| マッピングが 1 行も無い | エラーにしない。キー入力による上書きが無効なだけ（現行の固定表は消えるので、common.txt の既定は要る。§6 Q5） |
| `joy:*` を宣言したが Joystick が接続されていない | 意図された不在。起動時に 1 回 INFO、キーボードだけで動く。接続の hot-plug は追わない |
| 押下集合が複数の組み合わせに同率で部分一致 | 起動時に弾く形は、普通の設定（LunarLander で上と左の同時押しなど）でも成立しない。実行時の規則で解く（Q13）。Atari は 18 ラベルを全部宣言すればスティックからは同率が出ない |

### 3.6 システム操作との関係

`SHIFT`（Train トグル）/ `SPACE`（Eval トグル）/ `CTRL`（Eval 1 step）は行動ではないので、`app.input.map` の対象外で現行維持。
Joystick のボタンにこれらを割り当てたい（キーボード無しでデモする）需要はあるので、`app.input.system.[eval_toggle] = joy:button:7` のような
別キーで扱うかは §6 Q6。

### 3.7 Atari 視点の考慮点

§1.4・§1.5 の構造から出てくる論点。

1. **番号ではなくラベルで、全ゲーム共通の 1 枚にする。** 番号はゲームで変わり、`full_action_space` を切り替えても全部ずれる。
   Atari のラベルは方向と FIRE の組み立てで綴りが固定なので、スティック 9 状態 × ボタンの 18 ラベルを 1 枚宣言すれば全ゲームに効く（§3.2）。
   実機と同じく、装置の側はゲームを知らない。
2. **スティックと十字キーは 4 つの独立したボタンではなく、9 状態の 1 つの装置。** 18 ラベルを全部宣言すれば斜めも完全一致で引けるので、
   スティックからは同率が出ない。キーボードの逆方向の同時押し（上 + 下）は物理スティックでは起きないので、装置層で正規化する（§3.8）。
3. **ゲームに無い組み合わせは普通に起きる。** kung_fu_master のボタン単独、qbert の斜めと「方向 + ボタン」、name_this_game の上下。
   これはゲーム側の解釈の問題なので、入力層ではなく Env の別名表で解く。既定は汎用規則、ゲーム別の表で上書きできる（§3.8）。
4. **短い押下の取りこぼし。** 保持型は step ごと（Atari は約 67 ms ごと）に押下状態を見るので、step の間に押して離した FIRE は拾えない。
   押下の立ち上がりを次の step まで保持するか（Q12）。押し続けて同じ行動が続くのは実機でボタンを押し続けるのと同じ意味で、問題にならない。
5. **パドルのゲームは専用の扱いが要らない。** breakout / pong は ALE が左右と FIRE をつまみ操作へ読み替えるので、左右 + ボタンの離散入力で遊べる。
   スティックの倒し量をつまみの速さにする（`paddle_strength`）のは NG2 の連続値行動に当たる。
6. **sticky は Train 側で効く。** EvalPanel は sticky 0 なので押した行動がそのまま効く。Train へ向ける対 PRD では sticky 0.25 が効くため、
   強制行動への切り替わりもフレームごとに 25% の確率で前の行動に置き換わって遅れる（ALE 側の処理で、入力層では消せない）。

### 3.8 行動集合に無い行動の解決（汎用規則 + ゲーム別の上書き）

**置き場。** どの入力をゲームがどう扱うかはゲームの知識なので、AtariEnv が持つ。AtariEnv はゲームが決まった時点で、
18 ラベル全部について「行動集合のどの番号で実行するか」の別名表を作り、`ActionSpec` に載せる（欄の名前は実装時に決める。案 `label_aliases`）。
入力層（§3.1 の解決層）は `value_labels` と別名表を引くだけで、Atari の綴りの意味を知らない。別名表を持たない Env では、行動集合に無いラベルは §3.5 の起動時エラー。

**別名表の作り方。** 18 ラベルそれぞれについて、次の順で決める。

1. 行動集合にあるラベル → その番号。
2. ゲーム別の上書き `AtariEnv.action_alias.[<game>].[<ラベル>] = <ラベル>` があればそれ。
3. 無ければ汎用規則: ラベルを（上下, 左右, FIRE）の成分に分け、成分を落として行動集合にある最初のものを採る。落とす順は
   1 つずつ FIRE → 上下 → 左右、次に 2 つずつ {FIRE, 上下} → {FIRE, 左右} → {上下, 左右}、最後に全部（`NOOP`。全ゲームで 0 番にある、§1.5）。

| ゲーム | 入力 | 汎用規則の結果 | ゲーム自身の解釈（§1.5） |
|---|---|---|---|
| kung_fu_master | `FIRE` / `UPRIGHT` / `UPFIRE` | `NOOP` / `RIGHT` / `UP` | 同じ |
| name_this_game | `UP` / `UPRIGHTFIRE` | `NOOP` / `RIGHTFIRE` | 同じ |
| breakout | `DOWNRIGHT` / `RIGHTFIRE` | `RIGHT` / `RIGHT` | 同じ（`RIGHTFIRE` は 89% の状態で） |
| qbert | `UPRIGHT` | `RIGHT` | `NOOP`（斜めを無視） |
| tetris | `DOWNRIGHT` | `RIGHT` | `DOWN` |

汎用規則の結果が実機と違うゲームで実機どおりにしたいときだけ、上書きを書く（`ram_metric.[game]` と同じ形、[082](done/082_atari_ram_progress_metrics_10prd.md)）:

```
# qbert は斜めを無視する（§1.5）。実機どおりにするなら斜め 4 ラベルと斜め FIRE 4 ラベルを NOOP へ
AtariEnv.action_alias.[qbert].[UPRIGHT]       ?= NOOP
AtariEnv.action_alias.[qbert].[UPRIGHTFIRE]   ?= NOOP
# … UPLEFT / DOWNRIGHT / DOWNLEFT とそれぞれの FIRE 付きも同じ（計 8 行）。「方向 + FIRE」→ 方向は汎用規則のままで実機と一致
# tetris は斜めで下を優先する
AtariEnv.action_alias.[tetris].[DOWNRIGHT]    ?= DOWN
```

- **実行する行動 = 記録する行動。** 別名で解決した行動番号がそのまま強制行動になり、Q 値の表示・081 のログ・対 PRD の replay に入るのもその番号。
  汎用規則が実機と違うゲームでは操作感が実機と違うだけで、学習とログの整合は崩れない。
- **起動時に別名表を 1 回ログへ出す**（18 ラベル → 実行するラベル、上書き由来かどうか）。どの入力が何になるかを遊ぶ前に確認できる。
- **上書き表の中身は §1.5 の打ち比べで機械的に作れる。** 既定で同梱するか、空で始めるか、起動時に作るかは Q11。
- **逆方向の同時押し**（キーボードで上と下、右と左）は、その軸を押していない扱いにする（装置層で正規化する。物理スティックでは起きない）。
- **採らない代替: EvalPanel だけ `full_action_space = true`。** ALE に 18 行動をそのまま渡せば実機と完全に同じになるが、エージェントの出力番号（行動集合）を
  18 行動へ読み替える層が要り、行動集合に無い強制行動には Q 値の表示も行動番号も無くなる。対 PRD（Train）には使えない。別名表なら両方に効く。

## 4. 用語（CONTEXT.md へ足す候補。正式 PRD で確定）

| 語 | 意味 | _Avoid_ 候補 |
|---|---|---|
| **操作入力**（operator input） | 人が入力装置（キーボード / Joystick）で与える行動の指示。装置に依らず「押下中の操作の集合」として扱う | 手動入力（QValuePanel のクリックと紛れる）, キー入力（装置を限定する） |
| **強制行動**（forced action） | 操作入力によって Actor の選択を置き換えた実行行動。100 / 160 の「強制Action」を用語として確定する（2026-10-29 注記: [PRD105](done/105_eval_trace_arrow_key_forced_action_10prd.md) の D11 で確定し、[CONTEXT.md](../../CONTEXT.md) へ足した。定義は QValuePanel のクリックを含めて「人の操作（キー・クリックなど）によって」に広げた） | 手動 Action, manual action, 上書き（設定の上書き層と紛れる） |
| **操作マッピング**（input mapping） | 操作の集合から行動の指定（番号かラベル）への対応表。設定 `app.input.map` で宣言する | キーバインド（キーボード限定に読める）, key config |
| **行動の別名**（action alias） | Env の行動集合に無いラベルを、行動集合のどの行動で実行するかの対応。Atari が 18 ラベルについて持つ（汎用規則 + ゲーム別の上書き） | フォールバック（処理の失敗と紛れる）, 代替行動 |

## 5. 受入条件（暫定）

1. `RunnerFrame::OnKey` の行動 `switch` が消え、行動キーは全て `app.input.map` から引かれる。テンキー 0〜9 と矢印の現行挙動は
   common.txt の既定マッピングで再現される（LunarLander で矢印が 0/2/1/3 のまま）。
2. Joystick 1 台で Atari（kung_fu_master）の方向 + FIRE の複合行動を打て、EvalPanel の timer 駆動中に押している間だけ強制される。
   Atari の操作マッピングは全ゲーム共通の 1 枚で、ゲームを替えても書き直さない。
3. 行動集合に無いラベルは汎用規則で解決される（kung_fu_master の `FIRE` → `NOOP`、`UPRIGHT` → `RIGHT` など §3.8 の表のとおり）。
   `AtariEnv.action_alias` の上書きがあればそちらが勝つ。別名表は起動時に 1 回ログに出る。
4. 未知の操作識別子・範囲外の行動番号・どこにも無いラベル・重複する操作集合・不正な上書き表は起動時に fail-fast し、メッセージにキー・値・期待範囲が出る。
5. Joystick 未接続で `joy:*` を含む設定を起動しても落ちず、INFO 1 回でキーボードだけで動く。
6. [020 §6.2](../design/020_user_guide_run.jp.md) の操作表と [160](../design/160_applications_and_tools.jp.md) の EvalPanel 記述が新契約に同期している。

## 6. 未決（グリル対象）

| # | 論点 | 選択肢 / 備考 |
|---|---|---|
| Q1 | 装置 API | (a) wxJoystick polling（最小、core に同梱） (b) XInput 直叩き (c) SDL2 GameController。(a) から始め、パッドによって効かなければ (b) |
| Q2 | マッピングのキーは行動番号かラベルか | Atari はラベルで書く（§3.2・§3.8。ラベルは ALE の enum 名で安定、番号はゲームでも `full_action_space` でも変わる）。残る論点は番号も許すか: (a) ラベルだけ (b) 両方許す（番号は要件どおりの形）。LunarLander のラベルは空白を含む（`fire main`）ので、`[...]` のキーに書けるかは要確認 |
| Q3 | 押下型（1 押下 = 1 step）を残すか | (a) pause 中だけ残す (b) 全廃して保持型のみ (c) 現行どおり常時併存。コマ送りの需要が (a) の根拠 |
| Q4 | 何も押していない step | (a) Actor が選ぶ（現行の timer 挙動） (b) NOOP 固定 (c) 設定で選ぶ `app.input.idle = actor \| noop`。Train へ向けるとき（対 PRD）は (b) が要る場面がある |
| Q5 | 既定マッピングの置き場 | common.txt に矢印 + テンキーの既定（`?=`）を置き、Env 側で足す / 上書き。矢印 0/2/1/3 は LunarLander 固有なので LunarLander.txt へ移す案もある |
| Q6 | システム操作の Joystick 割当 | `app.input.system.[...]` を作るか、当面キーボード固定のままか |
| Q7 | スティック軸のしきい値・デッドゾーン | 固定値（例: ±50%）か設定か。最初は固定 |
| Q8 | 表示 | 押下中の操作と引けた行動をステータスバーへ出すか。デバッグにはほぼ必須 |
| Q9 | 設定階層 | `app.input.*` で確定してよいか。`app.online.` / `app.batchrun.` の層分けは要らない見込み（batchrun には人がいない） |
| Q10 | Atari の 18 ラベルの宣言の仕方 | (a) 18 ラベル × 装置を並べる（§3.2 の例。装置 2 つで 34 行） (b) 方向 4 つとボタン 1 つだけ宣言し、ラベルの組み立ては入力層の規則にする（入力層が Atari の綴りを知ることになる） (c) 当面 (a) |
| Q11 | ゲーム別上書き表の既定 | (a) 空（汎用規則だけ。§1.5 の 10 本は実機と操作感が違う） (b) §1.5 で食い違った 10 本分を Atari.txt に `?=` で同梱（表は打ち比べで機械的に作る） (c) AtariEnv の起動時に打ち比べて作る（1 ゲーム数秒〜十数秒、状態で変わる組の扱いが要る）。いずれも別名表の置き場は §3.8（AtariEnv が `ActionSpec` に載せる） |
| Q12 | 短い押下のラッチ | (a) しない（step 境界のサンプルだけ） (b) 押下の立ち上がりを次の step まで保持 (c) ボタンだけ保持。§3.7 の 4 |
| Q13 | 同率の部分一致の実行時規則 | §3.5 の「起動時に弾く」は LunarLander の上 + 左のような普通の設定でも成立しない。(a) 後から押した操作を優先 (b) 宣言順 (c) 引けない扱い（Q4 と同じ）。Atari は 18 ラベルを全部宣言すれば起きない |

## 7. スコープ外（再掲）

NG1〜NG5。加えて、Joystick の複数台対応（1 台で始める）、振動などの出力、キーリピートの調整。
Atari の 2P 操作（`PLAYER_B_*`）とコンソールスイッチ（`RESET` / `SELECT` / 難易度。ゲームモードと難易度は `AtariEnv.mode` / `difficulty` の設定で決まる）。

## 8. 実装に触れる見込みの箇所

- [RunnerFrame.cpp](../../apps/runner/src/RunnerFrame.cpp) `OnKey` の行動 `switch` の撤去、入力層の生成と EvalPanel への接続。
- [EvalPanel.cpp](../../apps/runner/src/EvalPanel.cpp) `OnTimer` の保持型サンプリング（`DoUpdateFrame` の `pre_step_func` 経由か、
  観測パイプライン PRD 後は `MakeAction` 入力）。
- [RunnerApp.cpp](../../apps/runner/src/RunnerApp.cpp) `Config` の `app.input.*` 読み込みと検証（`MakeSubConfigData`）。
- 入力層の新規ファイル（`apps/runner/src/` 配下。装置層・マッピング層・解決層。機能グループ単位で 1 組の `.hpp/.cpp`）。
- [rl.hpp](../../core/anet-core/include/anet/rl.hpp) の `ActionSpec` に別名表の欄（§3.8）。
- [AtariEnv.cpp](../../core/envs/atari1/src/AtariEnv.cpp) の別名表の構築（汎用規則 + `AtariEnv.action_alias.[<game>]` の上書き、検証、起動ログ）。
- [common.txt](../../apps/runner/config/common.txt) の既定マッピング、[Atari.txt](../../apps/runner/config/Atari.txt) の 18 ラベルのマッピングと上書き表（Q11）、
  [LunarLander.txt](../../apps/runner/config/LunarLander.txt) の Env 別マッピング。
- [020 §6.2](../design/020_user_guide_run.jp.md)、[160](../design/160_applications_and_tools.jp.md)、CONTEXT.md（§4 の用語）。

## 9. 関連

- [923_input_target_train_forced_action_10prd.md](923_input_target_train_forced_action_10prd.md) — 対 PRD。入力層（§3.1〜3.2）を共有する。
- [083](done/083_action_context_retire_10prd.md) / [921](921_agent_prev_action_obs_10prd.md) — forced action の `MakeAction` 入力化。§3.4 の乗り換え先。
- [081](081_evalpanel_episode_trace_10prd.md) — D6 の手動行動ログ `EvalPanel: manual action=`。保持型では step ごとに出ると spam になるので、
  押下状態の変化時だけ出す形へ寄せる（081 実装時に整合を取る）。
- [ADR 0016](../adr/0016-runnerframe-aui-layout-roundtrip.md) — RunnerFrame の pane 構成。ステータス表示（Q8）を足すときの前提。
- [082](done/082_atari_ram_progress_metrics_10prd.md)（Atari の RAM メトリクス） — ゲーム別の設定キー `AtariEnv.ram_metric.[<game>].[<name>]` の前例。§3.8 の `action_alias` は同じ形にそろえる。
