# PRD 051 ALE Atari Env 実装メモ

## 概要

- `core/envs/atari1` に ALE v0.12.0 C++ API 直結の `AtariEnv`、前処理、View、テストを追加する。
- 現行 `DefaultDQNAgent.baseline` を継承し、Nature CNN、256 Env、2.5M transitions を既定にした `Atari.txt` を追加する。
- Pong / Breakout 各 10M emulator frames の本走行は後続フェーズとし、今回は ROM 統合テストと短時間学習 smoke までを完了条件にする。
- 既存の未コミット変更を保持し、関連箇所へ局所差分だけを加える。

## 主な変更

### AtariEnv 公開契約

- `InitAtari()` から class ID `AtariEnv` の `SingleDiscreteEnvFactory` と View creator を登録する。既存 `SingleDiscreteEnv` API は変更しない。
- `GetSpec()` は `grid: uint8 [1,S,S]`、ROM 由来の action set と action 名、clip 設定に応じた reward range を返す。
- `Step()` は action index を検証し、ALE action を最大 `frame_skip` 回実行する。早期終了は real game over のみとし、`episodic_life` の lives 判定は窓完走後に一度だけ行う。
- real game over、life loss、frame 上限の順に判定する。窓内全報酬を生スコアへ加え、返却報酬だけを必要に応じて sign clip する。
- frame の正本は `ale.getEpisodeFrameNumber()` のみにする。Reset 中の NOOP/FIRE と soft-reset NOOP も frame 数へ含め、自前 frame counter は持たない。
- max-pool は当該 Step 内で実行できた最後の最大2フレームだけを使う。1フレームなら単独使用し、前 Step の画像は混ぜない。
- hard Reset は `reset_game -> random NOOP -> FIRE`、soft Reset は NOOP 1回とする。soft Reset では生スコアと ALE frame を継続する。
- `episode_score`、`episode_len`、`episode_frames` は real game over / truncation 時だけ確定し、auto-reset 後の集約まで保持する。`lives` は常時公開する。
- Step の AuxData に View 用の現在の `episode_score`、`episode_len`、`episode_frames`、`lives` を格納する。
- `GetTensor("rgb_frame")` は有効時に所有権を持つ `uint8 [3,210,160]` を Reset/Step ごとに更新する。
- `Reset()` と `Step()` の主要境界へ既存規約どおり profile range を置く。

### 設定、前処理、View

- PRD記載の config と v5/classic preset を実装し、snake_case game、値域、mode/difficulty、action index を fail-fast 検証する。
- ROM は非空の `rom_dir`、次に `ATARI_ROM_DIR` から `<game>.bin` を解決する。明示パス不備は環境変数へ fallback しない。
- ALE 非依存の `PixelwiseMax` と area resize を実装し、round/clamp 後の uint8 `[1,S,S]` を返す。
- `AtariView` は lane 0 の RGB を主、前処理後 grid を副として wxGLCanvas に描画し、生スコア、lives、episode step/frame、action 名、clipped reward を重ねる。
- `Atari.txt` は baseline の head/dueling/QR/PER/stack 設定を重複定義せず、`StackMerge -> Conv32(8,4) -> ReLU -> Conv64(4,2) -> ReLU -> Conv64(3,1) -> ReLU -> Flatten -> Linear512 -> ReLU` を追加する。
- train/eval の raw episode score、episode length/frame、lives metrics を追加する。

### ビルドと文書

- `ANET_ENABLE_ATARI=AUTO|ON|OFF` を追加し、`ALE_ROOT` とヘッダの存在だけで gate する。有効時だけ module、Runner link、`ANET_HAS_ATARI`、`InitAtari()` を組み込む。
- ALE build-tree の config 別 library と ZLIB を参照し、リリース workflow は `ANET_ENABLE_ATARI=OFF` を明示する。
- module に GPL 注意書きを追加する。
- PRD・Atari設計文書を ALE frame 正本、rolling max、life-loss 窓境界判定へ直し、Env設計索引へ module を追加する。
- PRD状態は「実装完了・長時間学習検証待ち」とし、実測前に `docs/experiments/` へ結果を書かない。

## テスト

- Public interface / surface: factory 経由の `GetSpec()` / `Reset()` / `Step()` / `GetScalar()`、`GetTensor()`、config 読み込み、Runner build gate。
- 優先 behavior:
  1. 公開 Env seam から Pong の spec と初期観測を取得できる。
  2. max-pool / area resize が合成入力の golden 値を返す。
  3. config、ROM、action、mode/difficulty の不正値が fail-fast する。
  4. real game over、truncation、reward clip、生スコア、hard/soft Reset が契約どおりになる。
  5. life loss が skip 窓を中断せず、`frame_skip=1` 比較 Env の報酬合計と一致する。
  6. seed 再現性、minimal/full actions、Vectorized/ThreadPool、auto-reset、RGB/AuxData が成立する。
- TDD 順序: 上記を1 behaviorずつ RED -> 最小 GREEN で進め、GREEN 後だけ重複を整理する。production API に test-only surface を追加しない。

## 検証

```powershell
# MSVC 環境は AGENTS.md の VsDevCmd.bat 経由で初期化する
cmake --preset x64-Debug -DANET_ENABLE_ATARI=ON -DALE_ROOT=C:/dev/ALE
cmake --build --preset x64-Debug --target AtariEnv-test AnetRLRunner
$env:ATARI_ROM_DIR='C:/dev/ale-sandbox/roms'
core\envs\atari1\bin\Debug\AtariEnv-test.exe
```

- AUTOでALEなし、OFF、ONでヘッダ不備、無効な三値も個別 configure で確認する。
- Runner smoke は 16 Env、10,240 transitions、eval 無効の一時 override で Pong を実行し、warmup 後の finite loss と終了を確認する。
- 単一 lane Runner で RGB/grid/overlay を目視確認する。

## 前提

- ALE v0.12.0 の Debug/Release build tree と Pong/Breakout ROM はリポジトリ外に存在する。
- CIへのALE取得、AtariVectorEnv、RGB/RAM学習観測、Atari 100k/Atari-5 protocol は対象外。
- header-only gate を維持し、ALE library/ZLIB の完全性は configure/link 時に検出する。
- Pong/Breakout 各2.5M transitions の本走行と方向性基準評価は後続フェーズで行う。
