---
title: Atari Env（ALE 統合）
page_kind: サマリ
material: "[[docs/design/220_atari_env.jp.md|220_atari_env.jp]]"
material_kind: 設計文書
concepts:
  - "[[docs/wiki/concepts/subjects/atari.md|Atari横断]]"
  - "[[docs/wiki/concepts/framework/322_env_individual.md|322 Env個別実装]]"
  - "[[docs/wiki/concepts/framework/320_env_core.md|320 Env 共通基盤]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
updated: 2026-11-26
---
# Atari Env（ALE 統合）

## 元の材料

[Atari Env（ALE 統合）](../../../design/220_atari_env.jp.md) blob:d5722d2d3d448ec82ed29771206f6365a0033d2b

## 要点

- AtariEnv は ALE を利用し、ROM を指定して共通の単体環境の口から Atari のゲームを実行する。
- ROM と外部ライブラリの準備、ゲームごとの設定とプロトコルを区別する。
- 観測画像の前処理、行動の繰り返し、報酬・終端、seed の扱いが学習・評価条件を決める。
- 生スコアと人間正規化スコアを区別する。正規化では人間とランダム方策の基準の差を符号付きのまま使う。
- 環境の accessor、RAM メトリクス、AtariView は、ゲームの状態を読み出す経路と表示側の責務を分ける。

## 内容

### 2. 全体像

- RB には単フレーム uint8 が保存され、stack はサンプル時に展開される（ADR 0026）。
- 責務分担: AtariEnv = 1 ALE の駆動と標準前処理。並列・stack・Step内の終端lane自動リセット・seed 配布はフレームワーク既存機構。

### 3. 外部依存（ALE v0.12.0）

- バージョン — tag `v0.12.0` 固定。vcpkg 依存は manifest の builtin-baseline で固定。
- AtariEnv が設定しない ALE のキーは、依存バージョンを更新するときに v0.12.0 の既定値との差を点検する。
- `setInt("random_seed", ...)` 等は ROM ロード前に設定する（`loadROM()` が設定を反映して環境を構築する）。
- ROM（.bin）はリポジトリ非同梱・自己調達。ファイル名は snake_case（`pong.bin`）。
- `loadROM()` は ROM md5 で対応ゲームを判定するため、非対応/破損 ROM は ALE 側でも検出される。
- ALE は GPL-2.0。①非同梱（外部参照）②オプショナルビルド ③リリースパッケージ除外（リリースビルドは `ANET_ENABLE_ATARI=OFF`）④`src/env/atari/NOTICE.md` に GPL 注意書き、で本体 Apache-2.0 を維持する。

### 4. AtariEnv 仕様

- classic の `max_episode_frames=108000` は歴史的 v4（gym 登録の TimeLimit=100,000 フレーム）と 8% 違う。SB3 `AtariWrapper` 自体は上限を持たないため、v5 と揃えた 108,000 を採用している。
- AtariEnv は `grid` に単チャネルの正方形 uint8 画像を返し、vector と action_mask は返さない。
- `fire_reset=true` でも、行動が3個以上あり、index 1 が FIRE のゲームだけで FIRE と index 2 を順に実行する。この条件を満たさないゲームでは両方を省く。
- env 内乱数（noop_max の回数抽選のみ）は `RandomHolder`（lane seed 初期化）を使用し、ALE 内部 RNG（sticky）と分離する。
- Step結果は生成時の値を保持し、Resetや通知後も変わらない。env本体には完了値を持ち越さず、移した統計キーは未知とする。結果の`lives`は補助NOOP前、env本体の`lives`は現在ALEの値である。

### 5. AtariView

- 副表示: 前処理後 obs（S×S グレー、TrainEvent の step_result から lane 0 の grid を取得）。「エージェントが見ているもの」の確認用で、resize / max-pool の不具合を生画面との見比べで検出できる。
- 既定は生 RGB 画面（カラー）のみ。中クリックで「元画像のみ → Agent画像（前処理後 obs）のみ → 両方 → 元画像のみ」と循環する。描画面・テキストオーバーレイ・Panel のどこでも1クリックで1段階進み、停止中も即座に再描画する。
- 単独表示はキャンバス全体を使って中央配置し、両方表示は RGB を左 2/3、前処理後 obs を右 1/3 の領域に並べる。既存の余白と整数倍拡大（ドット絵の視認性優先、GL_NEAREST）を維持する。画像未取得の領域は背景色のままとする。
- 表示対象は lane 0 固定（全 lane 同一ゲーム・同一設定のため代表 1 lane で挙動確認の目的を満たす。lane 選択 UI は将来拡張）。
- 更新は既存 View 機構の標準契約（`UIDataStore`、既定 `force_update_interval_ms=200` ≈ 5Hz。TrainEvent 毎に描画するわけではない）。

### 6. ビルド統合

- 三値 cache 変数 `ANET_ENABLE_ATARI` = `AUTO`（既定）/`ON`/`OFF`。AUTO は `ALE_ROOT` の存在と `src/ale/ale_interface.hpp` の実在で判定。ON で不備なら configure エラー。判定結果は STATUS ログへ。
- 有効時のみ: `add_subdirectory(src/env/atari)`、runner へ `AtariEnv` リンク + `ANET_HAS_ATARI` 定義、`RunnerApp.cpp` の `#ifdef ANET_HAS_ATARI` で `InitAtari()`。
- 縮小方式の変更時はRFD0099検証手順を参照する。固定24画像は各画像で全画素差1以下・99.9%以上一致を要求し、既定の学習系列のchecksumを保持する。性能受入は84×84・64 envの交互比較で速度比中央値0.98以上、現行同士の対照が1%以内に収まらない場合は判定保留とする。

## 主要な概念

- [Atari横断](../../concepts/subjects/atari.md)
- [322 Env個別実装](../../concepts/framework/322_env_individual.md)
- [320 Env 共通基盤](../../concepts/framework/320_env_core.md)
- [評価と比較の条件](../../concepts/measurement/evaluation.md)
