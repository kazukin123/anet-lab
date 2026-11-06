# PRD 059 Phase 1 素材化 + Phase 2 幹化 実装メモ: CartPole / GridMaze / ImageCls

- 作業日: 2026-08-24
- 対象: `apps/runner/config/CartPole.txt` / `GridMaze.txt` / `ImageCls.txt` / `nn_cnx.txt`(ImageCls 専用 include。include 元は ImageCls.txt の 1 箇所のみを確認済み)
- 正本: `059_config_concept_tree_alignment_30mat_guide.md`(手順)/ `..._10prd.md`(設計)
- 見本: `LunarLander.txt`(全般)/ `Atari.txt`(BODY 素材 `net.branch.@<Name>`・`@random`・`@vars` ペアの先例)

## 結果サマリ

| 項目 | 結果 |
|---|---|
| 等価性(値の差分 / 新規キー) | **0 件 / 0 件**(3 overlay とも) |
| 消滅キー | CartPole **0**(無変更)/ GridMaze **6**(`B.*`)/ ImageCls **30**(`net.branch.*` 素材 15 種 ×2)— 全て意図分 |
| 幹の等価性(ImageCls 2 本) | `run.@resnet18ish_hr` = 素 dump と**完全一致** / `run.@convnext_atto_hr` = BODY+run_name を CLI 手動切替した dump と**完全一致** |
| `[config]` テスト | 全緑(91 test cases / 608 assertions) |
| `git diff --check` | クリーン |
| 全 core テスト | **未実施**(DropMerge 枠と並行作業中のため。横断整合チェック時にまとめて実施) |

## 変更内容

### GridMaze.txt

1. **`B` → `DefaultDQNAgent.@random`**(6 行+チェーンコメントラダー 1 行)。Atari(25impl)・DropMerge(作業中)と同名で統一し、横断議題 1(3 ファイル不揃い)を解消方向へ。
2. **予算の一点化**: online/batchrun 同値(400K)のため `app.@vars : max_exp_step` 1 定数+`${}` 参照 2 行へ。
3. ALGO 素材は導入しない(GridMaze に `net.qr/iqn` 配線が無く、quantile は `@baseline` 供給の qr のまま。IQN 切替の実需なし)。

### ImageCls.txt + nn_cnx.txt

1. **BODY 素材 15 種の `@` 化**(`net.branch.@<Name>`、Atari 先例に追随): ImageCls.txt 内 11 種(FoodResNet13/_GAP、ResNet18、ResNet18ish_hr/_mr/_MaxPool/_std、Hybrid_CLS/_GAP、ViT_S16_CLS/_GAP)+nn_cnx.txt 内 4 種(ConvNeXtTiny/Nano/Femto/Atto)。選択ラダー(コメント込み 15 行)の RHS も追随。
2. **予算の一点化(`@vars` 隣接ペア)**: `max_exp_step` × `main_cosine_steps`(= 予算 − warmup 500K)のペアラダー(10M/15M/20M★/30M)。cosine 終端が予算に連動する手動同期を隣接ペア+方針コメントで局所化(D7 に演算なし)。`app.batchrun.exp_exit_step = 100M` は online と別値のためリテラル据置(Atari と同じ扱い)。
3. **named 幹 2 本(全て既定 OFF)**: 選定基準=「定着済+実験対象+今後試したくなる可能性」(Atari/DropMerge と共通)。
   - `run.@resnet18ish_hr`(定着: 86% ★現行既定)/ `run.@convnext_atto_hr`(実験対象: 83%、hr 化の本命候補)
   - 中身は BODY 選択+run_name の 2 スロット(この 2 箇所の手動同期が ImageCls の実在契約だった)
   - 作らなかったもの: ViT 系(37-43%、深追い非推奨)・Hybrid 系(56-60%)・Femto(Atto と同格で遅い)— D4

### CartPole.txt

**変更なし**。named プリセット・ALGO 配線・予算行・幹の実需のいずれも存在しない(before/after 完全一致で確認)。

## 等価性検証

一時テスト(`config_test.cpp` へ `[tempdump-cgi]` タグ+環境変数 `ANET_CGI_DUMP_STAGE=before|after|trunk`、**検証後に削除済み**)で 3 overlay の `ToPropertiesString()` を before/after 取得し、キー単位で比較(比較スクリプトは scratchpad、値差分と新規キーで fail)。幹の等価性は CLI 第 1 相(`run.$=run.@<name>`)有効化 dump と手動 CLI dump の全文一致で確認(7 assertions)。

**DropMerge 枠(Codex)と同一ワークツリーで並行作業だったため**、一時テストのタグ・環境変数を DropMerge 枠の `[tempdump]`/`ANET_DROPMERGE_DUMP_STAGE` と分離し、ビルドは彼らのビルド完了直後の窓で 1 回だけ実施した。一時テスト削除後の再ビルドは行っていない(現 exe に残る `[tempdump-cgi]` は env var 必須のため通常実行に影響しない。次回の通常ビルドで自然に消える)。

## 横断整合チェックへの申し送り

1. **stale コメントの掃除判断**: `CartPole.txt` の `#DefaultDQNAgent.$ = DefaultDQNAgent.trunk > A` / `trunk2`(定義が存在しない参照)、`GridMaze.txt` の `#GridMazeEnv.$ = GridMazeEnv.baseline`(定義されたことがない参照)と `#GridMazeEnv.test1.*` コメント群。等価性検証を汚さないため本作業では据置した。
2. **`net.branch.@<Name>` 命名の全 env 追認**(Atari 申し送り 2 と同件): 本作業で ImageCls も同形式に揃えた。残るは DropMerge の Suika* (DropMerge 枠が対応中)。
3. **`@random` の共通化**: Atari / GridMaze / DropMerge(作業中)の 3 ファイルに同名・同内容(6 行)の `DefaultDQNAgent.@random` が並ぶ。ALGO 素材 4 行(Atari 申し送り 3)と同様、agent.txt への吸い上げ候補。
4. GridMaze の `#R.quantile_mode = none # qr` コメント(旧 2 行同期時代の実験痕跡)は据置。ALGO 素材を GridMaze に導入する時が来たら合わせて整理する。
