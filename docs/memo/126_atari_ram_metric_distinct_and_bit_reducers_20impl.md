# Atari RAM メトリクスと設定読み取りの実装メモ

## 概要

[PRD126](126_atari_ram_metric_distinct_and_bit_reducers_10prd.md) の D1〜D16 と受入 1〜10 に従い、`distinct_seen`・`bit_rise_count:MASK`・`bit_fall_count:MASK` を追加する。RAM 定義の解析をまとめ、縮小方式を設定読み取り時に列挙型へ変換する。

## 主な変更

- `core/envs/atari1/src/AtariEnv.hpp`・`.cpp`: 3 種の畳み方、256 値の既訪問集合とマスクを追加する。語の名前・種別・引数表記を 1 表にまとめ、診断もその表から組み立てる。
- RAM 定義の全体読み取りを `AtariEnvConfig` の専用関数へ集め、既定キーの検査だけを残す。ゲーム名の形を 1 つの定数で共有する。届かないキー再検査・番地の catch を除き、記録は既存の `RecordConfigValue` を使う。
- `AtariPreprocess.hpp`・`.cpp`: `atari::ResizeMode` を追加し、`GrayscaleResizer` はその型を受け取る。寸法の前提検証と画素計算は維持する。
- `AtariEnvConfig::Validate` は game・frame_skip などの検査を維持し、縮小方式・screen_size の再検査を除く。設定なしでの構築の既定引数は除き、設定から構築する契約に合わせる。実効設定と verbose ログは文字列を保つ。
- `AtariEnv_test.cpp`: 新語の設定から集計までの経路、マスク、番号と実効設定、評価上書きの未読検出を検証する。縮小方式の型変更に呼び出しを合わせ、固定参照実装と既存の期待値を保持する。
- `docs/design/220_atari_env.jp.md` §4.3・§4.9、`core/anet-core/testdata/prd082/README.md` を更新する。英訳は PRD §6.4 に従って別工程とする。
- 対象 issue 949e8a・9ff3b7 の工程状態・対応内容・AI 判定と、一覧の派生区画を更新する。既存変更は保持する。
- 追加依頼（2026-11-12、「起票せずこの場で修正」）: `AtariEnv_test.cpp` の出荷設定を読む縮小方式テストを、合成の `ConfigData` による共通値の継承・評価個別の上書き・実効設定の検査へ置き換える。`[shipped_config]` を外して `[config]` とし、テスト名を実際に確認する挙動へ合わせる。実験用の Atari.txt をテストの期待値へ固定しないための追加対応で、本体・出荷設定・縮小画素の期待値は変えない。

## テスト

- Public interface / surface: `AtariEnvConfig`、`RamMetricState::Begin/Observe/Value`、`AtariEnvFactory::ValidateConfig`、`ConfigData::RequireAllKeysRead`、`GrayscaleResizer::Resize`。
- 優先 behavior / TDD 順序:
  1. `distinct_seen` の設定を番号へ結び、状態に初期値 1、続けて 0・1・2・6・2 を与える経路を 1 テストで RED → GREEN。初期値のみ、全 256 値、ゲーム再開の確認を加える。
  2. 立つビットをマスク内で個別に数える経路を RED → GREEN。次に落ちる側を RED → GREEN。PRD §5.1 の各行を別ゲームとし、開始値 2 の不変も確認する。
  3. マスクの受理・拒否と ERROR の全文、知らない語の 8 語の一覧を 1 挙動ずつ RED → GREEN。既存の reach_count の受理・文言を維持する。
  4. 評価上書きを env が読まず、切り出したタグの未読キー検査で止める挙動を RED → GREEN。旧い拒否テストの該当行を移す。
  5. GREEN のあと RAM 読み取りと縮小方式の整理を行い、関連テストを再実行する。3 語の番号と実効設定を確認し、既存の Atari・PRD117 の期待値を変えず検証する。
  6. 追加依頼のテスト整理は、既に確認済みの GREEN から行う。新しい本体の挙動は足さず、既存の継承・上書き・実効設定の検査を合成入力で維持する。`[atari][prd099][resize-profile]` と新語の `[prd126]` を再実行し、独立レビューを受ける。

## 検証

- MSVC 初期化後に Debug 全体ビルドし、`core/envs/atari1/bin/Debug/AtariEnv-test.exe "[atari]~[.],[prd117]~[.]"` と `core/anet-core/bin/Debug/anet-core-test.exe "[atari]~[.],[prd117]~[.]"` で既定の関連テストを実行する。ROM は既存の FindRom の探索先を利用し、skip は成功と混同せず記録する。
- ビルドは `cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug'`。TDD の反復では同じ初期化で `--target AtariEnv-test` を指定する。
- 詳細出力は `.scratch/prd126/` の build・test ログへ保存し、応答には終了コードと失敗の要約を返す。実行中はプロセス終了とログの進捗を確認し、経過時間だけで停止しない。
- PRD 受入 5・6・8・9 の検索、品質観点の機械検査（差分、LF、BOM、名前・重複文字列）を行う。
- 自己点検後、独立した `anet-review-agent` で `impl-plan` と `impl` のレビューを受け、差し戻しを修正して同じエージェントへ再依頼する。

## 前提

- ユーザー判断が必要なブロッカー: 0。PRD はグリル済みで暫定の印が無い。
- repo evidence: `ConfigData::MakeSubConfigData` は元の読んだ印を共有し、タグの `RequireAllKeysRead` が RAM 上書きの未読行を止める。RAM の既定キーの正規表現を通ればラベルの再検査と番地の例外処理には届かない。
- `AtariEnvConfig` は設定から構築する。game など既存の `Validate` の検証時機を維持し、screen_size は元の読み取り箇所へ集める。`GrayscaleResizer` の正の寸法・pixel_area の縮小先の検査は残す。
- 新語の初期化・観測時機・NaN・出口は既存の RAM 経路を使い、報酬と観測を変えない。`Atari.txt`、固定参照実装、過去資料と ADR は変えない。
- PRD 範囲外: Run 全体の和集合、2 レベル目の区別、Montezuma の設定と ROM の意味の探索。
- 基準コミット: `92e9226a8e68c753389259331a1e0d0d8e8238e1`。開始時に Atari の対象コードには変更がなく、CONTEXT・wiki・INDEX などの別作業の変更があるため保持する。

## 経過と結果

- 2026-11-12: PRD、関連設計、ADR、用語、コード・テスト・設定検査と未決事項を照合し、計画を保存した。
- TDD 1: `distinct_seen` を設定で拒否する RED（exit 42）を確認し、最小実装後に 1 test / 5 assertions が GREEN。ログ: `.scratch/prd126/red-distinct-atari.log`・`green-distinct.log`。
- TDD 2: `bit_rise_count:30` を拒否する RED（exit 42）を確認し、PRD の遷移表と開始値不変のケースを実装後に 2 tests / 33 assertions が GREEN。ログ: `.scratch/prd126/red-rise.log`・`green-rise.log`。
- TDD 3: `bit_fall_count:30` を拒否する RED（exit 42）を確認し、PRD の表と同時 2 ビットの減少・再初期化を実装後に 3 tests / 75 assertions が GREEN。ログ: `.scratch/prd126/red-fall.log`・`green-fall.log`。
- TDD 4: マスクの受理・拒否 56 assertions は通り、未知語の旧一覧だけで 2 assertions が RED（exit 42）。一覧更新後、4 tests / 133 assertions が GREEN。ログ: `.scratch/prd126/red-masks.log`・`green-masks.log`。
- TDD 5: 評価上書きが旧い専用検査で止まる RED（exit 42）を確認し、専用検査を除いて 5 tests / 136 assertions が GREEN。タグを先に切り出した設定でも、元の未読状態を共有して拒否する。ログ: `.scratch/prd126/red-eval.log`・`green-eval-direct.log`。
- 整理前の ROM あり Atari 検証は 46 tests / 1445 assertions が通り、隠し表示テスト 1 件だけが `SDL_VIDEODRIVER=dummy` 未設定で失敗した（exit 42、`.scratch/prd126/before-refactor-atari.log`）。通常のテストは `~[.]` で分け、隠し表示は必要な環境変数を指定して別に実行する。
- 整理では内部関数 `ValidateResizeMode` → `ParseResizeMode` とした。検証だけでなく列挙型への変換を行うため、処理を表す動詞へそろえた。固定参照の `PixelAreaReferenceResizer` は変更しない。
- 整理後の Debug 全体ビルドが exit 0（`.scratch/prd126/build-debug-all.log`）。自己点検でテストの共用ゲーム名を定数へまとめたあと、`--target AtariEnv-test` の Debug ビルドも exit 0（`build-final-tests.log`）。
- 最後の ROM あり通常検証: プロセス内で `ATARI_ROM_DIR=C:/dev/ale-sandbox/roms` を指定し、`core/envs/atari1/bin/Debug/AtariEnv-test.exe "[atari]~[.],[prd117]~[.]" --reporter compact --rng-seed 123` が exit 0、55 tests / 1602 assertions。skip なし。`.scratch/prd126/atari-final.log`。
- core 側: `core/anet-core/bin/Debug/anet-core-test.exe "[atari]~[.],[prd117]~[.]" --reporter compact --rng-seed 123` が exit 0、49 tests / 1410 assertions。`.scratch/prd126/core-regression.log`。
- 追加の隠し表示検証: `ATARI_ROM_DIR` と `SDL_VIDEODRIVER=dummy` を指定し、`AtariEnv-test.exe "[prd117_display]" --reporter compact --rng-seed 123` が exit 0、1 test / 7 assertions（`.scratch/prd126/display-final.log`）。最初は環境変数、次に SDL2.dll の読み込みで失敗した。ALE の読み込みは PATH を使わないため、既存の `C:/dev/ale-sandbox/build/Debug/SDL2.dll` をテストの `bin/Debug` に未存在の場合だけ配置して確認した。永続的な環境変数と本体コードは変えていない。
- 受入 5・6・8・9 の検索: env 本体の Run 階層名 0、語の名前は表の 8 行のみ、縮小方式の診断 1、旧縮小器の方式診断 0、`my_config_json_` 0、ゲーム名の正規表現の文字列 1。受入 1〜4・7・9 は上記のテスト、受入 10 は 220 §4.9 と既存 CONTEXT の 3 語で確認した。
- 自己点検: 実装品質確認観点の全項目を適用範囲ごとに通した。差分の空白エラー、追加行の直訳語、ヘッダの using namespace、本体の dynamic_cast・TODO は 0。対象ファイルは LF、既存 9 ファイルの BOM を保持。重複文字列の検査で見つかった新規テストのゲーム名は定数化し、設定キーは規約の除外に当たる。Atari.txt・固定参照の本体・既存の期待値は変更していない。
- 未実施の検証: PRD099 の隠し性能測定と大規模な固定参照比較は実行しない。PRD126 は性能の受入を置かず、画素の受入は通常の preprocess / prd099 を使うため。実装の受入に必要な検証の未実施はなし。
- 課題 949e8a で Run 全体の和集合を出せない事象は残るが、D7 により作らないと決まっている。実装範囲は完了しても、課題の AI 判定は事象が残る規則に従い「残件有り」とする。9ff3b7 は「残件無し」とする。状態はどちらも対応中のままにする。
- テスト実行体は `AtariEnv-test` であることを CMake と既存検証手順で確認し、計画の反復対象を訂正した。最初の `anet-core-test [prd126]` は該当テストがなく exit 2 で、検証の成功には数えていない。
- ROM のプロセス環境変数は未設定だが、既存の `C:/dev/ale-sandbox/roms` に Pong などの ROM がある。回帰検証ではこのパスをコマンドのプロセス内だけで指定する。

- レビュー 1 往復目: 計画 OK、実装は新規テストの定義・番号付け文字列 5 種の重複で差し戻し。使う範囲に合わせた定数へまとめた。5 種の文字列を固定文字列で検索し、それぞれ定数の宣言 1 行だけになったことを確認。関連 6 tests / 146 assertions が exit 0（`.scratch/prd126/review-fix-tests-direct.log`）。本体・期待値の変更はない。原文: `.scratch/prd126/review-20261112-0719.md`。

- レビュー 2 往復目: 計画・実装とも OK。修正要求・開発者への確認事項なし。レビューはログと差分を照合し、ビルド・テストの再実行はしていない。前回の範囲外の検出は下記の候補として保持する。原文: `.scratch/prd126/review-20261112-0729.md`。
- レビュー指摘の修正後、Debug AtariEnv-test ビルドが exit 0、関連 6 tests / 146 assertions が exit 0（`.scratch/prd126/build-review-fix.log`・`review-fix-tests.log`）。
- 課題の AI 判定を記録: 949e8a は残件有り（D7 の Run 全体の和集合）、9ff3b7 は残件無し。いずれも状態は対応中のまま。この時点では工程末の候補の選択待ちとして停止した。
- 2026-11-12 追加依頼: 開発者が範囲外の検出を「起票せずこの場で修正」と選んだ。同じ実装記録に対象と検証を追加し、既存テストの整理を行う。新しい課題は作らない。
- テスト用の共用ゲーム名は RAM 以外の設定検査でも使うため、`kRamMetricTestGame` → `kAtariTestGame` へ改名する。新しいテスト名は `Atari resize configuration inherits common values and keeps explicit overrides` とする。
- 追加修正: 出荷設定ファイルの読み取りと CLI の解析を除き、ゲーム名・共通の縮小方式・評価側の縮小方式を明示した `ConfigData` に置き換えた。共通値の継承、評価個別の上書き、実効設定を確認する 5 つの検査を残した。共通値の記録は入力そのものの読み返しから `AtariEnvConfig` が記録した実効設定の検査へ変えた。不要になった `wx/cmdline.h` を除いた。本体・出荷設定の変更はない。
- 追加修正の検証: MSVC 初期化後の `cmake --build --preset x64-Debug --target AtariEnv-test` が exit 0。`AtariEnv-test.exe "[prd126],[resize-profile]" --reporter compact --rng-seed 123` が exit 0、7 tests / 151 assertions（`.scratch/prd126/build-synthetic-resize.log`・`synthetic-resize-tests.log`）。本体は前段から変更していないため、今回変えたテストと共用定数を使う関連テストを再実行した。
- 追加修正の自己点検: 実装品質確認観点の全項目を適用範囲ごとに通し、追加した整理がユーザーの選択範囲内であることを確認。テストファイルから `ConfigManager`・`Atari.txt`・`shipped_config` の参照が無くなった。現用の core・apps・設計文書・CONTEXT で旧テスト名と旧定数名の参照は 0。LF・BOM を保持し、diff --check 成功。未実施の必要な検証・設計判断待ちはない。

## 決まらなかったこと

- なし。範囲外として提示した出荷設定テストの件も、開発者の選択によりこの場で修正し、検証とレビューを終えた。新しい課題は起票しない。

## 完了報告

- 実施内容: PRD126 の 3 語の追加と RAM 定義・縮小方式の読み取り整理、および開発者が選んだ出荷設定テストの合成設定への置換を完了した。
- 変更ファイル: 主な変更に列挙した AtariEnv / AtariPreprocess の本体・テスト、220 日本語版、PRD082 の検証手順、対象課題・派生区画と本記録。追加対応は `AtariEnv_test.cpp` と本記録。既存の別作業の変更は保持した。
- レビュー結果: 3 往復目で実装計画、PRD126 の実装、追加修正がすべて OK。原文: `.scratch/prd126/review-20261112-1051.md`。旧名の現用参照、LF・BOM・重複文字列と、ビルド・テストの記録を照合した。レビュー側ではビルド・テストを再実行していない。根拠がなく見送った指摘と、開発者への確認事項はない。
- 検証結果: 前段の Debug 全体ビルドと Atari 55 tests / 1602 assertions、core 49 tests / 1410 assertions、隠し表示 1 test / 7 assertions は成功。追加修正後の Debug AtariEnv-test ビルドと関連 7 tests / 151 assertions も成功。今回の変更に必要な未実施の検証はない。
- 決まらなかったこと・追加の起票候補: なし。Run 全体の和集合は D7 どおり作らず、既存課題 949e8a の残件として記録済み。9ff3b7 は残件無し。課題の状態はいずれも対応中で、解決は書いていない。
- 判断してほしいこと: なし。実装の工程を終えて停止する。
