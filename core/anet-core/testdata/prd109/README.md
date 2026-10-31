# PRD109 起動の非同期化の検証

正本は [実装メモ](../../../../docs/memo/109_runner_startup_background_init_20impl.md)。結果、コマンド、未実施と理由は実装メモへ追記する。

## ビルドと変更前の採取

MSVC の VsDevCmd.bat 経由で Debug 全体と AnetRLRunner-gui-test をビルドする。変更前の exe・DLL・config・hash は `.scratch/prd109/baseline-20261031/` に保存する。既定スイートはリポジトリルートを作業ディレクトリとし、JUnit の失敗したテスト名の集合を前後で比較する。採取先は毎回新規にし、前の記録を上書きしない。

`run_validation.py` の `snapshot` は exe・DLL・config の保存、`suites` は core/GUI の既定スイート、`fatal` は製品 Runner の起動失敗、`compare` は4環境の前後比較を実行する。GUI test は通常の全体 build には含まれないため、別にビルドする。

今回の実行記録（同じ保存先は再使用しない。再実行は新しい出力先にする）:

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd109/run_validation.py suites --output .scratch/prd109/after-suites --baseline .scratch/prd109/before-suites-repo --baseline-gui .scratch/prd109/before-gui.xml
```

`--baseline-gui` は保存済み変更前GUIバイナリから採取したJUnitを指定する。今回のcore基準はbefore-suites-repo/core.xml、GUI基準はbefore-gui.xmlであり、before-suites-repo/gui.xmlは採取時のテスト入口の判定不足があったため使用しない。

## GUI の自動検証

RunnerApp を起動するテストは隠しタグ `[.][prd109_startup]` を持ち、個別のタグで1プロセス1ケースを実行する。`[prd109_startup]` 全体を同一プロセスで実行しない。既存の `[eval_steps_menu]` と `[prd105_ui_keys]` もそれぞれ別プロセスで実行する。

テスト用 SlowInitTestAgent は gui-test だけに登録する。製品設定には追加しない。3秒の構築中の100ms timer、完了前のログ、menu/pane/resize、終了要求の重複、OS終了問い合わせ、強制Close、構築失敗、RUNNINGでない結果、CallOnExitでFrameを先に削除する道を確認する。待機上限は30秒とする。

待機上限30秒は、承認済み計画と改訂PRD §6の6の明示値を使う。GUI子process全体の上限は60秒とし、待機のほかに起動と後始末を含める。

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd109/run_validation.py gui --output .scratch/prd109/gui-validation
```

`fatal` は未知のeval scheduleで実Runnerを起動し、終了コード1、Runログに `Unknown run.eval_schedule tag 'nosuch'` があり、`stderr.log` には無いことを確かめる。Frame表示後のログ先はLogPanelとRunログであり、stderrへは重ねて出さない（改訂PRD受入5、PRD068のD5）。

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd109/run_validation.py fatal --output .scratch/prd109/fatal-routing-fixed-20261031-1947
```

## 学習値の比較

PRD098 の短縮条件生成と PRD084 の正規化を再利用する。前後の Runner には同じ凍結 config を渡し、現在の実験設定の編集を比較へ混ぜない。過去の PRD105 の実行結果は使い回さない。

GridMaze・LunarLander・Atari qbert・DropMerge を環境ごとに前後各1本、逐次実行する。seed=1、8lane、batch=16、warmup=256、replay ratio=1、8,704experience、評価停止、FP32、決定的backend。GridMaze/LunarLanderはCPU、qbert/DropMergeはCUDA。90_perf/・91_resource/・92_resource/を除く全scalar点をtag・step・値とも完全一致で比較し、点数とchecksumを保存する。

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd109/run_validation.py compare --baseline .scratch/prd109/baseline-20261031 --output .scratch/prd109/learning-comparison
```

## 実機での受入

人が Atari5 qbert の online 起動で、初期化中にFrameを移動・resizeし、別の窓を重ねて外す。乱れが残らず「応答なし」にならないこと、構築中のログが流れること、初期化中の✕が一度で受理されて構築後に閉じることを確認する。失敗設定を show_error_dialog=true で起動し、描画済みFrameの上にエラーダイアログが出ることも確認する。

OSの終了確定通知は Close を通さない。自動テストは実際にログオフせず、公開の CallOnExit でウィンドウ削除→OnExit の順序を通す。実際のOS通知の配送は使用中のwxWidgetsソースで確認する。

自動テストで人の目視確認を代替しない。未実施の受入がある場合は完了としない。

## 2026-10-31 の自動検証結果

- Debug全体・GUI testのビルドとレビュー修正後の12プロセスのGUI検証（`gui-review-fixed/`）は成功。製品fatalの旧記録（`fatal-review-fixed/`）はstderrにも診断を要求していたため、改訂PRDの合格根拠には使わない。stderrへの追加出力を撤去した検証結果は実装メモの「追加指摘への対応」に記録する。
- 追加指摘修正後もDebug全体・GUI testのビルドは成功。`fatal-routing-fixed-20261031-1947/` は終了コード1、Runログに診断あり、stderr.logは0バイトで成功。`gui-routing-fixed-20261031-1947/` の12プロセスとJUnit12件も全件成功した。修正前の `fatal-routing-red-20261031-1947/` はstderrへの診断出力で失敗しており、RED→GREENを保存している。
- 既定スイートの失敗集合は前後同じ。coreは失敗0、GUIは既存の折り返しインデントの1ケースだけである。
- `learning-comparison/comparison.json` で4環境の全105,682点が完全一致した。比較に使用した変更後exeは `.scratch/prd109/validated-after.exe` に保存した。その後のproduction変更は段階コメントとstderrへの追加出力の撤去で、学習処理への変更は無いため比較は再実行しない。

## 2026-10-31 の実機確認・受入結果

- ユーザーから「もろもろ実動作確認してoｋ」と報告を受領し、人による実機受入6の確認待ちを解消した。これは自動テストとは別の、ユーザーによる確認結果である。
- 修正後のcc-reviewもCodex・Claudeとも計画・実装がOK。既存の自動検証結果と合わせ、PRD109の受入完了とする。レビュー原文、全コマンド・成果物の保存先、各時点の履歴は実装メモを参照する。
