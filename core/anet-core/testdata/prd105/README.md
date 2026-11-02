# PRD105 Evaluation Trace 検証

実装と結果の正本は[実装メモ](../../../../docs/memo/105_eval_trace_arrow_key_forced_action_20impl.md)。

## 自動検証

MSVCのVsDevCmd.bat経由でDebug全体とAnetRLRunner-gui-testをビルドする。anet-core-testの `[prd105]` はEvalRunnerのforced、通知からの列、既定列、数値の並べ替え、再生順、ダンプの書式を確認する。既存の `[prd094]` とTrainer/GUIの関連回帰も実行する。

実wx操作はAnetRLRunner-gui-testのPRD105テストで確認する。RunnerAppを初期化するテストは通常のfixtureと別processで実行し、既存のユーザーのRunnerへイベントを送らない。

実Envで同じキー操作とログ欄を通す場合は `PRD105_UI_CONFIG` に、この節の準備コマンドで生成した `Atari-ui.txt` または `LunarLander-ui.txt` の絶対パスを指定し、`[prd105_ui_keys]` だけを実行する。学習とtimerのstepは停止し、実EvalRunnerを手動stepする。forced/prioのソート、QValueのクリック、info/verboseの表示差、ダブルクリックからログ欄へ反映するまでの実時間も確認する。外部ツールでのフォーカス移動や実キー入力の確認とは区別する。

## 学習値の比較

比較はユーザーが許可した並行実行条件で行う。変更前のexe・DLL・configは `.scratch/prd105/baseline-20261030-0138/` に保存した。既存の未コミット変更を含む変更前Debugビルドであり、過去のPRD098のバイナリとは別である。

2026-10-30 08:11 JSTに、ユーザーは評価Runと並行する追加検証を許可した。以降の比較と時間測定は並行実行の条件を付けて記録する。

専用ドライバ `run_validation.py` はPRD098の段0の条件生成とPRD084の正規化を呼び、今回の保存先へbefore/afterを環境ごとに逐次実行する。`--output` は新しいディレクトリに限る。各Runの実行体/設定のhash、終了値、実時間、正規化した点数とchecksumを保存する。失敗済みのディレクトリも上書きしない。

実行済みの比較コマンド（同じ保存先で再実行はできない）:

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd105/run_validation.py --baseline .scratch/prd105/baseline-20261030-0138 --output .scratch/prd105/validation-20261030-0811b
```

実行済みの実wx入力準備コマンド（Runは開始しない）:

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd105/run_validation.py --baseline .scratch/prd105/baseline-20261030-0138 --output .scratch/prd105/ui-20261030-0815 --prepare-ui
```

PRD098の段0と同じ条件でGridMaze/LunarLander/Atari/DropMergeの4環境を比較する。seed=1、8lane、batch=16、warmup=256、replay ratio=1、8,704experience、評価停止、FP32、決定論的backendとする。Atariはqbert。LunarLander/GridMazeはCPU、Atari/DropMergeはCUDA。比較対象は `90_perf/`・`91_resource/`・`92_resource/` を除く全scalar点で、tag・step・値の完全一致とする。forcedをscalar定義へ追加しない。

## 実画面での受入

[PRD105の受入基準](../../../../docs/memo/105_eval_trace_arrow_key_forced_action_10prd.md#7-受入基準)の4・5・10を行う。

- qbertのTraceに21行以上ため、先頭行を選択する。EvalのViewへ左クリックでフォーカスを移し、↓を20回押す。選択だけが21行目へ移り、Evaluation ViewのGame stepと最新行のstepが変わらないことを確認する。←→は行・選択・Game stepを変えない。Train Viewへフォーカスを移し↑20回で先頭へ戻る。
- StopでLiveへ戻して↓を押すと1stepの強制行動になり、forced=1と `Forced action: 2:UP` を確認する。テンキーとQValueの行のクリックも同じ。Agentの通常stepはforced=0。Trace閉鎖時も通知は出る。LunarLanderではLive/閉鎖時の矢印0〜3を確認する。
- forced/prioの見出しで降順→昇順、見出しの印、再生が表の下端から上へ進むこと、選択行と再生状態の維持、列幅のドラッグ、閉じて開き直したときのstep降順を確認する。
- 任意の行をダブルクリックし、Runのテキストログに同じ `Trace dump step=... lane=...:` で始まるINFO1件とVERBOSE1件を確認する。VERBOSEは複数行の既存の文字列化を含む。Log Levelがinfoならログ欄はINFOのみ、verboseなら両方を表示する。
- verboseでダブルクリックしてからログ欄へ詳細が現れるまでの実時間を測り、条件と結果を実装メモへ記録する。重い出力を毎stepへ移さない。

## 結果

コマンド・合否・実測・未実施と理由は実装メモへ追記する。自動テストの成功で実画面の受入と4環境比較を代替しない。

2026-10-30の4環境比較は `.scratch/prd105/validation-20261030-0811b/comparison.json` に保存し、全105,682点（片側合計）が一致した。実Envを使う実wxテストはqbert/LunarLanderとも73assertion成功。外部のUI操作ツールは起動不能だったため、実キー入力・フォーカス・外観の外部確認は未実施である。

qbertの全要素ダンプはverbose約175万文字で、ログ欄へ反映するまで約14秒かかった。info表示でもVERBOSEの生成とファイル出力は行うため約15秒かかった。ログ欄のlevelによる表示差と処理時間の軽減を混同しない。
