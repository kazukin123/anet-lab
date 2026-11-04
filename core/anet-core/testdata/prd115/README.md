# PRD115の検証資材

画像メモリの確保・解放と確保失敗を、実際のheat_map.cppとwxWidgetsの組み合わせで確かめる。通常のC++回帰テストは`anet-core-test.exe "[image]"`と`DropMergeEnv-test.exe "[noop_penalty]"`にある。

## 前提

- Windows x64、MSVC、リポジトリのDebugビルドを使う。先に`anet-core-test`をビルドし、compile_commands.json・build.ninja・依存DLLを生成する。
- 画像生成の実ソースと検証用のmemory_probe.cppをコンパイルし、既存Debugのリンク先を使う。本体に検証用APIや設定は足さない。
- ASanではSTLの文字列・vectorのコンテナ注釈だけを未instrumentの依存ライブラリにそろえる。`alloc_dealloc_mismatch=1`でヒープの確保・解放を検査する。
- 修正前の基準はheat_map.cppのGit blob `1fff8bfb5629b1e23b4d732f3ddb3a7991286707`。保存済みファイルがなければ履歴から復元し、違う内容の基準ファイルでは停止する。

## 実行

リポジトリルートのPowerShellから実行する。MSVCの初期化とPythonを同じcmd内で起動する。

```bash
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && .venv\Scripts\python.exe core\anet-core\testdata\prd115\verify_memory.py --phase baseline --asan'
```

```bash
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && .venv\Scripts\python.exe core\anet-core\testdata\prd115\verify_memory.py --phase current --asan'
```

```bash
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && .venv\Scripts\python.exe core\anet-core\testdata\prd115\verify_memory.py --phase current'
```

`--case empty`、`--case time`、`--case histogram`、`--case sweep`で1経路に絞れる。省略時はすべてを独立プロセスで実行する。`--dump-only`は全画素の採取専用であり、生成・破棄の合格には数えない。

## 判定と記録

- ASanの基準側: 各経路が`alloc-dealloc-mismatch`で非0終了することを期待する。
- ASanの修正後: 4経路がそれぞれ100回の生成・コピー・破棄を完走し、0終了することを期待する。生成元と元画像は各回で先に破棄する。
- 確保失敗: 検証用Debug CRTフックをRenderRawの直前だけ設定し、画像の要求バイト数に一致する確保を1回失敗させる。フックが実際に発火した印と、ANET_SYSTEM_ERRORの対象名・幅・高さを確認する。注入は失敗直後に解除し、検証区間の終了時に依存ライブラリの既存フックを戻す。
- 画素の採取は破棄を検査しない別モードで、出力をflushした後に終了する。生成・破棄の検査では対象画像とコピーをすべて破棄してから、TerminateProcessで自分自身を終了する。計測やTorchなどのDLLの静的終了処理は対象に含めない。実際の描画・確保・解放・コピーの破棄は差し替えない。
- 出力は`.scratch/prd115/memory-<phase>-<asan|crt>/`のresult.jsonと各コマンドのlog。実行したソースのSHA-256も残す。コンパイル済みオブジェクトはソース・anetヘッダ・コンパイル条件の一致時だけ再利用し、リンクと実行は毎回行う。
- timeout・コンパイル失敗・検出できない経路・注入できない経路は未達として扱う。1経路の失敗でも、構築できた実行体の残りの経路は続けて記録する。通常Debugの完走で代替しない。

DropMergeのテストはfixed seed 123、罰0と-0.25の差を絶対誤差1e-5で比較する。観測・物理状態・生スコア・終了フラグを一致させ、現在のエピソード累積報酬も比較する。終了ログは既存のLogCaptureGuardで捕捉して非対話で実行する。
