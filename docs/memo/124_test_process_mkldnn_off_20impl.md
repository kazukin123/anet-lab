# anet-core-test の開始時に MKLDNN を切る 実装メモ

> 対象の PRD: [124_test_process_mkldnn_off_10prd.md](124_test_process_mkldnn_off_10prd.md)。題材: [issue-20261109-fa2f69](../management/issues/issue-20261109-fa2f69.md)。2026-11-11 に計画を保存した。

## 概要

anet-core-test の main で Catch2 を回す前に、`at::globalContext().setUserEnabledMkldnn(false)` を 1 回呼ぶ。テストの中の 2 か所（`EnsureNNInitialized` と `PrepareNativeStackNetworkTest`）からは切る行を消し、`InitNN` を呼ぶ部分は残す。テストが始まった時点で MKLDNN が切れていることを確かめる見張りのテストを 1 本足す。PRD116 の検証資材の README から、`OMP_NUM_THREADS=1` とする文を消す。ビルド構成は問わない。本番の初期化と、ほかの 6 つのテストの実行体の main は変えない（PRD124 の D1〜D7）。

## 主な変更

| ファイル | 変更 | PRD |
| --- | --- | --- |
| [main_test.cpp](../../core/anet-core/src/main_test.cpp) | `SetupTestFailureDialog` と `StderrLogGuard` のあと、`Catch::Session` を作る前に MKLDNN を切る。理由のコメントを添える | D1・D2・D6 |
| [nn_test.cpp](../../core/anet-core/src/nn_test.cpp) | `EnsureNNInitialized` から切る行とそのコメントを消す。`InitNN` を 1 回だけ呼ぶ働きは残す | D1 |
| [dqn_based_agent_test.cpp](../../core/anet-core/src/dqn_based_agent_test.cpp) | `PrepareNativeStackNetworkTest` から切る行とそのコメントを消す。`InitNN` を呼ぶ働きは残す | D1 |
| [test_util_test.cpp](../../core/anet-core/src/test_util_test.cpp) | `TEST_CASE("anet-core-test disables MKLDNN before any test runs", "[test_util]")` を足し、`CHECK_FALSE(at::globalContext().userEnabledMkldnn())` を断定する | D5 |
| [PRD116 の検証資材の README](../../core/anet-core/testdata/prd116/README.md) | 17 行から `OMP_NUM_THREADS=1` とする文を消す。「本番コードのbackend設定は変更しない」は残す | D7 |
| [題材の課題](../management/issues/issue-20261109-fa2f69.md) | 計画の保存で対応中にする。レビューの前に対応内容を書き、完了の報告の前に AI の判定を「残件無し」にする | A8 |

## テスト

- Public interface / surface: anet-core-test の実行体を外から回したときの終了コードと失敗の数。テストが始まった時点の libtorch のプロセス状態（`at::globalContext().userEnabledMkldnn()`）。
- 優先 behavior:
  1. 見張りのテストを単独で回すと通る（main が切っている）。
  2. 題材のテスト `RND shipped Atari networks and Taiga settings match their contracts` を単独で回すと終了コード 0（A1）。
  3. `[shipped_config]` と、PRD124 §2.1 の 5 で止まった組（`--order lex`）が終了コード 0（A2・A3）。
  4. テストの中の切り替えを消したあとも、`[nn]` と native stack のテストが通る。
- TDD 順序:
  1. RED: 見張りのテストを足してビルドし、単独で回して落ちることを見る（今は main が切っていない）。題材のテストの単独は終了コード 3（PRD124 §2.1 の 5 で再現済み）。
  2. GREEN: main に 1 行足してビルドし、見張りのテスト、A1、A3 を通す。
  3. REFACTOR: テストの中の 2 か所の切り替えを消し、`[nn]`、`[test_util]`、native stack のテスト、A2 を回し直す。
  4. A4: main の 1 行を一時的に外してビルドし、見張りのテストが単独で落ちることを見てから戻す。外した行と落ちた断定を経過に書く。
  5. 文書: README の文を消す。文書だけの変更なので TDD の対象外。

## 検証

ビルドは AGENTS.md の「AI エージェントでのビルド注意事項」の形で行う。

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test'
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug'
```

テストはリポジトリのルートで、環境変数を足さずに回す。単独・タグ・組の実行は、標準出力と標準エラーを `.scratch/prd124/<段>.log` に振り向け、終了コードと最後の要約行（`All tests passed` か `test cases:` の行）だけを返す。

既定の集合（A7、1 回約 10 分・標準出力約 3.5 MB）は、PowerShell の道具のバックグラウンド実行で回し、完了の通知を待つ（待つあいだにポーリングしない）。JUnit を `<before|after>.xml`、console の reporter を `<before|after>-console.log`、標準出力と標準エラーを `<before|after>-stdout.log` へ振り向け、返すのは終了コードと所要秒数の 1 行だけにする。結果は `<before|after>-console.log` の末尾 2 行（件数の要約）と、JUnit の XML をテストケースの名前で比べた結果で読む。

```powershell
core\anet-core\bin\Debug\anet-core-test.exe "anet-core-test disables MKLDNN before any test runs"
core\anet-core\bin\Debug\anet-core-test.exe "RND shipped Atari networks and Taiga settings match their contracts"
core\anet-core\bin\Debug\anet-core-test.exe "[shipped_config]"
core\anet-core\bin\Debug\anet-core-test.exe --order lex "DQN intrinsic state scoring ignores next observations and terminal masks,RND shipped Atari networks and Taiga settings match their contracts"
$sw=[Diagnostics.Stopwatch]::StartNew(); & core\anet-core\bin\Debug\anet-core-test.exe --rng-seed 124001 --reporter JUnit::out=.scratch\prd124\<before|after>.xml --reporter console::out=.scratch\prd124\<before|after>-console.log > .scratch\prd124\<before|after>-stdout.log 2>&1; "exit=$LASTEXITCODE seconds=$([int]$sw.Elapsed.TotalSeconds)"
git grep -n "setUserEnabledMkldnn" -- core apps
git grep -n "OMP_NUM_THREADS" -- core/anet-core/testdata/prd116/README.md
```

- A7 の基準（直す前）は、どのファイルも直す前に、今の作業ツリーで Debug ビルドし直してから既定の集合を回して取る。oneDNN の assert で終了コード 3 になったら、その実行は使わず、`--rng-seed` を変えて最後まで回るまで取り直す。あわせて `git status --short core apps` を控える。
- 直したあとは全体の Debug ビルドを通して既定の集合を回し、テストケースの名前で直す前と比べる。直す前に通っていたテストが落ちていないことを確かめる。`git status --short core apps` を直す前と比べ、自分の 4 ファイルのほかに core が変わっていれば（ほかのセッションの作業）、基準を取り直す。
- 課題の対応内容と判定を書いたら、`reports/tools/wiki_support.py maps` を回す。

## 前提

- ユーザー判断が要るブロッカーは 0。実装はこのセッションで行う（2026-11-11、ユーザーの回答）。
- main_test.cpp は `anet/test_util.hpp` → `anet/common.hpp`（14 行）を経由して `<torch/torch.h>` を読むので、`at::globalContext()` を使うためのインクルードは足さない。
- テストのソースは `core/anet-core/CMakeLists.txt` の 18 行の GLOB で集まり、ファイルを足さないので CMake は変えない。
- MKLDNN を有効に戻すテストや、`userEnabledMkldnn` を読むテストは無い。
- `EnsureNNInitialized` と `PrepareNativeStackNetworkTest` の名前と呼び出しは変えない。
- 内部リンケージの書き方は、[実装 品質確認観点](../quality/impl.md) の「C++ コード」の規約に従う。
- PRD 範囲外: Debug 版の Runner を CPU で回したときの同じ assert（PRD124 §7。工程末の候補で開発者が見送った）。

## 経過

- 2026-11-11: 計画を保存した。課題 fa2f69 を対応中にした。
- 2026-11-11: 直す前の基準。`--target anet-core-test` のビルドは作り直すもの無し（`ninja: no work to do`）。既定の集合を `--rng-seed 124001` でバックグラウンド実行し、完了の通知を待った。返したのは `exit=0 seconds=622` の 1 行。出力は `.scratch/prd124/` の `before.xml`（JUnit）、`before-console.log`（console の reporter）、`before-stdout.log`（標準出力と標準エラー、約 3.5 MB）。`before-console.log` の末尾で、899 本（898 通過・1 スキップ）、175,431 assertions がすべて通過。assert で止まらなかったので取り直しは 0 回。`git status --short core apps` を `.scratch/prd124/status-before.txt` に控えた。
- 2026-11-11: TDD 1（RED）。見張りのテストを test_util_test.cpp に足してビルドし、単独で回すと終了コード 42（`CHECK_FALSE( at::globalContext().userEnabledMkldnn() )` が失敗）。
- 2026-11-11: TDD 2（GREEN）。main_test.cpp の `StderrLogGuard` のあと、`Catch::Session` の前に切り替えを足した。見張りのテストの単独、A1（題材のテストの単独。21 assertions）、A3（`--order lex` の組。2 本・33 assertions）がどれも終了コード 0。
- 2026-11-11: TDD 3（REFACTOR）。nn_test.cpp の `EnsureNNInitialized` と dqn_based_agent_test.cpp の `PrepareNativeStackNetworkTest` から切る行とコメントを消し、main が切る旨の 1 行のコメントに替えた。`[native_stack]`（5 本）、`[nn],[test_util]`（144 本・2,481 assertions）、A2 の `[shipped_config]`（12 本・2,604 assertions）、A1、A3、見張りのテストがどれも終了コード 0。
- 2026-11-11: TDD 4（A4）。main_test.cpp の `at::globalContext().setUserEnabledMkldnn(false);` の 1 行を一時的に外してビルドし、見張りのテストを単独で回すと終了コード 42 で落ちた（test_util_test.cpp の 129 行の `CHECK_FALSE( at::globalContext().userEnabledMkldnn() )`）。1 行を元に戻し、`git diff` で戻したことを確かめた。
- 2026-11-11: TDD 5（文書）。PRD116 の検証資材の README の 17 行から `OMP_NUM_THREADS=1` とする文を消した。
- 2026-11-11: 基準を取ったあとに、ユーザーのコミット（PRD124 の文書、PRD122 の実装、課題の一覧の道具）が HEAD に入った。基準のとき（`status-before.txt`）、PRD122 の C++ のファイルは index にだけ変更があり、作業ツリーとの差は無かった。PRD122 の C++ のファイルの更新時刻はどれも基準のビルド（03:55）より前（最も新しいもので 03:24）で、基準のあとに書き換えられていない。今は `git diff HEAD --stat -- core apps` で C++ のファイルに差があるのは本実装の 4 ファイルだけである。このため、基準のときと今とで、本実装のほかの C++ のコードは同じと判断し、基準を取り直さずに使った。
- 2026-11-11: 仕上げ。全体の Debug ビルド（`cmake --build --preset x64-Debug`）は終了コード 0。既定の集合を同じ seed でバックグラウンド実行し、完了の通知を待った。返したのは `exit=0 seconds=532` の 1 行。出力は `.scratch/prd124/` の `after.xml`、`after-console.log`、`after-stdout.log`（約 3.5 MB）。`after-console.log` の末尾で、900 本（899 通過・1 スキップ）、175,432 assertions がすべて通過。テストケースの名前で直す前と比べ、直す前に通っていたテストで通らなくなったものは 0、増えたのは見張りのテスト 1 本、スキップは両方とも `PRD070 reports every explicit CUDA key before NN initialization` だけ（A7）。
- 2026-11-11: A5 `git grep -n "setUserEnabledMkldnn" -- core apps` は main_test.cpp の 43 行だけ。A6 `git grep -n "OMP_NUM_THREADS" -- core/anet-core/testdata/prd116/README.md` は 0 件。
- 2026-11-11: 機械検査（実装 品質確認観点の項番 3・7・62・72・76）はどれも期待値どおり。項番 1・2 は対象のファイルが無い（ヘッダと production のコードを変えていない）。
- 2026-11-11: 自己点検。実装 品質確認観点の全項目を適用範囲に従って通した。直したもの: なし。計画に無いものとして、`EnsureNNInitialized` と `PrepareNativeStackNetworkTest` に「main が切るのでここでは切らない」旨の 1 行のコメントを置いた（切る行を消した場所で、切り替えを書き戻さないための目印。項番 16 の範囲内と判断）。項番 35（グローバル状態）: 自前の可変な変数は足していない。libtorch のプロセス全体の状態を書き換える場所を、テストの中の 2 か所から main の 1 か所へ移した（PRD124 §2.2）。課題 fa2f69 の対応内容に今回の行を書いた（項番 74）。
- 2026-11-11: レビュー 1 往復目（`anet-review-agent`、成果物 `impl-plan` と `impl`、観点 `docs/quality/impl.md`）: `impl-plan` は OK、`impl` は差し戻し 1 件（項番 48。既定の集合の実行の標準出力・標準エラーの保存先、返した要約、完了の待ち方が記録に無い）。「検証」と「経過」に書き足した。原文は `.scratch/prd124/review-20261111-0437-impl.md`。
- 2026-11-11: レビュー 2 往復目: どちらも OK。原文は `.scratch/prd124/review-20261111-0450-impl.md`。再確認ポイント: 81 行の更新時刻の根拠はレビューでは確かめていない（A7 の結論はこれに依らない）。概要が D8 と run_performance.py（触らないもの）を書いていないが、主な変更の表が範囲を閉じていて実装も触れていない。
- 2026-11-11: 課題 fa2f69 の AI の判定を「残件無し」（2026-11-11）にし、`reports/tools/wiki_support.py maps` を回した（A8）。状態は対応中のまま。解決はコミットの準備で持ちかける。

## 未実施の検証

- Release（RelWithDebInfo）の実行体での見張りのテスト: PRD124 の受入に無く、Debug で確かめた。Release 版 libtorch では assert が出ないので、題材の現象は Debug でしか確かめられない。

## 完了報告

- 実施内容: anet-core-test の main で MKLDNN を 1 回切り、テストの中の 2 か所の切り替えを消し、見張りのテストを足し、PRD116 の検証資材の README から `OMP_NUM_THREADS=1` の手順を消した。受入 A1〜A8 を満たした。
- レビュー結果: 2 往復で OK（1 往復目に差し戻し 1 件。項番 48）。見送った指摘は無い。
- 未実施の検証: Release の実行体での見張りのテスト（上の「未実施の検証」）。
- 決まらなかったこと: なし。記録する課題の候補は無い。
