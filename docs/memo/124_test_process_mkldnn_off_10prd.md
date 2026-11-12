# PRD124: anet-core-test の開始時に MKLDNN を切り、テストの順序で oneDNN の assert が出ないようにする

> 2026-11-11 起票（グリル）。行番号は起票時点の作業ツリー（HEAD `fea6d850` に未コミットの変更が乗った状態）の位置で、実装時は近傍の記述で探し直す。
> 起点: 2026-11-11 の PRD づくりの枠。課題 issue-20261109-fa2f69 を題材に渡され、再現と原因の確かめ（§2.1 の 5）のあとにグリルした。
> 題材: [issue-20261109-fa2f69](../management/issues/issue-20261109-fa2f69.md)。
> 関連: [PRD028 の実装記録](done/028_cnblock_convnext_20impl.md)（テストの中で MKLDNN を切る今の方式を決めた）、[PRD116](116_rnd_lifelong_novelty_10prd.md)（題材のテストと、`OMP_NUM_THREADS=1` で避ける手順）、[PRD119 の実装記録](119_small_doc_comment_config_fixes_20impl.md)（題材を見つけた経過）。

## 1. 問題

Debug 版の anet-core-test で、MKLDNN を切るのは一部のテストの中だけで、そのテストが回ったあとでないと切れていない。テストの実行順は Catch2 の既定でランダムなので、CPU で Conv を組むテストが oneDNN の経路を通るかどうかが、回すたびに、また回すテストの選び方で変わる。Debug 版 libtorch の oneDNN は、CPU の Conv で thread 数を確かめる assert（`nthr_ == nthr`）を持ち、それに当たると実行体ごと終了コード 3 で止まる。

その結果、次のことが起きている。

- 設定を変えたあとに回す決まりの `[shipped_config]`（[実装 品質確認観点](../quality/impl.md) の 156・417 行）が、RND の出荷設定のテストで毎回止まり、そのあとに回るテストが走らない。
- 既定のテストの集合も、確率で止まる（§2.1 の 4）。
- 避け方が、手順ごとに違う形で文書に散らばっている（§2.1 の 7）。

## 2. 現状（2026-11-11 に確かめたこと）

### 2.1 事実

1. 題材のテスト `RND shipped Atari networks and Taiga settings match their contracts`（[intrinsic_test.cpp](../../core/anet-core/src/intrinsic_test.cpp) の 157 行）は、`IntrinsicMetricsScope`（同 99〜108 行）で `InitNN` を呼ぶが、MKLDNN は切らない。Atari の RND の Network を CPU で組み、`Estimate` と `Learn` を回す。この Network は CNBlock を使わない普通の Conv の連なりである（[Atari.txt](../../apps/runner/config/Atari.txt) の 10・13 行の `AtariRndConv1 > … > AtariRndConv3`）。
2. MKLDNN を切るのは、テストの中の 2 か所である。どちらもビルド構成を問わず `at::globalContext().setUserEnabledMkldnn(false)` を呼ぶ。
   - [nn_test.cpp](../../core/anet-core/src/nn_test.cpp) の 101〜112 行の `EnsureNNInitialized`（無名の名前空間にあり、`InitNN` を 1 回だけ呼んで切る）。nn_test.cpp の中の 59 か所から呼ぶ。
   - [dqn_based_agent_test.cpp](../../core/anet-core/src/dqn_based_agent_test.cpp) の 684〜690 行の `PrepareNativeStackNetworkTest`（`InitNN` を呼んで切る）。同じファイルの 5 か所から呼ぶ。

   本番の `InitNN`（[nn_modules.cpp](../../core/anet-core/src/nn_modules.cpp) の 3274 行）と Run の初期化（[init.cpp](../../core/anet-core/src/init.cpp) の 172 行）は切らない。PRD028 で、本番は変えず、テストの中で切ると決めた（[PRD028 の実装記録](done/028_cnblock_convnext_20impl.md) の 15〜17 行）。
3. Debug ビルドだけが `libtorch/debug` をリンクし、RelWithDebInfo と Release は `libtorch/release` をリンクする（[CMakeLists.txt](../../CMakeLists.txt) の 141〜165 行）。assert は Debug 版 libtorch のものである（[PRD028 の実装記録](done/028_cnblock_convnext_20impl.md) の 16 行。PyTorch issue #104421）。
4. Catch2 は 3.14 で（[catch.hpp](../../third_party/catch2/include/catch.hpp) の 7539〜7540 行）、既定の実行順はランダム（同 3852 行）、順序を決める seed も既定で毎回作る（同 3836 行）。anet-core-test の main（[main_test.cpp](../../core/anet-core/src/main_test.cpp) の 24〜41 行）は順序を指定しない。既定の集合（899 本）では、題材のテストが、MKLDNN を切る約 60 本のどれよりも先に回ると止まる。約 60 本だけで数えた確率は約 1/61（約 1.6%）である。ほかに、切らずに CPU で Conv を組むテストがあれば、それが切るテストより先に回ったときにも止まりうる（その数は数えていない）。PRD119 の実装の検証で既定の集合が 2 回とも通ったのは、この確率の中のことである（[PRD119 の実装記録](119_small_doc_comment_config_fixes_20impl.md) の 59・62 行）。
5. 再現した。Debug の実行体（2026-11-11 01:09 のビルド）を、リポジトリのルートで、環境変数を足さずに回した。ログは `.scratch/prd124/` に置いた。

   | 回したもの | 結果 |
   | --- | --- |
   | 題材のテストを名前で単独に | 終了コード 3。`Assertion failed: nthr_ == nthr`（`dnnl_thread.hpp` の 294 行） |
   | `--order lex` で、MKLDNN を切る `LinearHeadFactory emits configurable output key` を先に置き、題材のテストと 2 本 | 2 本とも通過。終了コード 0 |
   | `--order lex` で、切らない `DQN intrinsic state scoring ignores next observations and terminal masks` を先に置き、題材のテストと 2 本 | 終了コード 3。同じ assert |

   止まるかどうかは、先に回ったテストが MKLDNN を切ったかで決まる。PRD028 の実装記録（16 行）は、assert が CNBlock の演算の連鎖で出るとしていたが、CNBlock を使わない題材のテストでも出る（1 の Network）。
6. テストの実行体は 7 つあり、それぞれが自分の main を持つ（anet-core-test の [main_test.cpp](../../core/anet-core/src/main_test.cpp)、5 つの Env のテスト（例: [AtariEnv_test.cpp](../../core/envs/atari1/src/AtariEnv_test.cpp) の 1395 行）、Runner の GUI のテスト）。CPU で NN を組むのは anet-core-test だけである。5 つの Env のテストのファイルは、Network や Agent を組む呼び出しを持たない。Runner の GUI のテストは偽の Agent を使う（[RunnerApp_test.cpp](../../apps/runner/src/RunnerApp_test.cpp) の 82 行、[EvalTracePanel_test.cpp](../../apps/runner/src/EvalTracePanel_test.cpp) の 90 行）。
7. assert を避ける手順が、文書に残っている。PRD116 の検証資材の [README](../../core/anet-core/testdata/prd116/README.md) の 17 行は「CNN を含むテストはそのプロセスだけ `OMP_NUM_THREADS=1` とする」と指示する。過去の記録にも、同じ避け方で回した記録がある（PRD116 の 23impl の 38・88・118 行、PRD918 の 20impl の 96・112・200・231 行、[212 設計文書](../design/212_rnd_lifelong_novelty.jp.md) の §5.2 の確認の表の 478 行）。

### 2.2 規約と過去の判断

- 暗黙のグローバル状態を増やさない（[AGENTS.md](../../AGENTS.md) の 48 行）。MKLDNN の切り替えは libtorch のプロセス全体の状態で、今はテストの中から、回った順に書き換わる。本 PRD は、書き換える場所を実行体の開始時の 1 か所にする（D1）。
- クリーンブレーク（[AGENTS.md](../../AGENTS.md) §クリーンブレーク方針）。要らなくなった切り替えと避ける手順を残さない。
- 過去の PRD・実装記録は書き換えない（[PRD 品質確認観点](../quality/prd.md) の項番 17）。

## 3. ゴール・非ゴール・成功の測り方

### 3.1 ゴール

- G1. Debug 版の anet-core-test を、どのテストの組み合わせ・順序で回しても、oneDNN の assert で止まらない。
- G2. 題材の課題が片づく。題材のテストを単独で回しても、`[shipped_config]` を回しても通る。

### 3.2 非ゴール

| 非ゴール | これで消える案 | 本当の制約か |
| --- | --- | --- |
| 本番の Runner の Debug 版を CPU で回したときの同じ assert | 本番の初期化（`InitNN` か Run の初期化）で、Debug ビルドのときだけ切る案 | 制約ではない。PRD028 の判断（本番は変えない）を保ち、テストの関心と分けた。クリーンブレークで移せるが、Debug 版の Runner を CPU で Conv の NN と回す必要が、まだ無い |
| テストの実行順を固定すること | main で既定の順序を宣言順にする案 | 制約ではない。単独やタグを指定した実行は、それでは直らない |
| oneDNN の CPU 経路をテストで確かめること | main で CPU のスレッド数を 1 にして、oneDNN の経路を残す案（§7） | 制約ではない。今も大半のテストは切った状態で回り（§2.1 の 4）、oneDNN の経路を確かめるためのテストは無い |

### 3.3 成功の測り方

- 題材のテストの単独、`[shipped_config]`、§2.1 の 5 で止まった組を、Debug で、環境変数を足さずに回して、どれも終了コード 0 になる（A1〜A3）。
- 見張りのテスト（D5）が、main の切り替えを一時的に外すと落ちる（A4）。
- `setUserEnabledMkldnn` を呼ぶのが main だけになる（A5）。

### 3.4 足すものと、ゴールへの寄与

| 足すもの | 寄与 |
| --- | --- |
| 実行体の開始時の切り替え（D1） | G1・G2。全テストが同じ状態で始まる |
| 見張りのテスト（D5） | G1 を保つ。Release では assert が出ないので、切り替えが消えたことに、Debug で単独に回す前に気づける |

先送りするものは無い。

## 4. 確定事項

2026-11-11 のグリルで決めた。D1 はユーザーが案を比べて選んだ。D5・D6 は、AI が示した推奨の既定を、ユーザーがまとめて採った。D2〜D4・D8 は、規約・現行コード・D1 から導いて示し、ユーザーが変えなかった。D7 は、起草で移行対象を検索して見つけ、D1 とクリーンブレークから導いた。

| # | 決定 | 理由と前提 |
| --- | --- | --- |
| D1 | **切る場所。** anet-core-test の main で、Catch2 を回す前に 1 回 MKLDNN を切る。テストの中の 2 か所（`EnsureNNInitialized` と `PrepareNativeStackNetworkTest`）からは切る行を消し、`InitNN` を呼ぶ働きは残す | 全テストが同じ状態で始まるので、順序と組み合わせで結果が変わらない（G1）。止まる 1 本だけに足す案は、題材の 1 本を直すが、切り替えを通らずに CPU で Conv を組むテストが順序で止まる仕組みが残り、写しが 3 か所になる（§7）。代わりに、anet-core-test のテストは oneDNN の CPU 経路を通らなくなる。これは非ゴールで（§3.2）、今も約 60 本のどれかが先に回れば以後は切れている（§2.1 の 2・4）。依存: なし |
| D2 | **ビルド構成。** 構成を問わず切る | 今の 2 か所が構成を問わず切っている（§2.1 の 2）のを保つ。Debug だけで切ると、Release のテストが oneDNN の経路を通るようになり、今と違う計算の経路で値を比べることになる。依存: D1 |
| D3 | **範囲。** anet-core-test の main だけを変える。ほかの 6 つの実行体の main は変えない | CPU で NN を組むのは anet-core-test だけである（§2.1 の 6）。ほかの実行体で NN を組むようになったら、Debug では順序によらず毎回止まるので、そのときに気づける。依存: D1 |
| D4 | **本番。** `InitNN` と Run の初期化（init.cpp）は変えない | 非ゴール（§3.2）。PRD028 の判断を保つ（§2.1 の 2） |
| D5 | **見張りのテスト。** テストが始まった時点で MKLDNN が切れていることを確かめるテストを 1 本足す。ほかのテストに頼らず、単独で回して通る形にする | Release の実行体は `libtorch/release` をリンクしていて assert が出ない（§2.1 の 3）ので、main の切り替えが消えても、Debug で CPU の Conv を組むテストを単独で回すまで気づけない。テストの中の切り替えは D1 で消すので、このテストが通るのは main が切ったときだけになる。依存: D1 |
| D6 | **コメント。** main の切り替えに、理由（Debug 版 libtorch の oneDNN が持つ thread 数の assert。PyTorch issue #104421）と、本番に入れない理由（PRD028 の判断と D4）を書く | テストの中の 2 か所にあるコメントを、1 か所へまとめて移す（D1）。依存: D1・D4 |
| D7 | **避ける手順。** PRD116 の検証資材の README の 17 行から、`OMP_NUM_THREADS=1` とする文を消す。同じ行の「本番コードのbackend設定は変更しない」は残す | D1 のあとは要らない手順で、残すとまだ要ると読める（§2.2 のクリーンブレーク）。本番を変えないのは D4 のとおりである。依存: D1・D4 |
| D8 | **過去の記録。** PRD028・PRD116・PRD918・PRD119 の実装記録と、212 設計文書の §5.2 の確認の表（当時の確かめ方の記録）は書き換えない | 過去の記録を書き換えない（§2.2）。212 の表の行は、当時どう確かめたかの記録で、D1 のあとも事実として正しい |

## 5. 実装契約

### 5.1 変える箇所

| ファイル | 変えること | 決定 |
| --- | --- | --- |
| [main_test.cpp](../../core/anet-core/src/main_test.cpp) | Catch2 を回す前に MKLDNN を切る。理由のコメントを添える | D1・D2・D6 |
| [nn_test.cpp](../../core/anet-core/src/nn_test.cpp) | `EnsureNNInitialized` から、切る行とそのコメントを消す。`InitNN` を 1 回だけ呼ぶ働きは残す | D1 |
| [dqn_based_agent_test.cpp](../../core/anet-core/src/dqn_based_agent_test.cpp) | `PrepareNativeStackNetworkTest` から、切る行とそのコメントを消す。`InitNN` を呼ぶ働きは残す | D1 |
| 見張りのテスト（置くファイルとタグは実装計画で決める） | テストが始まった時点で MKLDNN が切れていることを確かめる | D5 |
| [PRD116 の検証資材の README](../../core/anet-core/testdata/prd116/README.md) | 17 行から `OMP_NUM_THREADS=1` とする文を消す | D7 |
| 題材の課題の記録 | 対応内容と AI の判定 | — |

### 5.2 移行対象と、触らないもの

移行対象は §5.1 の表のとおりである。設定とサンプルに移すものは無い。設計文書・ADR・用語集は変えない。変えるのは実行体の中の初期化で、利用者の手順が変わるのは PRD116 の検証資材だけであるため（D7）。

変える経路を書いている箇所を検索した結果は次のとおり。

- `git grep -n -i -E "mkldnn|onednn|dnnl|nthr_" -- ':!third_party'` の当たりは、本 PRD を除いて 34 行である。そのうち 9 行は、本 PRD の起票に合わせて足した行（wiki の 311 の 7 行と目録 index.md の 1 行、課題の一覧 INDEX.md の 156 行の本 PRD の行）で、移行の対象として数えない。残りの 25 行の扱いは次のとおり。dqn_based_agent_test.cpp の 687〜689 行と nn_test.cpp の 105・106・108 行（6 行）は移行対象である（D1）。PRD116 の検証資材の README の 17 行は移行対象である（D7）。題材の課題の記述（課題の単票の 4 行、INDEX.md の 8 行、wiki の 321 の課題の区画の 1 行。計 6 行）は、課題の記録として扱う。PRD116 の 23impl の 2 行、PRD119 の 20impl の 3 行、PRD918 の 20impl の 2 行、PRD028 の実装記録の 5 行は、過去の記録で触らない（D8）。
- `git grep -n -E "EnsureNNInitialized|PrepareNativeStackNetworkTest"` の当たりは、nn_test.cpp（定義と 59 か所の呼び出し）、dqn_based_agent_test.cpp（定義と 5 か所の呼び出し）、題材の課題の 1 行、PRD028 の実装記録の 2 行である。名前と呼び出しは変えず、関数の中身だけを変える。
- `git grep -n -E "OMP_NUM_THREADS|MKL_NUM_THREADS"` の当たりは 10 行である。PRD116 の検証資材の README の 17 行は移行対象である（D7）。[run_performance.py](../../core/anet-core/testdata/prd116/run_performance.py) の 97・165 行は、GridMaze の性能の採取器の実行条件で、2026-11-07 の測定と同じ条件で回し直すために残す。212 設計文書の 478 行、PRD116 の 23impl の 3 行、PRD918 の 20impl の 3 行は、過去の確かめ方の記録で触らない（D8）。

触らないもの:

- 本番の `InitNN` と Run の初期化（D4）。init.cpp の 126 行の `torch::set_num_threads` も含む。
- ほかの 6 つの実行体の main（D3）。
- `EnsureNNInitialized` と `PrepareNativeStackNetworkTest` の名前と呼び出し。
- run_performance.py と、過去の記録（D8）。

### 5.3 クラス一覧

足す・変える・消すクラスは無い。変えるのは、anet-core-test の main 関数と、テストの補助の関数 2 つの中身である。

### 5.4 issue の対応表

| issueのID | 題 | area | 現在のコードで確認した根拠の位置 | 直す確定事項 | 確かめる受入 |
|---|---|---|---|---|---|
| [issue-20261109-fa2f69](../management/issues/issue-20261109-fa2f69.md) | [shipped_config]を単独で回すとRNDの出荷設定のテストがDebug版libtorchのoneDNNのassertで止まる | 321_agent_core | core/anet-core/src/intrinsic_test.cpp:157、core/anet-core/src/nn_test.cpp:101、core/anet-core/src/dqn_based_agent_test.cpp:684、core/anet-core/src/main_test.cpp:24 | D1・D2・D3・D5・D7 | A1・A2・A3・A4・A5・A6・A7・A8 |

## 6. 検証と受入

コマンドは、[AGENTS.md](../../AGENTS.md) の手順で Debug ビルドを通したあと、リポジトリのルートで回す。`OMP_NUM_THREADS` などの環境変数は足さない。

### A1. 題材のテストの単独

`core\anet-core\bin\Debug\anet-core-test.exe "RND shipped Atari networks and Taiga settings match their contracts"` が終了コード 0 で終わり、失敗が 0 である（G2）。

### A2. 出荷設定の検査

`core\anet-core\bin\Debug\anet-core-test.exe "[shipped_config]"` が終了コード 0 で終わり、失敗が 0 である（G2）。

### A3. 止まった順序

`core\anet-core\bin\Debug\anet-core-test.exe --order lex "DQN intrinsic state scoring ignores next observations and terminal masks,RND shipped Atari networks and Taiga settings match their contracts"` が終了コード 0 で終わる（G1）。§2.1 の 5 で終了コード 3 だった組である。

### A4. 見張りのテスト

- 足したテストを名前で単独に回すと、Debug の実行体で通る（D5）。
- 実装の工程で、main の切り替えを一時的に外してビルドし、同じテストを単独で回すと落ちることを確かめ、元に戻す。外した行、落ちた断定、元に戻したことを実装記録に書く。

### A5. 切り替えの置き場

`git grep -n "setUserEnabledMkldnn" -- core apps` の当たりが、core/anet-core/src/main_test.cpp の 1 行だけである（D1）。

### A6. 避ける手順

`git grep -n "OMP_NUM_THREADS" -- core/anet-core/testdata/prd116/README.md` の当たりが 0 件である（D7）。

### A7. ビルドと既存のテスト

Debug ビルドを通し、既定のテストの集合を回す。直したあとの実行は、終了コード 3 で止まらずに最後まで回り、直す前に通っていたテストが落ちない。比べる単位はテストケースの名前で、範囲は既定の集合の全部（隠しタグのテストは含まない）である。

比べる直す前の結果は、同じ手順で、実装に入る前に取る。直す前の実行が oneDNN の assert（題材の現象）で終了コード 3 で止まったら、その実行は比べる結果に使わず、`--rng-seed` を変えて最後まで回るまで取り直す。止まった点より後ろのテストは、直す前に通っていたかが決まらないため、回ったテストだけで比べると、そこで落ちたテストを見逃す。止まる確率は、切るテストだけで数えると約 1.6% で（§2.1 の 4）、取り直しは多くても数回で済む見込みである。使った seed、取り直した回数、両方の結果を実装記録に書く。

### A8. 課題の記録

題材の課題の対応内容に、直した内容と場所が書かれ、AI の判定が `残件無し` である。状態を `解決` にするのは開発者である。

## 7. 作らないもの・先送り

| 項目 | 裁定 | 理由と再開条件 |
| --- | --- | --- |
| 止まる 1 本（RND の出荷設定のテスト）の中だけで MKLDNN を切る | 不採用（D1） | 題材の 1 本は直るが、切り替えを通らずに CPU で Conv を組むテストが順序で止まる仕組みが残る。写しが 3 か所になる |
| 本番の初期化で、Debug ビルドのときだけ切る | 範囲外（非ゴール、D4） | 本番の Runner に触る。`InitNN` を呼ぶ前に Conv を組むテストは守られない。再開条件: Debug 版の Runner を CPU で Conv の NN と回す必要が出たとき。そのときは、本番の初期化に同じ切り替えを足せば消える（[PRD028 の実装記録](done/028_cnblock_convnext_20impl.md) の 17 行） |
| テストの実行順を宣言順に固定する | 不採用（非ゴール） | 単独やタグを指定した実行は直らない。順序に依存するテストを隠すことにもなる |
| main で CPU のスレッド数を 1 にして（`OMP_NUM_THREADS=1` か `at::set_num_threads(1)`）、oneDNN の経路を残す | 不採用（非ゴール） | 残す利点は oneDNN の経路のテストで、非ゴールである。CPU のテストがすべて 1 スレッドになり、Debug の集合が遅くなりうる（測っていない）。PRD116・PRD918 の実装記録が手順で避けた方法（§2.1 の 7）を、実行体の中へ移すだけになる |
| ほかの 6 つの実行体の main でも切る | 不採用（D3） | NN を組まない（§2.1 の 6）。再開条件: CPU で NN を組むテストを、ほかの実行体に足すとき |

## 8. 未決

なし。

## 9. 依存と順序

- 先に要る PRD は無い。
- [PRD117](117_warn_policy_and_unread_config_10prd.md) は、dqn_based_agent_test.cpp と nn_test.cpp の WARN を断定する行を見直す（PRD117 の 520 行）。本 PRD は、同じ 2 ファイルのテストの補助の関数 2 つの中身だけを変えるので、どちらを先に入れてもよい。

## 10. 参照

- 規約: [AGENTS.md](../../AGENTS.md) の §基本方針と §クリーンブレーク方針、[実装 品質確認観点](../quality/impl.md) の「テスト」、[PRD 品質確認観点](../quality/prd.md)。
- 地図: [311 開発ツール・テスト基盤](../wiki/categories/311_dev_tools.md)、[315 ニューラルネットワーク](../wiki/categories/315_nn.md)、[321 Agent 共通基盤](../wiki/categories/321_agent_core.md)。
- 課題の記録: [anet-update-issue](../../.agents/skills/anet-update-issue/SKILL.md)。
