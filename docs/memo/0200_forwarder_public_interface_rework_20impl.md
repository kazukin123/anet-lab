# Forwarder 公開インターフェース 実装メモ

## 概要

[RFD0200](0200_forwarder_public_interface_rework_10rfd.md) に従い、設定から変換を組む `ForwarderBuilder` と、依頼ごとに変換を実行する `SpecForwarder` を分ける。DefaultDQN は、Agent の `frame_stack.*` で決めた枚数とキーを、ReplayBuffer と各依頼へ渡す。Sample 用の入力 spec は ReplayBuffer が返す。

## 主な変更

- `anet/forwarder.hpp` に依頼、Builder、部品と工場の登録口を置く。`anet/init.hpp` に `anet::rl::detail::InitForwarderModules` を宣言する。基盤の `anet::rl::InitRL` と `src/core/base/src_testing/main.cpp` のテスト初期化から呼び、標準の工場を登録する。Builder は種類の共通部分を求め、設定どおりに部品を並べる。依頼で作った Forwarder が入力・出力 spec と図を返す。
- Builder は未読キー、値の型と値域、並び全体で使える Forward の種類との組み合わせの順に、全 block を検査する。RFD §6.7 の報告順を外部の部品にも当てるため、工場に `ReadConfigKeys`、部品に `ValidateForwardKinds` を追加した。`CreateModule` は値の読み取りと構築を受け持つ。外部の部品が守る契約を RFD §6.3 と設計文書に記した。
- 標準の `FrameStack`、`Normalize`、`DropKeys` を移し、`SetSpec` を加える。経験の受入れ用の依頼で正規化の統計を確保し、Perception 用では届いた宣言と統計を照合する。`SetSpec` は指定された属性だけを検査して変える。
- `ReplayBuffer::GetSampleObsSpecs` に通常実装とプリフェッチの委譲を加え、テストの差し替えも更新する。DefaultDQN は経験の受入れ用を最初に作り、可視化用の出力を Network に渡す。行動選択用と Sample 用の出力を構築時に照合する。必要な長さ 1 の stack 軸と開始の印の除去は DQN 内で扱う。
- 出荷設定 4 ファイル、固定入力の移行、基準 17 ファイルとその README を更新する。既存の Montezuma 起動は出荷設定の既定で 4 枚・全キーを重ねる。日本語設計文書の新旧の契約を照合し、拡張点を追加する。実装結果を課題と wiki に反映する。

| 旧名 | 移行先 |
| --- | --- |
| `ForwarderChain` | `ForwarderBuilder` |
| `ForwarderProvider` | `ForwarderModule` と工場 |
| `anet::ForwardExecutor` | 依頼で作る `SpecForwarder` |
| `ForwardRequest` | `ForwarderRequest` と `ForwardKind` |
| `ForwardSpec(` | 作られた変換の `GetInputSpecs` と `GetOutputSpecs` |
| `MakeForwarder(` | Builder の `BuildForwarder` と部品の `CreateForwarder` |
| `GetStackCount`、`GetStackKeys` | Agent が `frame_stack.*` から決めて配る |
| `perception.block.[stack].count`、`perception.block.[stack].keys.*` | `frame_stack.count`、`frame_stack.keys.*` |

## テスト

各行で、先に失敗を観測してからその挙動を成立させる。複数条件の行では一つずつ追加して実行する。不要になった旧口は呼び出し元の移行に合わせて取り除く。

| 順 | 入口 | 確かめる挙動 |
| --- | --- | --- |
| 1 | `[forwarder]` の公開ヘッダ、工場の登録、`BuildForwarder`、`Forward` | テスト独自の種類と部品を設定で並べ、入力 spec を返し、実行結果と出力 spec が合う |
| 2 | 同じ公開 API と設定 | 二重登録、未知の `type`、扱えない種類、空 spec、読まれない完全キーの拒否 |
| 3 | ReplayBuffer の `Sample` と `GetSampleObsSpecs` | 1 枚と 4 枚で軸の有無が実データと合い、対象外のキーは重ならない |
| 4 | `[perception]` の Builder と公開 ReplayBuffer | 4 種類の入出力と図、1 枚・4 枚の行動選択と Replay 復元、終端と skip の窓での一致 |
| 5 | DefaultDQN、Actor、Learner の構築と実行 | 軸と開始の印の扱い、Agent 設定の検査、外部部品が作る不一致の拒否。Rainbow の経路も既存テストで確認する |
| 6 | `[perception]` の標準部品と設定 | 正規化の統計の用意と読み出し、両方の並べ方、`SetSpec` の適用範囲と拒否条件、尺度で統計の対象が切り替わること |

## 検証

- Debug をビルドし、`out/test/Debug` に生成された `*-test.exe` をすべて引数なしで実行する。関連する `[forwarder]`、`[perception]`、`[replay_buffer]` と出荷設定のテストも確認する。RFD §7 の固定文字列を指定範囲で検索し、`git diff --check` を実行する。
- 固定 17 構成は `src/core/base/testdata/rfd0072/prepare.py` の `prepare` と `compare` を順に実行する。最初の `compare` が書いた `actual/` から、対応する `frame_stack.count`、`frame_stack.keys.mode` の位置と値だけを採る。HEAD と比較して、他の値と順序、`baseline_commit`、`resolution`、manifest と別の resolution 基準が同一であることを確認し、`compare` を再実行する。
- 変更直前の HEAD `87530c58` の Release Runner と出荷設定を確保する。LunarLander の作業中の変更は比較入力から外し、比較後側の設定は HEAD の出荷設定に今回の設定キー移行を当てたものを使う。GridMaze、LunarLander、DropMerge、Atari Q*bert の前後に同じ seed と `backend.$=backend.@deterministic` を指定する。Q*bert には `E1.game=qbert` を足す。`run.$`、ゲーム名、seed、両決定性フラグを実効設定で確認する。
- 学習 1,000 ステップに当たる経験数は、設定解決後の warm-up と学習頻度をもとに決め、実際の終了経験数と併せてここに記録する。旧側に TD 誤差か学習リターンがなければ、学習 10,000 ステップまで延ばして前後を再取得する。`src/core/base/testdata/rfd0097/compare_metrics.py` の `exact_match`、両レコードの件数、新側の構造図と詳細図を確認する。

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --parallel 4'
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Release --target AnetRLRunner --parallel 4'
git diff --check
```

## 前提

- RFD はレビュー済み。基準コミット `0411d5eb` から現行 HEAD までに `src`・`apps`・`data` の追跡内容の変更はない。用語集と ADR 0084・0086・0094〜0096 は起草時に更新済み。
- 枚数のコード既定は 1 で、出荷の基準プロファイルは 4。重ねるキーの既定は全キー。ReplayBuffer の 1 枚時の実データは軸なしのまま、DefaultDQN は入力に必要な軸をそろえる。Builder が渡らない Rainbow は現在と同じ経路を使う。
- 評価専用の `learner.enabled=false` では、行動選択と可視化の出力 spec を照合する。この設定では ReplayBuffer の構築を省くため、Sample 用の変換を作る段階に入らない。
- RFD §10.4 の直近系列バッファ、RND の観測加工、正規化統計の並行読み書きと checkpoint、NN の呼び出し情報は今回の範囲外。別作業の差分がある出荷設定、課題一覧、wiki には必要な変更だけを加える。
- Actor は観測だけを実行 device に移し、開始の印は CPU の真偽値で渡す。FrameStack の履歴は依頼ごとに持ち、各 Tensor の device で初回に用意する。毎 step の追加は小さい TensorDict の組み直しと鍵の参照で、新しい device 同期は入らない。正規化の統計を用意するのは変換の構築時とする。

## 経過

- 2026-11-24。設定と公開 API、ReplayBuffer、呼び出し元、用語集と ADR を照合し、開発者の判断が必要な未決は 0 件と判断した。
- 2026-11-24。独自の Forward の種類と部品で実行するテストは、変更前に公開口がなくビルド失敗し、登録と Builder を追加して 6 断定が通った。続いて拒否条件のテストは 5 断定の失敗を確認し、拒否を追加したあとに `[forwarder]` 全 5 件・24 断定が通った。ログは `.scratch/rfd0200/forwarder-*.log`。
- 2026-11-24。変更前の Release Runner をビルドし、SHA-256 `6C9EE014CF0AF28C5C6BAACFCEC4627C7FC7EB9B365C3C999CEC9DEE035383B0` の実行体と HEAD の設定を保存した。各試行は `_main.txt` と環境別設定を順に読み、seed 1、決定的なバックエンドを指定した。前側の GridMaze 131,072、LunarLander 40,704、DropMerge 282,112、Q*bert 456,320 経験が終了コード 0。TD 誤差と学習リターンの件数は順に 500/499、125/92、1,000/878、1,000/1,065。Run の保存先は `.scratch/rfd0200/before/verified-runs`。
- 2026-11-24。ReplayBuffer の公開口を追加し、1 枚と 4 枚の Sample の shape・dtype に対する 19 断定が通った。4 種類の依頼の入力 spec と出力の 62 断定、正規化統計を経験の受入れで用意して再利用する 9 断定、SetSpec の種類指定と素通しの 6 断定も通った。旧口撤去後の関連テストは Core 21 件・421 断定、DQN 9 件・46 断定が通った。追加の受入テストは全体実行で確認する。
- 2026-11-24。固定 17 構成は `prepare.py prepare` と `prepare.py compare` を使用し、旧キーが残る比較の 17 失敗を確認した。新しい解決結果から対応する 2 種類の設定キーの位置と値を採り、ほかの値と並び、`baseline_commit` と `resolution`、manifest と resolution の期待値を基準コミットと照合した。`prepare.py compare` の再実行は 157 断定通過、終了コード 0。ログは `.scratch/rfd0200/fixed17-*.log`。
- 2026-11-24。Debug 全体と Release Runner のビルドは終了コード 0。`[shipped_config]` は 8 件・955 断定が通った。`check_default_leaves.py` は変更した出荷設定の指摘 0 件、既存の `apps/tools/metrics_viewer.py` の Python 代入 4 行を設定代入として拾い終了コード 1。`git diff --check`、改行、BOM、RFD §7 の旧名検索を確認した。ログは `.scratch/rfd0200/full-debug-build.log`、`.scratch/rfd0200/after-release-build.log`、`.scratch/rfd0200/shipped-config.log`、`.scratch/rfd0200/config-default-audit.log`。
- 2026-11-24。追加の受入テストで、SetSpec のラベルと範囲値の個数が不正な場合に完全キーが出ないことを確認した。読み取った設定値と出典を既存の記録に載せるよう修正した。全体テストの初回は 9 実行体が通過し、Core はこの 3 断定に加え、既存の並行ログ捕捉テスト 2 断定で出力行数の差が出た。診断修正後の Debug 全体と Release の前後比較を再実行する。初回の全体テストは `.scratch/rfd0200/debug-tests/`。
- 2026-11-25。診断修正後の Debug ビルドと Release Runner ビルドが通った。Debug の Core `[perception],[forwarder],[shipped_config]` は 32 件・1,415 断定、DQN `[perception],[native_stack],[obs_config]` は 11 件・59 断定が通った。ログは `.scratch/rfd0200/final-debug-build.log`、`final-release-build.log`、`final-core-scoped.log`、`final-dqn-scoped.log`。計画と実装を独立レビューに依頼した。
- 2026-11-25。独立レビューの初回は実装計画 OK、実装は差し戻し。未読キーと値・組み合わせの報告順、受入テストの条件、補助関数の置き場、個別の計測、現用バッチの旧キーを修正した。複数の設定エラーを重ねたテストは修正前に 3 断定とも失敗した（`.scratch/rfd0200/validation-order-red.log`）。英語設計文書の修正指摘は RFD §6.9 が翻訳の道具による再生成を指定しているため、実装工程の範囲から外した。
- 2026-11-25。レビュー修正後の Debug 全体と Release Runner のビルドが通った。対象の Core は 34 件・1,428 断定、DQN は 11 件・65 断定が通った。出荷設定の検索条件と `git diff --check` に指摘はなかった。固定設定比較の README のコマンドを現在のパスに揃え、固定 17 構成が 157 断定、既定キー検査器の回帰テストが 16 件通った。既定キーの検査本体は従来の Metrics Viewer 用 Python 代入 4 行で終了コード 1。ログは `.scratch/rfd0200/review-*.log`。全 Debug テストを最終実行体で再実行する。
- 2026-11-25。バッチの `STK` 定義を読み取り、その値を Runner に直接渡した短い Montezuma Run が終了コード 0。実効設定のゲーム名、4 枚、全キーを照合した。レビューの 2 回目で公開拡張口の責務を RFD に記す指摘を受け、同 RFD を改訂した。設定の固定キーは既定のマクロで読む形に揃え、テストの補助関数を無名名前空間から移した。
- 2026-11-25。改訂した RFD、実装計画、実装の独立レビューはいずれも 3 回目で OK。設定エラーの優先順、外部の工場と部品の責務、固定キーの読み取り、テスト補助関数の配置を照合した。原文は `.scratch/rfd0200/review-20261125-0043-rfd.md` と `.scratch/rfd0200/review-20261125-0045-impl.md`。開発者の判断が必要な事項は無い。
- 2026-11-25。最終 Debug、Release Runner のビルドが通った。最終 Debug の Core `[forwarder],[perception],[shipped_config],[replay_buffer]` は 147 件・20,631 断定、`prepare.py compare` は 157 断定を通過した。改行と BOM は既存の構成を保ち、バッチは CP932 と CRLF を照合した。ログは `.scratch/rfd0200/acceptance-*.log`。

### Codex・Claude のレビュー対応

- `FrameStack` の構築時に、開始の印の spec が、各行で stack 軸のないスカラーの `bool`（空の shape）であることを確認する。Actor は、部品列の実際の出力に開始の印が残っていれば行動選択の前にエラーにする。
- 正規化だけの構成で、通常の `Actor::MakeAction` を通すテストに変えた。開始の印の除去だけを外すと、テストは行動選択前のエラーで失敗した。開始の印を出力 spec から除いたと宣言した外部部品が実際には印を返すケースを、通常の `Actor::MakeAction` が拒むことも確かめる。`FrameStack` の不正な dtype と shape は、構築時に拒否するテストを加えた。
- 既存の Montezuma 起動に使われていなかった `STK` 定義を外した。重ね方は出荷設定の既定で 4 枚・全キーとする。固定 17 構成は、RFD0200 適用前と同じ末尾の改行なしの形に戻した。
- 部品ごとの処理順をコメントに記した。`DropKeys` に従来名の個別計測を戻し、`SetSpec` の設定値を実装とテストでそれぞれ定数にまとめた。

**再レビュー用の版での検証**

- `cmake --build --preset x64-Debug --target anet-core-base-test anet-algo-dqn-test` と `cmake --build --preset x64-Release --target AnetRLRunner` に相当する対象ビルドを MSVC 環境で実行した。Core の `[forwarder],[perception],[shipped_config],[replay_buffer]` は 148 件・20,633 断定、DQN の `[perception],[native_stack],[obs_config]` は 11 件・64 断定が通過した。DQN では、登録した外部部品が実際に開始の印を返したとき、通常の Actor が行動選択の前にエラーにすることを確認した。ログは `.scratch/rfd0200/cc-round2-{core,dqn}.log`。
- 固定 17 構成は `prepare.py prepare` と `prepare.py compare` を通過した。計測名修正後の版でも `compare` の 157 断定が通過した。JSON 17 件の変更は末尾の LF の除去だけで、現用バッチは CP932 と CRLF を確認した。対象の `git diff --check` も通過した。ログは `.scratch/rfd0200/cc-round2-fixed17-compare.log`。
- 現行の出荷設定を使い、重ね方を指定せずに Release Runner で Montezuma を 128 ステップ実行した。終了コード 0、実効設定は `AtariEnv.game=montezuma_revenge`、`DefaultDQNAgent.frame_stack.count=4`、`DefaultDQNAgent.frame_stack.keys.mode=all`。起動手順は `.scratch/rfd0200/run-cc-montezuma-smoke.ps1`、実効設定の保存先は `.scratch/rfd0200/after/cc-round2-default-smoke/montezuma_defaults/config/config_data.txt`。
- 1 回目の修正後に全 10 実行体を実行し、Core を除く 9 実行体は計 550 件が通過した。Core では seed `2924472815` の順で `Logger prefixes an info message exactly once` が 1 件であるはずのログを 4 件捕捉し、最新版で同じ seed を指定しても再現した。単独実行は通過した。最新版の Core 全体を実装工程で通過した seed `2899524567` で再実行し、693 件が通過、CUDA が利用可能なため 1 件をスキップした。ログは `.scratch/rfd0200/debug-tests-cc/` と `.scratch/rfd0200/cc-round2-core-full-{retry,known-order}.log`。

## 4 環境の等価性

変更後の Release Runner の SHA-256 は `E96E9458CE3666247131F4B9819AC8F63C0C9FE1B8843B7A1CAEDAF81B4D4CC2`。前後とも seed 1、`backend.deterministic_algorithms=true`、`backend.cudnn_deterministic=true`。後側の実効設定は、出力先と重ねる設定キーの移行を除いて前側と一致した。全 Run の終了コードは 0。後側の `dot/` に `perception.structure.dot` と `perception.detail.dot` がある。

| 環境 | 経験数 | 解決した Run プロファイル | TD 誤差 / 学習リターンの件数 | 一致した scalar・trace |
| --- | ---: | --- | ---: | ---: |
| GridMaze | 131,072 | 指定なし | 500 / 499 | 30,335 |
| LunarLander | 40,704 | `run.@munch > run.@skip` | 125 / 92 | 14,007 |
| DropMerge | 282,112 | 指定なし | 1,000 / 878 | 101,146 |
| Atari Q*bert | 456,320 | `run.@v5_iqn_impala_x2 > run.@a5 > run.@a5_apex > run.@va_base > run.@hard125 > run.@munch > run.@rfit > run.@a5_metrics > run.@cap2m > run.@eval2ch_r1 > run.@to_100m` | 1,000 / 1,065 | 72,784 |

前後とも `AtariEnv.game=qbert`。追加除外を付けず `compare_metrics.py` を実行し、4 組とも `exact_match=true`。後側の Run は `.scratch/rfd0200/after/runs`、比較結果は `.scratch/rfd0200/after/*-comparison.json` に保存した。

SetSpec の診断を修正した Release Runner（SHA-256 `EF7F3C98541CED4B8E0FCB2040B57F3C97A95BF3555108894A89E29100705D69`）でも 4 組で同じ 218,272 点が完全一致した。後側の Run は `.scratch/rfd0200/after/final-runs`、比較結果は `.scratch/rfd0200/after/*-final-comparison.json`。

設定エラーの報告順と計測を修正した Release Runner（SHA-256 `A0A21887BD894D44FE2FA8BA12C5DFADB1C72F0B374E77EDD49515106C67A648`）でも、全 4 組が終了コード 0、218,272 点完全一致、実効設定の差 0 件、各 Run の DOT 2 点を確認した。後側の Run は `.scratch/rfd0200/after/reviewed-runs`、比較結果は `.scratch/rfd0200/after/*-reviewed-comparison.json`。

実装工程の完了時点（`3545c16a`）の Release Runner（SHA-256 `A6FE29AFF926038634D9B9CF87530E2A75053622F0C7D7C332DEB9FED1E00969`）でも、4 環境が同じ経験数で終了コード 0、上表の scalar と trace 218,272 点が完全一致した。出力先を外し、重ねる設定キーを読み替えた実効設定の差は 0 件で、両方の決定論設定が有効、各 Run の DOT は 2 点。後側の Run は `.scratch/rfd0200/after/acceptance-runs`、比較結果は `.scratch/rfd0200/after/*-acceptance-comparison.json`。

## 完了報告

- 依頼ごとの Forwarder、ReplayBuffer の Sample spec、DefaultDQN の重ね方の設定と spec 照合を実装した。出荷設定、固定 17 構成、日本語設計文書と wiki を更新した。既存の Montezuma 起動は出荷設定の既定を使う。
- 実装工程の完了時点（`3545c16a`）の Debug テストは、10 実行体すべてが終了コード 0。計 1,242 件が通過し、CUDA が使えない場合に実行する Core のテスト 1 件がスキップされた。ログは `.scratch/rfd0200/debug-tests-acceptance/`。同じ版の Release は、4 環境で上表のとおり完全一致した。
- 実装工程の完了時点（`3545c16a`）の RFD、実装計画、実装はいずれも独立レビューが OK。英語設計文書は RFD §6.9 のとおり翻訳の道具で再生成する対象。既定キー検査に残る既存の Python 代入 4 件は今回の設定変更の外にある。
- 決まらなかったことと、開発者の判断を要することはなし。記録する課題の候補はなし。
