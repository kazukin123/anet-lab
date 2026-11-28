# RFD0206 基盤とアプリの小修正 実装メモ

## 概要

[RFD0206](0206_core_and_app_small_fixes_10rfd.md) に沿い、Replay・メトリクス・ImageCls・Runner の境界を修正する。比較は RFD0103 の 6 構成、seed 1、8 lane、8,704 experience、FP32、決定論的 backend で行う。

## 主な変更

- Replay の n_step を構築時に検証し、先読みされた Push の結果を受け取る前に列から取り出す。
- ImageCls の値域と学習率の下限、QValuePanel の hist_bins、数値属性の読み切りを検証する。
- 行動の device の控えを補完後の device に揃え、対象の画像テストで wx を初期化する。
- ImageCls の組み立てとログを簡素化。StateSpec は void の AssertObservation とし、RunManager はアプリの予約を `std::vector<std::string>` で受け取る。
- bat 2 本、設計 100・140・150・160 の和英版、対象課題の記録を更新する。

## テスト

- 最初に公開 API `CreateReplayBuffer` の n_step 境界（負・0・1）を確かめる。
- 先読み Replay の Sample と優先度更新で worker の元の例外と回復、ObserverFactory で正しい表記と不正属性の診断、BatchActionInfo の CUDA 行動の再要求を確かめる。
- ImageClsAgentFactory の構築時に 4 キーの範囲外と境界、構築出力の INFO と JSON の件数を確かめる。QValuePanel の構築時に 0、負、1、510 を確かめる。
- RunManager の予約有無を公開 API で確かめ、既存の Env 観測テストと画像テストの実行順の検証を通す。
- 挙動ごとにテストを追加して RED を確認し、最小修正の後に同じテストを GREEN にする。名前・配置変更は関係テストと Debug 全ビルドで確認する。

| 観測する挙動 | 配置先 |
| --- | --- |
| Replay の構築時検証と先読み Push の例外 | `src/core/base/src/replay_buffer_test.cpp` |
| 数値属性の読み切り、旧字句の受理と CUDA 行動の再取得 | `src/core/base/src/observers_test.cpp` |
| ImageCls の設定値と構築出力 | `src/algo/imagecls/src/image_cls_agent_test.cpp` |
| QValuePanel の bins 境界 | `src/apps/runner/src/QValuePanel_test.cpp` |
| RunManager の予約有無 | `src/core/base/src/trainer_test.cpp` |
| wx 画像のテスト初期化 | `src/env/imagecls/src/ImageClsEnv_test.cpp` |
| 観測名の呼び出し先移行 | `src/env/atari/src/AtariEnv_test.cpp`、`src/env/dropmerge/src/DropMergeEnv_test.cpp`、`src/env/lunarlander/src/LunarLanderEnv_test.cpp` |

## 検証

- `VsDevCmd.bat -arch=x64 -host_arch=x64` を初期化する `cmd` 内で Debug 全ビルドと `AnetRLRunner-gui-test` をビルドする。
- 対象タグと既定スイート、ImageCls のテスト 3 通り、RFD0072 の検査を行う。標準出力・標準エラーと終了値は `.scratch/rfd0206/` に記録する。
- 短縮 Run の比較は `.scratch/rfd0206/validation.py` を使う。時間と資源の 3 接頭辞を除き、非空の正規化した点を照合する。
- RFD §7 の検索、bat の CP932・CRLF、和英の構造・リンク・ハッシュを確認する。
- 移行対象の旧名は `StateSpec::ValidateObservation`。新名 `StateSpec::AssertObservation` への移行を現用コード・テスト・設定・設計・用語集・トピック別まとめから検索する。

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug && cmake --build --preset x64-Debug --target AnetRLRunner-gui-test'
.\.venv\Scripts\python.exe -B .scratch/rfd0206/suites.py all
.\.venv\Scripts\python.exe -B .scratch/rfd0206/validation.py after
.\.venv\Scripts\python.exe -B .scratch/rfd0206/validation.py compare
```

## 前提

RFD0205 の改訂を土台にする。変更前の Debug ビルドは `.scratch/rfd0206/baseline-build.log` と `baseline-manifest.json` に記録する。採用した理由は「経過」の各行。未決事項監査の開発者判断待ちは 0 件。

| 固定した前提 | 採用内容 |
| --- | --- |
| U1（実装順と基準） | RFD0205 の次に実装し、変更前の Debug 実行体を保存する |
| U2（Replay n_step） | 構築時の下限を 1 とし、完全キーと値を診断に出す |
| U3（ImageCls の設定値） | 勾配 clip は正、平滑化は [0, 1]、減衰・学習率は 0 以上 |
| U4（QValuePanel hist_bins） | コンフィグ構築時に 1 以上を要求する |
| U5（数値属性） | 最初のコロンから末尾まで読み切り、従来の空白・数値表記と値域を保つ |
| U6（行動 device） | 実効 device を控えのキーにする |
| U7（先読みの例外） | 2 つの片づけ経路で future を取り除いてから結果を受ける |
| U8（wx 画像の初期化） | 対象ケースの先頭で EnsureWxImageSupport を呼ぶ |
| U9（既定宣言の検査） | 現行の検査とテストで探索対象を確認する |
| U10（ImageCls の構築） | 未使用引数と重複する末尾ログを削除する |
| U11（観測の検証） | void の AssertObservation とし、Trainer からは assert 有効時に呼ぶ |
| U12（アプリの予約名） | RunManager が予約を受け取り、Runner で EvalPanel を共有する |
| U13（bat の題） | 2 本にタイトルを付け、CP932・CRLF を維持する |
| U14（設計文書） | 100・140・150・160 の和英を更新する |
| U15（正しい設定の等価性） | 現行設定を凍結し、6 構成の非空の全 scalar を前後比較する |
| U16（無効時の n_step） | Learner 側の検証は後続候補にする |

## 経過

- 2026-11-27 入口レビューは初回差し戻し。数値属性の値域と字句、移行文書、比較の基準・受入、公開 API の契約を直し、再レビュー OK。
- 2026-11-27 夜間に推奨で採った。U1（実装順と基準）は RFD0205 の次にまとめて実装し、変更前の Debug ビルドを保存。比較にこの RFD の変更だけを反映するため。
- 2026-11-27 夜間に推奨で採った。U2（Replay n_step）は 構築時に 1 以上を検証し、完全キーと値を示す。不正値で学習が進まない問題を構築時に見つけるため。
- 2026-11-27 夜間に推奨で採った。U3（ImageCls の設定値）は 勾配 clip は正、平滑化は [0,1]、減衰と学習率の下限は 0。勾配 clip は常に使われ、学習率は予定の bounds で検証できるため。
- 2026-11-27 夜間に推奨で採った。U4（QValuePanel hist_bins）は 既存と同じ構築時チェックで下限 1。1 bin を描画できるため。
- 2026-11-27 夜間に推奨で採った。U5（数値属性）は 最初のコロンの後を読み切り、従来の数値表記と値域を維持。不正な末尾を見つけ、正しい設定を引き続き受理するため。
- 2026-11-27 夜間に推奨で採った。U6（行動 device）は 補完済み req_dev を控えのキーにする。同じ実効 CUDA device の再要求を認識するため。
- 2026-11-27 夜間に推奨で採った。U7（先読みの例外）は 二つの片づけ経路で future を取り出してから結果を読む。受け取った失敗を列に残さないため。
- 2026-11-27 夜間に推奨で採った。U8（wx 画像の初期化）は 対象テストの先頭で EnsureWxImageSupport。同ファイルの既存の初期化に揃えるため。
- 2026-11-27 夜間に推奨で採った。U9（既定宣言の検査）は 現行の検査とテストで確認する。探索先は 125075dc で修正済みのため。
- 2026-11-27 夜間に推奨で採った。U10（ImageCls の構築）は 未使用引数と重複する末尾ログを削除。既存の呼び出しは 1 引数で、冒頭に必要な記録を出すため。
- 2026-11-27 夜間に推奨で採った。U11（観測の検証）は AssertObservation に改名して void にし、Trainer からは assert 有効時に呼ぶ。成功または例外という挙動を名前と型で表すため。
- 2026-11-27 夜間に推奨で採った。U12（アプリの予約名）は RunManager はアプリの予約を受け取り、Runner は EvalPanel の定数を共有。予約を使うアプリが名前を定めるため。
- 2026-11-27 夜間に推奨で採った。U13（bat の題）は 対象 2 本に題を付け、CP932 と CRLF を保持。起動窓でファイルを識別できるため。
- 2026-11-27 夜間に推奨で採った。U14（設計文書）は 100・140・150・160 の和文を更新し英訳を追随。利用者が現行契約を確認できるようにするため。
- 2026-11-27 夜間に推奨で採った。U15（正しい設定の等価性）は 6 構成の共通入力で非空の全 scalar を前後比較。変更後の学習結果を同じ条件で確かめるため。
- 2026-11-27 夜間に推奨で採った。U16（無効時の n_step）は Learner 側の検証は後続候補。Replay 構築より外側の境界であるため。

## 検証結果

- 比較の基準は HEAD `a60bae83cbf8c43d872882f910ac9318249e0df7` の Debug Runner と DLL 50 本。前後共通の現行設定を保存し、修正前の短縮 Run 6 構成から 179,687 点を採取した。各入力・実行体の識別情報は `.scratch/rfd0206/` に保存。
- A2（n_step）。修正前は負・0 の 2 入力を誤って受理。修正後は負・0 を完全キー付きの診断で拒否し、1 を受理した。
- A3（ImageCls の値域）。構築経路で修正前に境界の失敗を確認。学習率の定数指定は fixture が使う完全キーに揃えて再検証し、範囲外と受理する境界の 15 アサーションを通した。
- A5（数値属性）。修正前は不正属性の 13 アサーションが失敗し、修正後は符号、指数表記を含む 22 アサーションが通った。
- A6（CUDA の行動）。修正前は 2 回目の `cuda` と `cuda:0` がキャッシュ不一致。実機 CUDA で修正後の 4 アサーションが通った。
- A7（先読み Push）。修正前は元の例外に続いて `no state` が発生してテストプロセスが終了。修正後は Sample、同期・優先度更新の 2 経路で元の例外を一度だけ返し、再同期と破棄が成功した。
- A8（画像の初期化）。対象テストだけの修正前実行は 60 秒で打ち切り、終了値を 124 と記録。
- A9（既定値宣言の検査）。`check_default_leaves.py` は 39,369 件の代入を収集し、4,296 件を検査。既存の `apps/tools/metrics_viewer.py` の 4 行を報告し、終了値 1。付属テスト 16 件は終了値 0。
- A10（ImageCls 構築出力）。修正前に設定 JSON が 2 件出ることを再現。ログの捕捉を基盤の INFO に当たる wxLOG_Message に揃え、修正後は設定の INFO と JSON が各 1 件。
- A14（文書）。和英 4 組の見出し階層、表行数とフェンスが一致し、未訳の日本語は本文に 0 行。変更文書のリンク切れは 0 件。翻訳元のハッシュは 100 `3855305b9680d0b6fabe8d1ba4b39dcf3ea87113`、140 `8253b800db44f69798dabec409dd25c6cc742494`、150 `a2fbc6126741c084b63107a392f4be418379a07f`、160 `daba95059f94284ebbbee0b500fff58224281862`。

- A4（hist_bins）。修正前は −1 と 0 を受理。修正後は完全キー・値・下限を示して拒否し、1 と 510 を受理した。
- A9（探索対象）。出力 JSON の `file` を確認し、`apps/` から 4,132 行、`src/` から 164 行。
- A11（観測検証）。旧 API 名の現用呼び出しは 0、宣言・定義を含む新 API 名は 27、うち Env テストは 23。Trainer の 2 呼び出しは `#if ANET_ENABLE_ASSERT` の有効時に限り、Release では既定で無効になる定義を CMake と diag.hpp で確認。
- A12（予約名）。予約ありでは Env 構築前に元と同じ衝突診断を出し、予約なしでは同じタグを構築できる。旧実装で予約なしのテストが失敗し、修正後は 6 アサーションが通った。基盤コードに Runner 固有の名前の固定文字列は 0 件。
- A13（コンソールタイトル）。全 21 本の先頭 2 行を確認し、変更 2 本は CP932・CRLF、`git ls-files --eol` の作業ツリー欄は `w/crlf`。
- `git diff --check` は RFD0206 の対象範囲で終了値 0。文書の前段階の版は `.scratch/rfd0206/docs-before/` に保存した。
- A1（ビルド）。標準の MSVC 初期化から Debug 全体と Runner GUI テストのビルドが終了値 0。ログは `.scratch/rfd0206/final-debug-build.log`。
- A16（既定スイート）の基盤の初回全体実行は 696 件通過、1 件スキップ、評価ログの 1 件失敗（seed 2739814420、ログ `.scratch/rfd0206/suite-core-base.log`）。失敗時に拾った終了行 2 件の値は、直前の別テストが出したログに一致した。同じ順で 2 ケースに絞ると、保存済みの RFD0192 着手前実行体と今回の実行体で各 207 アサーションが通り、今回の単独実行も 115 アサーションが通った。

- A8（画像の初期化）。修正後は単独 4.11 秒、宣言順 4.64 秒、seed 1094244747 の乱順 4.17 秒で終了値 0。
- A16（既定スイート）。基盤は seed 2899524567 の再実行で 697 件通過、1 件スキップ（34,310 アサーション）。ImageCls Agent 31、MuZero 7、Atari 56、CartPole 1、DropMerge 33、GridMaze 12、ImageCls Env 20、LunarLander 8、Runner GUI 43 件が通過。DQN の初回は検証スクリプトの 600 秒上限で途中終了した。処理の進行を確認し、上限を画像テストだけに限定して DQN を再実行した。

- A15（短縮 Run）。GridMaze off 27,382、LunarLander off 24,463、Atari off 23,229、DropMerge off 31,274、GridMaze on 38,633、LunarLander on 34,706 点の全 179,687 点が一致した。比較は終了値 0（`.scratch/rfd0206/validation-compare.log`）。
- A16（DQN の再実行）。384 ケース、161,422 アサーションが通過（`.scratch/rfd0206/dqn-full-retry.log`）。基盤・DQN の初回の非成功と後続の成功を別々に記録した。

- A5（数値属性の空白互換性）。独立レビューで従来受理したコロン直前のタブが拒否されることが分かった。追加テストで失敗を再現してから、左右の従来の空白処理と数値の読み切りを併用し、基盤の RFD 対象 5 ケース・43 アサーションを通した。
- 最終確認。Debug 全体と Runner GUI の再ビルド、ImageCls 対象の再ビルドが成功。Runner GUI 43 ケース・926 アサーションと ImageCls 対象 2 ケース・19 アサーションが成功。固定設定の数値属性 470 件には末尾修正の影響を受ける空白がなく、最終 Runner でも GridMaze on の 38,633 点が一致した（`.scratch/rfd0206/validation-review-gridmaze.log`）。

### 再レビューに向けた追加確認

- A15（短縮 Run の前後比較）。RFD0207 着手前に保存した `.scratch/rfd0207/baseline/Debug/AnetRLRunner.exe` を RFD0206 の最終版として使った。SHA-256 `9b29565051609032bd3628dae091f400f222623369cd47cda40a5f43783c645d` が先に比較した GridMaze on の実行体と一致し、各構成の凍結入力の SHA-256 も前後で一致した。
- 残り 5 構成は `validation.py after --after after-cc-review --after-runner` に上記実行体の絶対パスと `--cases GridMaze_off LunarLander_off Atari_off DropMerge_off LunarLander_on` を指定して実行。続けて `validation.py compare --after after-cc-review --cases GridMaze_off LunarLander_off Atari_off DropMerge_off LunarLander_on`。5 構成の終了値はすべて 0。GridMaze off 27,382、LunarLander off 24,463、Atari off 23,229、DropMerge off 31,274、LunarLander on 34,706 点が一致した。先に確認した GridMaze on の 38,633 点を合わせ、最終版の 6 構成で計 179,687 点が一致した（`.scratch/rfd0206/validation-cc-final.log`）。
- A16（既定スイート）。初回と同じ既定順、seed `2739814420` の全体実行で、評価ログの 1 件・11 アサーションの失敗を再現した（`cc-core-original-seed.log`）。同じ `eval1` を使う Trace テストが出した所要時間 0.00 秒・平均報酬 2 と 5.6666665 の終了行を拾い、テスト内の終了件数が 2 から 4 になっていた。RFD0192 着手前の保存済みテストはテスト集合と現在の設定の互換条件が異なるため、当時の同一条件を示す証拠には採用しなかった（`cc-core-pre-rfd0192-original-seed.log`）。
- 評価ログのテストに専用の Eval 名を付け、その Runner のログだけで既存の開始・待機・終了の件数と順序、時刻・値を確かめる。修正先は `src/core/base/src/episode_end_test.cpp` の `EvalRunner RunSession emits adopted episodes then one session event`。
- 専用の Eval 名に変更後、`anet-core-base-test` の再ビルドが成功した（`cc-core-isolation-build.log`）。`.\out\test\Debug\anet-core-base-test.exe --anet-test-failure-dialog=off --rng-seed 2739814420` を既定順で実行して終了値 0。697 ケースが成功、1 ケースがスキップ、34,310 アサーションが成功した（`cc-core-isolated-original-seed.log`）。
- Debug 全体と Runner GUI テストの再ビルド、ImageCls の既定スイートの 31 ケース・365 アサーションが通過した（`cc-followup-debug-build.log`、`cc-imagecls.log`）。Observer の数値属性名と ImageCls の JSONL 欄名・判別値はテスト内の定数に揃えた。
- 設計 100・160 の和英の図を `reserved_env_names` に揃え、設計 100 英語版の検証ラベルも英訳した。翻訳元は 100 `6c1dc9f466692e79ca9709adf19884f7effbf537`、160 `7792af6316f42225183b97bc53aebc625ac8a12f`。和英の見出し・表・フェンスとリンク、図の該当ラベルを確認した。

## 完了報告

RFD、実装計画、実装の独立レビューはいずれも再レビュー OK。実装の初回と再レビューの原文は `.scratch/rfd0206/review-20261128-0003-impl-initial.md` と `.scratch/rfd0206/review-20261128-0007-impl-followup.md`。数値属性の空白互換性、Runner 起動図、ImageCls の共通テスト設定と宣言順を見直し、観点全体の自己点検を終えた。

12 課題の AI 判定を残件無しに記録する。後続の候補は Replay が無効な場合の Learner 側 `learner.n_step`、最初にデストラクタで遭遇する worker 例外、wx 内部ダイアログの抑止。既定値宣言の現行検査で出た `apps/tools/metrics_viewer.py` の 4 件は A9（探索範囲の確認）の結果に記録した。
