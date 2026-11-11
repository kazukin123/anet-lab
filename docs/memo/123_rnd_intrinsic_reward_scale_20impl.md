# RND の内的報酬の尺度 実装メモ

## 概要

[PRD123](123_rnd_intrinsic_reward_scale_10prd.md) の D1〜D16 と §6・§7.2 を実装する。whiten の既定を `warmup_mode=learn_predictor` にし、warm-up 中は predictor を学習させ、誤差を統計に入れない。統計は累積が既定で、EMA と平均を引かない式を選べる。終端の内的報酬の mask は既定 true のまま選べるようにする。

## 主な変更

- `intrinsic_rnd.hpp/.cpp`: 新しい設定を検証・通知する。非公開の `RndErrorStatistics` が既存の累積と直近重みを持つ。累積は既存 `RunningMeanStd` に委ね、計算と保存順序を変えない。EMA は重み付きの平均・二乗偏差を CPU double で lane 順に更新する。採点前の統計で倍率を作り、その後更新する。
- `default_dqn_agent.hpp/.cpp`: facade が `intrinsic.use_done_mask` を読み、`next_state` の done lane を必要な場合だけ 0 にする。state では値を使わず、既定と違う値だけ WARN。
- `intrinsic_test.cpp`: 現行挙動を確認する既存テストに `collect_errors` を明示し、`[prd123]` で新しい挙動と設定・checkpoint を検証する。出荷設定の検査に新しい 5 キーを足す。
- `agent.txt`: baseline に 5 キーの既定を `?=` で足す。Atari の warm-up 200,000 行と、それを断定するテスト・文書はキュー終了確認まで足さない。
- `testdata/prd123/README.md` と比較の道具を用意し、着手前実行体と設定を独立保存して、whiten collect_errors・variance_feedback・4 env の RND 無効の全 scalar 点と定義を比較する。旧実行体の checkpoint を新実行体で読む。
- `testdata/prd123/compare_config.py` で、保存した7環境の全キー・文字列値を照合し、共通設定の5キーだけが追加されたことを確かめる。
- PRD116 の smoke を実効 warm-up 件数より後の点で判定し、KFM は 12,800 件を明示する。
- `testdata/prd123/smoke_validation_test.py` で、Runner を起動せず合成指標から smoke 判定の変更を確かめる。
- 設計文書 212・200 の日英版を追随させ、今回の検証資材へリンクする。課題の対応内容・判定と派生一覧も更新する。
- 追加の二モデルレビューへの対応として、`error_norm.use_centering` と `intrinsic.use_done_mask` の既定値を、それぞれの設定型の定数へまとめ、初期値・通知の判定・文言から参照する。値はどちらもtrueのまま。

## テスト

- Public interface / surface: `LifelongNoveltyEstimatorFactory::Create`、`Estimate`、`Learn`、`Diagnostics`、`Save/Load`、`DefaultDQNAgent::CreateLearner` が返す `Learner::UpdateFromBatch` の学習報酬、設定読み込み、出荷設定、Runner の scalar 出力。
- TDD 順序: (1) Factory で構築し warm-up 中に採点・学習して NaN の誤差統計と有限 loss を確認する tracer bullet、(2) warm-up の境界と 0、(3) EMA の既知列・batch 分割・時定数 1、(4) 中心化と clip の 4 通り、(5) 設定の ERROR と読まない値の通知、(6) checkpoint の継続・識別子の不一致、(7) facade の done/truncated と診断値、(8) 出荷設定、の順に 1 テスト → RED → 最小実装 → GREEN を進める。既存の `[prd116]`・`[rnd_reference_feedback]` も回帰確認する。

## 検証

- 基準コミット: `dc41e4cba7ffbfc7eddf8b2aa7800a125d2a8ef5`。着手時の staged PRD・wiki・課題と、その他の無関係な変更を保持する。
- ビルドは `AGENTS.md` の MSVC 初期化経由とし、Debug 全体と関連テストを実行する。通常ログも `.scratch/prd123/` に保存し、終了コード・集計を返す。
- `[prd123]` → `[prd116],[scaler],[shipped_config]`、設定の既定葉・解決値の前後比較、3 構成の scalar checksum と定義、旧 checkpoint 読込、差分・改行・BOM・リンクを確認する。
- 全品質確認観点を適用範囲で自己点検し、独立した `anet-review-agent` のレビューを受け、直せる指摘を直して同じ agent へ再依頼する。

## 前提

- ユーザー判断が必要な実装ブロッカー: 0。既存コードの warm-up 判定は採点前 `<`、学習は採点後 `<=` を保持する。variance_feedback は変更しない。
- `ema_steps=1,000,000`、`use_centering=true`、`use_done_mask=true`、warm-up 12,800 件、β 0.1 が実装・共通設定の既定。RND 無効時は関連値を読まない。
- EMA は母分散、累積は既存の統計と完全に同じ計算を使う。cumulative と variance_feedback は `RndNoveltyEstimator-3`、EMA は `-4`。設定に対応する識別子だけを読む。
- 新しい ADR・用語集項目・メトリクスは作らない。PRD116 の過去の実験記録は変えない。
- PRD §7.1 のサブ機 KFM 10 本は、段取りを別に示して承認後に実行する。本工程では起動しない。U1〜U3 はその結果待ちで、実装を止めない。
- PRD §7.2 の 8 はキュー終了の確認待ち。終了を確認できない場合は Atari.txt を変えず、課題は残件有りとする。

## 経過と結果

- 2026-11-11: PRD の再グリルとレビュー OK の記録を確認。暫定の印は改訂で外されている。現行の設定・採点と学習・保存を照合し、上記の前提で未決事項監査を通した。
- 着手前の Debug 全体ビルドは no work。Runner の SHA256 は `e01759e3bda254ef2715684bd91334cae9042317e6f1d2bb701869b004b30d3b`。実行体・DLL・設定は `.scratch/prd123/baseline-20261111/` へ複製し、`[prd093-config-snapshot]` は15断定・1件で成功した（baseline-config.log）。
- TDD: warm-up の公開 Factory テストは RED で5断定中4失敗、最小実装後に5断定すべて成功。EMA の既知列・batch 分割・時定数1は RED で13断定中5失敗、実装後は warm-up と合わせ18断定・2件すべて成功（red/green-warmup、red/green-ema の各ログ）。中心化4通りは RED で4断定中1失敗を確認した。
- 実装中に足す同期転送は無い。EMA は既存の採点で CPU に戻した誤差を一括 double 化して読み、lane のループは CPU のスカラー計算だけを行う。`RndErrorStatistics::Update` は計測し、累積の軽量な読み口は既存 RunningMeanStd の読み口へ委譲する。
- 英語版212は着手時点で翻訳進捗が chapter 0/5 だった。今回の変更に関わる箇所を追随させ、既存の未完了部分を全訳完了と報告しない。
- 終端報酬の検証は、偽 Learner を差し込む公開の口が無い。テスト専用の取り出し口を本体へ足さず、既存の `DQN learner consumes the scaled external reward plus intrinsic bonus` と同じく、Q=0・学習率0・gamma=0 の実 Learner の公開 TD 平均で、受け取った学習報酬そのものを照合する。容量8の Replay から初期の倍率1の経験を押し出し、残る経験の倍率を上限へ揃える。done と truncated、mask の true/false、state と next_state を独立して確認する。
- checkpoint の RED は63断定中2失敗（EMA の識別子と取り違えの早期拒否）。統計に応じた識別子の選択後は、ここまで120断定・5件すべて成功した。設定の INFO は logger の実装で wxLOG_Message に対応するため、テストの捕捉もそのレベルに合わせた。
- 指定された stage20 の Run directory は現時点で見つからず、終了の印も確認できなかった。Atari の warm-up 行は保留する。既存のキューや Run は操作していない。
- Debug 全体ビルドは成功した（full-debug-build.log）。最初の関連テストは60件中59件成功で、既存の done/truncated テストが warm-up 明けの最初の batch で非ゼロを期待していたため2断定が失敗した。このテストにも collect_errors を明示して、mask 自体を確かめる従来の断定を保つ（PRD の既知4件に加える移行対象）。追加の境界テストでは、同じ観測を繰り返すと正規化後の2 lane の誤差が同じになり分散0になるため、統計を使う採点を確認する batch は異なる観測にした。
- smoke 判定の合成テストは1件成功。既定葉検査は assignments=32,503、audited=4,017、errors=0。変更文書のローカルリンクに切れは無く、Python3ファイルの構文と、日英2組の翻訳元ハッシュを確認した。
- 正しさの100k件の独立条件は、同じ保存済み実行体の準備後なら並列で確認してよい。性能の比較ではなく、時刻・資源以外の全 scalar を比較する。条件ごとの execution.json を分け、集計のファイル名にも条件を含めて衝突を避ける。

## 実行した検証

通常の stdout・stderr は `.scratch/prd123/` の各ログへ保存し、終了コードと集計を確認する。長い Run は条件ごとの execution.json と開始・終了ログを残し、実行中の session を待って完了を確かめる。

| コマンド・確認 | 結果 | 記録（`.scratch/prd123/` から） |
|---|---|---|
| `cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug'` | 終了0 | `full-debug-build.log` |
| 同じ MSVC 初期化で `cmake --build --preset x64-Debug --target anet-core-test`、続けて `core/anet-core/bin/Debug/anet-core-test.exe "[prd123],[prd116],[scaler],[shipped_config]"` | 終了0、60ケース・4,804断定。読まない値の変更前後で採点・学習・全診断値・保存識別子が同じことも確認 | `ignored-behavior-build.log`、`focused-tests-r3.log` |
| `core/anet-core/bin/Debug/anet-core-test.exe "[shipped_config]"` | 終了0、12ケース・2,614断定 | `shipped-tests.log` |
| `.venv/Scripts/python.exe core/anet-core/testdata/prd123/smoke_validation_test.py` | 終了0、合成テスト1件 | `smoke-validation-final.log` |
| PRD072 の `check_default_leaves.py` | 終了0、32,503代入・4,017検査・違反0 | `default-leaves.log` |
| 設定 snapshot と `.venv/Scripts/python.exe core/anet-core/testdata/prd123/compare_config.py` | 終了0、7環境それぞれ新規5キー、既存値の変更・削除0 | `baseline-config.log`、`after-config.log`、`config-comparison.log` |
| `.venv/Scripts/python.exe core/anet-core/testdata/prd123/run_validation.py before` | 全6条件終了0 | `baseline-20261111/` の各 `before_*/execution.json` |
| 同スクリプト `after --label after-r1` の RND 無効4環境と `compare` | 全4条件終了0、全 scalar 点の checksum と定義が一致 | `after-gridmaze.log`、`after-off-rest.log`、`compare-gridmaze.log`、`compare-off-rest.log` |
| 同スクリプト `load --label after-r1 --case collect` | 着手前 checkpoint を読み、16件進めて終了0 | `old-checkpoint-load.log` |
| 同スクリプト `after --label after-r1 --case collect` と `compare --label after-r1 --case collect` | 100,000件を終了0、全371,051点・97タグの checksum と定義が一致 | `after-collect.log`、`compare-collect.log` |
| 同スクリプト `after --label after-r1 --case feedback` と `compare --label after-r1` | 100,000件を終了0。feedback の全371,796点・97タグが一致。全6条件の計848,529点と全定義も一致、終了0 | `after-feedback.log`、`compare-final.log` |

RND 無効の一致は GridMaze 26,946点、LunarLander 24,399点、Atari 23,185点、DropMerge 31,152点。除外は時刻・資源の群だけで、NoisyNet 関連の scalar も含める。RND 有効の2条件も各100,000件を完走し、全点の一致を確認した。実装後の固定 Runner の SHA256 は `8062402090011036174d019542e5c28ee0e55e42dc56afdcad7e31b9828b6431`。collect の checksum は `384f8b659704acf0456644bd08e26faad8edfb936202f134719fee9c3d044811`、feedback は `aecb37c8265dfb72e4c3337fefafddc71556b6f0b194a8f0d80d8fd1bb9547e5`。効果の10M×10本は別の段取りの承認待ちなので未実施。

## 受入との対応

| PRD §7.2 | テスト・観測結果 | 状態 |
|---|---|---|
| 1 既存挙動 | RND 無効4環境、collect_errors と variance_feedback 各100k件の全 scalar checksum・定義が一致 | 成功 |
| 2 直近重みの式 | `RND EMA weights scored observations independently of batch splitting`: 誤差1,4,9,16、H=2の手計算、1×4と4×1、H=1 | 成功 |
| 3 warm-up | `RND learns during warmup without collecting errors` と `RND zero warmup makes both warmup choices equivalent`、従来の warm-up テスト。件数3/4でbatchをまたぐ境界・空の最初・次の採点、NaN/有限lossとLearnを照合 | 成功 |
| 4 倍率の式 | `RND centering and clipping are independent scoring choices`: 中心化とclipの4通り、更新前の統計と非ゼロ誤差で照合 | 成功 |
| 5 設定 | `RND validates new settings before notifying unused values` と `DQN intrinsic done mask controls the actual learning reward`: 5キーの不正値・構築時1回の通知・使わない値で挙動不変・無効時無視 | 成功 |
| 6 checkpoint | `RND checkpoint format follows error statistics and resumes scoring`: 全3構成の識別子と継続、異なる識別子だけの入力を早期拒否。実行体の前後でも旧collect checkpointを読んで16件進行 | 成功 |
| 7 出荷設定 | `[shipped_config]` の新5葉と実装の既定を照合。既存7環境の全キー・文字列値も照合 | 成功 |
| 8 Atari warm-up | stage20終了を確認できず、指定どおり行とテストを未追加。課題に残す | PRDが認める残作業 |
| 9 終端報酬 | `DQN intrinsic done mask controls the actual learning reward`: state/next_state、mask true/false、done/truncatedの8通りを実Learnerの公開TD平均で照合。state falseの通知1回も照合 | 成功 |

## 自己点検

`docs/quality/impl.md` の全項目を適用範囲で走査した。新しい公開の取り出し口・共通層の個別機能分岐・同期転送・循環依存・背景スレッド・UI 処理・新規メトリクスは足していない。設定の検証は持ち主の設定読み込みにあり、読まない値も有効時は検証し、無効時は読まない。累積の計算と保存 payload は既存処理を使う。新規の検証用ファイルを含む変更と「主な変更」を照合した。

| 観点のまとまり | 確認・対象外の理由 |
|---|---|
| 計画・受入・範囲・課題の記録 | PRD §7.2 の8はPRDが認める残作業。他は公開経路のテストと検証で成功。§7.1は承認後の別実験。課題へ今回の行を追記済み |
| 契約・失敗・名前・設定 | 廃止する識別子は無し。新5キーは baseline の `?=`、実装既定と一致。enum・H・bool の不正値、読まない条件の通知、無効時の無視をテスト。旧 artifact は保持 |
| C++・コメント・計測・所有権・ログ | private な入れ子の統計型、既存 State 内の所有、固定キーのマクロ、CPU lane 順更新。主要更新の計測、日本語コメント・英語ログを確認。新規 C++ ファイル、借用の保持、UI ロック、取り出し口、初期化の組み替えは対象外 |
| テスト・検証・文書 | Factory・facade・実 Learner の公開経路を使用。日英の設計文書を追随。新資材は testdata/prd123 に置き設計文書からリンク。翻訳元情報は200完了、212部分確認の範囲を記載 |
| 検索・形式・改行 | `using namespace` のヘッダ追加、production の `dynamic_cast`、大文字 TODO、禁止された直訳の追加は無し。差分の空白・LF・既存 BOM・ローカルリンク・翻訳元 hash・Python 構文を確認。bat、skill、改名の旧名、プロファイル境界と metrics 連鎖は変更が無く対象外 |

自己点検でユーザー判断を要する新たな未決事項は無い。

## レビューの経過

- 1往復目: 計画はOK、実装はテストに追加した同じモード名・入力値の反復直書き1件で差し戻し。原文は `.scratch/prd123/review-20261111-0709.md`。今回追加行を既存の定数と用途の狭い定数へ揃えた。テスト対象だけを再ビルドし終了0、関連60ケース・4,804断定がすべて成功した（`review-fix-build.log`、`focused-tests-r4.log`）。比較に使う Runner は変えていない。範囲外の検出・開発者への確認事項は無し。100k件の照合と修正後テストを次回再確認する。
- 修正後の自己点検: 追加した定数は持ち主の既存の定数を優先し、テスト固有値は使う範囲の狭い場所に置いた。変更範囲・入力元との対応、BOM/LF、差分の空白、通知と保存の検証、完走した全6比較を再確認した。自己点検で残る不足は無し。課題は Atari の反映と効果未確認のため残件有りとし、`wiki_support.py maps` は終了0で派生区画を更新した。
- 2往復目: 計画・実装ともOK。定数化の指摘、100k件比較と修正後テストの再確認を解消した。範囲外の検出・開発者への確認事項は無し。原文は `.scratch/prd123/review-20261111-0725.md`。最後のリンク確認は206参照すべて実在し、今回の差分全体の `git diff --check` も終了0。

## 完了報告

実施内容: PRD123 の本工程で行える実装・検証・自己点検・独立レビューを完了した。whiten の warm-up の既定、EMA、中心化の選択、done mask の選択と設定・文書・検証資材を更新した。Debug 全体ビルドと関連60ケース・4,804断定が成功し、変更前後の全6条件・848,529点と指標定義が一致した。旧 checkpoint の読み込み、既存設定値の維持も確認した。

変更ファイル: `intrinsic_rnd.hpp/.cpp`、`default_dqn_agent.hpp/.cpp`、`intrinsic_test.cpp`、`agent.txt`、PRD116 の検証スクリプトと README、PRD123 の検証用4ファイル、設計200・212の日英、実装計画、課題単票と一覧・wiki の派生区画。着手時からあった無関係な変更を保持した。

レビュー結果: 2往復目で計画・実装ともOK。テスト値の定数化1件を修正し、再ビルドと関連テスト、全比較結果を再確認した。再確認ポイントのうち、本工程で完了できるものは解消した。Atari の設定反映と効果実験は、PRD の条件・承認待ちとして未実施である。根拠が無く見送った指摘、範囲外の検出、開発者への確認事項は無し。

決まらなかったこと: 新たな未決・起票候補は無し。既存の `issue-20261111-77ac3c` に、Atari の warm-up 200,000 行と効果確認の残作業を記録した。判定は残件有り、状態は対応中とした。

判断してほしいこと: 本実装を止める判断事項は無し。KFM の10M×10本は具体的な段取りを別に示して承認後に行い、その結果から PRD の U1〜U3（β・観測の正規化とwarm-up・EMAの既定）を判断する。

次にできること: メイン機の Montezuma キューの stage20 終了を確認したあと、Atari の warm-up 行と対応テスト・文書を追加する。効果実験は別の段取りを作る。本工程でコミット・ステージ・push は行っていない。

## 追加の二モデルレビュー

- 2026-11-11: 開発者から `cc-review` と、指摘があれば修正して再レビューする指示を受けた。対象は同じ計画と実装。現在の差分基準は `3b6450ac1ea9e42ea12a5e059c97c20a3feefa13`、依頼は `.scratch/prd123/request-20261111-0919.md`。
- 1往復目: Codexは計画・実装ともOK、Claudeは計画OK・実装差し戻し1件。Claudeの指摘は、boolの既定値を初期値と通知の判定・文言で重ねて直書きしていたこと。原文は `.scratch/prd123/review-20261111-0919.md` と `review-20261111-0919-claude.md`。Claude費用は約1.8768 USD。
- 修正: `RndNoveltyEstimatorConfig::kDefaultErrorNormUseCentering` と `DefaultDQNAgentConfig::kDefaultIntrinsicUseDoneMask` を設け、初期値・通知で共有した。done mask の通知は指定値と既定値をJSONのbool表記で組み立て、従来の英文を保つ。公開ヘッダの追加はこの既定値定数だけで、設定・状態・アルゴリズムの契約は変えない。
- 確認: 変更3ファイルのLF・BOM・禁止構文・重複リテラルと差分空白は成功（`.scratch/prd123/cc-fix-static.log`）。重複文字列の候補2件は設定キーで、定数化の例外に当たる。前掲と同じMSVC初期化経由のDebug全体ビルドが終了0、`core/anet-core/bin/Debug/anet-core-test.exe "[prd123],[prd116],[scaler],[shipped_config]"` は60ケース・4,804断定すべて成功した。通常stdout/stderrは `.scratch/prd123/cc-fix-debug-build.log` と `cc-fix-tests.log` へ保存した。
- 修正後RunnerのSHA256: `7aa4a240a4fc84bad4ead21069f1bb6aabbe436bf622bb3929bd05e1d739224c`。今回変更した既定値と通知の検証は上の関連テストで確認した。品質確認観点の全項目を適用範囲で再点検し、変更起因の不足は無い。新たな所有・同期転送・契約・設定キー・アルゴリズムを足していない。
- 既存の100k件比較6条件は `after-r1` 実行体の結果として保持する。今回の変更は同じtrueを表す定数への置換と同文の通知の組み立てで、統計・採点・学習・保存・設定値を変えていない。今回の修正後の確認と、先に行った比較結果を区別して記録する。
- 2往復目: Codex・Claudeとも計画・実装がOK。Claudeの既定値共有1件は両モデルが解消を確認した。原文は `.scratch/prd123/review-20261111-0932.md` と `review-20261111-0932-claude.md`、依頼は `request-20261111-0932.md`。Claudeは前回セッションの続きで、再レビューのSDK費用表示は3.22078 USD（初回表示1.8768054 USD）。
- 再確認ポイント: Claudeは道具の制約で検査スクリプトを実行せず、Codexと親の確認記録を根拠とした。計画の見出しの形式、feedback比較に足した使われないモード値、偽Learnerを実Learnerの公開TD平均で代えた点は、前回の説明・検証から許容する判定を維持した。長い比較を繰り返していない扱いも、両モデルが今回の変更範囲と記録を確認した。範囲外の検出・開発者確認事項・根拠なく見送った指摘は無し。
- 追加レビューで新たな未決・起票候補は無し。Atariの設定反映とKFM効果実験は引き続き既存課題の条件・承認待ち。ステージ・コミット・pushは行っていない。
