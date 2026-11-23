# Evaluation Trace の β を学習前から用意する 実装メモ

## 概要

PRD129 を実装する。RND が有効な Agent は構築時に exp step 0 の β を保持し、学習を止めた入口でもその入口の exp step で更新する。採点・補助学習・診断値の公開は学習を有効にした場合だけ行う。checkpoint の書式、公開 API、設定、同期とロックの契約は変えない。

## 主な変更

- `default_dqn_agent.cpp` の構築と学習の入口、ヘッダーの直近の β の注記を変更する。
- `intrinsic_test.cpp` の checkpoint と同期のテストを更新し、予定付き β の進行と学習停止中の全内的報酬診断の不変性を追加する。既存 Trace の初回ボーナスの期待値も新しい契約に合わせる。
- 設計文書 020・200・212 の日英と PRD127 検証資材のテスト名を更新する。英訳の翻訳元情報は対応を確認した範囲で更新する。
- wiki の内的報酬ページにある「評価だけの Run では空欄になる」を PRD127 当時の挙動へ限定し、現在の設計文書と今回の確認を出典として更新を記録する。`docs/wiki/log.md` に fix の行を足す。
- 対象課題 `issue-20261113-317d32` の工程の状態・対応内容・判定、および一覧の派生区画を更新する。他作業の変更は保持する。
- 工程末に開発者が選んだ追加対応として、`intrinsic_test.cpp` の既存テスト 3 箇所の関数内 `RemoveCheckpoint` も、すでに無名 namespace にある `IntrinsicCheckpointCleanup` の利用へ置き換える。保存・削除の時機とテストの断定は変えない。PRD129 の機能追加とは別の、入力元なしの小さな実装として同じレビュー担当へ出す。

## テスト

- Public interface / surface: Agent の公開 Save と auto_load_file、CreateLearner / UpdateFromBatch、Actor::Sync、CreateStepDiagnosticsProvider / Observe / GetScalar、および Montezuma の Evaluation Trace。
- 優先 behavior と TDD 順序:
  1. tracer bullet: checkpoint を読む新しい Agent（learner.enabled は偽と真）を同期し、最初のボーナスが定数 β × clamp(z, 0, 上限−1) に一致するテストを RED にし、構築時の評価を足して GREEN にする。
  2. 既存の起動直後の同期・Trace の期待値を、標準化した値 NaN・ボーナス 0 に更新して確認する。
  3. 学習停止中の linear β（0.2→0.6、10 step）が、入口を通る前に 0.2、exp step 5 の入口後に 0.4 となるテストを RED にし、β の評価を学習有効の分岐の前へ移して GREEN にする。
  4. 学習停止中に入口を 3 回通しても、群56の全キーが変わらないテストを追加する。既存実装ですでに成立する不変性なので、失敗を人為的に作らず回帰確認として扱う。
  5. PRD127 の報酬一致・交互採点の学習不変性テストは期待値を変えずに確認する。

## 検証

- Debug のテスト対象を VsDevCmd 経由でビルドし、各 RED / GREEN のログを `.scratch/prd129/` へ保存する。
- Debug 全体ビルド、`anet-core-test.exe "[prd129]"`、`anet-core-test.exe "[prd127]"`、内的報酬の関連テストを実行する。標準出力と標準エラーを同じログへ保存し、応答は終了値と件数のみ。非同期 session を長めに待つ。
- Montezuma の `run_20261113-113201_mz_tg_r123_s1_c50m` の `agent_close.anet` を、評価専用で train 自動開始あり・なしの 2 通りで開く。それぞれ最初の行と選んだ 3 行の rnd_bonus が、0.01 × clamp(rnd_norm, 0, 4) と表示の 4 桁で一致することを確認する。専用の使い捨て Run だけを操作する。
- PRD §7 の旧文言検索、BOM / LF / diff check、日英の見出し・表・コードブロック・リンクと内容対応を確認する。
- 品質確認観点の全項目を適用判定して自己点検し、計画と実装を独立レビューへ出す。指摘は修正して同じレビューへ出し直す。

実行するビルドの形:

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test --parallel 2'
```

## 前提

- ユーザー判断が必要なブロッカーは0。PRD は確定済みで未決表は空。既存コードと §2 の現状が一致し、PRD128 の移し替え前の置き場を使用する。
- 直近の β を読むのは Trace だけで、学習では入口の値を使う。Evaluate は既存の ProfiledValue を用い、新しい所有者・乱数・ロック・device 転送は増やさない。
- Run の step 復元、β の永続化、性能測定、PRD121 / 128 の実装は範囲外。CONTEXT と ADR は変更しない。
- 基準コミット: `68cb2eed545f5257c96d11ed9dd6e175585ed090`。開始時の変更ファイルに今回の本体・テスト・設計文書は含まれない。一覧と Runner の wiki にある既存変更は保持する。

## 経過

- 2026-11-14: PRD と現行実装・テストを照合し、未決事項監査を通過。上の計画から実装する。
- 工程の状態は `anet-update-issue` の「工程の出来事」で対応中にし、`maps-start.log`（exit 0）で一覧を更新した。
- tracer bullet: `red-checkpoint-build.log` は exit 0、`red-checkpoint-test.log` は exit 42（NaN による 4 assertions の失敗）。構築時の β 初期化後、`green-checkpoint-build.log` と `green-checkpoint-test.log` は exit 0、1 case / 10 assertions。
- 学習停止中の予定: `red-schedule-build.log` は exit 0、`red-schedule-test.log` は exit 42。exp step 5 でもボーナスが 0.8 のまま（期待値 1.6）で、Actor 複製有無の 2 assertions が失敗した。β 評価を学習有効の分岐の前へ移し、`green-schedule-build.log` と `green-schedule-test.log` は exit 0、3 cases / 34 assertions。
- 全診断値の不変性は、学習した checkpoint を読み、13 キーすべてを 3 回の入口の前後で照合する回帰テストを追加した。保存前の学習有効な Agent で learn_count が正であることを要求し、学習済みの checkpoint を使っていることを確かめる。初回と r2 の関連テストは各 60 cases 中 1 case が、学習停止中の Agent の診断値を有限とした誤った事前条件（error_step、learn_count）で失敗した。学習停止中の Agent の GetScalar はこれらを NaN とする既存契約なので、有限性の事前条件を保存前へ移した。これらは実装の RED には数えない。
- GUI 用の一時設定 `.scratch/prd129/gui_main.txt` は、出荷の `run.@evalonly` と既存 Run の凍結設定を使う。自動評価 session の 4 タグは interval 0 にして、手動 EvalPanel の確認へ絞る。元の Run と出荷設定は書き換えない。使い捨て Run は `.scratch/prd129/runs/` に保存する。
- 翻訳元情報は変更前の日本語 blob と一致していることを確認し、変更した段落・表の対応を確認して更新した。変更していない本文は保持する。
- Debug 全体ビルドは `final-debug-build.log`、r2、r3 とも exit 0。r3 の `[prd129]` は 5 cases / 134 assertions が成功した。
- GUI: `run_20261114-001948_tmp_prd129_start` は train 自動開始あり、`run_20261114-002225_tmp_prd129_pause` は自動開始なしで、同じ 50M checkpoint を読んだ。Trace を開き、Step で最初の 3 行を作って次の値を読み、0.01 × clamp(rnd_norm, 0, 4) と表示の 4 桁で一致することを確認した。後者では train の exp / train 欄は未進行の `-` のままである。

  | Run の条件 | eval step | rnd_err | rnd_norm | rnd_bonus | 式の値 |
  |---|---|---|---|---|---|
  | 自動開始あり | 1 | 0.007975 | -0.2692 | 0 | 0 |
  | 自動開始あり | 2 | 0.00748 | -0.2878 | 0 | 0 |
  | 自動開始あり | 3 | 0.007208 | -0.2981 | 0 | 0 |
  | 自動開始なし | 1 | 0.007984 | -0.2688 | 0 | 0 |
  | 自動開始なし | 2 | 0.007471 | -0.2881 | 0 | 0 |
  | 自動開始なし | 3 | 0.007206 | -0.2982 | 0 | 0 |

- GUI の最初の Run は、出荷の run.@evalonly の優先順位でファイル内の interval 0 が上書きされ、4 つの自動評価が走った。終了要求後の待ちが長かったため、パスを照合した専用 PID 43536 だけを停止してログを保持した。2 回目は CLI の interval 0 で自動評価を止め、通常の閉じる操作後に PID 14428 が終了したことを確認した。既存の Release 実験は操作していない。
- `verify_changes.py`（`.scratch/prd129/`）で、対象 11 ファイルの LF、既存 10 ファイルの BOM、日英 3 対の見出し階層・表の行数・コードフェンス、追加リンク、翻訳元 blob、PRD §7 の旧文言 0 件、`git diff --check` を確認した（`verify-changes.log`・r2、exit 0）。wiki の訂正後は wiki とログを含む 13 ファイル（既存 12）を同じ検査へ含める。
- 新しい公開 API・設定・所有者・ロック・device 転送はない。β の数回の算術は既存の計測と書き込みロックの中にある。ログと失敗の扱いは変更しない。設定の変更は受入用の `.scratch` の一時設定だけで、出荷設定は変更していない。
- `intrinsic-regression-r3-test.log` は exit 0、60 cases / 4,954 assertions。PRD127 の学習報酬一致と交互 Trace の学習不変性も、断定を変えずに通った。自己点検で繰り返す列挙値 `clipped` を既存の `kMultiplierClipped` へそろえたので、最後に同じ Debug 全体ビルドと `[prd129]` をもう一度実行する。
- 上の定数への変更後、`final-debug-r4-build.log` と `prd129-r4-test.log` も exit 0、5 cases / 134 assertions。`verify-changes-final.log` は exit 0、対象の全テキストは w/lf。レビュー用差分は `.scratch/prd129/diff-20261114-0027/` に保存した。

## 実行した検証

作業ディレクトリは `C:\dev\anet-lab`。stdout / stderr を下記ログへまとめて保存し、exec の session を write_stdin で待った。

| 確認 | 実行したコマンド | ログと結果 |
|---|---|---|
| Debug 全体 | `cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --parallel 2'` | `.scratch/prd129/final-debug-r3-build.log`、exit 0 |
| 定数への変更後の Debug 全体 | 上と同じ | `.scratch/prd129/final-debug-r4-build.log`、exit 0 |
| PRD129 | `& core/anet-core/bin/Debug/anet-core-test.exe '[prd129]'` | `.scratch/prd129/prd129-r3-test.log`、exit 0、5 cases / 134 assertions |
| 定数への変更後の PRD129 | 上と同じ | `.scratch/prd129/prd129-r4-test.log`、exit 0、5 cases / 134 assertions |
| 関連回帰 | `& core/anet-core/bin/Debug/anet-core-test.exe '[prd129],[prd127],[prd116],[prd123]'` | `.scratch/prd129/intrinsic-regression-r3-test.log`、exit 0、60 cases / 4,954 assertions |
| 文書・BOM・LF・移行 | `& .venv/Scripts/python.exe .scratch/prd129/verify_changes.py` | `.scratch/prd129/verify-changes-r2.log`、exit 0 |
| 重複リテラル | 品質確認観点 72 の追加行の候補抽出 | `.scratch/prd129/duplicate-literals.log`。設定キーは例外。clipped は既存定数へ変更 |
| 共有する契約 | 品質確認観点 4 の 2 検索を protocol とレビューエージェント定義へ実行 | ともに 0 件 |

## 自己点検

- 実装品質確認観点の全項目を適用判定した。計画の前提・TDD の順・PRD §7 の入力と回数・受入 1〜6 を、コードと検証結果へ照合した。起動時のボーナス、予定進行、13 キーの 3 回の不変性、checkpoint の真偽両経路、2 通りの GUI を確認した。
- 本体は構築時の β の設定と既存の入口での評価位置の変更だけ。公開 API・設定・保存形式・失敗・ログ・乱数の消費・所有・借用・namespace・include を変えていない。汎用層と UI の変更、互換コード、追加の同期転送はない。それらだけを対象とする項目は対象外。
- コメントは日本語、変えたテストには `[prd129]` を付け、既存 `[prd127]` を保持。NaN を 0 で偽装せず、新しい契約で倍率 1 のときだけ実際のボーナスが 0 となることを確認した。
- 日英 3 対の変更箇所の内容・構造・リンクと翻訳元、既存 BOM / LF、旧文言の移行と過去記録の保持を確認した。PRD 本文・CONTEXT・ADR・出荷設定は変更しない。課題の対応内容も追記した。
- 自己点検後にユーザーへ判断を求める未決はない。独立レビューへ計画と実装を渡す。
- レビューと並行する終了確認で wiki の過去の事象が現行の制約にも読めると分かり、PRD127 当時という時期を補って、PRD129 実装後の確認を出典つきで追記した。出典とページの食い違いを直す AGENTS の規約に従い、この wiki と fix ログも同じ初回レビューへ追加した。

## レビューと修正

- 1 往復目: 計画は OK、実装は差し戻し。原文は `.scratch/prd129/review-20261114-0033.md`。新設した 2 つのテスト関数内の後始末用 struct の置き場が規約に合わないという 1 件を、自分で直す指摘として受けた。無名 namespace に `IntrinsicCheckpointCleanup` を置き、新設の 2 テストで共用する形へ直した。PRD079 D1〜D4 を読み、型だけをインデントした既存 namespace へ足した。
- 帰結の確認: `RemoveCheckpoint` は既存の 3 型だけが残り、新設の 2 つにはない。`IntrinsicCheckpointCleanup` は型宣言 1 と使用 2。新しい API・所有・保存形式は増えていない。該当する補助型の配置・指示付き初期化・namespace・テスト経路を再点検する。
- 修正後の Debug 全体ビルド（`review-fix-build.log`）と `[prd129]`（`review-fix-test.log`）は exit 0、5 cases / 134 assertions。既存の 60 cases の結果は本体が同じ r3 の記録を用い、今回の型の配置変更に関わる 5 cases を再確認した。補助型の配置と namespace の規約を自己点検し、同じレビューエージェントへ出し直す。
- 2 往復目: 計画・実装とも OK（2 件）、差し戻し 0 件。原文は `.scratch/prd129/review-20261114-0037.md`。新しい範囲外の検出・開発者への確認事項はない。機能の受入は初回の確認を引き継ぎ、補助型の配置と修正後の検証記録の不足が解消した。レビュー自身はテストを再実行せず、保存済みの記録で確認した。
- PRD129 の対象課題を `anet-update-issue` の「判定」で残件無し、ai_judged を 2026-11-14 にした。status は対応中のままで、解決・監理の項目は変更しない。ステージ・commit・push はしていない。

## 工程末で選ばれた追加対応

- 2026-11-14: 開発者が「既存3箇所もこの場で直す」を選んだ。対象は既存の checkpoint 検証 2 本と、PRD129 で断定を更新した checkpoint Trace の検証 1 本。近くの変更と現在の Git 状態を確認し、並行する文書・skill・wiki の変更は保持する。
- 純粋な型の配置の整理なので、新しい挙動の RED は作らない。すでに通った公開経路のテストを維持し、Debug 全体ビルドのあと、既存 2 テストの名前と `[prd129]` を合わせた 7 cases を確認する。`RemoveCheckpoint` が 0 件、共通型の使用が 5 箇所になったこと、BOM / LF / diff check を確認する。
- ビルド・テストの stdout / stderr は `.scratch/prd129/selected-cleanup-build.log` と `selected-cleanup-test.log` に保存し、session で待つ。自己点検後、同じレビュー担当へ追加差分を出す。
- 結果: Debug 全体ビルドと上記の 7 cases / 356 assertions がともに exit 0。`verify-selected-cleanup.log` も exit 0。`RemoveCheckpoint` は 0 件、`IntrinsicCheckpointCleanup` は宣言 1・destructor 1・使用 5 箇所。既存のテスト名・タグ・断定と checkpoint のファイル名・lifetime は変えていない。
- 追加分も実装品質確認観点の全項目を適用判定し、局所的な型の利用への変更、指示付き初期化、BOM / LF / diff check を確認した。新しい挙動・API・設定・補助型・ログ・失敗はない。設計・ガイドへの追加追随は不要。同じ内容の既存課題は無いので、課題記録の新設や別件への追記はしない。
- 追加対応のレビュー（通算 3 往復目）も計画・実装とも OK、差し戻し 0 件。原文は `.scratch/prd129/review-20261114-0046.md`。保存・削除の時機、ファイル名、名前・タグ・断定の維持と、ビルド・7 cases の成功記録を確認された。新たな範囲外の検出・開発者への確認事項はない。

## 決まらなかったこと

なし。レビューで見つかった既存の 3 箇所は工程末に提示し、開発者がこの場での修正を選んだ（上の追加対応）。`RemoveCheckpoint`・後始末の型・ローカル struct・補助型の配置で課題を検索し、同一の既存課題は見つからなかった（検索に当たった cfe994 は Observer の複製で別件）。新規起票はしない。

## 完了報告

- 2026-11-14: PRD129 の実装、必要な自動テストと 2 通りの GUI 受入、日英文書と wiki の追随、自己点検、独立レビュー、選択された既存 3 箇所の追加修正と再レビューを完了した。
- 評価専用・学習未開始・起動直後でも、構築時の β によるボーナスが出る。学習停止中も入口の exp step で β の予定を更新し、内的報酬の採点・補助学習は休止する。公開 API・設定・checkpoint 書式は変えていない。
- Debug 全体ビルド成功。関連 60 cases / 4,954 assertions が成功し、その後の型の整理は影響する 7 cases / 356 assertions で再確認した。機械検査と文書確認は成功。必要な受入で未実施のものはない。
- レビューは通算 3 往復。初回の型配置の指摘 1 件を修正し、2 往復目に計画と実装が OK。ユーザーが選んだ追加対応も 3 往復目に OK。レビュー自身はビルド・テストを再実行せず保存済み記録で確認した。判断待ち・起票候補・残件はない。
- 課題の AI 判定は残件無し、status は対応中。別作業の変更と index は保持し、ステージ・commit・push は行っていない。
