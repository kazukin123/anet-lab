# 課題の正本を issue に移す 実装メモ

## 概要

PRD114 の承認済み計画を実装する。課題の正本を issue に移し、PRD は指示文書として保持する。台帳の退役、既存データの移行、skill・一覧・wiki・統計の更新を同じ変更で行う。

## 主な変更

1. 初版の課題 skill 3 本、発番の文書、品質観点の検査4、レジストリを改修する。共有コアは変更しない。
2. PRD の索引 issue と親子関係、対応表、監査と実装完了の AI 判定、工程末の4区分と「この場で直す」を接続する。skill は両配置を同期する。
3. 台帳を全件移し、F番号の対応を検証して旧ファイルとCLIを退役させる。使用済み番号を再集計して999を振り直し、直下のPRDへ索引issueを作る。
4. wiki_support の読取・検証・並べ替えを統計側でも使い、INDEXとwikiの課題区画、実装済み検出を更新する。設計文書の日本語版と英語版、既存テストも更新する。
5. 試走、自己点検、独立レビューと修正、動線マップの再生成を行う。監理項目の棚卸しは領域ごとに提案まで行う。

## テスト

- Public interface / surface: wiki_support.py の scan / maps、stats.py の出力、課題 skill と PRD 起票・実装の工程。
- 優先 behavior: issue から INDEX と地図の課題区画への生成、本版の既定、依存と WSJF 継承、複数 area、アプリケーションの範囲、不正入力時の全出力不変、区画外保持、冪等性、同 blob の実装済み判定、過去の done の扱い。
- TDD 順序: 一時 repo の issue を maps へ渡して一覧と地図が生成される1テストを RED→GREEN とする。その後は上の behavior を1つずつ RED→GREEN で進め、GREEN のあとだけ整理する。
- 試走Aは実データの一時コピーで起票とレビューまで行う。B・Dも隔離環境で実際の skill を動かし、元の index・ref・仕事一覧へ試験用データを加えない。
- 検証資材は core/anet-core/testdata/prd114/ に置く。PRD095の関連テストも移行する。

## 検証

- PRD114 §7 の受入を項目ごとに記録する。
- F番号の過不足、索引issueの一対一対応、番号の重複・再利用、リンク切れ、skill同期、LF・BOM、共有コアの不変を確認する。
- 関連Pythonテストと、AGENTS.md の VsDevCmd 経由の Debug ビルドを実行する。
- PRD の補足は prd、実装と本計画は impl / impl-plan として別々にレビューし、応答原文を .scratch/prd114/ に保存する。

## 前提

- 基準コミット: 1cbe76d65c3087f7d6b06645cf7243ddb2e2179b。開始時のユーザー変更は保持する。
- 計画時の対象: 台帳44件、999のPRD47本、直下のPRD64本。実装開始時に再集計し、未追跡のPRDも含める。
- ユーザー判断: 引数なしの一覧とINDEXは本版（次版・解決を除く）を既定とする。依存・WSJF継承は全件で計算してから絞る。閉じた課題は状態指定で確認する。
- ユーザー判断: 試走AのPRDと親子関係は一時コピーだけに作り、検証記録を残す。
- issueの追加欄は area / type / ai_judge / ai_judged / docs。AIの自発的な変更は判定欄だけに限定する。
- Python CLI は scan / maps を維持する。YAMLは safe_load を使い、PyYAML 6.0.3 を明示依存にする。
- 全入力と地図の目印を検証してから書き、不正入力時は INDEX と地図を更新しない。
- 過去資料は改名に必要なリンクだけを直し、歴史的な本文とwikiログを保持する。
- 試走C・C2・Eは所定の実際の場面がなければ理由付きで未実施とする。最初の5回の工程末の成功指標は後続の場面で記録する。
- コミット・ステージ・push は行わない。

## 経過

- 2026-11-03: ユーザーが提案計画を承認し、実装を指示した。実装計画を保存。開始時の作業ツリーにあるPRD114・ADR0049・wiki311と無関係なbat・未追跡ファイルの既存変更を保持する。
- 2026-11-03: .scratch/prd114/baseline.json と baseline/ に開始時の本文・hashを保存。台帳44件、直下PRD64本、999のPRD47本を再確認し、改名対応表を固定した（末尾959）。migration.json に F番号→issue、PRD→索引issue、改名の全対応を保存した。44件と64本の計108件を移し、旧台帳とCLI、専用テストを退役した。
- 2026-11-03: 課題skillの追加欄と判定モード、PRD起票と工程末の接続を反映。PRDへ本版の既定と隔離試走の合意を追記した。
- 2026-11-03: PythonのTDDで、mapsによるINDEX・課題区画の生成、本版の既定、依存とWSJF継承、不正な追加欄で全出力不変、同blobのPRDの実装済み検出、統計のissue・未追跡PRD集計を、それぞれRED→GREENで実装した。
- 実装済みの取り込み記録: scanのpending行のimplementedにAIの判定日を返す。取り込む側は既存のingest/writeback行へ `implemented:YYYY-MM-DD` を添える。同じ本文・同じ判定日を取り込み済みなら再び候補にしない。本文のblobだけではPRDの起草時と実装後を区別できないためである。


- 2026-11-03: 開始時からのリンク切れ23か所について、ユーザーは今回の受入を増分0件とし、別issueへ記録することを選んだ。直す場所と判断が異なるため、現行ガイド・過去PRD・消失した検証出力の3件に分けた。記録先: issue-20261103-c1c608, issue-20261103-fb8a45, issue-20261103-ee32bd。

## 決まらなかったこと

- 2026-11-04: PRDレビュー1往復目はADR0049のHEADとの差分を理由に差し戻し。開始時hashとの比較で、当該ADRはこの実装中に一切変わっていないと確認した（SHA256 8207b11cf8687afe28d58f66385a802bdebaf1440008dffa1f3d3d4d1b7ac565）。PRD §10で入力と編集境界を明記して再依頼し、2往復目でOK、最初の指摘は取り下げとなった。原文: .scratch/prd114/review-20261104-0007-prd.md、review-20261104-0009-prd.md。

2026-11-03に決着した事項: 旧CLIへの過去資料のリンクとdocs全体のリンク検査の両立。ユーザーは「旧CLIが存在するコミットへの固定リンク」を選んだ。PRDの改名分に加え、退役資材へのリンクだけも実装開始コミットへ向け、本文の歴史的記述とwikiログを保つ。

未解決: 設計160英語版 §5.1 の起動図が、RunManagerをRunnerAppから直接構築する旧フローを示す。日本語版とRunnerApp.cppの現在の実装はRunInitThreadで構築する。実装レビューの最後に見つかった当初は、開発者に質問していなかった。今回変えた節の外なので受入の差し戻し理由ではない。

## 検証結果

2026-11-04。PRDの補足と実装は別の観点でレビューする。Pythonの変更は、1つの挙動ごとに修正前の失敗を確かめ、実装後に同じテストを通した。受入以外のC++実装は変更していない。

| 検証 | コマンド・手順 | 結果 |
| --- | --- | --- |
| PRD114の公開CLI | `.\.venv\Scripts\python.exe -B core/anet-core/testdata/prd114/issues_test.py -v` | 引用符なしの日付診断も修正後、16件成功、65.898秒。INDEXと地図、scan、統計を一時repoから通す |
| PRD095の既存wiki | `.\.venv\Scripts\python.exe -B core/anet-core/testdata/prd095/wiki_test.py -v` | 日付検証の修正後、8件成功、131.684秒。退役した台帳専用テストは削除した |
| Debugビルド | `cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug'` | 成功（終了値0）。11出力を再構築。既存のD9025警告あり |
| 実データの統計 | `.\.venv\Scripts\python.exe -B reports/tools/stats.py` | 成功、縮退なし。本版108件・直下PRD64本。期間2026-10-23〜11-04のdigestもsummaryの数字だけで追記 |
| 移行・文字コード・リンク | 開始時の本文とhash、Git履歴、F番号・索引・改名の対応を突き合わせる。作業用検査は `.scratch/prd114/check.py` | 成功、errors=[]。F44件、直下PRD64本、単票111件、改名47本、移行による新しいリンク切れ0件。LF・BOM・同期・共有コアの不変も成功。`git diff --check`成功。既存リンク切れ23か所は別issue |
| 動線マップ | Archifyのvalidate→deliver→visual-check。公開版・個人版の2図ずつ | 4図ともshowcase 9/9、errors 0、warnings 0。1440×900・1600×1000・1920×1080・2048×1320の包含検査と、最小・最大サイズの明暗画像を確認。工程図は行の整理とラベルの修正を2回の限定修正で実施 |

自己点検では、保留を経由するWSJFの継承、解決済みを経由する依存循環、2つの索引が共通の設計文書を参照する場合を追加でRED→GREENにした。継承元は活動中の課題の元の点数に限定し、保留自身の点数は伝えない。循環は直接の依存が解決済みでも「待ち」とする。一対一の制約はPRDにだけ適用する。

実装レビュー1往復目（原文 `.scratch/prd114/review-20261104-0128-impl.md`）: 計画はOK、実装は日付形式の受理、現行文書の旧説明・節参照・WIP、テストの段階コメントで差し戻し。raised・resolved・ai_judgedの整数・週日付・不正な暦日を公開CLIで拒否し、全出力が不変であるテストを追加してRED→GREENとした。YAMLの日付型と空欄は維持する。列挙された規約・参照を両skill配置で揃え、15テストの段階コメントを追記した。レビュー依頼一覧では、開始時の未追跡改名元14本を新規と誤記していたため、改名先321パスの主表と改名元の参考表に分けて直した。

統計で未分類だった公開ヘッダ6本は、`core/envs/atari1/include/anet/env/Atari.hpp`、`core/envs/cartpole2/include/anet/env/CartPole.hpp`、`core/envs/dropmerge1/include/anet/env/DropMerge.hpp`、`core/envs/gridmaze1/include/anet/env/GridMaze.hpp`、`core/envs/imagecls1/include/anet/env/ImageCls.hpp`、`core/envs/lunarlander1/include/anet/env/LunarLander.hpp`。分類追加案は、categories.jsonの322_env_individualに `^core/envs/[^/]+/include/` を含めること。categories.jsonの値は変えていない。

実装レビュー2往復目（原文 `.scratch/prd114/review-20261104-0149-impl.md`）: 前回の3件は解消し、計画はOK。追加で、引用符なしの不正暦日をPyYAMLが構築するときのValueErrorが欄の診断を失う点が差し戻しとなった。3つの日付欄を単票へ直接書く公開CLIテストを追加してRED→GREENとし、構築前の構文木から問題の欄と値を特定して診断するよう修正した。safe_loadは維持し、全出力不変も確認した。

2026-11-04、ユーザーは通常レビュー完了後にcc-reviewも通して指摘対応まで進めるよう指示した。実装と計画を同じ依頼でCodexとClaudeへ出し、指摘を直して再確認する。

実装レビュー3往復目は計画・実装ともOK。引用符なしの不正暦日への診断と16件の成功記録を確認した（原文 `.scratch/prd114/review-20261104-0203-impl.md`）。その後、ユーザー指定のcc-reviewを開始し、PRDと実装・計画の2組に分けた同一依頼を各モデルへ送った。

cc-reviewのPRDはCodex・ClaudeともOK。Codex原文は `.scratch/prd114/review-20261104-0204-cc-prd.md`、Claude原文は `review-20261104-0203-cc-prd-claude.md`。Claude側は検査スクリプトを実行できない制約がある。補足の再確認として、保存した試走Dの初回本文と再監査後の本文を `git diff --no-index` で比較し、変化がai_judgeだけと確認した（ai_judgedは同日で同値）。ADR0049と0068も保存した開始時本文とバイト一致した。証拠は `.scratch/prd114/review-rechecks.json` と `trials/d-ai-fields.diff`。PRD本文の追加変更はない。ClaudeのPRDレビュー費用は約0.8672 USD。

cc-reviewの実装1往復目: Codexは計画・実装ともOK、Claudeは計画OK・実装差し戻し。［Claude］地図16ページとwiki規約に残った台帳による作り直しの説明を、issueの道具へ修正した。［Claude］現用の地図に残った旧999名の表示と、リンクのない名指しも固定対応の新名へ揃えた。過去の本文とwikiログの既存行は保持し、wikiにfix行を追記した。検査も接尾辞なしの名指しまで見るよう補強した。［Claude］「無題のファイル.md」を対象一覧に混ぜていた。作成者は特定できないが、今回の生成コードはIDから名前を作り、この名前を生成しない。実ファイルの本文・mtime・hashを保持し、今回の対象一覧だけから外した。対象は321から320パスへ訂正した。対応記録は `.scratch/prd114/cc-fixes.json`。Claudeの実装初回費用は約2.0956 USD。コードは変更していない。

cc-reviewは最終的にPRD・計画・実装すべて、Codex・ClaudeともOK。実装のClaude2往復目で3件の解消を確認し、補足の修復ログのblobも同じ実行で追加した行へ加えた。3往復目でClaudeがOK、Codexも19文書と追加の1行を合わせて確認してOKとなった。原文は `.scratch/prd114/review-20261104-0220-cc-impl.md`（Codex）と `review-20261104-0214-cc-impl-claude.md`（Claude）。統合した説明は `cc-review-summary.md`。セッション別の最終費用表示はPRD 0.8672125 USD、実装3.7589027 USD、合計約4.63 USD。各往復の累積表示は重ねて足していない。

### 隔離環境でのskill試走

`.scratch/prd114/trials/repo/` に、実データとskill・関連コードを複製し、独立したGit repoを作った。元のrepoのGit index・refは共有していない。テンプレートの書き込みと比較を補助スクリプトで行い、採用・拒否・監査の意味判断は実際のskillの手順を読んで行った。

| 試走 | 操作と観測 | 結果 |
| --- | --- | --- |
| A | 312_utilのF0030・F0032の実データ2件を題材に暫定PRD115を起こした。現行InitRL・ParseDevice・ThrowErrorを読み、テストの既存名とタグまで指定。独立レビューの指摘後に修正し、対応表の38・39を含めOK。索引issue `issue-20261104-2cc161` を作成し、題材2件はparentだけを変更した | 合格。試験用PRDと索引は一時repoだけに存在 |
| B | 存在しないID、ID書式違反、次版、解決、索引、他PRDとの結び付き、既存parentをD11の順に確認して拒否。未知areaとmax=0も拒否。各拒否前後のPRD・issue本文を比較 | 合格。拒否したID・値と理由を示し、書き込みなし |
| D | 一時コピーでanet-list-issuesのCodex表示YAMLを片側だけ除いた。実際のprocess観点で検出し、phase=監査の `issue-20261104-c2c600` を記録。YAML復元後の再監査でAIの2欄だけを更新 | 合格。status・監理項目・索引issueは不変 |

試走Aのレビュー原文は `.scratch/prd114/review-20261104-0010-trial-a-prd.md` と `review-20261104-0014-trial-a-prd.md`。A・B・Dの入力、判定、前後hashは `.scratch/prd114/trials/state.json`。元のHEADは開始時のままで、indexのSHA256は `9cf009a3535d396bd25402ea2d8d5a5b974620d634debeb53a5b1500a8ca42d3` のままである。

### PRD §7の受入との対応

| 項 | 結果 | 根拠・残る確認 |
| --- | --- | --- |
| 1 | 合格 | skillの両配置をバイト比較。差なし |
| 2 | 合格 | 追加欄と操作、表示YAML、節参照、検査4を確認。共有コア3ファイルは開始時hashと一致 |
| 3 | 合格 | newid.jsでissue-20261104-0788dc・eac7eb・8d0da4を発行して形式を確認（ファイルは作らない） |
| 4 | 合格 | issue-recordの出力先と工程末の読み替えをレジストリに反映 |
| 5 | 合格 | F0001〜0044の一対一対応、旧5ファイルの削除、現用範囲の旧参照0件 |
| 6 | 合格 | 47本を改名。9xxは57本、999は0本、使用済み番号の再利用なし。旧10本は不変。改名・退役・移行によるリンク切れ増分0件 |
| 7 | 合格 | 未追跡を含む直下64本に索引64件。docsとtypeを検証 |
| 8 | 合格 | 本版の既定と全件での継承をCLIテストで確認。area/type・束ねの条件はskillを点検し、不正引数の拒否は試走B。棚卸しでAI判定とstatusの食い違いを提案へ含めた |
| 9 | 合格 | 隔離試走A、レビューOK、親子関係以外の題材項目は不変 |
| 10 | 合格 | 隔離試走B。不適格な入力では一覧・起草・更新を行わない |
| 11 | 未実施 | 新規・既存へ追記の選択まで成立した工程末が無く、Cの条件をすべて満たす場面はまだ無い。既存リンク切れの記録は実装途中で直接指示されたため、Cの工程末として数えない |
| 12 | 未実施 | 開発者が「この場で直す」を選ぶ場面がまだ無い。今回の承認範囲内の修正をC2と読み替えない |
| 13 | 合格 | 隔離試走D。新規記録と既存AI判定だけの更新を確認 |
| 14 | 未実施 | Aの試験用PRDや、PRD114完了後の最初のPRDはまだ実装していない。今回の索引判定の更新は別に行う |
| 15 | 合格 | INDEX・全地図の課題区画、areaの合致、同blobの実装済み再検出と取り込み済み判定、不正type・後続ページの不正目印で全出力不変をテスト |
| 16 | 合格 | 2配置の正式なレビューエージェント名と共有コアの名前・検査4を確認 |
| 17 | 合格 | memo README・AGENTS・issue-trackerから新規999と新規PRDを移動する指示を除去。過去の保管場所は保持 |

成功指標の「最初の5回」は、PRD114完了後の実際の工程末で記録するため未実施である。上の未実施はPRDが理由付きの持ち越しを認める場面待ちであり、未完の実装を合格扱いしたものではない。

単票111件のうち、次版2件と解決1件を除いた本版108件を確認した。束ねも実データへ読取だけで適用し、311_dev_toolsの3件、120_metrics_viewerの4件、317_metricsの4件の順で最大3組となった。全件にPRD参照とparentがなく、索引でもなく、各組5件以下。各組のID・題名とWSJFは `.scratch/prd114/bundles.json` に保存した。束ねの理由は、それぞれ資料の参照修復、Viewerの表示と再取得、画像メトリクスの境界の整備である。新しいPRDは作っていない。

動線マップの実測は、死んだ動線2、実体コピーのドリフト0、同名skillはClaude6・Codex6、条件付きの移設候補20節532行。公開の固定5ファイルと `.scratch/harness-map/` の個人版5ファイルを更新した。個人版構成図にはユーザー環境の実ノード4つを加え、工程図は交差制約のためカードへ畳んだ。Archifyの元の資材は変更せず、生成HTMLのLFを守るため一時コピーのtemplateだけをLFへ揃え、同じvalidate・deliver・visual-checkで出力した。

自己点検: docs/quality/impl.mdの全項目を走査し、適用する機械検査・実装計画・PRD適合・失敗の扱い・名前・コメント・ログ・テスト・検証報告・skill同期を確認した。C++、学習設定、所有権、UIロック、GetScalarの項目は変更が無いため対象外。設計160の日本語版・英語版、wiki規約、skill一覧、検証資材の導線を同時に更新した。未決の実装判断は無い。

## 課題の棚卸し提案

2026-11-04、anet-list-issuesの棚卸し手順により本版108件（索引64件を含む）の本文を読んだ。areaの先頭で分け、複数areaのものは重複して数えない。以下は提案であり、statusと監理の値は変更していない。起票直後に見つけたカテゴリ照合の不具合による初期areaの訂正34件は、監理の見直しとは分け、`.scratch/prd114/index-area-corrections.json` に前後を保存した。

| 主領域 | 確認件数 | 提案の要点 |
| --- | --- | --- |
| 110_runner_app | 8 | 設定所有の主領域と、局所修正の作業量 |
| 120_metrics_viewer | 7 | LOD文書整合の作業量、リリース組立の主領域 |
| 311_dev_tools | 9 | ハーネスのkind、tag番号の主領域、CIの作業量、AI判定とstatus |
| 312_util | 5 | deviceの2課題は同時に扱えるが原因が異なるため統合しない |
| 314_tensor | 1 | テスト空白の3経路は別々に検証できる。現時点で監理値の変更案なし |
| 315_nn | 4 | NNブロック別metricsと公開拡張点は索引を維持。変更案なし |
| 316_config | 2 | Runキューの主領域をrunnerへ |
| 317_metrics | 5 | 画像の確保方法の不一致と重複除去を統合せず、優先度と作業量を見直す |
| 318_rl_core | 26 | 性能候補の独立した判断、MuZeroの領域、コメント修正の作業量 |
| 319_gui | 1 | 3か所のdynamic_castは1判断で直せるため分割案なし |
| 320_env_core | 2 | 評価trace・閾値率は複数領域の索引を維持。変更案なし |
| 321_agent_core | 14 | インクルードガードの作業量、実装済み判定の状態確認、異なる原因の重複は統合しない |
| 322_env_individual | 4 | AdaptiveMaxPoolの主領域をNNへ |
| 324_dqn | 3 | stack_keysの性質とpriority、実装済み判定の状態確認 |
| 未分類 | 17 | 本文から領域を付けられる10索引と7課題を下表へ |

| # | 課題 | 項目 | 今 → 提案 | 理由 |
| --- | --- | --- | --- | --- |
| T1 | [PRD112の索引](../management/issues/issue-20261103-58696c.md)・[PRD114の索引](../management/issues/issue-20261103-b713c2.md) | kind | 成果物 → ハーネス | 主な題材がskill・プロセス・品質観点である |
| T2 | [PRD108の索引](../management/issues/issue-20261103-0ef7b8.md) | areaの先頭 | 311_dev_tools → 317_metrics（既存の他areaは維持） | 変更の中心はメトリクスのtag契約である |
| T3 | [PRD949の索引](../management/issues/issue-20261103-b13ed2.md) | areaの先頭 | 120_metrics_viewer → 311_dev_tools（既存の他areaは維持） | Viewerの機能ではなくCMakeとリリース組立が中心である |
| T4 | [PRD952の索引](../management/issues/issue-20261103-66f453.md) | areaの先頭 | 316_config → 110_runner_app（316_configは維持） | キューの計画・順序・実行をrunnerが所有する |
| T5 | [PRD088の索引](../management/issues/issue-20261103-7f1789.md) | areaの先頭 | 322_env_individual → 315_nn（322は維持） | Envで使うが、追加する契約は汎用NNブロックである |
| T6 | [PRD934の索引](../management/issues/issue-20261103-55b5f1.md) | priorityとWSJF | 中、5/5/3/5 → 低、2/2/1/1、wsjf=5.0 | 本文が示す直しは設計160のLOD説明の整合である |
| T7 | [PRD906の索引](../management/issues/issue-20261103-61d771.md) | size・wsjf | 5・2.6 → 8・1.6 | 複数workflow・ビルド・定点ジョブの設計を含む |
| T8 | [旧キー専用WARN](../management/issues/issue-20261103-2c9a07.md) | size・wsjf | 5・2.2 → 3・3.7 | 本文の修正先は1関数とその呼び出しである |
| T9 | [GetStatsの注記](../management/issues/issue-20261103-5a5373.md) | size・wsjf | 3・1.7 → 1・5.0 | 現呼び出しは契約を守り、必要なのは数行の注記である |
| T10 | [インクルードガード](../management/issues/issue-20261103-03a1b6.md) | size・wsjf | 3・1.7 → 2・2.5 | ガードと先頭コメントの局所修正である |
| T11 | [wxImageの確保不一致](../management/issues/issue-20261103-dd8058.md) | priority | 中 → 高 | 本文は描画時の未定義動作と異常終了を示す |
| T12 | [stack_keysの無検証](../management/issues/issue-20261103-b754ff.md) | type・priority・WSJF | ずれ・低・2/2/1/3 → 不具合・高・8/5/8/3、wsjf=7.0 | 指定した観測stackが無効のまま学習し、誤った結果が残る |
| T13 | [MuZeroのテスト空白](../management/issues/issue-20261103-60f11c.md) | area | [318_rl_core] → [325_muzero,318_rl_core] | 本文の対象がMuZero固有の価値targetとmaskである |
| T14 | [PRD909の索引](../management/issues/issue-20261103-182ca1.md) | 分割 | 子なし → 5候補を判断別の子にする | 本文が1件ずつ直すか決めるとしており、独立した判断が5件ある。起票する内容は各候補を再確認してから選ぶ |
| T15 | [PRD098の索引](../management/issues/issue-20261103-2349d0.md)・[PRD107の索引](../management/issues/issue-20261103-f33167.md)・[PRD114の索引](../management/issues/issue-20261103-b713c2.md) | status | 未着手 → 解決にするか人が確認 | ai_judgeが実装済み。人の判断として自動反映しない |
| T16 | [PRD905の索引](../management/issues/issue-20261103-1edf84.md) | area | [] → [324_dqn,315_nn] | Qの複数HeadとBody共有が題材 |
| T17 | [PRD951の索引](../management/issues/issue-20261103-322a83.md) | area | [] → [315_nn] | residualとspectralの両立をNNが所有 |
| T18 | [PRD932の索引](../management/issues/issue-20261103-51220d.md) | area | [] → [317_metrics,318_rl_core] | episode捕捉と事後調査の記録が題材 |
| T19 | [PRD947の索引](../management/issues/issue-20261103-7227bf.md) | area | [] → [311_dev_tools,324_dqn] | 多環境でのスクリーニングの実行が題材 |
| T20 | [PRD941の索引](../management/issues/issue-20261103-a39330.md) | area | [] → [324_dqn,318_rl_core] | IQNとPERの初期優先度が題材 |
| T21 | [PRD930の索引](../management/issues/issue-20261103-aa8701.md) | area | [] → [318_rl_core,321_agent_core] | Run全体のSave/Loadが題材 |
| T22 | [PRD933の索引](../management/issues/issue-20261103-afe841.md) | area | [] → [324_dqn,318_rl_core] | DQNのbatchとreplay scheduleが題材 |
| T23 | [PRD946の索引](../management/issues/issue-20261103-c5c791.md) | area | [] → [313_profiling,318_rl_core,322_env_individual] | PRD097の性能再測定とDropMerge追加費用の調査 |
| T24 | [PRD915の索引](../management/issues/issue-20261103-dbe666.md) | area | [] → [322_env_individual,317_metrics] | DropMergeのNEETをtraceで診断する |
| T25 | [PRD936の索引](../management/issues/issue-20261103-f993bc.md) | area | [] → [315_nn,321_agent_core] | 全AgentのNN実行設定とAMPの契約 |
| T26 | [nn.txtのサンプル](../management/issues/issue-20261103-2751c8.md)・[nn.txtのコメント](../management/issues/issue-20261103-ef0c2b.md) | area | [] → [316_config,315_nn] | 設定サンプル・記法とNNの登録typeを照合する |
| T27 | [Doxygen zipの名前](../management/issues/issue-20261103-554849.md) | area・size・wsjf | []・3・1.7 → [311_dev_tools]・1・5.0 | リリース物の案内の数行を直す |
| T28 | [merginの綴り](../management/issues/issue-20261103-66100f.md) | area | [] → [110_runner_app,317_metrics] | GUIパネルと画像メトリクスの設定である |
| T29 | [Agentプロファイル配置](../management/issues/issue-20261103-7c9a70.md) | area | [] → [316_config,321_agent_core] | 設定の層と所有の問題である |
| T30 | [scalar記法のコメント](../management/issues/issue-20261103-879a70.md) | area | [] → [317_metrics,316_config] | metric tokenの説明と設定の整合である |
| T31 | [実効deviceの用語](../management/issues/issue-20261103-d4258c.md) | area | [] → [316_config,312_util] | 設定の解決結果と用語を合わせる |

統合は提案しない。InitRLの重複fatalと指定値欠落、HeatMapの確保不一致と描画の重複、Notifierの排他とscoped detach、DoStepのイベント不整合と共通化は、同じファイルに触れるが別々に判断して直せる。前提として必須だと本文から言えないものをdependsへ足す提案もしない。PRD909以外は現時点で具体的な分割案を出さない。

## 移行の対応表

F番号→issue、PRD→索引issue、旧名→新名の固定した対応は [migration.json](../../core/anet-core/testdata/prd114/migration.json) に保存する。使用済みの900・910・911・912・913・920・921・930・931・932・933・940・941・998・999を飛ばした。元のフォルダと歴史的な本文は保持し、過去資料では改名と退役資材のリンクだけを更新した。

## 工程末の状況

2026-11-04、承認済みの実装・移行・検証・レビューと指摘対応は済み、全レビューがOKとなった。公開CLI16件と既存wiki8件、Debugビルドが成功。移行44件・索引64件・改名47本・全単票111件の対応が一致し、新しいリンク切れは0件。元のGit index・HEAD、共有コア、開始時のユーザー変更を保持した。動線マップは公開後のファイルと配送・目視検証のhashが一致する。保存状態の最終確認は `.scratch/prd114/final-receipt.json`。

場面待ちのC・C2・Eと最初の5回は、上の理由付き未実施として残す。監理の棚卸し31提案は提案までの依頼であり、値へは反映していない。

## 完了報告

- 実施内容: PRD114と承認済み計画に沿う実装、移行、検証、レビューと指摘対応を完了した。
- 変更ファイル: 課題skillと両配置、issueと索引、PRD改名、wiki_supportとstats、設計・wiki・プロセス関連文書、検証資材、動線マップ。レビュー対象は320パス。固定対応と各検証結果は本記録の各節にある。
- レビュー結果: PRDは通常2往復・cc-review1往復、実装と計画は通常3往復・cc-review3往復で最終OK。Codex・ClaudeともOK。Claude側の検査コマンド実行不可は制約として区別し、Codex側の確認と実装側の検証結果を保持した。PRD通常初回のADR0049に関する指摘は、開始時本文との一致を根拠に取り下げとなった。
- 決まらなかったこと: 既存リンク切れの記録はissue-20261103-c1c608・issue-20261103-fb8a45・issue-20261103-ee32bd。その他の範囲外の未解決は「決まらなかったこと」の記載を保持する。
- 判断してほしいこと: 今回の工程を止める事項はなし。
- 次にできること: 次工程のコミットは人間が行う。
