# 開発プロセス・開発ツール・テスト基盤（311）

[anet-lab](../../../README.md) › [wiki の目録](../index.md) › [機能カテゴリ](../index.md#機能カテゴリ)

## 概要・責務

- CLIはGUIによる標準開発を補い、AIエージェントのビルド・検証とCI相当の再現、問題の診断に使う（出典: [開発環境構築ガイド](../../design/040_development_environment.jp.md)、§1.1・§2.4）。
- Optuna harnessの共通runtimeは探索、実行、集計、成果物を扱い、RunnerProcessManagerが子processの起動・timeout・中断・終了を管理する（出典: [アプリケーションとツール](../../design/160_applications_and_tools.jp.md)、§2.3・§3.3・§7.3）。
- wikiの取り込み・点検は正本のまとめと機械的な一覧を保ち、issueの道具はINDEXと地図の課題区画を更新する（出典: [アプリケーションとツール](../../design/160_applications_and_tools.jp.md)、§7.4）。
- reports/toolsの統計ツールと、各CI workflowの個別の概要・責務は、設計文書に記述なし。

## 成立の経緯

- 2026-08-10: Run 分類は物理 workspace フォルダで行い、workspace を自己完結の箱とする（出典: [ADR 0021](../../adr/0021-run-classification-by-workspace-folder.md)）。
- 2026-08-12: MLflow DB は選択 workspace の runs 配下に置く（出典: [ADR 0022](../../adr/0022-mlflow-database-lives-with-workspace-runs.md)）。
- 2026-08-12: Workspace 機構(Run 分類の物理フォルダ化と Runner/Viewer/optuna 統合)（出典: [PRD046](../../memo/done/046_workspace_10prd.md)、題・目的）。
- 2026-08-15: AI向けRun解析CLI `inspect_run.py`（出典: [PRD054](../../memo/done/054_inspect_run_10prd.md)、題・目的）。
- 2026-08-17: interval 発火の bucket-crossing 化 と perf メトリクスの時間重み EMA 化（出典: [PRD053](../../memo/done/053_interval_gate_perf_ema_10prd.md)、題・目的）。
- 2026-08-17: 解析用メタデータは Runner が出力し、解析側は設定から再導出しない（出典: [ADR 0029](../../adr/0029-analysis-metadata-emitted-by-runner.md)）。
- 2026-08-21: DropMerge探索を起動するOptuna harnessを追加した（出典: [PRD016](../../memo/done/016_dropmerge_optuna_10prd.md)、題・目的）。
- 2026-08-21: 043: GitHub Actions リリースパッケージング(実行用 bin パッケージ)（出典: [PRD043](../../memo/done/043_release_packaging_10prd.md)、題・目的）。
- 2026-09-05: メトリクス trace チャネル: 統計（scalar）とは別に「1 件 = 1 レコード」を識別子付きで出す（出典: [PRD069](../../memo/done/069_metrics_trace_channel_10prd.md)、題・目的）。
- 2026-09-05: 個体メトリクスは trace チャネルで出し、評価セッションは採用 episode ごとの EpisodeEndEvent と SessionEndEvent を分けて発火する（出典: [ADR 0037](../../adr/0037-metrics-trace-channel-and-session-end-event.md)）。
- 2026-09-19: Actor 設定カタログと eval スロットごとの方策 / network 指定（出典: [PRD061](../../memo/done/061_eval_slot_policy_override_10prd.md)、題・目的）。
- 2026-09-22: 終了時の評価セッション排水（出典: [PRD076](../../memo/done/076_eval_session_drain_on_exit_10prd.md)、題・目的）。
- 2026-10-10: 比較粒度と保存LODを改める変更に合わせ、inspect_runの現用キャッシュ判定をschema2へ追随させた（出典: [PRD089](../../memo/done/089_metricsviewer_comparison_resolution_10prd.md)、確定事項）。
- 2026-10-18: categories.jsonの対応を直し、apps/runner/srcのファイルを319 GUI共通基盤から110 RunnerAppへ分類するようにした（出典: [PRD095](../../memo/done/095_llm_wiki_10prd.md)、§3「実装を分ける語」）。
- 2026-10-20: 解析ツールが遷移結果のtarget env_stepを読むように移行した（出典: [PRD097](../../memo/done/097_env_auto_reset_after_notification_10prd.md)、§4.2の17・§9）。
- 2026-10-22: Evaluation Traceの読み口と表示の検証資材を整えた（出典: [PRD094](../../memo/done/094_evalpanel_step_history_10prd.md)、§8・§10）。
- 2026-10-28: TempoRLの等価性・性能比較用に、実行体と設定の着手前snapshotを使う検証資材を整えた（出典: [PRD103](../../memo/done/103_temporl_reference_alignment_10prd.md)、§4 D14・§7）。
- 2026-10-31: リリース物の組み立て（中身・検査・zip化）をrelease.ymlのPowerShellからCMakeのinstall・CPackとワークフローpresetへ移し、GitHubを使わずに手元でも同じコマンドで作れるようにする案を暫定で起票した（出典: [949_release_packaging_cmake](../../memo/949_release_packaging_cmake_10prd.md)、題・§3のゴール・§4 D1・D4）。
- 2026-11-02: Claude Codeのmemoryをgit・フォルダ・正本・規約と定期的に照合して直すskill（anet-tidy-memory）と、機械的な検査を受け持つreports/tools/memory_support.py、housekeepingの候補memory-tidyを足す案を暫定で起票した（出典: [PRD112](../../memo/112_claude_memory_tidy_10prd.md)、題・§1・§4 D9・D10）。
- 2026-11-02: 課題の置き場を台帳からissue（docs/management/issues/に1課題1ファイル）に替え、課題のskillをanetのskillとして持ち、すべてのPRDにissueを伴わせて作業の状態と順序をissueで持ち、新しいPRDはフォルダを移さず起票時に正式な番号を取り、999のPRDを使われたことの無い番号へ振り直す案を、再グリルで決めた（出典: [PRD114](../../memo/114_findings_intake_and_fix_prd_10prd.md)、題・§1・§4 D1・D2・D9・D13・D14）。
- 2026-11-02: 作業の状態と順序はissueが持ち、PRDは指示文書としてフォルダを動かさない（出典: [ADR 0068](../../adr/0068-work-state-and-order-live-in-issues-prd-stays-an-instruction-document.md)）。
- 2026-11-02: 共有コア（プロセス契約・レビューエージェントの定義）とハーネスに出てくるskill・エージェント・アダプタの名前はanet-で始まる名前とし、レビューエージェントは入口のskill anet-reviewと重ならないようanet-review-agentとする（出典: [PRD114](../../memo/114_findings_intake_and_fix_prd_10prd.md)、§4 D3）。
- 2026-11-02: 共有コアにanet-で始まるskill・エージェント・アダプタの名前を置けると補足し、名前以外はプロジェクトに依存しない文面に保つ判断を維持した（出典: [ADR 0049](../../adr/0049-process-contract-kept-project-neutral-anet-specifics-in-registry-and-viewpoints.md)、冒頭の2026-11-02注記）。
- 2026-11-03: issueの一覧の見せ方（依存のツリー、スプレッドシート、見やすい表）を、発番と同じ置き場の道具と一覧のskillの引数で作る案を暫定で起票した（出典: [925_issue_list_views](../../memo/925_issue_list_views_10prd.md)、題・§1・§4 D1〜D3）。
- 2026-11-10: 課題のtypeを仕事の種類（機能追加・機能改善・性能改善・リファクタリング・不具合・不整合・調査）にし、PRD専用の索引の課題をやめて各課題がprd欄でPRDを1本だけ指し（N:1）、PRDの実装済みを課題の判定と状態から導き、parent・depends・prdをObsidianのリンクで書き、domain・tags・resolutionを足す案を起票した（出典: [PRD120](../../memo/120_issue_work_type_and_prd_link_10prd.md)、§1・§4 D1〜D8）。

## 選ばなかった形

- inspect_runによるcacheの生成・更新・修復・migration・削除 — 不採用: 読み取り専用の抽出ツールから分ける（出典: [PRD054](../../memo/done/054_inspect_run_10prd.md)、§10）。
- inspect_runによるmetricの意味解釈・採否判断・Run ranking — 不採用: 分析の判断をツールへ統合しない（出典: [PRD054](../../memo/done/054_inspect_run_10prd.md)、§10・§11）。
- inspect_runのraw全点の無制限出力・plot画像生成・interactive viewer — 不採用: 分析用の材料抽出と分ける（出典: [PRD054](../../memo/done/054_inspect_run_10prd.md)、§10）。
- 監査の指摘の本文や対処方針を地図にも要約する — 不採用: 台帳の派生一覧と同じ正本の要約が2か所になる（出典: [PRD095](../../memo/done/095_llm_wiki_10prd.md)、§1の非ゴール・§6）。
- リリースのconfigureの引数だけをpresetへ寄せる案（packageとワークフローpresetは使わない） — 不採用: 手元でリリース物を作れない（出典: [949_release_packaging_cmake](../../memo/949_release_packaging_cmake_10prd.md)、§8）。
- apps/runner/bin/Releaseやフォルダを丸ごとリリースのzipに包む形 — 不採用: 手元の作業ツリーでは試行用のexe、管理外のDLL、Gitが無視するruns/が混ざる（出典: [949_release_packaging_cmake](../../memo/949_release_packaging_cmake_10prd.md)、§2の9・§8）。
- exeのimportの表からリリースに入れるDLLを自動でたどる形 — 不採用: 実行時に読み込むDLL（third_party/runtime_dllsのSDL2など）を拾えない（出典: [949_release_packaging_cmake](../../memo/949_release_packaging_cmake_10prd.md)、§2の6・§8）。
- リリースのインストーラ（NSIS・WiX・Inno Setup） — 先送り: 2GiBの上限は形式で変わらず、要求が無い。再開条件: 展開ではなくインストールで配る需要が出たとき（出典: [949_release_packaging_cmake](../../memo/949_release_packaging_cmake_10prd.md)、§8）。
- 7Zなど圧縮率の高い形式、LibTorchのDLLを別のzipに分ける形 — 先送り: 今のzipは上限の内。再開条件: 実行用zipが2,000MBの上限に近づいたとき（出典: [949_release_packaging_cmake](../../memo/949_release_packaging_cmake_10prd.md)、§8）。
- Linux・macOSのリリース物 — 範囲外: 先にビルドの移植が要る。再開条件: Linuxでビルドが通ったとき（出典: [949_release_packaging_cmake](../../memo/949_release_packaging_cmake_10prd.md)、§2の14・§8）。
- windows-ci.ymlのconfigureをリリースのpresetへ寄せる — 範囲外: CIのビルド確認の関心で、906 CI/CD改善のL1で扱う。再開条件: リリースのpresetが入ったあと（出典: [949_release_packaging_cmake](../../memo/949_release_packaging_cmake_10prd.md)、§8）。
- 依存の準備（LibTorchの取得など）をCMakeで行う — 範囲外: 環境整備は別の関心で、CUDA Toolkitの導入はCMakeで行えない。再開条件: 環境整備も1つのコマンドにしたくなったとき（出典: [949_release_packaging_cmake](../../memo/949_release_packaging_cmake_10prd.md)、§3の非ゴール・§8）。
- memoryの点検をanet-lint-wiki（wiki_support.py）に足す形 — 不採用: wikiの点検の材料はrepoの中の文書だけで、memoryはrepoの外にありClaude Codeからしか見えない（出典: [PRD112](../../memo/112_claude_memory_tidy_10prd.md)、§8）。
- 汎用のconsolidate-memoryをそのまま定期に回す形 — 不採用: git・フォルダ・正本・規約と照合しないので、古い状態の記述と知見の二重持ちが残る（出典: [PRD112](../../memo/112_claude_memory_tidy_10prd.md)、§8）。
- memoryの整備の1回分の記録をreports/housekeeping/に置く形 — 不採用: repoは公開しており、memoryには私的な事柄がある（出典: [PRD112](../../memo/112_claude_memory_tidy_10prd.md)、§3.2・§4 D6）。
- Codexのmemoryの整備 — 先送り: CodexのmemoryはCodexアプリが更新し、書き換えの可否を確かめていない。再開条件: 段階1を2回以上回し、Codexのmemoryの書き換えの可否を確かめたとき（出典: [PRD112](../../memo/112_claude_memory_tidy_10prd.md)、§8）。
- 工程末の起票の区分をプロセス契約（protocol.md）§7に書き足す — 不採用: 共有コアはプロジェクトに依存しない文面に保ち、anet固有の決まりはレジストリの読み替えに置く（出典: [PRD114](../../memo/114_findings_intake_and_fix_prd_10prd.md)、§3.2・§8）。
- 課題の仕組みのファイルを置いた版のまま持ち、プロセス契約の節とanetの値をレジストリの読み替えで当てる — 不採用: skillが合わない節を指したまま残り、skillと読み替え表を行き来して読むことになる。足す欄も中立の語に縛られる（出典: [PRD114](../../memo/114_findings_intake_and_fix_prd_10prd.md)、§4 D2・§8）。
- 台帳を監査の出力用に残し、選んだものだけissueへ上げる — 不採用: 置き場が2つに分かれる（出典: [PRD114](../../memo/114_findings_intake_and_fix_prd_10prd.md)、§8）。
- 監査は候補を出すだけにし、開発者が選んで起票する — 不採用: housekeepingの中で選ぶ問いで止まり、定期の監査が回らない（出典: [PRD114](../../memo/114_findings_intake_and_fix_prd_10prd.md)、§4 D17・§8）。
- 工程の最後に「暫定PRDを新規」の区分を残す — 不採用: 工程末に残った件がissueと999に分かれ、入口でPRD案件かの見立てが要る（出典: [PRD114](../../memo/114_findings_intake_and_fix_prd_10prd.md)、§4 D6・§8）。
- 同じ話題のissueとPRDの置き場を1つにし、PRDを書いたらissueを「移管」で閉じる — 不採用: PRDの優先度と作業順を番号でしか持てない（出典: [PRD114](../../memo/114_findings_intake_and_fix_prd_10prd.md)、§4 D9・§8）。
- issueから起こしたPRDだけissueを残して状態を揃える — 不採用: ユーザーが起こしたPRDが仕事の一覧から欠ける（出典: [PRD114](../../memo/114_findings_intake_and_fix_prd_10prd.md)、§4 D9・§8）。
- 分類をscopeの先頭に機能カテゴリのkeyを書く決まりで持つ — 不採用: 自由記述の欄に機械が読む決まりを重ねるので壊れやすく、性質で絞れない（出典: [PRD114](../../memo/114_findings_intake_and_fix_prd_10prd.md)、§4 D5・§8）。
- issueをPRDにする出口の別skillを作る、束ねる提案を持たない — 不採用: 別skillは一覧の並べ方と束ね方の写しになり、提案を持たないと既定5件の提案が無くなる（出典: [PRD114](../../memo/114_findings_intake_and_fix_prd_10prd.md)、§4 D11・D12・§8）。
- issueのareaを1つの値にする — 不採用: 2つの機能カテゴリにまたがる課題が、主でない側の地図と絞り込みから漏れる（出典: [PRD114](../../memo/114_findings_intake_and_fix_prd_10prd.md)、§6の3・§8）。
- 台帳へ積む入口専用のskill（anet-record-finding） — 不採用: anet-create-issueが持つ（出典: [PRD114](../../memo/114_findings_intake_and_fix_prd_10prd.md)、§4 D2・§8）。
- registry.py validateのF番号の重複検査 — 不採用: issueの番号は日付と乱数で、ブランチをまたいでも重ならない（出典: [PRD114](../../memo/114_findings_intake_and_fix_prd_10prd.md)、§8）。
- 9xxのPRDをすべて900からの連番へ振り直す — 不採用: 使われた番号を再び使い、過去の記録の番号が別のPRDを指す（出典: [PRD114](../../memo/114_findings_intake_and_fix_prd_10prd.md)、§4 D13・§8）。
- 今あるdone/・frozen/・dropped/のPRDを直下へ戻す — 不採用: 123ファイル・1,049個のリンクを直すことになる（出典: [PRD114](../../memo/114_findings_intake_and_fix_prd_10prd.md)、§4 D14・§8）。
- PRDの実装済みを、番号をスコープに持つコミットで拾う — 不採用: PRDの工程でこの場で直したコミットや起草のコミットも拾い、誤る。PRDのissueのAIの判定（実装済み）で拾う（出典: [PRD114](../../memo/114_findings_intake_and_fix_prd_10prd.md)、§4 D22・§8）。
- issueの一覧の作り込み（依存のツリー、スプレッドシート、見やすいHTML、anet-list-issuesへの機能の足し方） — 先送り: 見せ方は別の関心で、試作で見比べないと決めにくい。再開条件: 最小の一覧が動いたとき（出典: [PRD114](../../memo/114_findings_intake_and_fix_prd_10prd.md)、§3.2・§8）。
- PRDの起票をanet-housekeepingの候補にする — 不採用: 起票にはグリルの質疑が要り、了承後は質問しない約束と合わない（出典: [PRD114](../../memo/114_findings_intake_and_fix_prd_10prd.md)、§3.2・§8）。
- GitHub Issueを課題の置き場にする — 不採用: エージェントがrepoの中を検索して読む流れの外にあり、投稿のたびに外部への書き込みになる（出典: [PRD114](../../memo/114_findings_intake_and_fix_prd_10prd.md)、§3.2・§8）。
- 索引の課題を残し、typeだけ仕事の種類にする — 不採用: 索引の課題69件のうち61件はそのPRDのただ1件の仕事の課題で、窓口を分けると同じ仕事が2行になる（出典: [PRD120](../../memo/120_issue_work_type_and_prd_link_10prd.md)、§2.3・§7）。
- 課題に複数のPRDを書ける（N:M） — 不採用: 課題から導くPRDの実装済みが、ほかのPRDの進み具合を待つ（出典: [PRD120](../../memo/120_issue_work_type_and_prd_link_10prd.md)、§4 D2・§7）。
- PRDのファイルに状態を書く — 不採用: 状態の置き場が2つになり食い違う（出典: [PRD120](../../memo/120_issue_work_type_and_prd_link_10prd.md)、§7）。
- 課題の大まかな分類をareaから導く、またはtagsに人が書く — 不採用: 導く形はObsidianで式を書かないと絞れず課題ごとに直せない。tagsは書き忘れと表記のぶれが起き、areaと同じ軸を2か所で持つ（出典: [PRD120](../../memo/120_issue_work_type_and_prd_link_10prd.md)、§4 D6・§7）。

## 変えるときの条件

一般の条件は[リファクタリング観点](../../quality/refactor.md)の「ガードレール」、テストの充足は[定点統計](../../../reports/stats/trend.md)の「テスト充足」を見る。

- workspaceの箱を移動・削除・退避しても、Run・cache・Optuna履歴・基底configが一体で移動する自己完結性を保つ（出典: [PRD046](../../memo/done/046_workspace_10prd.md)、§1の不変条件）。
- inspect_runのstreaming・1 pass・128点上限をfocused testで確かめる（出典: [PRD054](../../memo/done/054_inspect_run_10prd.md)、§9・§10）。
- inspect_runで128点以下は全点を保持し、間引く場合も先頭・末尾と決定性を保つ（出典: [PRD054](../../memo/done/054_inspect_run_10prd.md)、§9）。
- issueの分類をstatsと共有する変更では、statsのカテゴリ集計が変わらないことを確かめる（出典: [PRD095](../../memo/done/095_llm_wiki_10prd.md)、§4.7の台帳のツール・受入15）。
- リリースの組み立てを移すときは、実行用zipの中身（ファイルのentryのパスの一覧）を変えない。移行の直前のコミットで旧workflowのドライランを回したartifactと、新しい入口で作ったzipの一覧を比べる（出典: [949_release_packaging_cmake](../../memo/949_release_packaging_cmake_10prd.md)、§3のG2・§7の受入3）。
- リリースの構成を足しても、日常のDebug・Release構成の出力先と出力物を変えない。前後でexe・DLL・静的ライブラリのファイル名の一覧を比べる（出典: [949_release_packaging_cmake](../../memo/949_release_packaging_cmake_10prd.md)、§4 D2・§7の受入1）。
- housekeepingに候補を足しても、reports/housekeeping/runs.jsonlの既存の行は書き換えない（出典: [PRD112](../../memo/112_claude_memory_tidy_10prd.md)、§5.5）。
- 過去のPRD（done/・frozen/・dropped/）・実装記録・ADRの本文は書き換えない。999の振り直しでは、ファイル名を指すパスの表記だけを直す（出典: [PRD114](../../memo/114_findings_intake_and_fix_prd_10prd.md)、§5.9・§5.12）。
- PRDの番号は、一度使ったものを二度と使わない（コミットの件名・CHANGELOG・ADR・実験記録が番号でPRDを指すため）（出典: [PRD114](../../memo/114_findings_intake_and_fix_prd_10prd.md)、§4 D13）。
- 課題の仕組みのファイルは共有コアに含めず、anetのskillと文書として持つ。プロセス契約の節とanetの値はskillに直接書き、レジストリの読み替えで当てない（出典: [PRD114](../../memo/114_findings_intake_and_fix_prd_10prd.md)、§4 D2・D4・D5）。
- 共有コアの文面はskill・エージェント・アダプタの名前以外をプロジェクトに依存させず、anet固有のことはレジストリと観点に置く（出典: [ADR 0049](../../adr/0049-process-contract-kept-project-neutral-anet-specifics-in-registry-and-viewpoints.md)、冒頭の2026-11-02注記）。
- 新しい設定が既定のときは、着手前に保存した実行体・設定を使い、スキップ無効と既存の有効構成のmetrics checksumを一致させる。古い設定と新しい実行体を組み合わせる比較を保つ（出典: [PRD103](../../memo/done/103_temporl_reference_alignment_10prd.md)、§4 D14・§5.7・§7の受入1）。

## 関係

- [ANET フレームワーク全体概要](../../design/010_framework_overview.jp.md)。
- [ANET ドキュメント](../../design/README.jp.md)。
- 設計: [アプリケーションとツール](../../design/160_applications_and_tools.jp.md) §2.3・§3.3・§7.3・§8。
- ユーザーガイド: [開発環境構築ガイド](../../design/040_development_environment.jp.md) §6・§7、[Run分析ガイド](../../design/030_user_guide_analysis.jp.md) §1。
- 含まれるアプリケーション: [RunnerApp](../apps/110_runner_app.md)。

## ファイル

点検が作り直す。手で直さない。

- [push_review.py](../../../.github/scripts/push_review.py)
- [gemini_ai_review.yml](../../../.github/workflows/gemini_ai_review.yml)
- [pages.yml](../../../.github/workflows/pages.yml)
- [release.yml](../../../.github/workflows/release.yml)
- [windows-ci.yml](../../../.github/workflows/windows-ci.yml)
- [batchrun_fatal_error_handling_test.ps1](../../../apps/runner/tools/batchrun_fatal_error_handling_test.ps1)
- [compress_workspace_metrics.py](../../../apps/runner/tools/compress_workspace_metrics.py) / [compress_workspace_metrics_test.py](../../../apps/runner/tools/compress_workspace_metrics_test.py)
- [dropmerge_optuna.py](../../../apps/runner/tools/dropmerge_optuna.py)
- [optuna_common.py](../../../apps/runner/tools/optuna_common.py)
- [optuna_metrics_gzip_test.py](../../../apps/runner/tools/optuna_metrics_gzip_test.py)
- [optuna_workspace_test.py](../../../apps/runner/tools/optuna_workspace_test.py)
- [resolve_workspace.bat](../../../apps/runner/tools/resolve_workspace.bat) / [resolve_workspace_test.ps1](../../../apps/runner/tools/resolve_workspace_test.ps1)
- [symbolize.py](../../../apps/runner/tools/symbolize.py)
- [watch_log.ps1](../../../apps/runner/tools/watch_log.ps1)
- [categories.json](../../../reports/tools/categories.json)
- [requirements.txt](../../../reports/tools/requirements.txt)
- [stats.py](../../../reports/tools/stats.py)
- [wiki_support.py](../../../reports/tools/wiki_support.py)
- [atari5.py](../../../viewers/metrics-tools/atari5.py) / [atari5_test.py](../../../viewers/metrics-tools/atari5_test.py)
- [btr_to_metrics.py](../../../viewers/metrics-tools/btr_to_metrics.py)
- [inspect_run.py](../../../viewers/metrics-tools/inspect_run.py) / [inspect_run_test.py](../../../viewers/metrics-tools/inspect_run_test.py)
- [metrics_source.py](../../../viewers/metrics-tools/metrics_source.py) / [metrics_source_test.py](../../../viewers/metrics-tools/metrics_source_test.py)
- [metrics_viewer.py](../../../viewers/metrics-tools/metrics_viewer.py)
- [mlflow_bridge.py](../../../viewers/metrics-tools/mlflow_bridge.py) / [mlflow_bridge_test.py](../../../viewers/metrics-tools/mlflow_bridge_test.py)
- [requirements.txt](../../../viewers/metrics-tools/requirements.txt)
- [start_mlflow.bat](../../../viewers/metrics-tools/start_mlflow.bat)
- [start_tb.bat](../../../viewers/metrics-tools/start_tb.bat)
- [tb_bridge.py](../../../viewers/metrics-tools/tb_bridge.py) / [tb_bridge_test.py](../../../viewers/metrics-tools/tb_bridge_test.py)
- [reload.lua](../../../viewers/metrics-tools/vlc/lua/extensions/reload.lua)

## PRD の一覧

点検が作り直す。手で直さない。

- [PRD046](../../memo/done/046_workspace_10prd.md) — Workspace 機構(Run 分類の物理フォルダ化と Runner/Viewer/optuna 統合)
- [PRD053](../../memo/done/053_interval_gate_perf_ema_10prd.md) — PRD 053: interval 発火の bucket-crossing 化 と perf メトリクスの時間重み EMA 化
- [PRD054](../../memo/done/054_inspect_run_10prd.md) — PRD 054: AI向けRun解析CLI `inspect_run.py`
- [PRD061](../../memo/done/061_eval_slot_policy_override_10prd.md) — PRD 061: Actor 設定カタログと eval スロットごとの方策 / network 指定
- [PRD069](../../memo/done/069_metrics_trace_channel_10prd.md) — メトリクス trace チャネル: 統計（scalar）とは別に「1 件 = 1 レコード」を識別子付きで出す
- [PRD076](../../memo/done/076_eval_session_drain_on_exit_10prd.md) — PRD 076: 終了時の評価セッション排水
- [PRD094](../../memo/done/094_evalpanel_step_history_10prd.md) — Evaluation Trace — EvalPanel の遷移を step trace として表で見て、その時点を再生する PRD
- [PRD095](../../memo/done/095_llm_wiki_10prd.md) — LLM-Wiki 型のまとめの層（PRD の上流で、知見を題材と機構ごとに、実装の地図をアプリケーションと機能カテゴリごとにまとめる）PRD
- [PRD097](../../memo/done/097_env_auto_reset_after_notification_10prd.md) — 遷移の結果を Step の結果で運ぶ — 通知の中で読む遷移の結果を、終端の lane でもリセット前の値にする PRD
- [PRD103](../../memo/done/103_temporl_reference_alignment_10prd.md) — TempoRL 第二弾（著者の実装との差の洗い出しと是正）PRD
- [PRD114](../../memo/114_findings_intake_and_fix_prd_10prd.md) — 課題の記録を issue（1 課題 1 ファイル）に替え、仕事は issue・指示は PRD で持つ（台帳の退役、課題の仕組み、PRD の採番とフォルダの改め）PRD
- [PRD116](../../memo/116_rnd_lifelong_novelty_10prd.md) — RND による一生の新しさだけの腕（内的報酬の第一の腕）暫定 PRD
- [PRD120](../../memo/120_issue_work_type_and_prd_link_10prd.md) — PRD120: 課題の種類を仕事の種類にし、PRD との結び付けを課題の prd 欄（Obsidian のリンク）に替える

## 課題

issueの道具（`wiki_support.py maps`）が作り直す。手で直さない。

<!-- issues:begin -->
| # | ID | 状態 | WSJF | 待ち | 対象 | 工程 | 領域 | 分類 | 種類 | PRD | AIの判定 | 課題 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | [issue-20261104-52ffba](../../management/issues/issue-20261104-52ffba.md) | 対応中 | 1.4 | — | ハーネス | ハーネス保守 | 311_dev_tools | 開発 | 機能改善 | — | 残件有り | 課題の仕組みに残る改善点をまとめて管理する（段階に分けた PRD の「実装済み」の時機ほか） |
| 2 | [issue-20261103-955e50](../../management/issues/issue-20261103-955e50.md) | 調査中 | 1.2 | — | 成果物 | PRD | 318_rl_core, 317_metrics, 110_runner_app, 311_dev_tools, 120_metrics_viewer | UI | 機能追加 | [PRD081](../../memo/081_evalpanel_episode_trace_10prd.md) | 残件有り | EvalPanel のゲームを trace へ出す — dormant タグの trace を鏡写しインスタンスへ結び付け、trace 行の時刻で振り返る |
| 3 | [issue-20261103-23d185](../../management/issues/issue-20261103-23d185.md) | 対応中 | 0.9 | — | ハーネス | PRD | 311_dev_tools | 開発 | 機能追加 | [PRD925](../../memo/925_issue_list_views_10prd.md) | 残件有り | issue の一覧の見せ方（依存のツリー、スプレッドシート、見やすい表） |
| 4 | [issue-20261103-5addea](../../management/issues/issue-20261103-5addea.md) | 未着手 | 2.2 | — | 成果物 | PRD | 311_dev_tools, 318_rl_core, 315_nn, 312_util, 317_metrics, 321_agent_core, 319_gui, 316_config, 322_env_individual, 324_dqn, 110_runner_app, 325_muzero, 313_profiling | 開発 | リファクタリング | [PRD079](../../memo/079_cpp_internal_linkage_convention_10prd.md) | 残件有り | PRD 079: .cpp の内部リンケージと namespace インデント規約 |
| 5 | [issue-20261103-58696c](../../management/issues/issue-20261103-58696c.md) | 未着手 | 1.9 | — | ハーネス | PRD | 311_dev_tools | 開発 | 機能追加 | [PRD112](../../memo/112_claude_memory_tidy_10prd.md) | 残件有り | Claude Code の memory の定期整備（`anet-tidy-memory` と housekeeping の候補 `memory-tidy`） |
| 6 | [issue-20261103-0ef7b8](../../management/issues/issue-20261103-0ef7b8.md) | 未着手 | 1.6 | — | 成果物 | PRD | 317_metrics, 311_dev_tools, 316_config, 318_rl_core, 321_agent_core, 120_metrics_viewer, 110_runner_app | UI | 機能改善 | [PRD108](../../memo/108_metrics_tag_numbering_10prd.md) | 残件有り | メトリクスの tag の番号を全体で振り直す（群の番号と群の中の番号） |
| 7 | [issue-20261103-61d771](../../management/issues/issue-20261103-61d771.md) | 未着手 | 1.6 | — | 成果物 | PRD | 311_dev_tools | 開発 | 機能追加 | [PRD906](../../memo/906_cicd_improvement_10prd.md) | 残件有り | CI/CD 改善（ビルド・テスト自動化、定点ジョブ、LLM 補助） |
| 8 | [issue-20261103-52bda0](../../management/issues/issue-20261103-52bda0.md) | 未着手 | 1.4 | — | 成果物 | PRD | 318_rl_core, 321_agent_core, 120_metrics_viewer, 311_dev_tools | UI | 機能改善 | [PRD096](../../memo/096_metrics_step_axis_interval_10prd.md) | 残件有り | メトリクスの横軸と間隔を数える単位を分ける |
| 9 | [issue-20261110-9d4e5a](../../management/issues/issue-20261110-9d4e5a.md) | 未着手 | 1.4 | — | 成果物 | その他 | 311_dev_tools | 開発 | 機能追加 | — | 未確認 | C++ の単純な書き方の問題を機械で検出する仕組み（lint・静的解析）が無い |
| 10 | [issue-20261103-71bb9a](../../management/issues/issue-20261103-71bb9a.md) | 未着手 | 1.2 | — | 成果物 | PRD | 311_dev_tools | 開発 | 機能追加 | [PRD959](../../memo/959_test_line_coverage_10prd.md) | 残件有り | 行カバレッジ計測（OpenCppCoverage） |
| 11 | [issue-20261103-7227bf](../../management/issues/issue-20261103-7227bf.md) | 未着手 | 1.2 | — | 成果物 | PRD | 311_dev_tools, 317_metrics | 開発 | 機能追加 | [PRD947](../../memo/947_q_statistics_config_screening_10prd.md) | 残件有り | Q 値統計による多環境ハイパラ健全性スクリーニング |
| 12 | [issue-20261103-dbe666](../../management/issues/issue-20261103-dbe666.md) | 未着手 | 1.2 | — | 成果物 | PRD | 322_env_individual, 318_rl_core, 311_dev_tools | Env | 調査 | [PRD915](../../memo/915_dropmerge_neet_diagnostics_10prd.md) | 残件有り | DropMerge の NEET 診断（trace で足りるか確かめてから指標化） |
| 13 | [issue-20261103-c16aab](../../management/issues/issue-20261103-c16aab.md) | 未着手 | 1.0 | — | 成果物 | PRD | 320_env_core, 322_env_individual, 312_util, 120_metrics_viewer, 311_dev_tools | UI | 機能追加 | [PRD954](../../memo/954_scalar_threshold_rate_10prd.md) | 残件有り | スカラー閾値越え率の汎用化（Observation 層 / Viewer 層） |
| 14 | [issue-20261105-f4b870](../../management/issues/issue-20261105-f4b870.md) | 未着手 | 1.0 | — | 成果物 | その他 | 311_dev_tools, 315_nn | アルゴリズム | 調査 | — | 未確認 | LLM が NN の重みを直接読み書きする試み（Weight-Space Programming）を anet で試す |
| 15 | [issue-20261103-b13ed2](../../management/issues/issue-20261103-b13ed2.md) | 未着手 | 0.8 | — | 成果物 | PRD | 311_dev_tools | 開発 | 機能改善 | [PRD949](../../memo/949_release_packaging_cmake_10prd.md) | 残件有り | リリース物の組み立てを CMake（install・CPack・ワークフロー preset）へ移し、手元でも同じコマンドで作れるようにする |
| 16 | [issue-20261103-2dd820](../../management/issues/issue-20261103-2dd820.md) | 未着手 | 1.0 | [issue-20261103-785e68](../../management/issues/issue-20261103-785e68.md) | 成果物 | PRD | 321_agent_core, 315_nn, 324_dqn, 312_util, 318_rl_core, 325_muzero, 110_runner_app, 311_dev_tools, 322_env_individual | 開発 | リファクタリング | [PRD957](../../memo/957_source_layout_10prd.md) | 残件有り | ソース配置の見直し（トップ階層を `src/` に、Env と同じ具象別のライブラリへ Agent を分ける） |

着手できる 15 / 待ち 1 / 保留 0 / 次版 0 / 解決 0
- 関連する課題: [issue-20261103-785e68](../../management/issues/issue-20261103-785e68.md) — anet-core の外で実装する拡張点とコード配置の規約
<!-- issues:end -->
