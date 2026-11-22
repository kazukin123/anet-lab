# 開発プロセス・開発ツール・テスト基盤（311）

[anet-lab](../../../README.md) › [wiki の目録](../index.md) › [機能カテゴリ](../index.md#機能カテゴリ)

## 概要・責務

- CLIはGUIによる標準開発を補い、AIエージェントのビルド・検証とCI相当の再現、問題の診断に使う（出典: [開発環境構築ガイド](../../design/040_development_environment.jp.md)、§1.1・§2.4）。
- Optuna harnessの共通runtimeは探索、実行、集計、成果物を扱い、RunnerProcessManagerが子processの起動・timeout・中断・終了を管理する（出典: [アプリケーションとツール](../../design/160_applications_and_tools.jp.md)、§2.3・§3.3・§7.3）。
- wikiの取り込み・点検は正本のまとめと機械的な一覧を保ち、issueの道具はINDEXとトピック別まとめの課題区画を更新する（出典: [アプリケーションとツール](../../design/160_applications_and_tools.jp.md)、§7.4）。
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
- 2026-10-18: categories.jsonの対応を直し、src/apps/runner/srcのファイルを319 GUI共通基盤から110 RunnerAppへ分類するようにした（出典: [PRD095](../../memo/done/095_llm_wiki_10prd.md)、§3「実装を分ける語」）。
- 2026-10-20: 解析ツールが遷移結果のtarget env_stepを読むように移行した（出典: [PRD097](../../memo/done/097_env_auto_reset_after_notification_10prd.md)、§4.2の17・§9）。
- 2026-10-22: Evaluation Traceの読み口と表示の検証資材を整えた（出典: [PRD094](../../memo/done/094_evalpanel_step_history_10prd.md)、§8・§10）。
- 2026-10-28: TempoRLの等価性・性能比較用に、実行体と設定の着手前snapshotを使う検証資材を整えた（出典: [PRD103](../../memo/done/103_temporl_reference_alignment_10prd.md)、§4 D14・§7）。
- 2026-10-31: リリース物の組み立て（中身・検査・zip化）をrelease.ymlのPowerShellからCMakeのinstall・CPackとワークフローpresetへ移し、GitHubを使わずに手元でも同じコマンドで作れるようにする案を暫定で起票した（出典: [185_release_packaging_cmake](../../memo/185_release_packaging_cmake_10prd.md)、題・§3のゴール・§4 D1・D4）。
- 2026-11-02: Claude Codeのmemoryをgit・フォルダ・正本・規約と定期的に照合して直すskill（anet-tidy-memory）と、機械的な検査を受け持つreports/tools/memory_support.py、housekeepingの候補memory-tidyを足す案を暫定で起票した（出典: [PRD112](../../memo/112_claude_memory_tidy_10prd.md)、題・§1・§4 D9・D10）。
- 2026-11-02: 課題の置き場を台帳からissue（docs/management/issues/に1課題1ファイル）に替え、課題のskillをanetのskillとして持ち、すべてのPRDにissueを伴わせて作業の状態と順序をissueで持ち、新しいPRDはフォルダを移さず起票時に正式な番号を取り、999のPRDを使われたことの無い番号へ振り直す案を、再グリルで決めた（出典: [PRD114](../../memo/114_findings_intake_and_fix_prd_10prd.md)、題・§1・§4 D1・D2・D9・D13・D14）。
- 2026-11-02: 作業の状態と順序はissueが持ち、PRDは指示文書としてフォルダを動かさない（出典: [ADR 0068](../../adr/0068-work-state-and-order-live-in-issues-prd-stays-an-instruction-document.md)）。
- 2026-11-02: 共有コア（プロセス契約・レビューエージェントの定義）とハーネスに出てくるskill・エージェント・アダプタの名前はanet-で始まる名前とし、レビューエージェントは入口のskill anet-reviewと重ならないようanet-review-agentとする（出典: [PRD114](../../memo/114_findings_intake_and_fix_prd_10prd.md)、§4 D3）。
- 2026-11-02: 共有コアにanet-で始まるskill・エージェント・アダプタの名前を置けると補足し、名前以外はプロジェクトに依存しない文面に保つ判断を維持した（出典: [ADR 0049](../../adr/0049-process-contract-kept-project-neutral-anet-specifics-in-registry-and-viewpoints.md)、冒頭の2026-11-02注記）。
- 2026-11-03: issueの一覧の見せ方（依存のツリー、スプレッドシート、見やすい表）を、発番と同じ置き場の道具と一覧のskillの引数で作る案を暫定で起票した（出典: [187_issue_list_views](../../memo/187_issue_list_views_10prd.md)、題・§1・§4 D1〜D3）。
- 2026-11-10: 課題のtypeを仕事の種類（機能追加・機能改善・性能改善・リファクタリング・不具合・不整合・調査）にし、PRD専用の索引の課題をやめて各課題がprd欄でPRDを1本だけ指し（N:1）、PRDの実装済みを課題の判定と状態から導き、parent・depends・prdをObsidianのリンクで書き、domain・tags・resolutionを足す案を起票した（出典: [PRD120](../../memo/120_issue_work_type_and_prd_link_10prd.md)、§1・§4 D1〜D8）。
- 2026-11-11: Debug版libtorchのoneDNNのassertがテストの順序で出ないよう、基盤テストの実行体が開始時に1回MKLDNNを切り、テストの中の切り替えを消す案を起票した（出典: [PRD124](../../memo/124_test_process_mkldnn_off_10prd.md)、題・§3.1・§4 D1）。
- 2026-11-17: 9xxの番号のPRD 57本を、起票の日の順に132〜188へ振り直し、この振り直しを番号の不再利用の例外として9xxを再び使える番号に戻し、旧い番号を各PRDの題の直下の振り直しの行で引くようにした（wikiのPRDの一覧の道具もその行を読む）（出典: [PRD131](../../memo/131_prd_9xx_renumber_by_creation_date_10prd.md)、題・§1.2・§4 D1〜D3・D7・D8）。
- 2026-11-16: PRDの9xxの番号を、起票の日の順に900未満の連番へ振り直し、9xxを再び使える番号に戻す（出典: [ADR 0077](../../adr/0077-prd-9xx-renumbered-by-creation-date-and-9xx-reusable.md)）。
- 2026-11-18: 置き場の規則（ビルドの入力はsrc、実行時の配置はapps、実行時にできるものはdata、配らないビルドの出力はout）、テストの参照範囲、基盤のテスト用の口、テスト実行体の出力先の1か所化、道具のapps/toolsへの集約を、グリルで決めた（出典: [PRD182](../../memo/182_source_layout_10prd.md)、§4.1 D2・D12・D13・D19・D20）。
- 2026-11-18: トップの階層は、ビルドの入力・実行時の配置・実行時にできるもので分け、ビルドはappsへ直接出す（出典: [ADR 0078](../../adr/0078-top-level-layout-separates-build-inputs-runtime-layout-and-runtime-data.md)、本文）。
- 2026-11-18: テストは自分のライブラリの内部まで読み、ほかのライブラリは公開ヘッダとテスト用の口だけを通して使う（出典: [ADR 0080](../../adr/0080-tests-read-own-library-internals-and-reach-other-libraries-through-public-headers-and-test-support.md)、本文）。
- 2026-11-19: PRDのグリルで、goal anchorの応答に共通の前提を並べ、ゴールの合意と同じ返事で説明が要る前提の番号を受けて理解を確かめ、欠けた前提は先にそろえるかをその都度選び、PRDに前提の表を置き、PRD品質確認観点に項番43を足す案を、グリルで決めた（出典: [PRD189](../../memo/189_grill_premise_understanding_10prd.md)、§1.2・§6.1 D2〜D8・§6.2 D13）。
- 2026-11-21: 文書の呼び名「PRD」が製品の要求書と中身で食い違うので、中身に合う名前（暫定の既定はSPEC）へ改め、ファイル名・skill名・課題の欄・コミット件名・テストのタグ・道具など今使っている場所をすべて移し、過去の記録は書き換えずに道具が旧い件名の番号も読む案を暫定で起票した（出典: [PRD195](../../memo/195_rename_prd_document_term_10prd.md)、題・§1・§6）。
- 2026-11-22: 会話の枠ごとに 1 枚のまとめページ（冒頭・いまの位置・記号の一覧に、場面ごとの節を足したもの）を、場面に入ったときか前の応答の記号を引き始めたときに作って、中身が変わった応答で更新し、Claude Code は非公開ページ、Codex は ChatGPT Sites で同じ URL のままスマホから見返せるようにする案を、グリルで決めた。手順は skill、始める条件は AGENTS.md、場面ごとの指針とひな形は docs/process/summary-page/ に置く（出典: [PRD197](../../memo/197_conversation_options_summary_page_10prd.md)、§1.2・§6 D1・D2・D5〜D9・D11・D25・D26）。

## 選ばなかった形

- inspect_runによるcacheの生成・更新・修復・migration・削除 — 不採用: 読み取り専用の抽出ツールから分ける（出典: [PRD054](../../memo/done/054_inspect_run_10prd.md)、§10）。
- inspect_runによるmetricの意味解釈・採否判断・Run ranking — 不採用: 分析の判断をツールへ統合しない（出典: [PRD054](../../memo/done/054_inspect_run_10prd.md)、§10・§11）。
- inspect_runのraw全点の無制限出力・plot画像生成・interactive viewer — 不採用: 分析用の材料抽出と分ける（出典: [PRD054](../../memo/done/054_inspect_run_10prd.md)、§10）。
- 監査の指摘の本文や対処方針をトピック別まとめにも要約する — 不採用: 台帳の派生一覧と同じ正本の要約が2か所になる（出典: [PRD095](../../memo/done/095_llm_wiki_10prd.md)、§1の非ゴール・§6）。
- リリースのconfigureの引数だけをpresetへ寄せる案（packageとワークフローpresetは使わない） — 不採用: 手元でリリース物を作れない（出典: [185_release_packaging_cmake](../../memo/185_release_packaging_cmake_10prd.md)、§8）。
- apps/runner/bin/Releaseやフォルダを丸ごとリリースのzipに包む形 — 不採用: 手元の作業ツリーでは試行用のexe、管理外のDLL、Gitが無視するruns/が混ざる（出典: [185_release_packaging_cmake](../../memo/185_release_packaging_cmake_10prd.md)、§2の9・§8）。
- exeのimportの表からリリースに入れるDLLを自動でたどる形 — 不採用: 実行時に読み込むDLL（third_party/runtime_dllsのSDL2など）を拾えない（出典: [185_release_packaging_cmake](../../memo/185_release_packaging_cmake_10prd.md)、§2の6・§8）。
- リリースのインストーラ（NSIS・WiX・Inno Setup） — 先送り: 2GiBの上限は形式で変わらず、要求が無い。再開条件: 展開ではなくインストールで配る需要が出たとき（出典: [185_release_packaging_cmake](../../memo/185_release_packaging_cmake_10prd.md)、§8）。
- 7Zなど圧縮率の高い形式、LibTorchのDLLを別のzipに分ける形 — 先送り: 今のzipは上限の内。再開条件: 実行用zipが2,000MBの上限に近づいたとき（出典: [185_release_packaging_cmake](../../memo/185_release_packaging_cmake_10prd.md)、§8）。
- Linux・macOSのリリース物 — 範囲外: 先にビルドの移植が要る。再開条件: Linuxでビルドが通ったとき（出典: [185_release_packaging_cmake](../../memo/185_release_packaging_cmake_10prd.md)、§2の14・§8）。
- windows-ci.ymlのconfigureをリリースのpresetへ寄せる — 範囲外: CIのビルド確認の関心で、156 CI/CD改善のL1で扱う。再開条件: リリースのpresetが入ったあと（出典: [185_release_packaging_cmake](../../memo/185_release_packaging_cmake_10prd.md)、§8）。
- 依存の準備（LibTorchの取得など）をCMakeで行う — 範囲外: 環境整備は別の関心で、CUDA Toolkitの導入はCMakeで行えない。再開条件: 環境整備も1つのコマンドにしたくなったとき（出典: [185_release_packaging_cmake](../../memo/185_release_packaging_cmake_10prd.md)、§3の非ゴール・§8）。
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
- issueのareaを1つの値にする — 不採用: 2つの機能カテゴリにまたがる課題が、主でない側のトピック別まとめと絞り込みから漏れる（出典: [PRD114](../../memo/114_findings_intake_and_fix_prd_10prd.md)、§6の3・§8）。
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
- コミットの準備で出荷設定の検査のテスト（`[shipped_config]`）を流す形 — 不採用: 使わない定義の誤りは、使う Run の構築時に必ず止まるので害の経路が無く、テストの実行体が古いときの扱いという別の不確かさを持ち込む（出典: [PRD117](../../memo/117_warn_policy_and_unread_config_10prd.md)、D13・§8）。
- 止まるテストの中だけでMKLDNNを切る形 — 不採用: 切り替えを通らずにCPUでConvを組むテストが、Catch2の既定のランダム順で止まる仕組みが残る（出典: [PRD124](../../memo/124_test_process_mkldnn_off_10prd.md)、§4 D1・§7）。
- 本番の初期化でDebugビルドのときだけMKLDNNを切る形 — 範囲外: 本番のRunnerに触り、InitNNを呼ぶ前にConvを組むテストは守られない。再開条件: Debug版のRunnerをCPUでConvのNNと回す必要が出たとき（出典: [PRD124](../../memo/124_test_process_mkldnn_off_10prd.md)、§3.2・§7）。
- テストの実行順を宣言順に固定する形 — 不採用: 単独やタグを指定した実行は直らない（出典: [PRD124](../../memo/124_test_process_mkldnn_off_10prd.md)、§3.2・§7）。
- テストの実行体でCPUのスレッド数を1にしてoneDNNの経路を残す形 — 不採用: 残す利点のoneDNNの経路のテストは非ゴールで、CPUのテストがすべて1スレッドになる（出典: [PRD124](../../memo/124_test_process_mkldnn_off_10prd.md)、§3.2・§7）。
- NNを組まないテストの実行体（Envの5本、RunnerのGUIのテスト）のmainでもMKLDNNを切る形 — 不採用: NNを組まない。再開条件: CPUでNNを組むテストをそれらの実行体に足すとき（出典: [PRD124](../../memo/124_test_process_mkldnn_off_10prd.md)、§4 D3・§7）。
- 9xxのPRDを振り直したあと、9xxを使われた番号として飛ばし続ける（退役） — 不採用: 番号の並びが900台に届いたときに予約の帯が残り、規約にも9xxの例外が要る（出典: [PRD131](../../memo/131_prd_9xx_renumber_by_creation_date_10prd.md)、§4 D3・§8）。
- 9xxのPRDを振り直したあとの900台の扱いを、番号が890に届いたときに4桁の形と一緒に決める（先送り） — 不採用: 規約が暗に9xxを予約し続ける。得るのは100個（今の速さで3〜4か月）だけという反対を聞いたうえで、予約の帯を残さない形を保った（出典: [PRD131](../../memo/131_prd_9xx_renumber_by_creation_date_10prd.md)、§4 D3・§8）。
- 9xxのPRDの振り直しでリンクとファイル名だけを直し、番号だけの表記を残す — 不採用: 今使っている文書に番号の食い違いが500か所を超えて残る（出典: [PRD131](../../memo/131_prd_9xx_renumber_by_creation_date_10prd.md)、§4 D4・§8）。
- 改名したPRDを指すADRのリンクを、先頭の注記だけで済ませるか切れたまま残す — 不採用: リンク切れが検査に毎回混ざり、次の改名・移動でも同じ食い違いに当たる（出典: [PRD131](../../memo/131_prd_9xx_renumber_by_creation_date_10prd.md)、§4 D6・§8）。
- PRDの旧い番号と新しい番号の対応表のファイルを置き、道具はそれを読む — 不採用: 置き場が増え、PRDを開いても旧い番号が分からない（出典: [PRD131](../../memo/131_prd_9xx_renumber_by_creation_date_10prd.md)、§4 D7・§8）。
- 番号が1000に届いたときの4桁の形 — 先送り: 別の関心。再開条件: 番号が990に届いたとき（出典: [PRD131](../../memo/131_prd_9xx_renumber_by_creation_date_10prd.md)、§4 D15・§8）。
- .libをsrcの下のlibフォルダやout/libに出す形 — 不採用: 前者はsrcにビルドの出力が入り、どちらも.libのパスを人が使う場面が無い（出典: [PRD182](../../memo/182_source_layout_10prd.md)、§4.3）。
- テスト用の口を、別のライブラリのフォルダや、基盤の中の3つ目のフォルダにする形 — 不採用: 前者は別のフォルダのsrcを読む箇所が残り、後者はライブラリの形が3つになる（出典: [PRD182](../../memo/182_source_layout_10prd.md)、§4.3）。
- 組み合わせのテストをsrc/tests/やRunnerのテストに置く形 — 不採用: テストは各フォルダの中という形から外れるか、Runner自身のテストと別物が混ざる（出典: [PRD182](../../memo/182_source_layout_10prd.md)、§4.3）。
- 一方の道具だけを apps の下へ移し、置き場を2か所に分ける形 — 不採用: 探す場所が2つ残る（出典: [PRD182](../../memo/182_source_layout_10prd.md)、§4.3）。
- ソース配置の受入の検査をCIに載せること — 範囲外: CIの整備のPRDが扱う。再開条件: CIの整備のPRD（出典: [PRD182](../../memo/182_source_layout_10prd.md)、§4.1 D25・§7）。
- グリルの前提を全部、最初に1つの応答で一覧にする形 — 不採用: 論点が多いと読み流され、問いでもう一度言い直して二重になる（出典: [PRD189](../../memo/189_grill_premise_understanding_10prd.md)、§6.1 D2・§9）。
- グリルの前提を問いごとに添えるだけにする形 — 不採用: 全体に効く前提が最初の問いまで欠けたまま進む（出典: [PRD189](../../memo/189_grill_premise_understanding_10prd.md)、§6.1 D2・§9）。
- 共通の前提を、足す仕組みの確認のあとの別の応答で示して確かめる形 — 不採用: 何も無いときの返事が承認だけの往復になり、改訂では前提の画面が2つ並ぶ（出典: [PRD189](../../memo/189_grill_premise_understanding_10prd.md)、§6.1 D2・§9）。
- 前提の理解を、前提ごとの申告、開発者自身の言葉での要約、AIの確認の問いで確かめる形 — 不採用: 質問欄に収まらないか、毎回、自由記述の答え（中央値4.4分）か複数の往復を足す（出典: [PRD189](../../memo/189_grill_premise_understanding_10prd.md)、§6.1 D3・§9）。
- 前提が欠けたら常にグリルを止めて暫定のPRDにする形 — 不採用: 小さな欠けでも止まり、それまでの決定が暫定の既定に下がって開き直しになる（出典: [PRD189](../../memo/189_grill_premise_understanding_10prd.md)、§6.1 D4・§9）。
- 欠けた前提を今のまま結果ごとの分かれ方だけで決める形 — 不採用: 全体に効く前提が欠けたままグリル全体が進む（出典: [PRD189](../../memo/189_grill_premise_understanding_10prd.md)、§6.1 D4・§9）。
- グリルの前提をPRDの既存の節（現状・規約と過去の判断）へ分けて書く形 — 不採用: 書き漏れを会話を知らないレビューで見つけにくく、確かめを飛ばした記録が散らばる（出典: [PRD189](../../memo/189_grill_premise_understanding_10prd.md)、§6.1 D5・§9）。
- 暫定の起票で、前提の表に加えて前提をそろえる段まで書く形 — 不採用: 開発者の選択（出典: [PRD189](../../memo/189_grill_premise_understanding_10prd.md)、§6.1 D6・§9）。
- 前提の段と、前提をそろえるための停止を、プロセス契約に中立の文面で足す形 — 先送り: 手順を試しておらず、anetの外のプロジェクトの工程にまで入る。再開条件: anetで手順を試し、開発者が中立の文面に上げてよいと判断したとき（出典: [PRD189](../../memo/189_grill_premise_understanding_10prd.md)、§6.1 D7・§9）。
- 前提の表の検査を、観点29の確認内容を広げて済ませる形 — 不採用: 表そのものが無くても見つからない（出典: [PRD189](../../memo/189_grill_premise_understanding_10prd.md)、§6.1 D8・§9）。
- 共通の前提の各行に、効く論点と、違うと推奨がどう変わるかを添える形 — 不採用: goal anchorの時点では論点が決まっていないことがあり、誤りやすい（出典: [PRD189](../../memo/189_grill_premise_understanding_10prd.md)、§6.1 D30・§9）。
- 実装の入口でレビューを受けるPRDに、前提をそろえるためにグリルを止めたPRDを足す形 — 範囲外: 課題が範囲をanet-create-prdに限っている。再開条件: 未決表のU1を起票し、その課題を扱うとき（出典: [PRD189](../../memo/189_grill_premise_understanding_10prd.md)、§1.3 NG3・§9・§10）。
- グリルの手順の試走（実装のあとの最初のグリル1本）を受入に入れる形 — 不採用: 実装の完了が次のグリルを待つ（出典: [PRD189](../../memo/189_grill_premise_understanding_10prd.md)、§6.1 D32・§9）。
- 会話のまとめページの決まりを skill だけに置く形 — 不採用: 自動で作るのが、skill の説明文に AI が気づくかどうかに頼る（出典: [PRD197](../../memo/197_conversation_options_summary_page_10prd.md)、§6 D5・§13）。
- 会話のまとめページの決まりを AGENTS.md だけに置く形 — 不採用: 毎回読む AGENTS.md が手順で数千バイト増え、細則を外に置く方針と逆向きで、頼む入口も無い（出典: [PRD197](../../memo/197_conversation_options_summary_page_10prd.md)、§6 D5・§13）。
- まとめページの場面ごとの指針を skill のフォルダに置く形 — 不採用: skill の本体ではないものを .agents と .claude の 2 か所にコピーすることになる（出典: [PRD197](../../memo/197_conversation_options_summary_page_10prd.md)、§6 D6）。
- まとめページの場面ごとの指針を registry.md に置く形 — 不採用: 成果物の対応表に別の関心が混ざり、レビュー役にも毎回読ませる（出典: [PRD197](../../memo/197_conversation_options_summary_page_10prd.md)、§6 D6）。
- まとめページの目印を一覧ファイル 1 つにする形 — 不採用: 並行する会話の書き込みがぶつかり、終わった行を消す決まりも要る（出典: [PRD197](../../memo/197_conversation_options_summary_page_10prd.md)、§6 D17）。
- まとめページの URL を課題や PRD に書く形 — 不採用: 公開リポジトリに非公開ページの URL を残す（出典: [PRD197](../../memo/197_conversation_options_summary_page_10prd.md)、§6 D14・§13）。
- Codex のまとめページを、スレッドの中で Markdown を見せる形や ChatGPT Space の Page にする形 — 不採用: 中身が Markdown になり、Claude Code の HTML と型を 2 つ持つ（出典: [PRD197](../../memo/197_conversation_options_summary_page_10prd.md)、§6 D2）。
- まとめページを主題ごとに 1 枚にする形 — 不採用: 1 つの会話で主題を行き来する長い会話で、行き来のたびに作り直すか、URL が複数できる（出典: [PRD197](../../memo/197_conversation_options_summary_page_10prd.md)、§6 D25・§13）。
- まとめページのひな形を置かない形、場面ごとに置く形 — 不採用: 前者はページごとに見た目が戻りうり、後者は直すとき複数のひな形がずれうる（出典: [PRD197](../../memo/197_conversation_options_summary_page_10prd.md)、§6 D26・§13）。

## 変えるときの条件

一般の条件は[リファクタリング観点](../../quality/refactor.md)の「ガードレール」、テストの充足は[定点統計](../../../reports/stats/trend.md)の「テスト充足」を見る。

- workspaceの箱を移動・削除・退避しても、Run・cache・Optuna履歴・基底configが一体で移動する自己完結性を保つ（出典: [PRD046](../../memo/done/046_workspace_10prd.md)、§1の不変条件）。
- inspect_runのstreaming・1 pass・128点上限をfocused testで確かめる（出典: [PRD054](../../memo/done/054_inspect_run_10prd.md)、§9・§10）。
- inspect_runで128点以下は全点を保持し、間引く場合も先頭・末尾と決定性を保つ（出典: [PRD054](../../memo/done/054_inspect_run_10prd.md)、§9）。
- issueの分類をstatsと共有する変更では、statsのカテゴリ集計が変わらないことを確かめる（出典: [PRD095](../../memo/done/095_llm_wiki_10prd.md)、§4.7の台帳のツール・受入15）。
- リリースの組み立てを移すときは、実行用zipの中身（ファイルのentryのパスの一覧）を変えない。移行の直前のコミットで旧workflowのドライランを回したartifactと、新しい入口で作ったzipの一覧を比べる（出典: [185_release_packaging_cmake](../../memo/185_release_packaging_cmake_10prd.md)、§3のG2・§7の受入3）。
- リリースの構成を足しても、日常のDebug・Release構成の出力先と出力物を変えない。前後でexe・DLL・静的ライブラリのファイル名の一覧を比べる（出典: [185_release_packaging_cmake](../../memo/185_release_packaging_cmake_10prd.md)、§4 D2・§7の受入1）。
- housekeepingに候補を足しても、reports/housekeeping/runs.jsonlの既存の行は書き換えない（出典: [PRD112](../../memo/112_claude_memory_tidy_10prd.md)、§5.5）。
- 過去のPRD（done/・frozen/・dropped/）・実装記録・ADRの本文は書き換えない。999の振り直しでは、ファイル名を指すパスの表記だけを直す（出典: [PRD114](../../memo/114_findings_intake_and_fix_prd_10prd.md)、§5.9・§5.12）。参照先のPRDの改名・移動に合わせてMarkdownのリンクのパスだけを直すことと、先頭へ注記を足すことは、書き換えに当たらない（出典: [PRD131](../../memo/131_prd_9xx_renumber_by_creation_date_10prd.md)、§4 D6）。
- PRDの番号は、一度使ったものを二度と使わない（コミットの件名・CHANGELOG・ADR・実験記録が番号でPRDを指すため）（出典: [PRD114](../../memo/114_findings_intake_and_fix_prd_10prd.md)、§4 D13）。9xxの振り直しはこの例外で、振り直しの日（振り直しの行の日付）以前の9xxの使用を数えない（出典: [PRD131](../../memo/131_prd_9xx_renumber_by_creation_date_10prd.md)、§4 D3）。
- PRDを改名するときは、改名と表記の直しを1回のコミットにまとめ、ステージしたあとの`git diff --cached -M`で改名（`R`）として見分けられることを確かめる。定点統計の起票の月は`git log --follow`で取るため（出典: [PRD131](../../memo/131_prd_9xx_renumber_by_creation_date_10prd.md)、§2の15・§4 D14・§6のA12）。
- 課題の仕組みのファイルは共有コアに含めず、anetのskillと文書として持つ。プロセス契約の節とanetの値はskillに直接書き、レジストリの読み替えで当てない（出典: [PRD114](../../memo/114_findings_intake_and_fix_prd_10prd.md)、§4 D2・D4・D5）。
- 共有コアの文面はskill・エージェント・アダプタの名前以外をプロジェクトに依存させず、anet固有のことはレジストリと観点に置く（出典: [ADR 0049](../../adr/0049-process-contract-kept-project-neutral-anet-specifics-in-registry-and-viewpoints.md)、冒頭の2026-11-02注記）。
- 新しい設定が既定のときは、着手前に保存した実行体・設定を使い、スキップ無効と既存の有効構成のmetrics checksumを一致させる。古い設定と新しい実行体を組み合わせる比較を保つ（出典: [PRD103](../../memo/done/103_temporl_reference_alignment_10prd.md)、§4 D14・§5.7・§7の受入1）。
- 基盤テストの実行体でMKLDNNを切るのは実行体の開始時の1か所にし、テストの中で切らない。テストの開始時に切れていることを見張りのテストで確かめ、Release（assertが出ない）でも切り替えの消失に気づけるようにする（出典: [PRD124](../../memo/124_test_process_mkldnn_off_10prd.md)、§4 D1・D5・§6のA4）。
- テスト用の口の中身は、組み立ての補助と読むだけの口に限り、製品の処理を置かない。anet/test/の下のヘッダはガードを持ち、製品のコードは読まない（出典: [PRD182](../../memo/182_source_layout_10prd.md)、§4.1 D13・§6 のS3-3）。
- リリースのzipの中のパスを変えるときは、着手前の一覧と比べて、決めた差だけであることを確かめる（出典: [PRD182](../../memo/182_source_layout_10prd.md)、§6 のS1-4・S4-4）。
- PRDのグリルの手順を変えるときは、理解を深める往復（説明や具体例を求める返答と、その答え）を減らす対象にしない。前提の確かめと前提をそろえる段は、開発者が指定すれば飛ばせる形を保つ（出典: [PRD189](../../memo/189_grill_premise_understanding_10prd.md)、§1.3 NG1・§6.1 D1）。
- 会話のまとめページは見返し用の写しに保ち、決定の正本（PRD・実験記録・課題）にしない（出典: [PRD197](../../memo/197_conversation_options_summary_page_10prd.md)、§1.3）。
- まとめページの決まりは、Claude Code と Codex で同じ決まりと同じ中身の型を使い、違うのは出し方だけに保つ（出典: [PRD197](../../memo/197_conversation_options_summary_page_10prd.md)、§6 D3）。
- skill の本体ではないもの（まとめページの共通の型と場面ごとの指針）を、.agents と .claude の 2 か所にコピーしない（出典: [PRD197](../../memo/197_conversation_options_summary_page_10prd.md)、§6 D6）。
- 追跡するファイルに、まとめページの非公開ページの URL を書かない（出典: [PRD197](../../memo/197_conversation_options_summary_page_10prd.md)、§6 D14・§10 A8）。

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
- [.gitignore](../../../apps/tools/.gitignore)
- [atari5.py](../../../apps/tools/atari5.py) / [atari5_test.py](../../../apps/tools/atari5_test.py)
- [batchrun_fatal_error_handling_test.ps1](../../../apps/tools/batchrun_fatal_error_handling_test.ps1)
- [btr_to_metrics.py](../../../apps/tools/btr_to_metrics.py)
- [compress_workspace_metrics.py](../../../apps/tools/compress_workspace_metrics.py) / [compress_workspace_metrics_test.py](../../../apps/tools/compress_workspace_metrics_test.py)
- [dropmerge_optuna.py](../../../apps/tools/dropmerge_optuna.py)
- [inspect_run.py](../../../apps/tools/inspect_run.py) / [inspect_run_test.py](../../../apps/tools/inspect_run_test.py)
- [metrics_source.py](../../../apps/tools/metrics_source.py) / [metrics_source_test.py](../../../apps/tools/metrics_source_test.py)
- [metrics_viewer.py](../../../apps/tools/metrics_viewer.py)
- [mlflow_bridge.py](../../../apps/tools/mlflow_bridge.py) / [mlflow_bridge_test.py](../../../apps/tools/mlflow_bridge_test.py)
- [optuna_common.py](../../../apps/tools/optuna_common.py)
- [optuna_metrics_gzip_test.py](../../../apps/tools/optuna_metrics_gzip_test.py)
- [optuna_workspace_test.py](../../../apps/tools/optuna_workspace_test.py)
- [requirements.txt](../../../apps/tools/requirements.txt)
- [resolve_workspace.bat](../../../apps/tools/resolve_workspace.bat) / [resolve_workspace_test.ps1](../../../apps/tools/resolve_workspace_test.ps1)
- [start_mlflow.bat](../../../apps/tools/start_mlflow.bat)
- [start_tb.bat](../../../apps/tools/start_tb.bat)
- [symbolize.py](../../../apps/tools/symbolize.py)
- [tb_bridge.py](../../../apps/tools/tb_bridge.py) / [tb_bridge_test.py](../../../apps/tools/tb_bridge_test.py)
- [reload.lua](../../../apps/tools/vlc/lua/extensions/reload.lua)
- [watch_log.ps1](../../../apps/tools/watch_log.ps1)
- [categories.json](../../../reports/tools/categories.json)
- [requirements.txt](../../../reports/tools/requirements.txt)
- [stage_issue_memos.py](../../../reports/tools/stage_issue_memos.py) / [stage_issue_memos_test.py](../../../reports/tools/stage_issue_memos_test.py)
- [stats.py](../../../reports/tools/stats.py)
- [wiki_support.py](../../../reports/tools/wiki_support.py)

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
- [PRD131](../../memo/131_prd_9xx_renumber_by_creation_date_10prd.md) — PRD131: 9xx の番号の PRD 57 本を、起票の日の順に 900 未満の連番へ振り直す PRD
- [PRD182](../../memo/182_source_layout_10prd.md) — ソース配置の見直し（置き場の規則、Agent の系統ごとのライブラリ、DQN のファイル分割、テストの参照範囲）PRD
- [PRD191](../../memo/191_rnd_raw_error_and_recent_learning_10prd.md) — PRD191: RND の設定を段階ごとに組み替え、誤差で割らない内的報酬の式と、ためてからエポックで学ぶ predictor の補助学習を足す（Burda らの形へ寄せる） PRD

## 課題

issueの道具（`wiki_support.py maps`）が作り直す。手で直さない。

<!-- issues:begin -->
| ID | 状態 | WSJF | 待ち | 対象 | 工程 | 領域 | 分類 | 種類 | PRD | AIの判定 | 課題 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [issue-20261104-52ffba](../../management/issues/issue-20261104-52ffba.md) | 対応中 | 1.4 | — | ハーネス | ハーネス保守 | 311_dev_tools | 開発 | 機能改善 | — | 残件有り | 課題の仕組みに残る改善点をまとめて管理する（段階に分けた PRD の「実装済み」の時機ほか） |
| [issue-20261103-955e50](../../management/issues/issue-20261103-955e50.md) | 調査中 | 1.2 | — | コード | PRD | 318_rl_core, 317_metrics, 110_runner_app, 311_dev_tools, 120_metrics_viewer | UI | 機能追加 | [PRD081](../../memo/081_evalpanel_episode_trace_10prd.md) | 残件有り | EvalPanel のゲームを trace へ出す — dormant タグの trace を鏡写しインスタンスへ結び付け、trace 行の時刻で振り返る |
| [issue-20261103-23d185](../../management/issues/issue-20261103-23d185.md) | 対応中 | 0.9 | — | ハーネス | PRD | 311_dev_tools | 開発 | 機能追加 | [PRD187](../../memo/187_issue_list_views_10prd.md) | 残件有り | issue の一覧の見せ方（依存のツリー、スプレッドシート、見やすい表） |
| [issue-20261111-3986b7](../../management/issues/issue-20261111-3986b7.md) | 未着手 | 3.0 | — | ハーネス | ハーネス保守 | 311_dev_tools | 開発 | 機能改善 | — | 未確認 | テストを回す範囲・回数と、テストの控え・ログの置き場の決まりが無く、実装の時間と .scratch が膨らむ |
| [issue-20261103-5addea](../../management/issues/issue-20261103-5addea.md) | 未着手 | 2.2 | — | コード | PRD | 311_dev_tools, 318_rl_core, 315_nn, 312_util, 317_metrics, 321_agent_core, 319_gui, 316_config, 322_env_individual, 324_dqn, 110_runner_app, 325_muzero, 313_profiling | 開発 | リファクタリング | [PRD079](../../memo/079_cpp_internal_linkage_convention_10prd.md) | 残件有り | PRD 079: .cpp の内部リンケージと namespace インデント規約 |
| [issue-20261119-25e4d0](../../management/issues/issue-20261119-25e4d0.md) | 未着手 | 2.0 | — | ハーネス | ハーネス保守 | 311_dev_tools | 開発 | 機能改善 | — | 未確認 | 会話が要約されたあと、続ける工程の手順を読み直さずに進めてしまう（読み直しを促す仕掛けが無い） |
| [issue-20261103-58696c](../../management/issues/issue-20261103-58696c.md) | 未着手 | 1.9 | — | ハーネス | PRD | 311_dev_tools | 開発 | 機能追加 | [PRD112](../../memo/112_claude_memory_tidy_10prd.md) | 残件有り | Claude Code の memory の定期整備（`anet-tidy-memory` と housekeeping の候補 `memory-tidy`） |
| [issue-20261119-9afcf4](../../management/issues/issue-20261119-9afcf4.md) | 未着手 | 1.7 | — | コード | PRD | 311_dev_tools, 321_agent_core | 開発 | リファクタリング | — | 未確認 | テストごとに書かれている代役の Agent 8 個を、基盤のテスト用の口の代役へ寄せる |
| [issue-20261103-0ef7b8](../../management/issues/issue-20261103-0ef7b8.md) | 未着手 | 1.6 | — | コード | PRD | 317_metrics, 311_dev_tools, 316_config, 318_rl_core, 321_agent_core, 120_metrics_viewer, 110_runner_app | UI | 機能改善 | [PRD108](../../memo/108_metrics_tag_numbering_10prd.md) | 残件有り | メトリクスの tag の番号を全体で振り直す（群の番号と群の中の番号） |
| [issue-20261103-61d771](../../management/issues/issue-20261103-61d771.md) | 未着手 | 1.6 | — | コード | PRD | 311_dev_tools | 開発 | 機能追加 | [PRD156](../../memo/156_cicd_improvement_10prd.md) | 残件有り | CI/CD 改善（ビルド・テスト自動化、定点ジョブ、LLM 補助） |
| [issue-20261115-3e5b77](../../management/issues/issue-20261115-3e5b77.md) | 未着手 | 1.6 | — | 文書 | その他 | 311_dev_tools | 開発 | リファクタリング | — | 未確認 | 設計160（アプリケーションとツール）に開発環境の道具の記述が混ざっているので、開発環境の文書へ分けて整える |
| [issue-20261121-529662](../../management/issues/issue-20261121-529662.md) | 未着手 | 1.5 | — | ハーネス | PRD | 311_dev_tools | 開発 | リファクタリング | [PRD195](../../memo/195_rename_prd_document_term_10prd.md) | 未確認 | 文書の呼び名「PRD」が製品の要求書と中身で食い違うので、中身に合う名前へ改め、今使っている場所をすべて移す |
| [issue-20261103-52bda0](../../management/issues/issue-20261103-52bda0.md) | 未着手 | 1.4 | — | コード | PRD | 318_rl_core, 321_agent_core, 120_metrics_viewer, 311_dev_tools | UI | 機能改善 | [PRD096](../../memo/096_metrics_step_axis_interval_10prd.md) | 残件有り | メトリクスの横軸と間隔を数える単位を分ける |
| [issue-20261110-9d4e5a](../../management/issues/issue-20261110-9d4e5a.md) | 未着手 | 1.4 | — | コード | その他 | 311_dev_tools | 開発 | 機能追加 | — | 未確認 | C++ の単純な書き方の問題を機械で検出する仕組み（lint・静的解析）が無い |
| [issue-20261103-71bb9a](../../management/issues/issue-20261103-71bb9a.md) | 未着手 | 1.2 | — | コード | PRD | 311_dev_tools | 開発 | 機能追加 | [PRD157](../../memo/157_test_line_coverage_10prd.md) | 残件有り | 行カバレッジ計測（OpenCppCoverage） |
| [issue-20261103-7227bf](../../management/issues/issue-20261103-7227bf.md) | 未着手 | 1.2 | — | コード | PRD | 311_dev_tools, 317_metrics | 開発 | 機能追加 | [PRD153](../../memo/153_q_statistics_config_screening_10prd.md) | 残件有り | Q 値統計による多環境ハイパラ健全性スクリーニング |
| [issue-20261103-dbe666](../../management/issues/issue-20261103-dbe666.md) | 未着手 | 1.2 | — | コード | PRD | 322_env_individual, 318_rl_core, 311_dev_tools | Env | 調査 | [PRD163](../../memo/163_dropmerge_neet_diagnostics_10prd.md) | 残件有り | DropMerge の NEET 診断（trace で足りるか確かめてから指標化） |
| [issue-20261115-657248](../../management/issues/issue-20261115-657248.md) | 未着手 | 1.2 | — | コード | ハーネス保守 | 311_dev_tools | 開発 | リファクタリング | — | 未確認 | 既存のソースと設定を、最新の規約（include の順・クラス宣言の区切り・関数名の形など）に合わせてまとめて点検し直す |
| [issue-20261103-c16aab](../../management/issues/issue-20261103-c16aab.md) | 未着手 | 1.0 | — | コード | PRD | 320_env_core, 322_env_individual, 312_util, 120_metrics_viewer, 311_dev_tools | UI | 機能追加 | [PRD154](../../memo/154_scalar_threshold_rate_10prd.md) | 残件有り | スカラー閾値越え率の汎用化（Observation 層 / Viewer 層） |
| [issue-20261105-f4b870](../../management/issues/issue-20261105-f4b870.md) | 未着手 | 1.0 | — | コード | その他 | 311_dev_tools, 315_nn | アルゴリズム | 調査 | — | 未確認 | LLM が NN の重みを直接読み書きする試み（Weight-Space Programming）を anet で試す |
| [issue-20261116-fe09dc](../../management/issues/issue-20261116-fe09dc.md) | 未着手 | 1.0 | — | ハーネス | PRD | 311_dev_tools | 開発 | 機能改善 | — | 未確認 | PRD の番号が 1000 に届いたときの 4 桁の形（ファイル名の検査と道具の番号の読み方）を決める |
| [issue-20261103-b13ed2](../../management/issues/issue-20261103-b13ed2.md) | 未着手 | 0.8 | — | コード | PRD | 311_dev_tools | 開発 | 機能改善 | [PRD185](../../memo/185_release_packaging_cmake_10prd.md) | 残件有り | リリース物の組み立てを CMake（install・CPack・ワークフロー preset）へ移し、手元でも同じコマンドで作れるようにする |

着手できる 23 / 待ち 0 / 保留 0 / 次版 0 / 解決 0
<!-- issues:end -->
