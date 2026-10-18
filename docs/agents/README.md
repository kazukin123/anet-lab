# AI ハーネス動線マップ

anet-lab で AI エージェントの振る舞いを決めているファイル群と、その間の参照の連なりを、図 2 枚と表 4 つで表した地図です。
**人間が全体像を把握するための資料**であり、エージェントに読ませる正本ではありません。「いつ何を読むか」の正本は
従来どおり [AGENTS.md](../../AGENTS.md) と各文書側にあります。

本書は `anet-harness-map` skill が現行 checkout から**再生成**します。手で直さず、ハーネスが変わったら作り直してください。
目的は、重複(shadow・実体コピーのドリフト)・欠落・死んだ動線を**名指しできる**ようにすることです。
**修正は本書の範囲外**で、次フェーズの PRD が本書の表と発見事項を引用して起票します。

## 1. 凡例と図の索引

### 用語

| 語 | 定義 | 避け語 |
|---|---|---|
| **AI ハーネス** | エージェント(Claude Code / Codex)の振る舞いを決めるファイル群の総体。指示・規約・設計文書・PRD/ADR/用語集・検証資材・skill・memory・CI。コード本体は含まない | 環境、設定 |
| **入口** | 条件なしに毎セッション読まれるファイル。`CLAUDE.md`、`AGENTS.md`、global `CLAUDE.md`、memory の index | 常時読み、エントリ |
| **条件読み** | 作業種別や領域で読むかどうかが決まる読み。入口以外の読みはすべてこれ | 参照、オンデマンド |
| **動線** | 入口から末端(文書の章・skill・ツール)まで、エージェントが辿る参照の連なり。参照元の無い末端は「死んだ動線」 | 流路、経路、フロー |
| **shadow** | 別スコープ(user-level)に同名の skill があり、project 版の代わりに起動しうる状態。Claude Code は `~/.claude/skills`、Codex は `~/.agents/skills` が user-level | 上書き、重複 |
| **実体コピー** | 同じスコープ内で意図して同内容を 2 実体として置いている状態(`.agents/skills` と `.claude/skills`)。同期は手作業 | 重複、ミラー |

「重複」を避け語にするのは、**shadow は消すべき事故、実体コピーは意図した構造で同期だけが問題**と、対処が正反対だからです。

### 図

`表示` は GitHub Pages 上のレンダリング結果、`html` と `json` はリポジトリ内の実ファイルです。

| 順序 | 図 | 何が分かるか | 表示 | ファイル |
|---:|---|---|---|---|
| 1 | 構成 | ノードの種別、リポジトリとユーザー環境の境界、主経路、欠落しているリンク | [表示](https://kazukin123.github.io/anet-lab/docs/agents/harness_map_architecture.html) | [html](harness_map_architecture.html) / [json](harness_map_architecture.archify.json) |
| 2 | 動線 | PRD ライフサイクルの工程、役割ごとのレーン、側枝の分岐点 | [表示](https://kazukin123.github.io/anet-lab/docs/agents/harness_map_workflow.html) | [html](harness_map_workflow.html) / [json](harness_map_workflow.archify.json) |

俯瞰(構成)と動線(工程)は読み方が違うので 2 枚に分けています。構成図は「どこに何があり、どこが切れているか」、
動線図は「どの順で誰が何をするか」を答えます。

Pages のサイト index は `docs/archify/*.html` だけを自動掲載するため([pages.yml](../../.github/workflows/pages.yml) の Generate site index)、
本ディレクトリの図はこの表の `表示` 列からのみ辿れます。

### 生成根拠と言語

- 生成根拠の Git revision: `9b928dda407785ec8cb39b12641d8a8f836dcead` の**作業ツリー**(実測 2026-09-21)。
- 図の説明文は日本語ですが、日本語は Archify Viewer の対応 locale ではないため `meta.locale` を省略しています。
  **固定 Viewer UI(Light/Dark、Present、Export など)と `<html lang>` は英語**です。[docs/archify/README.md](../archify/README.md) と同じ扱いです。
- 構成図は 1440×900 で 1 画面に収まります。**動線図はレーンが 5 本あるため縦スクロールが要ります**(横スクロールはどの画面幅でも出ません)。
- 動線図のレーンは**役割名**(起票側 / 精査側 / 実装側)です。どの枠がどれに入るかはトークン予算で入れ替わるため、
  図はエージェント名で固定しません。工程の正本は [docs/memo/README.md](../memo/README.md) §工程と担当 です。
- 動線図の `実装` は計画 → 編集 → **設定ファイルと 020 §3 のドキュメントの同時更新** → 検証を畳んだノードです。
  設定変更を独立ノードにすると工程に見え、誤読のもとになるので畳んでいます。読む条件は表 2 が持ちます。動線図に入らなかった側枝(Run 分析・実験記録)の条件は表 2 が持ちます。

## 2. 表 1 ノード在庫

**参照元が空の行が「死んだ動線」**です。入口(常時読み)は参照元を必要としないので `—(入口)` と書きます。

参照元は、入口と常設文書(`AGENTS.md` / `CLAUDE.md` / `README.md` / `CONTEXT.md` / `docs/design` / `docs/agents` / `docs/archify` /
`docs/memo/README.md` / `docs/experiments/README.md` / `.agents/skills` / `.claude/skills`)からのリンクと名指しを数えます。
**本書自身と 2 図は参照元に数えません**(動線マップが自分で参照を作って死んだ動線を消してしまうため)。
個別 PRD(`docs/memo/NNN_*`)も、要件文書であって定常の動線ではないので数えません。

| 種別 | パス | 読む条件 | 誰が読む | 参照元 | 根拠 |
|---|---|---|---|---|---|
| 入口 | [CLAUDE.md](../../CLAUDE.md) | 常時(3 行) | Claude | —(入口) | 全文が `AGENTS.md` を読む指示 |
| 入口 | [AGENTS.md](../../AGENTS.md) | 常時(720 行・31 節) | 人間・Claude・Codex | 設計文書 3 本(010・020・040 の jp/en) | `CLAUDE.md` の指示。Codex は project 直下の本ファイルを読む |
| 用語集 | [CONTEXT.md](../../CONTEXT.md) | 設定キー・メトリクス名・公開クラス名を新しく付けるとき | Claude・Codex | `README.md`、設計文書 6 本(010/020/110/120/140/210) | AGENTS.md §用語(CONTEXT.md)を引く条件 |
| 記録 | [docs/adr/](../adr/)(44 件) | 契約・構造を変える / 古く見えるコードを消す / それらをレビューする | Claude・Codex | `AGENTS.md`(フォルダ単位) | AGENTS.md §設計判断の記録(ADR)を読む条件 |
| 記録 | [docs/memo/](../memo/)(直下 36・done 160・frozen 6・dropped 1) | 該当 PRD を実装するとき | 人間・Claude・Codex | `docs/memo/README.md` 経由(フォルダ単位) | docs/memo/README.md §採番・§運用 |
| 記録 | [docs/memo/README.md](../memo/README.md) | PRD を起票・移動するとき | 人間・Claude・Codex | `AGENTS.md`、`docs/agents/issue-tracker.md` | AGENTS.md §Agent skills |
| 設計索引 | [docs/design/README.jp.md](../design/README.jp.md) | 領域の文書を探すとき | 全員 | `README.md`、`010_framework_overview.jp.md` | §3 ドキュメント一覧 |
| 全体概要 | [010_framework_overview.jp.md](../design/010_framework_overview.jp.md) | 全体像を掴むとき | 全員 | 設計文書 7 本、`docs/archify/README.md` | §1 はじめに |
| ユーザーガイド | [020_user_guide_run.jp.md](../design/020_user_guide_run.jp.md) | 設定を書く・Run を起動するとき | 全員 | `AGENTS.md`、`README.md`、設計文書 6 本 | AGENTS.md §設定値の扱い が §3.6・§3.7 を名指し |
| ユーザーガイド | [030_user_guide_analysis.jp.md](../design/030_user_guide_analysis.jp.md) | Run 結果を分析するとき | 全員 | `AGENTS.md`、`README.md`、設計文書 7 本 | AGENTS.md §AI エージェントのRun結果分析ルール が冒頭で名指し |
| 開発ガイド | [040_development_environment.jp.md](../design/040_development_environment.jp.md) | 環境構築・CLI ビルド・Python ツール実行 | 開発者 | `README.md`、設計文書 5 本 | §6 CLI によるビルド・テスト、§7 Python 補助ツール |
| 設計(1xx) | [100_runtime_and_configuration.jp.md](../design/100_runtime_and_configuration.jp.md) | 起動・設定解決・Run 構築を編集する前 | Claude・Codex | 設計文書 8 本のみ(**入口からの直リンク無し**) | §8 テストと拡張時の確認事項 |
| 設計(1xx) | [110_agents_and_learning.jp.md](../design/110_agents_and_learning.jp.md) | Agent・Actor・Learner を編集する前 | Claude・Codex | 設計文書 9 本のみ(同上) | §8 テストと拡張時の確認事項 |
| 設計(1xx) | [120_environments.jp.md](../design/120_environments.jp.md) | Env・BatchEnv を編集する前 | Claude・Codex | 設計文書 7 本のみ(同上) | §8 テストと拡張時の確認事項 |
| 設計(1xx) | [130_neural_networks.jp.md](../design/130_neural_networks.jp.md) | NN module・optimizer を編集する前 | Claude・Codex | 設計文書 4 本のみ(同上) | §8 テストと拡張時の確認事項 |
| 設計(1xx) | [140_observability.jp.md](../design/140_observability.jp.md) | Event・Observer・メトリクス・profiling を編集する前 | Claude・Codex | 設計文書 10 本のみ(同上) | §9 テストと拡張時の確認事項(章番号が他と違う) |
| 設計(1xx) | [150_replay_buffer.jp.md](../design/150_replay_buffer.jp.md) | Experience・PER・転送を編集する前 | Claude・Codex | 設計文書 8 本のみ(同上) | §8 テストと拡張時の確認事項 |
| 設計(1xx) | [160_applications_and_tools.jp.md](../design/160_applications_and_tools.jp.md) | Runner GUI・補助ツールを編集する前 | Claude・Codex | 設計文書 8 本のみ(同上) | §8 テストと**変更**時の確認事項(章題が他と違う) |
| 設計(2xx) | [200_dqn_agents.jp.md](../design/200_dqn_agents.jp.md) | DQN 系 Agent を編集する前 | Claude・Codex | 設計文書 9 本のみ(同上) | §10 テストと拡張時の確認事項 |
| 設計(2xx) | [210_metrics_viewer.jp.md](../design/210_metrics_viewer.jp.md) | Metrics Viewer を編集する前 | Claude・Codex | `160`、`docs/design/README.jp.md` のみ | §12 テストと拡張時の確認事項 |
| 設計(2xx) | [220_atari_env.jp.md](../design/220_atari_env.jp.md) | AtariEnv・ALE 設定を編集する前 | Claude・Codex | `docs/design/README.jp.md` のみ | **テスト章が無い**(§5 AtariView → §6 ビルド統合 → §7 関連文書) |
| 設計(英訳) | `docs/design/*.en.md`(15 本) | 英語で読むとき | 人間 | `docs/design/README.en.md` | 翻訳成果物。AI は `.jp.md` を読む(AGENTS.md §AI エージェントの応答言語ルール) |
| 補助仕様 | [docs/design/optuna.md](../design/optuna.md) | Optuna 探索空間を変えるとき | 人間・Claude | `*.en.md` 3 本のみ。**`.jp.md` 側のリンク `../optuna.md` は存在しないパスを指す** | 030 §1、140 §関連文書、160 §詳細運用仕様 |
| skill 規約 | [docs/agents/domain.md](domain.md) | mattpocock 系 skill が用語集を引くとき | Claude | `AGENTS.md`、`setup-matt-pocock-skills` | AGENTS.md §Agent skills > Domain docs |
| skill 規約 | [docs/agents/issue-tracker.md](issue-tracker.md) | 実装 issue を `.scratch/` へ置くとき | Claude | `AGENTS.md`、`docs/memo/README.md`、`setup-matt-pocock-skills` | AGENTS.md §Agent skills > Issue tracker |
| skill 規約 | [docs/agents/triage-labels.md](triage-labels.md) | triage role を status へ対応させるとき | Claude | `AGENTS.md`、`setup-matt-pocock-skills` | AGENTS.md §Agent skills > Triage labels |
| 責任境界 | [docs/ownership_guideline.md](../ownership_guideline.md) | Agent 系の変数・オブジェクトを追加するとき | Claude・Codex | `AGENTS.md`、`README.md`、設計文書 4 本、`anet-harness-map` | AGENTS.md §Agent 系実装の所有権ルール |
| 実験記録 | [docs/experiments/README.md](../experiments/README.md) | 実験記録を書く・config を複製するとき | 人間・Claude | **`anet-harness-map` の調査対象としてのみ**。規約からの動線は無い | AGENTS.md §実験記録と実効 config の保存ルール は `docs/experiments/<agent>/<env>/config/` を名指しするが README へは到達しない |
| 検証資材 | [testdata/prd061/README.md](../../core/anet-core/testdata/prd061/README.md) | Actor カタログの契約を検証するとき | Claude・Codex | `110_agents_and_learning`(jp/en) | AGENTS.md §検証 の一般則 |
| 検証資材 | [testdata/prd072/README.md](../../core/anet-core/testdata/prd072/README.md) | 設定の等価性を検証するとき | Claude・Codex | `AGENTS.md`、`100_runtime_and_configuration`(jp/en) | AGENTS.md §設定値の扱い が手順を名指し |
| 検証資材 | [testdata/prd078/README.md](../../core/anet-core/testdata/prd078/README.md) | RB 履歴整合性を検証するとき | Claude・Codex | `150_replay_buffer`(jp/en) | AGENTS.md §検証 の一般則 |
| CI | [.github/workflows/windows-ci.yml](../../.github/workflows/windows-ci.yml) | push / PR(自動) | —(CI) | `040_development_environment`(jp/en) | 040 §6 CLI によるビルド・テスト |
| CI | [.github/workflows/pages.yml](../../.github/workflows/pages.yml) | `docs/**` の push(自動) | —(CI) | **(無し)** | ファイル冒頭コメントが配信規約を持つ |
| CI | [.github/workflows/release.yml](../../.github/workflows/release.yml) | tag push(自動) | —(CI) | **(無し)** | リリース資材の生成 |
| CI | [.github/workflows/gemini_ai_review.yml](../../.github/workflows/gemini_ai_review.yml) | PR(自動) | —(CI) | **(無し)** | 外部レビュー |
| ツール | [inspect_run.py](../../viewers/metrics-tools/inspect_run.py) | Run 名だけを渡されたとき(分析の入口) | Claude・Codex | **(無し)**。`AGENTS.md` はコマンドブロック内でパスを書くだけで、リンクにも `` ` `` 表記にもなっていない | AGENTS.md §AI エージェントのRun結果分析ルール |
| ツール | [check_default_leaves.py](../../core/anet-core/testdata/prd072/check_default_leaves.py) | 設定を変更したとき | Claude・Codex | **(無し)**。`AGENTS.md` はファイル名だけを書きパスが無い | AGENTS.md §設定値の扱い |
| ツール | [mlflow_bridge.py](../../viewers/metrics-tools/mlflow_bridge.py) | Run 名の prefix を変えようとしたとき | Claude・Codex | `AGENTS.md` | AGENTS.md §Run 命名・後片付けルール(`run_*` で glob する制約) |
| ツール | 標準テストスイート 9 本(Python 7・PowerShell 2) | コード変更後の検証 | Claude・Codex | `AGENTS.md`(コマンドブロック) | AGENTS.md §検証。読む条件が一律なので 1 行にまとめる |

## 3. 表 2 作業 → 読む文書

動線図はラベルしか持てないので、条件文はこの表が持ちます。前半が PRD ライフサイクルの工程、後半が領域ごとの作業です。

| これから何をするか | 読むもの | 読まなくてよいとき | 根拠 |
|---|---|---|---|
| グリルして PRD を起こす | [docs/memo/README.md](../memo/README.md)(採番と状態)、[issue-tracker.md](issue-tracker.md)、対象領域の設計文書 | 既存 PRD をそのまま実装するとき | AGENTS.md §Agent skills > Issue tracker |
| 起票済みの PRD を再グリルする(精査) | PRD 本文、対象領域の設計文書と該当 ADR、[CONTEXT.md](../../CONTEXT.md) | 軽い変更で枠をまとめると起票時に決めたとき | [docs/memo/README.md](../memo/README.md) §工程と担当。**起票側とは別の枠**が行う |
| 実装計画(`*_2ximpl.md`)を書く | PRD 本文、近い実装とテスト、設計文書、ADR | PRD が skill の実行そのものを計画の代わりにしているとき | [docs/memo/README.md](../memo/README.md) §工程と担当。**実装する枠**が作る。`implement-prd-with-docs` の手順自体は担当中立で、Claude / Codex の語を持たない |
| コミット前にレビューする | 変更差分、PRD の受入基準、`git diff <base>` | なし。**起票側の枠と人間の両方**が見る | [docs/memo/README.md](../memo/README.md) §工程と担当 |
| 領域のコードを編集する(Agent / Env / NN / RB / 可観測性 / Runner) | 対応する設計文書 1xx・2xx と各文書の §4 コードマップ | 設計・利用方法に影響しない局所修正 | AGENTS.md §AI エージェントの作業ルール。**文書番号の名指しが無いので [docs/design/README.jp.md](../design/README.jp.md) §3 で引く** |
| 既存の契約・構造を変える / 古く見えるコードを消す | 該当 [ADR](../adr/)(ファイル名で絞る)、領域の設計文書、AGENTS.md §クリーンブレーク方針 | バグ修正、テスト追加、ログ・計測の追加、既存 PRD に沿った実装 | AGENTS.md §設計判断の記録(ADR)を読む条件 |
| 新しい設定キー・メトリクス名・公開クラス名を付ける | [CONTEXT.md](../../CONTEXT.md)(`_Avoid_` を含む) | 既存の語で言い換えられるとき。日常の実装・修正 | AGENTS.md §用語(CONTEXT.md)を引く条件 |
| Agent 系の変数・オブジェクトを追加する | [docs/ownership_guideline.md](../ownership_guideline.md) | 既存 State の値を変えるだけのとき | AGENTS.md §Agent 系実装の所有権ルール |
| 検証する | AGENTS.md §検証 の標準スイート → 領域の設計文書のテスト章 → [testdata/prdNNN/README.md](../../core/anet-core/testdata/) | docs だけの変更 | AGENTS.md §検証 の一般則 |
| 設定ファイルを書き換える | [020](../design/020_user_guide_run.jp.md) §3.6・§3.7、[prd072/README.md](../../core/anet-core/testdata/prd072/README.md) の等価性検査手順 | 既存キーの値だけを変え、演算子と配置を動かさないとき | AGENTS.md §設定値の扱い |
| Run を起動する・workspace を選ぶ | [020](../design/020_user_guide_run.jp.md) | 既存の launcher(`apps/*.bat`)をそのまま使うとき | 020 §1〜§6 |
| Run 結果を分析する | [030](../design/030_user_guide_analysis.jp.md) と `inspect_run.py` の 5 サブコマンド | なし。Run 名だけを渡された時点で分析依頼として扱う | AGENTS.md §AI エージェントのRun結果分析ルール |
| 実験記録を書く | [docs/experiments/README.md](../experiments/README.md)、実効 config の複製規約 | 記録に Run 名が出ない捨て Run(`tmp` 付き) | AGENTS.md §実験記録と実効 config の保存ルール |
| コミット文案を出す | AGENTS.md §Git 操作・コミットメッセージルール の Topic Issue 対応表 | なし。git 操作は人間が行うが、文案は常にこの規約に従う | 同節(`git add` / `commit` / `push` を AI は実行しない) |
| ビルドが通らない・長時間の検証を回す | AGENTS.md §AI エージェントでのビルド注意事項、§長時間のビルド・テスト・バッチ検証、[040](../design/040_development_environment.jp.md) §8 | すでに `VsDevCmd.bat` 経由のコマンドを持っているとき | 同節(素の PowerShell から `cmake --build` しない) |

## 4. 表 3 AGENTS.md の節の配置

入口の削減余地を測るための表です。`AGENTS.md` の全 `##` 節を対象とし、行数は見出し行から次の `##` 見出しの直前まで
(子の `###` 節を含む)を数えています。冒頭のタイトル 2 行を除いた合計が 718 行です。
**分類は候補であって決定ではありません。** 移設は次フェーズです。

| # | 節 | 行数 | 分類 | 移設先候補 | 根拠(節本文が名指しする対象) |
|---:|---|---:|---|---|---|
| 1 | Viewing UTF-8 Japanese Text in AI Agent Terminals on Windows | 23 | 条件付き | [040](../design/040_development_environment.jp.md) | `chcp 65001`、`Get-Content -Encoding UTF8`。mojibake が出たときだけ要る |
| 2 | プロジェクト概要 | 16 | 条件付き | [010](../design/010_framework_overview.jp.md) | ディレクトリ一覧。010 §2 と重複 |
| 3 | 基本方針 | 9 | 常時必要 | — | 変更範囲・依存方向・グローバル状態 |
| 4 | クリーンブレーク方針 | 23 | 条件付き | [100](../design/100_runtime_and_configuration.jp.md) | 設定キー置換の例(`use_qr` → `quantile_mode`) |
| 5 | 設計判断の記録(ADR)を読む条件 | 21 | 常時必要 | — | 読む条件そのもの。`docs/adr/`、設計文書 1xx・2xx |
| 6 | 用語(CONTEXT.md)を引く条件 | 12 | 常時必要 | — | 読む条件そのもの。`CONTEXT.md` |
| 7 | 汎用機構と利用側の責任境界 | 37 | 条件付き | [100](../design/100_runtime_and_configuration.jp.md) / [130](../design/130_neural_networks.jp.md) | NN の `bind` 構文、TensorSpec、component lifecycle |
| 8 | Fail-Fast 原則 | 25 | 条件付き | [100](../design/100_runtime_and_configuration.jp.md) | `ANET_SYSTEM_ERROR`、`interval=0`、`auto` 系 fallback |
| 9 | 設定値の扱い(子節: 設定ファイルの代入演算子) | 19 | 条件付き | [020](../design/020_user_guide_run.jp.md) §3.6・§3.7 | 本文が 020 と `prd072/README.md` を名指ししている |
| 10 | コーディング規約 | 30 | 常時必要 | — | C++20、機能グループ単位のファイル、改行コード |
| 11 | コメント・TODO ルール | 10 | 常時必要 | — | `/// @todo` 形式 |
| 12 | 性能測定・ProfileRange ルール | 42 | 条件付き | [140](../design/140_observability.jp.md) | `ANET_PROFILE_SCOPE` 系マクロ |
| 13 | GetScalar 実装ルール | 9 | 条件付き | [140](../design/140_observability.jp.md) | `GetScalar()` の `std::optional<float>` 契約 |
| 14 | Agent 系実装の所有権ルール | 18 | 条件付き | [ownership_guideline.md](../ownership_guideline.md) | 本文が同ファイルを名指ししている |
| 15 | ビルド | 32 | 条件付き | [040](../design/040_development_environment.jp.md) | CMake preset、libtorch/wxWidgets、`Torch_DIR_*` |
| 16 | 検証 | 34 | 常時必要 | — | 標準スイートと `testdata/prdNNN/README.md` への一般則 |
| 17 | Python 補助ツールの実行 | 22 | 条件付き | [040](../design/040_development_environment.jp.md) §7 | `.venv\Scripts\python.exe` |
| 18 | 人間へ提示するコマンドの書き方 | 27 | 常時必要 | — | 応答規約。1 ブロック 1 行、```bash タグ |
| 19 | 編集しない・慎重に扱う領域 | 12 | 常時必要 | — | `out/`、`.vs/`、`docs/runs/`、`third_party/` |
| 20 | AI エージェントの応答言語ルール | 11 | 常時必要 | — | 応答規約 |
| 21 | AI エージェントの概念説明ルール | 11 | 常時必要 | — | 応答規約 |
| 22 | ログ出力ルール | 9 | 常時必要 | — | `LOG::info()` 系は英語 |
| 23 | LLM コーディング規律 (karpathy-guidelines) | 11 | 条件付き | `.agents/skills/karpathy-guidelines/SKILL.md` | 本文が「全文は SKILL.md」と書いており、要約が二重化している |
| 24 | AI エージェントの作業ルール | 26 | 常時必要 | — | 編集前・編集中・編集後の手順 |
| 25 | 長時間のビルド・テスト・バッチ検証 | 12 | 条件付き | [040](../design/040_development_environment.jp.md) | 外側スクリプト、終了コード記録 |
| 26 | Git 操作・コミットメッセージルール | 48 | 常時必要 | — | Conventional Commits と Topic Issue 対応表 22 行 |
| 27 | Run 命名・後片付けルール(子節: 重要 Run のマーク) | 35 | 条件付き | [020](../design/020_user_guide_run.jp.md) | `tmp` と `★` の付け方、`mlflow_bridge.py` の glob 制約 |
| 28 | 実験記録と実効 config の保存ルール | 31 | 条件付き | [docs/experiments/README.md](../experiments/README.md) | `docs/experiments/<agent>/<env>/config/` の置き場規約 |
| 29 | AI エージェントのRun結果分析ルール | 34 | 条件付き | [030](../design/030_user_guide_analysis.jp.md) | 本文が 030 と `inspect_run.py` を名指ししている |
| 30 | AI エージェントでのビルド注意事項 (Windows/MSVC) | 50 | 条件付き | [040](../design/040_development_environment.jp.md) §8 | `VsDevCmd.bat` 経由コマンドの書式 |
| 31 | Agent skills(子節: Issue tracker / Triage labels / Domain docs) | 19 | 条件付き | [docs/agents](.) の 3 本 | 本文が 3 ファイルを名指し。うち Triage labels と Domain docs は参照先 skill が project に無く、Codex からはどの作業からも到達しない |
| — | **集計** | **718** | 常時必要 13 節 260 行 / **条件付き 18 節 458 行** / どの作業からも到達しない 0 節 | — | 条件付きの 458 行が入口からの移設候補(全体の 64%) |

## 5. 表 4 skill 在庫

project の skill は `.agents/skills`(Codex が読む)と `.claude/skills`(Claude Code が読む)の**実体コピー 2 系統**で、
同期は `diff -rq` の手作業です([.agents/skills/README.md](../../.agents/skills/README.md) §追加と更新の手順)。
shadow 列は user-level に同名があるかで、`同一` は中身が一致、`中身違い` は差分あり、`—` は同名が無い状態です。

| 名前 | 所在 | 実体コピーの同期 | shadow(Claude) | shadow(Codex) | `openai.yaml` | 根拠 |
|---|---|---|---|---|---|---|
| anet-archify-atlas | project | 一致 | — | — | あり | skills README §プロジェクト固有 |
| anet-audit | project | **不一致(SKILL.md)** | — | — | あり | `diff -rq .agents/skills .claude/skills` |
| anet-harness-map | project | 一致 | — | — | あり | 本書を生成する skill |
| anet-housekeeping | project | **不一致(SKILL.md)** | — | — | あり | 同上。`.agents` 側が新しく、「実行前にユーザーの了承を得る」が `.claude` 側に届いていない |
| anet-stats | project | 一致 | — | — | あり | skills README §プロジェクト固有 |
| anet-survey-queue | project | 一致 | — | — | あり | 同上 |
| anet-translate-docs | project | 一致 | — | — | あり | 同上 |
| prepare-commit | project + user-level | 一致 | **lnk(別実体)** | 同一 | あり | Claude 側は `prepare-commit.lnk` で `~/.agents/skills/prepare-commit` を指す |
| implement-prd-with-docs | project | 一致 | — | — | なし | skills README §プロジェクト固有 |
| grill-me | project + user-level | 一致 | **中身違い** | **中身違い** | なし | SKILL.md と `agents/` の有無 |
| grill-with-docs | project + user-level | **不一致(SKILL.md)** | **中身違い** | **中身違い** | なし | user-level 版には `ADR-FORMAT.md` / `CONTEXT-FORMAT.md` が**無い** |
| to-prd | project + user-level | 一致 | 同一 | 同一 | なし | user-level 版と同内容 |
| handoff | project + user-level | 一致 | 同一 | 同一 | なし | user-level 版と同内容 |
| setup-matt-pocock-skills | project + user-level | 一致 | **中身違い** | **中身違い** | なし | SKILL.md ほか 4 ファイル |
| improve-codebase-architecture | project + user-level | 一致 | **中身違い** | **中身違い** | なし | SKILL.md、HTML-REPORT.md |
| tdd | project + user-level | 一致 | **中身違い** | **中身違い** | なし | project 版にだけ `deep-modules.md` / `interface-design.md` / `refactoring.md` がある |
| karpathy-guidelines | project | 一致 | — | — | なし | AGENTS.md §LLM コーディング規律 が名指し |

- project: **17 本**。うち実体コピーが不一致なのは **3 本**、`agents/openai.yaml` を持つのは **8 本**。
- user-level は Claude Code が `~/.claude/skills`(**38 本**)、Codex が `~/.agents/skills`(**43 本**)で、前者は後者の部分集合です
  (差の 5 本は `.agents` 側にのみ)。`~/.codex/skills` にも 1 本ありますが project とは無関係です。
  project と同名は **8 本**で、中身が違うのは **Claude 側 6 本 / Codex 側 5 本**(差は `prepare-commit` が Claude 側だけ `.lnk` であること)。名前の列挙は個人版に置きます。
- 入口の非対称は user-level 側にあります。Claude Code だけが global `CLAUDE.md` と memory 層を持ち、
  Codex の user-level 入口 `~/.codex/AGENTS.md` は空ファイルです。汎用 skill(`archify` を含む)は両者の user-level にあります。

## 6. 発見事項

**本書の範囲は発見と名指しまでで、修正はしません。** 次フェーズの PRD がここを引用して起票します。

### shadow

1. **project と同名の user-level skill が 8 本**あり、中身が違うのは **Claude 側 6 本 / Codex 側 5 本**です。
   同内容は `to-prd` と `handoff` だけ。とくに `grill-with-docs` の user-level 版には project 版にしかない
   `ADR-FORMAT.md` / `CONTEXT-FORMAT.md` が無く、起動されると ADR と用語集の書式が読まれません。
2. **`prepare-commit` だけが Claude 側で `.lnk`** で、`~/.agents/skills/prepare-commit` という第 3 の実体を指しています。

### 実体コピーのドリフト

3. **3 本が不一致**: `anet-audit`、`anet-housekeeping`、`grill-with-docs` の SKILL.md。
   **ドリフトは双方向**で、`anet-housekeeping` は `.agents` 側が新しく、「実行前にユーザーの了承を得る」への方針変更が `.claude` 側に届いていません。

### 欠落

4. **入口から領域の設計文書(1xx・2xx)へのリンクが 1 本も無い。** `AGENTS.md` が Markdown リンクで名指しする設計文書は
   [020](../design/020_user_guide_run.jp.md) と [030](../design/030_user_guide_analysis.jp.md) の 2 本だけで、
   §AI エージェントの作業ルール は「`docs/design/` 配下の関連ドキュメント」としか書いていません。
   [docs/design/README.jp.md](../design/README.jp.md) §3 の索引へのリンクもありません。
5. **[220](../design/220_atari_env.jp.md) にテスト章が無い。** 他の 1xx・2xx は「テストと拡張時の確認事項」章から
   `testdata/prdNNN/README.md` へ降りますが、220 は §5 AtariView → §6 ビルド統合 → §7 関連文書 で終わります。
6. **[160](../design/160_applications_and_tools.jp.md) だけ章題が「テストと**変更**時の確認事項」**で、他は「拡張時」です。
   章番号も 8(100/110/120/130/150/160)、9(140)、10(200)、12(210)とばらついています。
7. **`docs/design/*.jp.md` の `../optuna.md` リンクが存在しないパスを指す。** 実ファイルは
   [docs/design/optuna.md](../design/optuna.md) で、`*.en.md` 側の `optuna.md` は正しく解決します。jp 側 4 箇所がリンク切れです。
8. **PRD 冒頭の分担行が 5 通りに割れている。** 起票・精査・実装の工程は
   [docs/memo/README.md](../memo/README.md) §工程と担当 が正本になりましたが、個別 PRD の冒頭にはそれ以前の表記が残っています
   (`> 設計分担: Claude=設計/PRD、実装=Codex。` 10 件、同 `+ Run/commit=ユーザー` 4 件、
   `Claude/Codex=設計/PRD` 3 件、散文 3 件、注記付き 2 件の計 22 本)。正本ができたので、次フェーズで揃えるか落とせます。
   なお `AGENTS.md` には「Codex」の語が 1 件もなく、`implement-prd-with-docs` の `SKILL.md` と `PRD-IMPLEMENTATION.md` も
   担当中立で Claude / Codex の語を持ちません(どちらの枠が起動しても成立する設計です)。

### 死んだ動線(参照元が無い)

9. **[.agents/skills/README.md](../../.agents/skills/README.md) と [.claude/skills/README.md](../../.claude/skills/README.md)** —
   skill 一覧・起動記法・同期手順の正本ですが、常設文書からの参照元がありません。
   `AGENTS.md` §Agent skills が名指しするのは `docs/agents/` の 3 本だけです。
10. **[pages.yml](../../.github/workflows/pages.yml) / [release.yml](../../.github/workflows/release.yml) /
    [gemini_ai_review.yml](../../.github/workflows/gemini_ai_review.yml)** — 参照元がありません。
    設計文書から参照されている workflow は [windows-ci.yml](../../.github/workflows/windows-ci.yml) だけです。
11. **[inspect_run.py](../../viewers/metrics-tools/inspect_run.py) と
    [check_default_leaves.py](../../core/anet-core/testdata/prd072/check_default_leaves.py)** — `AGENTS.md` が
    前者をコマンドブロック内のパスとしてのみ、後者をファイル名だけ(パス無し)で書いており、リンクとして辿れません。
12. **[docs/experiments/README.md](../experiments/README.md)** — 規約からの動線がありません。
    `AGENTS.md` §実験記録と実効 config の保存ルール は `docs/experiments/<agent>/<env>/config/` を名指ししますが、
    記録の書き方を定めた README へは到達しません(現在の参照元は本書を生成する skill の調査対象リストだけです)。

### 機械化点(本書では実装しない)

13. **実体コピーの同期検査** — 現在は `diff -rq .agents/skills .claude/skills` を人が思い出して打つ運用です。
    上の 3 は、この検査が動線に載っていないために見つかりませんでした。再生成のたびに表 4 が検出します。
14. **CI のパス条件化** — `999_cicd_improvement` の範囲です。
15. **設計文書テスト章の章番号・章題の統一と 220 への追加** — 上の 5・6。次フェーズ。

### evidence gap

16. `~/.codex/` には `AGENTS.md`(0 バイト)と `skills/`(project 無関係の 1 本)しか確認していません。
    Codex が他に読む user-level 設定があるかは未確認です。
17. 表 1 の「誰が読む」は `AGENTS.md` の記述と skill の所在からの推定を含みます。実際の読み込みログでは検証していません。
18. 動線図の主経路にある「再グリル」と「調整の往復」は、PRD の改訂履歴行を根拠にした**実運用の再構成**です。
    規約に定義が無いため(上の 8)、工程の順序は実測ではなく利用者の説明に基づきます。
