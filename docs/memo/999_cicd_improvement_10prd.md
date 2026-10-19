# CI/CD 改善（ビルド・テスト自動化、定点ジョブ、LLM 補助）暫定 PRD

> 起点: 2026-09-17 の「トークン残量で流す空き時間スキル群」の整理。候補にあった「全構成ビルドとテストの既知失敗台帳」は、ローカルの LLM トークンを使う仕事ではなく CI の仕事だと判断し、CI/CD 改善として切り出した。
> 関連: [windows-ci.yml](../../.github/workflows/windows-ci.yml)（現行ビルド）、[gemini_ai_review.yml](../../.github/workflows/gemini_ai_review.yml) と [push_review.py](../../.github/scripts/push_review.py)（現行 LLM レビュー）、[release.yml](../../.github/workflows/release.yml)（配布物。043 で整備済み）、[pages.yml](../../.github/workflows/pages.yml)（docs 配信）、[AGENTS.md](../../AGENTS.md) のテスト実行規約。

## Context / Problem Statement

現行の GitHub Actions は 4 本あるが、開発の日常で回っているのは配信系の 2 本（release、pages）だけで、品質ゲートとしては機能していない。以下は 2026-09-17 にワークフロー定義と CMake を読んで確認した事実である。

### 1. ビルド CI が起動しない

[windows-ci.yml](../../.github/workflows/windows-ci.yml) のトリガは `pull_request: branches: [ ]` と `workflow_dispatch` である。branches が空リストのため PR では起動せず、実質は手動起動のみになっている（空リストの厳密な挙動は要確認だが、自動で走った形跡はない）。main へ直接 push する運用が中心なので、PR トリガだけでは入口にならない点も併せて見直しが要る。

### 2. テストを実行していない

同ワークフローは Configure と Build で終わっており、`ctest` を呼ぶステップが無い。CMake には `enable_testing()` と 5 本の `add_test`（anet-core-test と 4 つの Env テスト）が登録済みで、実行体は既に揃っている。テストの実行はローカルの手作業に依存し、既知の失敗テストは台帳を持たず記憶で管理している。

### 3. 毎回重い

CUDA Toolkit のインストールをキャッシュ無しで毎回行い、LibTorch は Release と Debug の 2 本を取得し、wxWidgets はソースからビルドする。マトリクスは Debug、RelWithDebInfo、Release の 3 構成で、wx を構成ごとに別々にビルドしてキャッシュする。1 回の所要時間は未計測。

### 4. LLM レビューが残らない

[gemini_ai_review.yml](../../.github/workflows/gemini_ai_review.yml) も `push: branches: [ ]` で手動起動のみ。[push_review.py](../../.github/scripts/push_review.py) は commit の差分または全ツリーを 60 万字単位に分割して `gemini-3-flash-preview` へ投げ、チャンク間で 70 秒待つ（無料枠の TPM 対策）。結果は Step Summary に書くだけで、PR コメントにも issue にも台帳にも残らない。

### 5. 定点情報の生成が人力

コード規模、git 履歴、設定キーの棚卸しなどの定点情報は、思い出した時に手で集計している。決定的に生成できるものなので、CI の定期ジョブに向く。

**したがって欲しいのは**: 人が手を動かさなくても「ビルドが通るか、テストがどこで落ちているか、規模と履歴がどう推移しているか」が定期的に残る仕組み。

## 効く事実

1. テスト実行体と `add_test` は揃っている。CI 側は `ctest` 呼び出しと結果の収集だけでよい。
2. Catch2 は `--reporter junit` で JUnit XML を出せる。失敗テストの機械的な台帳化に使える。
3. ホスト型 runner（windows-2022）に GPU は無い。CUDA 依存のテストは CPU 経路だけを通すか、self-hosted runner が要る。
4. 実験機を self-hosted runner にすると実験と競合する。実験機は 1 時間で最大 8% の throughput ドリフトが実測されており、性能比較中の CI 実行は結果を汚す。
5. LibTorch と wxWidgets のキャッシュ機構は既にあり、release.yml とキーを共有している。
6. Gemini の無料枠は TPM と RPD の上限が厳しく、70 秒待ちがその制約の現れである。LLM 補助は「小さい入力を少数回」に限定して設計する必要がある。
7. ローカル側では、同日に整理した空き時間スキル群が `reports/registry/findings.jsonl` を指摘の台帳として使う予定である。CI の失敗テスト台帳を同じレコード形式（category=test-failure）で書けば、派生ビューを共有できる。
8. 統計スクリプトはローカルの anet-stats と同一実装（`reports/tools/stats.py` 予定）を CI から呼ぶだけでよい。cloc は既に利用実績がある。

## 提案

### 骨子: 3 層に分けて段階導入

| 層 | 内容 | 起動 | LLM |
|---|---|---|---|
| L1 ゲート | 構成を絞ったビルドと CPU テスト。JUnit を artifact に残す | PR と main への push | 無し |
| L2 定点 | 統計スクリプトと設定キー・メトリクスのカタログ生成。失敗テスト台帳の更新 | 週次 schedule と workflow_dispatch | 無し |
| L3 補助 | Gemini による差分レビューの永続化、失敗テストの一次分類 | main への push と L2 の後段 | Gemini |

L1 は既存 windows-ci.yml の改修で済む。L2 は決定的なスクリプトなのでローカルの空き時間スキルと実装を共有する。L3 だけが LLM 依存で、API 制約とコストの影響を受ける。

### L1 ゲート

- トリガを `pull_request` と `push: branches: [main]` にする。
- マトリクスを Release と Debug の 2 構成に絞り、RelWithDebInfo は週次に回す（Q8）。
- CUDA Toolkit のインストールをキャッシュするか、必要なサブパッケージだけに絞る（Q9）。
- Build の後に `ctest --output-on-failure` を追加し、Catch2 の JUnit 出力を artifact として保存する。GPU 必須のテストは Catch2 の tag でラベル付けし、ホスト型 runner では除外する（Q2）。
- 失敗時は Step Summary に失敗テスト名の一覧を出す。

### L2 定点

- `reports/tools/stats.py` を週次で実行し、`reports/stats/<date>.json` と trend を生成する。
- 設定キーの 3 集合差（コードが読むキー、設定ファイルのキー、文書のキー）とメトリクス群の対応表を生成する。
- JUnit を `findings.jsonl` のレコード形式に変換し、first_seen と last_seen を維持して台帳を更新する。
- 生成物の置き場は未決（Q3）。候補は artifact のみ、専用ブランチへの bot commit、Pages 配信。

### L3 補助

- push_review.py を「差分だけ、PR コメントか commit コメントへ永続化、既存台帳との重複排除」に改修する。
- 失敗テストの新規分に対し、ログ抜粋を Gemini に渡して「回帰 / 環境起因 / flaky 疑い」の一次分類を付ける。
- docs/design 英訳の CI 化は候補にとどめる。用語の固定が要るので、ローカル側の anet-translate-docs の品質と比べてから判断する。

## 未決

| # | 論点 | 選択肢 |
|---|---|---|
| Q1 | トリガ方針 | PR のみ / PR と main push / main push のみ。PR を経由しない push が中心の運用では main push が実質の入口 |
| Q2 | ホスト型 runner で回すテスト範囲 | CPU 経路の全テスト / tag で GPU 依存を除外 / smoke のみ |
| Q3 | 生成物の永続化 | artifact のみ / `ci-reports` ブランチへ bot commit / Pages 配信。main へは bot commit しない（git 操作は人が行う方針と衝突しない場所に限る） |
| Q4 | self-hosted GPU runner | 置かない / 実験機に置き時間帯で制限 / 別機 |
| Q5 | LLM プロバイダ | Gemini 無料枠（現行、起点の想定） / GitHub Models / 有料 API。制約は TPM と RPD、secret の扱い、fork からの PR で secret が使えない点 |
| Q6 | ローカル台帳との統合 | 同一 jsonl 形式を共有 / CI 専用形式。共有が本命 |
| Q7 | 実行時間と minutes 予算 | 未計測。まず L1 を 1 回手動で回して計測してから決める。Windows runner は minutes 消費が Linux の 2 倍 |
| Q8 | マトリクス縮小 | 3 構成維持 / Release と Debug / Release のみ + 週次フル |
| Q9 | CUDA Toolkit の高速化 | action のキャッシュ機能 / サブパッケージ限定 / CPU-only ビルド構成を追加 |

## 受入条件（暫定）

1. PR または main への push で、ビルドとテストが自動で走り、失敗テスト名が Step Summary で読める。
2. 失敗テストが台帳に first_seen 付きで残り、直った時点で last_seen が止まる。
3. 定点統計が週次で生成され、規模と履歴の推移が 1 つの表で追える。
4. CI の 1 回の所要時間が計測され、Q7 で決めた上限以内に収まる。
5. Gemini レビューが Step Summary 以外の場所に残り、後から参照できる。

## スコープ外

- ローカルの空き時間スキル群。CI とは台帳形式とスクリプトだけを共有する。
- リリース配布物の作成。043 で整備済み。
- Pages 配信。既存の pages.yml のまま。
- 性能 A/B の CI 化。実験機ドリフトのため CI では比較しない。
- Doxygen 生成。時間だけかかり、品質ゲートにも定点情報にも寄与しないため対象外。
