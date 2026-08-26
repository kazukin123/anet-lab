# NN 設定ツリーは Agent config prefix 配下に置く

QR/IQN のようなアルゴリズム概念の切替が `net.$`(グローバル NN 設定ツリー)と `quantile_mode`(Agent 設定)の離れた 2 行同期を要求し、揃え忘れが原因キーを名指ししない遠いエラーか静かな不整合になる問題(059 PRD §1・付録 B)への対応として、**NN 設定の読み口を Agent config prefix 配下(`<agent>.net.*`、例: `DefaultDQNAgent.net.*` / `ImageClsAgent.net.*`)へ移す**ことにした。これによりアルゴリズムプロファイル(`DefaultDQNAgent.@iqn`)が agent 設定と NN 配線選択(`net.$`)を単一 namespace 内に束ねられ、cross-namespace 同期そのものが表現不能になる。ブロックカタログ `net.block.[*]` は全 Agent 共有のグローバル定義として現位置に残す。プロファイル(qr/iqn 配線プロファイル等)とテンプレ定義は本 rename の対象外 — AutoMerge/Resolver は RHS プロファイルを LHS prefix へ複製するため、プロファイルの置き場所は読み口の移動に追随する必要がない。プロファイルのキー名 `@` 化(`net.@qr/@iqn`。配置は root のまま)は 059 Phase 1 の完了条件として別途行われ、agent 配下への移設(`<agent>.net.@iqn`。相対参照が可能になる)のみ任意である。rename 対象は「最終ツリーへの直書き行」のみで、実測 112 行(`net.$` 7+`net.branch.[slot]` 63+`net.body.output/$` 37+bat CLI override 5。コメントラダー込み・機械置換可)+`NetworkConfig` の構築 prefix 1 箇所と関連テスト。optuna 生成 config は net キーを扱っておらず影響ゼロ。

## Considered Options

- **(A) 読み口を Agent 配下へ移す(採用)**: 所有関係(NN は Agent の持ち物)が設定階層に現れ、アルゴプロファイルが agent 内で完結する。ImageClsAgent を含む「各 Agent が自分の net を読む」一般規則になる(MuZeroAgent の実最終ツリーは root の `net.rep/dyn/pred` にあり、保留中のため本 rename の対象外 — 再着手時に同規則へ寄せる。059 §3.5/§10)。代償は直書き行 112 行の rename と読み口変更。
- **(B') 読み口は現状維持し、アルゴを root 横断プロファイルにする**: `@algo_iqn.*` の絶対キー束を root 持ち上げで展開する。rename 最小だが、所有表現を失い(アルゴが agent の持ち物であることが構造から読めない)、root 持ち上げ機構が Phase 0 に必須へ戻る。棄却。
- **専用検証の追加のみ(現状維持+契約検証)**: 2 行同期は残したまま `quantile_mode` と `net.$` の整合を設定境界で検証する。概念ツリー不一致そのものが残り、059 の目標 1(標準経路では概念の部分選択が書けない)を達成しない。単独案としては棄却(検証自体は目標 2 として別途採用)。

## Consequences

- ALGO プロファイルが `DefaultDQNAgent.@iqn : net.$ = net.@iqn` のように agent 配下で完結し、アルゴ切替はチェーン 1 項の変更になる(059 §0.3 受け入れ指標 1)。`net.$` の参照先はプロファイルが root に残るため絶対参照で書く(プロファイルを `DefaultDQNAgent.net.@iqn` へ移設した後は相対 `@iqn` に書き換え可能)。
- ADR 0018 の Consequences にあった「QR との切替が quantile_mode と NN 設定の差し替えだけで完結する」(2 行切替を利点とする記述)は本 ADR が supersede する。ADR 0018 の bind `*` DAG・Body/Head 役割分担・検証責任境界は不変。
- `NetworkConfig` を使う全 Agent(DefaultDQN / ImageCls / Rainbow)が同一規則に従う。RainbowAgent は Agent 柔軟性実証として保持され(059 D22)、本規則の適用対象(現用 env 構成に使用なし)。
- 過去 Run の `config_data.txt` との自動 diff は net 系キーで断絶する(実効値の意味は不変、キー名のみ変更)。移行は golden comparison(旧/新 resolver の実効 leaf map 突合)で検証する(059 §8)。
- 仕様詳細は `docs/memo/059_config_concept_tree_alignment_10prd.md`(D19 / §3.5)。
