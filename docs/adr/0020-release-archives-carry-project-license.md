# リリース zip 直下にプロジェクトのライセンス文書を置く

Apache License 2.0をanet-labのファーストパーティ部分へ適用し、実行用zipとDoxygen zipをそれぞれ単独で再配布できる状態にする必要がある。ADR 0017で採用したリポジトリ配置のミラーサブセットは維持する一方、法的文書はサブディレクトリ内ではなく各配布物の直下に置く。

## Decision

- リポジトリ直下の`LICENSE`と`NOTICE`を、実行用zipとDoxygen zipの直下へ同梱する。
- 実行用zipの`apps/`、`docs/`、`licenses/`は、ADR 0017の配置ミラーを維持する。
- `licenses/`は同梱する第三者ソフトウェアのライセンス専用とし、anet-lab本体の`LICENSE`と`NOTICE`を混在させない。
- Metrics Viewerのjarにも`META-INF/LICENSE`、`META-INF/NOTICE`、`META-INF/THIRD-PARTY.txt`を同梱する。

## Consequences

- 実行用zipのルート要素は`LICENSE`、`NOTICE`、`apps/`、`docs/`、`licenses/`となる。
- Doxygen zipのルート要素は`LICENSE`、`NOTICE`、`html/`となる。
- Release workflowはzip作成後に必須entryを検証し、法的文書が欠落した配布物を作成しない。
- ADR 0017の配置ミラー方針は継続し、法的文書だけを意図的な例外とする。
