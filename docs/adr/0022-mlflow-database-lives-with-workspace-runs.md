# MLflow DB は選択 workspace の runs 配下に置く

workspace は config と成果物を一体で移動できる自己完結フォルダとする一方、`runs/` 直下は原則としてRunだけを置く。MLflow bridgeとMLflow serverは同じSQLite DBを共有し、監視対象のRun群とDBを同じ操作単位で選択・退避できる必要がある。この補助ツールDBに限り、`<workspace>/runs/mlflow.db`へ置くことを明示的な例外として採用する。

## Considered Options

- **`<workspace>/mlflow/mlflow.db`**: `runs/`の純粋性は保てるが、既存launcherとbridgeの運用契約を追加のbucketへ分け、Run群との対応が見えにくくなる。
- **user-data配下のglobal DB**: 全workspaceを横断できるが、workspaceを移動・削除しても分類データが残り、自己完結性と衝突する。
- **`<workspace>/runs/mlflow.db`（採用）**: 既存のRun監視rootとDBの選択が一つになり、workspace単位の退避・復帰で一緒に移動する。

## Consequences

- `41_mlflow_bridge.bat`と`42_start_mlflow.bat`は同じ選択workspaceの`runs/mlflow.db`を使用する。
- MLflow DBはRun directoryではないため、Run走査側は従来どおり`run_*/metrics.jsonl`だけを対象にする。
- この例外はMLflow DBだけに限定し、他のworkspace metadataを`runs/`へ追加する根拠にはしない。
- 詳細なworkspace契約は`docs/memo/done/046_workspace_10prd.md`を参照する。
