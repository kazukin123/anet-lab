# Run 分類は物理 workspace フォルダで行い、workspace を自己完結の箱とする

runs ディレクトリに異なる実験系列(長期 Run / env×アルゴリズムの組合せ / optuna 探索)の Run が混流し、分類の仕組みが必要になった。分類の主データを**物理フォルダ構造**(`apps/runner/workspaces/<ws>/{config/, runs/, optuna/}`)に置き、workspace を「Run の入力(workspace config)と成果物(runs、optuna storage/artifacts)を一体で束ねる自己完結フォルダ」とする。理由は、本リポジトリの Run 管理が「フォルダ＝Run の完全な実体」(削除=rmdir、退避/復帰=別ドライブ移動、Metrics キャッシュも Run フォルダ同梱でパス非依存: ADR 0015)を意図した設計であり、分類だけをフォルダ外(タグ DB・メタデータファイル)に置くとファイル操作との整合性問題を新たに作るため。optuna の storage(optuna.db)と Dashboard artifacts も workspace 内に置き、探索履歴が Run 群・基底 config と一緒に移動するようにする(optuna study=workspace、trial seed run=Run の割当)。

境界規約として、runner の `--config` 明示起動は「完全自己記述モード」とし workspace 解決(last-used/_default の後読みと runs_dir 導出)を行わない。workspace config は env 選択(=NN 既定値を含む)を持つため、trial override より後に読まれると探索パラメータを潰す。合成順の責任は config を生成する側(optuna ハーネス)が `$include` の並びで持ち、runner が暗黙に重ねることを禁じて二重適用を構造的に排除する。

## Considered Options

- **タグ/メタデータ方式**(フラット runs + sidecar or DB でフィルタ): 横断検索は得意だが、フォルダ移動・削除と分類データの同期という新しい整合性問題を生む。「フォルダ＝真実」と衝突するため却下。
- **Viewer 側の論理ビューのみ**(保存フィルタ/セッション): ディスク上の混流が解消せず、退避もフォルダ単位でできない。却下。
- **workspace 物理フォルダ(採用)**: 分類・退避・削除がすべて OS のフォルダ操作で完結。横断検索は弱いが、必要になれば Viewer の後続機能(任意パスアタッチ等)で補う。
- optuna storage について、**グローバル 1 db 温存**(全 study 横断の Dashboard が可能)も検討したが、workspace 退避時に探索履歴が置き去りになるため却下。跨ぎ集約は既存の `summarize-study --target-storage` で代替できる。

## Consequences

- workspace の分類・退避・削除・復帰は git や DB を介さずフォルダ操作のみ。`workspaces/` は `.gitignore` 対象。
- env 選択は共通 `_main.txt` から workspace config へ移り、コメントアウト切替が消滅する。env 未選択は既存の env.class_id 解決失敗で fail する(専用バリデーションは持たない)。
- Metrics Viewer は 1 プロセス 1 workspace(サーバ側 current 方式)。複数 workspace の同時閲覧は別ポート起動で行う。
- Dashboard の study 横断閲覧は workspace 内に限定される。
- 旧 study の再開(`00_last_run_study_args`)には `--workspace` の追加が必要になる(env include の移管のため)。
- 詳細設計は `docs/memo/046_workspace_10prd.md`。
