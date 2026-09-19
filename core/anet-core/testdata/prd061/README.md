# PRD061 検証

Actor カタログの公開契約は通常テストの `[prd061]` で検証する。設定からの Actor 生成、scalar 購読、学習 counts、独立 RNG、IQN の異なる K、clone/shared、未定義参照、dormant、EvalPanel の参照タグ、Agent 固有の検証を含む。

`[shipped_actors]` は現用7環境×online/batchrunの設定を解決し、train・評価タグ・休眠タグ・EvalPanelのActor参照を検証する。EvalPanelタグ・Actor・deviceをテスト用に補正しない。MuZeroのCPU deviceとActor由来のtau、共通full metricsのtauも対象とする。

## 移行比較

`[prd061-capture]` は隠しテストで、PRD072 の固定17入力を現用設定に適用し、解決済み文字列値と Agent constructor の既定補完後の typed 設定を JSON に保存する。出力先は環境変数 `ANET_PRD061_OUTPUT` で指定する。既存の JSON は上書きしない。Env・Agent の学習資源は生成しない。

変更前、P1 後、P2 後を異なる出力先へ採取する。`compare_p1.py` は Runner 設定ルートの改名前後の全キー・文字列値および typed 設定を比較する。`compare_typed.py` は旧方策からカタログへの対応表に従い、未使用フィールドも含めて旧 typed 値を比較し、新規カタログフィールドを別途報告する。固定17入力の欠落や既存値の差分は失敗とする。PRD072 の golden は更新しない。

今回の採取済み入力に対する再比較:

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd061/compare_typed.py .scratch/prd061/p1/typed .scratch/prd061/p2/typed-final --report .scratch/prd061/p2/typed-comparison.json
```

これらの入力はローカル検証 artifact であり、リポジトリへ巨大な実行体・checkpoint を追加しない。入力を採取していない checkout では、差分比較を合格扱いにしない。

## 固定入力推論

`[prd061-inference]` は `ANET_PRD061_INFERENCE` が指す JSON manifest を読み、既存 Run の EnvSpec、指定した解決済み設定と checkpoint から固定観測を作り、greedy / fixed τ の Q 値と行動を保存する。manifest の `baseline` を指定すると保存済み行動との完全一致、Q 値の rtol/atol 各 1e-6 を検査する。出力ディレクトリの再利用は禁止する。

旧 Run artifact は変更しない。今回の Atari は既存の学習済み checkpoint、DropMerge は現行ネットワーク定義で変更前に新規作成した checkpoint を使う。後者は学習済み checkpoint の等価性証明ではない。採取時の識別情報は `.scratch/prd061/before/identifiers.json` と実装メモに記録する。
