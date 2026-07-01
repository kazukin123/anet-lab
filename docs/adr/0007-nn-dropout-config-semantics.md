# NN dropout 設定名と `Drop.p` 改名

NN モジュール内部の正則化を config から有効化できるようにするにあたり、設定名は module 内での適用位置と flavor が分かる名前にする。`ResBlock` では `res.dropout_rate` が channel dropout の `Dropout2d`、構造ブロック `Drop` では `dropout_rate` が任意点の要素 dropout を意味する。`TransformerEncoder` では hidden activations / residual branches 用の要素 dropout を `tf.hidden_dropout_rate`、attention weights dropout を `tf.attn_dropout_rate`、Stochastic Depth を `tf.droppath_rate` として分ける。既存の `Drop.p` は命名を揃えるため `dropout_rate` へ後方互換なしで改名し、旧キーを読む fallback や WARN は追加しない。

## Consequences

- `ResBlock` と `Drop` は `dropout_rate` を維持し、`TransformerEncoder` は `hidden_dropout_rate` を使うことで hidden/residual 用 dropout と attention/drop path を区別する。
- `Drop.p` が残った config は未読キーとして扱われ、`DropoutModule` は既定の `dropout_rate=0.0` で動く。repo 内 config は実装時に更新する。
