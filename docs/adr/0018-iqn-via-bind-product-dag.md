# IQN を bind `*` DAG 汎用化と最小 Head で表現する

IQN（Implicit Quantile Networks）の導入にあたり、専用ネットワーククラス（taus 埋め込み・fusion を内包した IQNNetwork）を実装するのではなく、既存の branch DAG 設定へ要素積演算子 `*`（feature-last elementwise product、rank 不一致は batch 直後への singleton 挿入で吸収）を追加し、taus ブランチ（CosineEmbedding + 既存 Linear/ReLU）と fusion ブランチ（`bind = features * tau_embedding`）を**ユーザー設定で**構成することにした。C++ 側の新規実装は CosineEmbedding モジュール・bind `*`・最終射影だけの IQNHead / IQNDuelingHead・TauGenerator に最小化される。taus は固定キー `kKey_Taus = "taus"` で NN 入力 TensorDict に載せ、iqn 有効時は input_specs へ能動追加する。検証は AGENTS.md「汎用機構と利用側の責任境界」に従い、フレームワークは局所契約のみを見る: NN 一般の「未使用入力 WARN」（構築時に input_specs と bind を突合する診断。到達性は保証しない）と、IQN Head factory の rank・B/K 検証（Head 自身の局所入力契約として初期化時 fail-fast。典型的な fusion 書き忘れは結果として検出されるが、taus への依存を証明するものではない）。DAG の推移的到達性検証は行わず、taus が最終出力へ意図どおり寄与することは NN 設定者の責任とする。

## Considered Options

- **IQN 専用ネットワーククラスを実装**: cos 埋め込み・積・MLP を C++ に固定化。直接的で設定は簡単だが、既存の branch DAG / block カタログ / config_profile / GraphViz 可視化と並ぶ第二のネット構成経路が生まれ、fusion 構造（積の位置・後段 MLP の深さ・Dueling 分岐）の実験に毎回 C++ 変更が要る。棄却。
- **bind `*` による DAG 汎用化（採用）**: fusion 構造が設定ファイルで見え、既存部品（Linear/ReLU/Dropout/config_profile）を tau 埋め込み側にもそのまま再利用できる。将来の別種 fusion（gating、FiLM 的変調）にも `*` が転用可能。代償として bind 記法が「積」を含む公開設定契約になり、パース・依存解決・GraphViz・ToJson の全消費箇所が項リスト構造に追従する必要がある。

## Consequences

- `bind = a * b, c` が公開設定契約になる（`*` は `,` より優先、(raw) は factor 単位、3 項以上は左結合）。後から専用クラス方式へ戻すと設定互換が壊れるため、実質不可逆。
- ネット構成ミス（taus 未 bind、次元不一致）はコンパイルではなく構築時/実行時エラー・WARN として現れる。未使用入力 WARN は書き忘れを知らせる診断、Head factory の rank・B/K 検証と batch 検証は局所契約の fail-fast であり、いずれも構成の意味的な正しさ（taus の到達性・寄与）は保証しない。意味的妥当性は NN 設定者の責任で、設定の自由度と引き換えに「設定を書けるのは仕様を理解した人」という前提が強まる（taus 以外由来の rank 3 誤結線は検出されない）。
- Head は「最終的な Q 分位の出力層のみ」という既存の Body/Head 役割分担が IQN でも維持され、QR との切替が quantile_mode（none/qr/iqn）と NN 設定の差し替えだけで完結する。
- 仕様詳細は `docs/memo/001_iqn_10prd.md`。
