# anet::nn::Network GraphViz 仕様

## Context

`anet::nn::Network` は `NetworkBody`、複数の `NetworkBranch`、各 Branch 内の `NetworkStruct` / `NetworkBlock`、任意の `NetworkHead` から構成される。学習開始時に構築済みのネットワーク構造を Graphviz dot として出力し、Properties から組み上がった NN の確認とレビュー共有を容易にする。

出力粒度は 2 種類とする。

- 構造版: デフォルト `NetworkGraphVizConfig{}` で生成し、入力、Branch、Block、Body output、Head、Head output の接続を描く。
- 詳細版: Agent の `config_.nn_viz` を渡して生成し、設定に応じて TensorSpec、Block 詳細、param count、Branch 設定、Head 詳細を追加する。

ダンプは Agent 初期化時に 1 回だけ行う。Network 構造は学習中に変化しないため、再生成や target network の重複出力は行わない。

## API / Config

- `Network` 自体は `anet::graphviz::GraphViz` を継承しない。GraphViz 生成は `Network::MakeGraphViz(const NetworkGraphVizConfig&) const` に一本化する。
- 構造版は `Network::MakeGraphViz(NetworkGraphVizConfig{})`、詳細版は `Network::MakeGraphViz(config_.nn_viz)` で生成する。
- `MakeGraphViz()` の戻り値は `std::unique_ptr<anet::graphviz::GraphViz>` とし、dot 文字列化は戻り値の `GraphViz::ToDotString()` が担う。
- `NetworkGraphVizConfig` は `anet::nn` 名前空間の素の struct とし、設定ファイルからの読み込みは各 Agent Config が `ANET_READ_CONFIG(config_data, nn_viz.<field>)` で行う。
- Properties key は `DefaultDQNAgent.nn_viz.*` / `MuZeroAgent.nn_viz.*` に統一する。過去の dot 系 key は読まない。

`NetworkGraphVizConfig` のフィールド:

| field | default | 説明 |
| --- | --- | --- |
| `show_param_shapes` | `false` | `NetworkModule::GetCurrentConfigData()` の key/value を Block label に表示する。 |
| `show_param_count` | `false` | `NetworkBlock` 単位で `parameters(true)` の `numel()` 合計を `params` として表示する。 |
| `show_tensor_specs` | `false` | input node に TensorSpec の `shape` / `dtype` / `num_classes` を表示する。 |
| `show_branch_config` | `false` | Branch node に確定済み `auto_format` / `raw_keys` を表示する。 |
| `show_head_info` | `false` | Head output shape と Head details を表示する。 |
| `layout` | `"LR"` | `"LR"` または `"TB"` のみ許可する。その他は fail-fast。 |
| `cluster_branches` | `true` | Branch node に Branch 用 style を使う。`false` の場合は default style を使う。 |
| `float_precision` | `3` | `LabelData::AddAttr(..., precision)` に渡す浮動小数点表示桁数。負値は fail-fast。 |

## Graph Structure

- input node は `input:<key>` を title に持つ ellipse node とする。`show_tensor_specs=true` のときだけ TensorSpec を属性行として表示する。
- Branch node は `branch:<name>` を title に持つ。Branch node 内に `bind` 行は表示しない。`show_branch_config=true` のときだけ `auto_format` と `raw_keys` を属性行として表示する。
- Branch へ入る edge label は `LabelData::SetText(bind_key)` を使う plain text とし、table label にはしない。
- Block node は `shape=plain` とし、node 外枠を消して HTML table label の枠だけを残す。構造版・詳細版の両方で同じ style を使う。
- Body output node は `body output:<key>` を title に持つ緑系 node とし、`NetworkConfig::output_keys` の対応元 Branch を `from` に表示する。
- Head node は head が存在する場合に追加する。`HeadGraphVizInfo::type` が空でなければ title はその type のみとし、`head:` prefix は付けない。未対応 Head では title は `Head` のままにする。
- Head output は `HeadGraphVizInfo::outputs` の各 entry から `output:<key>` node を作り、`Head -> output:<key>` として構造版でも描く。Body output と区別するため、Head output node は赤系 style を使う。
- `show_head_info=true` のときだけ、Head output node に shape を表示し、Head node に `HeadGraphVizInfo::details` を属性行として表示する。
- Head が存在しない場合は Body output node が最終出力になる。Head が存在しても outputs を自己申告しない場合は Head までを描き、追加の Head output node は作らない。

## Data Sources

- dot 生成は `NetworkBody` / `NetworkBranch` / `NetworkStruct` の読み取り専用 getter で構築済み構造を辿る。
- Block 詳細は `NetworkModule::GetCurrentConfigData()` から取得する。Network 側は module 型で分岐せず、返された `ConfigData` の key/value をそのまま label に追加する。
- 未対応 module は空の `ConfigData` を返し、Block title のみを表示する。
- Conv / Linear など lazy 構築される module では、未構築時点で未確定の `in_channels` / `in_features` は表示しない。dummy forward による warm-up はこの仕様の範囲外。
- Head 詳細は `NetworkHead::GetGraphVizInfo()` が返す `HeadGraphVizInfo` から取得する。`type`、`outputs`、`details` は Head 自身の自己申告とする。

## Logging

- `MetricsLogger::Log(const std::string& tag, const anet::graphviz::GraphViz& viz)` を Step なし GraphViz 出力として使う。
- 出力先は `run_dir/dot/<safe_tag>.dot` とし、既存の tag sanitize ルールを使う。
- DefaultDQN は main network のみ `net.structure` / `net.detail` を出力する。
- MuZero は representation / dynamics / prediction の各 network を構造版・詳細版で出力する。
- target network は構造が同一なので出力しない。

## Out Of Scope

- dot-to-png 変換 bat の変更。
- Tree への本格的な subgraph cluster サポート追加。
- DefaultDQNAgent / MuZeroAgent 以外の Agent への自動出力追加。
- dummy forward による lazy module 入力次元の確定。
- 学習中の再ダンプや重み差分ダンプ。
