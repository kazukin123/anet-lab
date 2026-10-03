# anet::nn::Network GraphViz 実装計画

## Summary

`004_nndot_10spec.md` に従い、`anet::nn::Network` を Graphviz dot として出力できるようにする。構造版は `Network::MakeGraphViz(NetworkGraphVizConfig{})`、詳細版は `Network::MakeGraphViz(config_.nn_viz)` で生成する。Agent 側の設定メンバ名と properties key は `nn_viz` に統一する。

## Key Changes

- `NetworkGraphVizConfig` を `anet::nn` に追加し、`DefaultDQNAgentConfig` / `MuZeroAgentConfig` には `nn_viz` メンバとして持たせる。
- `Network` 自体は `anet::graphviz::GraphViz` を実装せず、`MakeGraphViz(cfg)` が `cfg` に従った `anet::graphviz::GraphViz` を返す。
- dot 生成のため、`NetworkBody` / `NetworkBranch` / `NetworkStruct` に読み取り専用 getter を追加し、構築済み `NetworkModule` を辿れるようにする。
- `NetworkModule::GetCurrentConfigData()` を追加し、主要 module が自身の現在値を返す。未対応 module は空を返す。
- `HeadGraphVizInfo` と `NetworkHead::GetGraphVizInfo()` を追加し、Head の `type` / `outputs` / `details` を Head 自身が自己申告できるようにする。
- `LabelData::SetText()` を追加し、Branch edge label を plain text で表示できるようにする。
- `LabelData::AddAttr(const std::string& key, const T& value, int precision = 3)` に変更し、浮動小数点表示の桁数を呼び出し側から指定できるようにする。
- `MetricsLogger::Log(tag, GraphViz)` を Step なし出力として追加し、`run_dir/dot/<safe_tag>.dot` に保存する。

## 10spec 外で明記する実装ルール

- properties key は `DefaultDQNAgent.nn_viz.*` / `MuZeroAgent.nn_viz.*` とする。過去の dot 系 key は使わない。
- `net.detail` は常に生成するが、詳細項目は `config_.nn_viz` に従う。`show_param_shapes` / `show_param_count` をコードで強制 true にしない。
- `layout` は `LR` / `TB` のみ許可し、それ以外は fail-fast にする。`float_precision < 0` も fail-fast にする。
- `show_param_count` は `NetworkBlock` 単位で `parameters(true)` の `numel()` 合計を generic に表示する。
- Branch node 内の `bind` 行は構造版・詳細版とも表示しない。Branch へ入る edge label は bind key だけを plain text で表示する。
- `show_branch_config=true` のときだけ、確定済み Branch 設定の `auto_format` / `raw_keys` を Branch node に表示する。
- Block node は構造版・詳細版とも `shape=plain` にして、HTML table label の枠だけを残す。
- Head の type は Head node title に表示し、`head:` prefix は付けない。
- Head outputs は `Head -> output:<key>` の別 node として構造版でも表示する。output shape と Head details は `nn_viz.show_head_info=true` のときだけ表示する。
- Head output node は body output node と区別できるように、赤系の専用 style にする。
- `NetworkGraphVizConfig::float_precision` は `LabelData::AddAttr(..., cfg.float_precision)` に渡し、詳細ラベル内の浮動小数点値の表示精度に使う。
- DefaultDQN は main network のみ、MuZero は representation / dynamics / prediction の各 network を構造版・詳細版で出力する。target network は出力しない。
- dot-to-png bat、core 実装、テストコードはこの docs 同期では変更しない。

## Test Plan

- `LabelData::AddAttr` の既定 precision が 3 のまま既存出力を保ち、明示 precision 指定で浮動小数点表示が変わることを確認する。
- `LabelData::SetText()` が HTML table ではなく quoted plain label を返すことを確認する。
- `Network::MakeGraphViz(NetworkGraphVizConfig{})->ToDotString()` が構造のみを出し、`rankdir="LR"`、入力、Branch、Block、Body output、Head、Head output の接続を含むことを確認する。
- Branch node に `bind` 行が出ないこと、Branch edge label が plain text になることを確認する。
- `show_branch_config=false` では `auto_format` / `raw_keys` が出ず、`show_branch_config=true` の詳細版で表示されることを確認する。
- `Network::MakeGraphViz(detail_config)->ToDotString()` で `show_param_shapes` / `show_param_count` / `show_tensor_specs` / `float_precision` がラベルに反映されることを確認する。
- Block node が構造版・詳細版とも `shape=plain` になることを確認する。
- Head の type と `Head -> output` node が構造版に出て、Head output node が赤系 style になることを確認する。
- `show_head_info=false` では output shape / Head details が出ず、`show_head_info=true` の詳細版で表示されることを確認する。
- `nn_viz` config 読み込みで properties key が反映され、過去の dot 系 key は読まれないことを確認する。
- `MetricsLogger::Log(tag, GraphViz)` が `dot/<safe_tag>.dot` を生成することを確認する。
- Debug build、`anet-core-test.exe`、`git diff --check`、CRLF チェック、必要に応じて `dot -Tsvg` 変換を実行する。

## Assumptions

- 既存の未追跡・未コミット変更には触れない。
- 実装対象は `core/anet-core` と必要なテストに限定し、dot-to-png bat は変更しない。
