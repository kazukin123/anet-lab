# NN Head の最終射影と出力変換の責務分離

2026-09-30、[NoisyNet PRD](084_noisynet_10prd.md) のグリルから切り出した暫定 PRD。番号は 999 のままとし、方式・対象 Agent・公開契約は未裁定とする。別途グリルしてから正式化と実装範囲を決める。

## 1. 問題と目的

新しい射影機能を追加するとき、Body の Linear と Head 内の Linear で、構築・設定・検証の経路が分かれている。NoisyNet の検討では両方へ同じ演算を適用する必要があり、演算の共用に加えて、射影の所有や設定経路まで揃える価値があるかが論点となった。

本 PRD は、Head の最終射影と出力変換をどう分担すれば、機能追加時の重複実装・設定・検証を減らせるかを比較する。現行構造は意図された設計であり、Head が重みを持つこと自体を不具合とは扱わない。NoisyNet の演算共用だけで課題が解消する可能性も含め、構造変更の必要性から評価する。

NoisyNet は既存 Head が最終射影を所有する形で進める。本 PRD の裁定・実装を NoisyNet の前提条件にしない。

## 2. 現状と用語

本書では、Head が持つ処理を次の二つに分けて議論する。これらは比較のための整理であり、新しい公開クラスや設定名を確定するものではない。

| 責務 | 意味 | 例 |
|---|---|---|
| 最終射影 | 学習可能な重みで特徴量を、行動数・分位数・クラス数に対応する値へ変換する | DQN の行動別出力、Dueling の value / advantage の射影、分類 logits の射影 |
| 出力変換 | 射影後の値を、利用側が期待する軸・shape・意味・出力キーへ組み立てる | Dueling 合成、QR reshape、IQN 軸変換、分位平均、TensorDict の出力キー |

現在の DefaultDQN の主要な形は次のとおりである。B は batch、A は行動数、N は QR の分位数、K は IQN の分位点数を表す。

| Head | 最終射影と出力変換 |
|---|---|
| TD | 特徴量を A 個の値へ射影し、`q [B,A]` を出す |
| QR | A×N 個の値へ射影し、`q_dist [B,A,N]` へ reshape、分位平均で `q` を作る |
| IQN | `[B,K,D]` を `[B,K,A]` へ射影し、`q_dist [B,A,K]` へ軸変換、分位平均で `q` を作る |
| Dueling | value / advantage を別に射影し、行動軸の平均を引く合成を行う。TD・QR・IQN それぞれの shape と補助出力を持つ |

Body Linear は NN module の Factory・設定を通して作られる。一方、Head の最終射影は Head の構築と `head_init` 等の経路で作られる。演算が Linear であることは、同じ設定部品を使っていることを意味しない。

`NetworkHead` の interface は重みの所有を要求しない。既存の `PassThroughHead` は重みを持たないため、「Head は必ず重みを所有しなければならない」という interface 上の制約はない。

ただし、最終射影を Head に残すのは [ADR 0018](../adr/0018-iqn-via-bind-product-dag.md) の意図的な判断である。射影を Body へ移す案を採る場合は、この所有境界の判断を変更することを明示する。IQN の τ 融合を Body の bind / product DAG で表す判断まで、理由なく撤回するものではない。

現在は Network が Head へ渡す特徴量を FP32 化し、Head 全体を autocast の外で実行する。通常の `Forward` と診断用 `TensorDictFunction` の双方で、最終射影を含めて精度を保護している。

## 3. User Stories

- 射影機能を追加する実装者として、同じ演算を Body と Head に組み込むための重複実装や検証を減らしたい。
- NN を構成する実験者として、射影の設定位置と出力次元の決まり方を一貫して理解したい。
- Agent を利用する実験者として、構造整理によって Q・分位・logits の shape、出力キー、精度、診断の意味が意図せず変わらないことを確認したい。
- 保守する実装者として、所有する parameter が optimizer・同期・保存から漏れない責任境界を持ちたい。

## 4. 比較する案

いずれも未採用の候補である。案の優劣は、NoisyNet の演算共用後に残る問題と、移行・保守の費用を比較して決める。

| 案 | 狙い | 主な論点 |
|---|---|---|
| A. 射影を Body へ移す | Head を出力変換に限定し、最終射影を既存 NN 設定で扱う | 行動数・分位数・クラス数からの出力次元解決、Dueling の複数出力、最終射影を含む FP32 境界の再設計が必要 |
| B. Head が共通の射影部品を所有する | 現行の所有境界を保ち、最終射影を差し替え可能にする | Body と演算・初期化・設定・Factory・検証をどこまで共用するか。新しい部品の interface 自体が重複を増やさないか |
| C. 現行構造を維持する | NoisyNet で必要な演算共用にとどめる | 共用後も重複や設定上の不便が残るか。残る費用が構造変更の費用より小さいなら分離を採用しない |

A は Head を重み無しにする方向、B は重みの所有を残して共通部品を持たせる方向であり、同じ案ではない。C も、構造変更が必要かを評価するための正式な比較対象とする。

## 5. 別途グリルする事項

### 5.1 所有者・設定・出力次元

- 最終射影の parameter をどの module が所有するか。Agent が所有する network Resource の内側で、Body / Head の境界をどう置くか。
- 最終射影の初期化・NoisyNet 等をどこに設定し、既存 NN 設定とどこまで共用するか。
- `action_dim`、QR の分位数、クラス数等の正本をどこに置くか。手書きの出力幅と Agent が持つ値を二重管理せず、矛盾を早い段階で検出できるか。
- Dueling の value / advantage と、IQN の τ 軸を含む入力をどの粒度で宣言・検証するか。

### 5.2 最終射影を含む FP32 保護

射影を Body へ移すと、現在の「Head の入口から FP32」という境界だけでは、その射影を保護できない。低精度で計算した射影結果を後から FP32 へ cast しても、失われた精度は戻らない。

最終射影・Dueling 合成・分位平均のどこを FP32 とするかを先に定め、その境界を通常 forward と診断経路の双方で実現する方法を比較する。Body の他の部分の AMP 方針や NoisyNet の精度設定と、暗黙に衝突させない。

### 5.3 interface と診断

- `NetworkHead` / Factory が受け取る情報と、出力 shape の解決・検証の責任をどう分けるか。
- 通常の `Forward` と `TensorDictFunction` が同じ出力変換・精度契約を満たす構造にできるか。
- GraphViz に最終射影と出力変換をどう表示するか。ノードや parameter 名の変更を、利用側へどう反映するか。
- feature / readout の診断でどの parameter を集計するか。Body へ移動したという理由だけで、最終射影の所属や診断の意味が意図せず変わらないか。

### 5.4 対象 Agent と移行

- DefaultDQN の TD・QR・IQN、Dueling に限定するか、Rainbow・ImageCls・MuZero まで揃えるか。
- 他の Agent の Head が担う役割の差を保ったまま、どこまで共通化できるか。
- 設定、parameter 名、checkpoint、Factory、診断、テスト、現行ドキュメントのどこを同時に移行するか。
- 採用時は現行のクリーンブレーク方針に従う。過去の Run artifact は当時の記録として保持し、念のための alias・互換分岐・自動変換を積み重ねない。具体的な互換要求がある場合だけ、対象・期間・削除条件を別途裁定する。

### 5.5 費用に見合う効果

NoisyNet の共用化後にも、Body と Head の二重実装・二重設定・二重検証がどれだけ残るかを確認する。新しい射影機能を一つ追加する具体例で、変更箇所と確認箇所が減るかを比較する。

Head を重み無しにすること自体、クラス数を減らすこと自体は成功条件にしない。新しい公開 interface や次元解決機構が増える場合は、それも費用に含める。

## 6. 検証候補

以下は採用案を評価するための候補であり、未裁定の方式に対する実装指示ではない。

| 領域 | 確認すること |
|---|---|
| 数値 | 対応する同じ重み・入力・ノイズで、変更前後の出力と入力・parameter 勾配が一致する |
| shape・意味 | TD・QR・IQN と Dueling の軸、分位平均、TensorDict の出力キー・補助出力が契約どおりである |
| 精度 | 通常 forward と診断用 `TensorDictFunction` の両方で、最終射影を含む精度契約を満たす。近接した行動価値の比較も確認する |
| parameter 網羅性 | clone、hard / soft copy、snapshot、保存・読込、optimizer に全 parameter が一度ずつ正しく含まれる |
| 観測・表示 | GraphViz、parameter 名、feature / readout の診断対象が裁定した契約と一致する |
| fail-fast | 入出力 shape、必要キー、出力次元の不整合を、情報が揃う最も早い自然な境界で検出する |
| NoisyNet | ε の保持・共有・役割分離と、診断の非干渉が構造変更後も維持される |
| 拡張時の費用 | 新しい射影機能の追加に必要な重複実装・設定・検証箇所と、新たに増える機構を比較できる |

## 7. 正式化への昇格条件

次を裁定し、選択しなかった案の理由も残してから、正式な PRD・実装計画へ進む。

1. 採用案、または C を選んで構造分離を行わない判断。
2. 対象 Agent と Head、今回含める機能と対象外。
3. 射影の所有者、設定位置、出力次元の正本、公開 interface と TensorDict 契約。
4. 最終射影を含む精度境界と、通常 forward / 診断経路での実現方法。
5. 設定・parameter 名・checkpoint・診断・表示・文書の移行範囲と方法。
6. 検証対象と、拡張時の重複実装・設定・検証箇所が減ることを判断できる比較結果。

今回の起票では実装方式を確定しない。未裁定の構造や用語を、現行仕様書や `CONTEXT.md` に確定仕様として追加しない。

## 8. 参照する現行仕様・実装

現在の仕様は設計文書を正本とし、ADR は所有境界を選んだ理由を確認するために参照する。以下は候補案の採用を意味しない。

- [NN 設計](../design/130_neural_networks.jp.md)、[DQN 系 Agent 設計](../design/200_dqn_agents.jp.md)。
- [ADR 0018: IQN を bind / product DAG で表す](../adr/0018-iqn-via-bind-product-dag.md)。
- [NetworkHead interface](../../core/anet-core/include/anet/nn.hpp)、[Body Linear と module Factory](../../core/anet-core/src/nn_modules.cpp)。
- [DQN Head 群](../../core/anet-core/src/dqn_based_heads.cpp)、[PassThroughHead・汎用 LinearHead](../../core/anet-core/src/nn_heads.cpp)。
- [Network の FP32 境界と診断](../../core/anet-core/src/nn_impl.cpp)。
- [NoisyNet PRD](084_noisynet_10prd.md): 現行 Head を維持して演算を共用する、独立した先行課題。
