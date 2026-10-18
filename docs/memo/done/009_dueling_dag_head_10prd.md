# DefaultDQNAgent Dueling: V/A ストリームを body DAG で分岐できるようにする

## Context（なぜやるか）

古典的 Dueling DQN（Wang et al. 2016）は、共有バックボーンの後で **V 用と A 用に独立した隠れ層**（FC+活性）へ枝分かれし、それぞれ最終射影する。
しかし現状の `DuelingHead` / `QuantileDuelingHead` は **V/A 各ストリームが「単一 Linear 一枚」固定**で、`features` から直接 V→1・A→A（QR は V→N・A→A*N）へ射影しているだけ。body 末尾に共有隠れ層（`Linear_2048 > SiLU`）は足せるが、**V と A に別々の隠れ表現を持たせることができない**。

本フレームワークの body は `net.block`/`net.branch`/`net.body.output.[key]` という強力な config DSL を持ち、**body が複数キーを出力 → head が複数キーを読む** 機構は既に存在し、**MuZero が実運用している**（`value_branch`/`policy_branch` → pred ヘッドが `value_feature`/`policy_feature` を読む）。これを Dueling に転用すれば、最小のコード改修で「V/A 独立隠れ層」を **config だけで自由に組める** ようになる。

## 現状の要点（調査結果）

- **head は config 化されておらず**、`use_dueling_net`/`use_qr` の4択を factory で選ぶだけ（[default_dqn_agent.cpp:91-114](core/anet-core/src/default_dqn_agent.cpp:91)）。← この選択ロジックは**変更不要**。
- `DuelingHead`（[dqn_based_heads.cpp:83](core/anet-core/src/dqn_based_heads.cpp:83)）/ `QuantileDuelingHead`（[同:252](core/anet-core/src/dqn_based_heads.cpp:252)）が単一 `features` キー（`kKey_DefaultOutput`、[nn_heads.hpp:10](core/anet-core/src/nn_heads.hpp:10)）を読み、V/A を単一 Linear で射影。
- **body 複数キー出力は実装済み**: `net.body.output.[HeadKey] = BranchName` をパース（[nn_impl.cpp:363-377](core/anet-core/src/nn_impl.cpp:363)）し、`NetworkBody::Forward` が全 output_keys を head 用 Dict に詰める（[nn_impl.cpp:636-642](core/anet-core/src/nn_impl.cpp:636)）。
- **テンプレート（MuZero）**: `MuZeroPredictionHeadFactory::CreateHead` が `dummy_features` から `value_feature`/`policy_feature` の各次元を取得し（[muzero_based_agent.cpp:166-174](core/anet-core/src/muzero_based_agent.cpp:166)）、`MuZeroPredictionHead::Forward` が両キーを読んで各最終層を通す（[同:112-128](core/anet-core/src/muzero_based_agent.cpp:112)）。
- **factory コンストラクタの引数は変えない**（`action_dim`/`num_quantiles`/`init`）。construction site（DefaultDQN・Rainbow 双方）に波及しない。変更は `CreateHead` 内部と head クラスに限定。
- **消費側**: `forward.v`/`forward.a`/`v_values`/`a_values` を `HeatMapPanel.cpp` が利用。head が引き続き V/A を計算するので互換維持できる。

## 設計（MuZero 流・後方互換）

head に「2モード」を持たせ、**config で V/A ストリームを定義したら分岐、しなければ従来通り**にする。

- **分岐モード（opt-in）**: `dummy_features` に `value_feature` と `adv_feature` の両キーがあれば、それぞれを V/A の入力とする（独立隠れ層は body 側ブランチが担当）。
- **共有モード（既存互換）**: 両キーが無ければ従来通り `features` を V/A 双方の入力にする。→ **既存 config はそのまま不変で動作**。

最終射影（V→1 or N、A→A or A*N）と Dueling 集約（`q = v + (a - mean(a))`）は **引き続き head が担当**。head は `action_dim`/`num_quantiles` を知っているので、config 作者が `A*N` 等の次元を手書きする必要は無い（Option C の弱点を回避）。

キー名定数を追加: `kKey_ValueFeature = "value_feature"`, `kKey_AdvFeature = "adv_feature"`（`dqn_based_heads.cpp` 内の名前付き定数。無名 namespace は使わない方針に従い、ファイル内 static か `anet::rl::dqn` 配下に置く）。

## 変更内容

### 1. `core/anet-core/src/dqn_based_heads.cpp` — `DuelingHead`
- コンストラクタを `(int64_t v_in, int64_t a_in, int64_t action_dim, std::string v_key, std::string a_key, init)` に変更。
  `value_ = Linear(v_in, 1)`, `adv_ = Linear(a_in, action_dim)`。`v_key_`/`a_key_` をメンバ保持。
- `Forward`: `feature_dict.At(v_key_)` / `feature_dict.At(a_key_)` を読む。集約式は不変。
- `GetTensorDictFunction`（`forward` / `forward.q` / `forward.v` / `forward.a`）: `kKey_DefaultOutput` 直書きを `v_key_`/`a_key_` 参照に置換（HeatMapPanel 互換維持）。
- `GetGraphVizInfo`: `details` に入力キー（`value_feature`/`adv_feature` or `features`）を追記し、分岐有無を可視化。

### 2. `core/anet-core/src/dqn_based_heads.cpp` — `QuantileDuelingHead`
- 同様に `(v_in, a_in, action_dim, num_quantiles, v_key, a_key, init)` 化。
  `value_ = Linear(v_in, num_quantiles)`, `adv_ = Linear(a_in, action_dim*num_quantiles)`。
- `Forward` と各 `GetTensorDictFunction`（`forward.dist`/`forward.v`/`forward.a` 含む）を `v_key_`/`a_key_` 参照に。

### 3. factory（`DuelingHeadFactory` / `QuantileDuelingHeadFactory`）`CreateHead`
MuZero の `CreateHead` に倣い、モード判定を入れる（コンストラクタ引数・[.hpp 宣言](core/anet-core/src/dqn_based_heads.hpp:29) は不変）:
```text
if dummy_features に value_feature と adv_feature が両方ある:
    v_in = value_feature.size(-1); a_in = adv_feature.size(-1)
    v_key = "value_feature"; a_key = "adv_feature"
else:
    f = GetFeature(dummy_features, kKey_DefaultOutput)   // 従来エラーメッセージ流用
    v_in = a_in = f.size(-1); v_key = a_key = "features"
make_shared<DuelingHead/QuantileDuelingHead>(v_in, a_in, ..., v_key, a_key, init)
```
キー存在判定は `dummy_features.Get(key)`（`std::optional`）で行う。

### 4. （変更不要の確認）
`default_dqn_agent.cpp` の head 選択・`NetworkBody`・`NetworkConfig` パース・`Network::Forward`・factory construction site・Rainbow 側はいずれも無改修。

### 5. config サンプル追加（ドキュメント／運用）
`apps/runner/config/DropMerge.txt` 等に、分岐を有効化する記述例を追記（既定はコメントアウトで無効、従来動作を壊さない）:
```text
net.block.[HeadFC512].type = Linear
net.block.[HeadFC512].linear.out_features = 512
net.block.[HeadFC512].init.mode = 1

net.branch.[value_stream].bind = main_feature
net.branch.[value_stream].structure = HeadFC512 > SiLU
net.branch.[adv_stream].bind   = main_feature
net.branch.[adv_stream].structure = HeadFC512 > SiLU

net.body.output.[value_feature] = value_stream
net.body.output.[adv_feature]   = adv_stream
# 既存の net.body.output.[features] = main_feature は残してもよい（分岐モードでは未使用）
```
両ストリームは同一バックボーン出力（`main_feature`）を bind する独立ブランチ。トポロジカルソート（[nn_impl.cpp:733-756](core/anet-core/src/nn_impl.cpp:733)）が依存順を解決する。

## 後方互換・エッジケース
- 既存 config（`features` のみ）は **共有モード**で従来と完全一致。新キー未指定なら挙動不変。
- 片方のキーのみ定義された場合は誤設定。`else` 分岐で `features` を探し、無ければ既存の `GetFeature` 親切エラー（`net.body.output.[features]` を案内）。実装時に「片側だけ定義」を明示エラーにするか検討。
- 既存テスト（[dqn_based_agent_test.cpp](core/anet-core/src/dqn_based_agent_test.cpp)・[nn_test.cpp](core/anet-core/src/nn_test.cpp)）は `features` 系のみ設定 → 共有モードで通過するはず。head 直接構築箇所があれば新シグネチャに追従（実装時に確認）。
- 重み初期化は従来通り `head_init`（[agent.txt:73-74](apps/runner/config/agent.txt:73)）を V/A 最終層へ適用。

## 検証
1. **ビルド**: runner をビルド（既存手順）。`dqn_based_agent_test` / `nn_test` を実行し緑を確認。
2. **後方互換**: 分岐記述なしの既存 `DropMerge.txt` で起動 → `net.detail`/`net.structure`（[default_dqn_agent.cpp:144-151](core/anet-core/src/default_dqn_agent.cpp:144)）のログが従来と同型・学習が回ることを確認。
3. **分岐モード**: 上記サンプルを有効化して起動 → `net.detail` の GraphViz に `value_stream`/`adv_stream` の2独立ブランチと、head が `value_feature`/`adv_feature` を読む表示が出ることを確認。短時間 train で loss/Q が NaN なく更新されること。
