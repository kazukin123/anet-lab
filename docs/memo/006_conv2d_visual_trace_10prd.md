# Conv2d 可視化「実Forwardトレース化」実装指示書

## 0. このドキュメントの位置づけ
- `Conv2dPanel` / `Conv2dVisualizationObserver` の可視化を、**推論で実際に流れた activation を本番 Forward 中に横取り（trace）して描画する方式**へ変更する。
- 設計判断は確定済み。本書は実装手順書。**設計の再検討は不要**、記載どおり実装してほしい。
- 言語・コメントは日本語。

---

## 1. 背景（直すべきバグ）
現状の可視化は「推論に入力された FrameStack 済み Observation」を**再現できていない**。

- 実推論の入力は `Actor::MakeAction` で `context_->PushObservation(state)`（実フレーム履歴を保持する `DictFrameStacker`）→ `obs_norm_->Normalize` を通った **本物のスタック済み・正規化済み obs**。
- ところが可視化経路は `DefaultDQNAgent::GetTensorDictFunction`（`core/anet-core/src/default_dqn_agent.cpp:315`）が、1フレームを `unsqueeze(1)→expand` で **stack_count 個に複製した偽スタック**を作って再 Forward している（`default_dqn_agent.cpp:346-353`）。
- 結果、**実際の履歴とは異なる入力**が可視化されている。これがバグ。

## 2. 方針（確定：Case D = 実Forwardトレース）
- 再 Forward／再現をやめ、**本番 Forward 中に conv2d 中間 activation を sink で横取り**する。
- 横取りした env0 の activation を `BatchActionInfo` の **AuxData（transient・非永続）** に積み、`TrainEvent` 経由で Observer/Panel が読む。
- 利点：
  - 入力の取り違えが原理的に起きない（実際に流れたものそのもの）。
  - 将来 NoisyNet / Dropout 等の確率的レイヤを入れても、**推論時に引いた乱数そのものの activation** が出る（再 Forward だとノイズが食い違う）。
  - `Forward` と `GetConv2dOutputs` の二重実装が解消され、行数が減る。

## 3. スコープ
### やること
1. trace 用の薄い sink 型（`std::function`）を追加。
2. Forward チェーン4関数に `sink` 引数（デフォルト空）を通し、`NetworkStruct::Forward` に**唯一の捕捉点**を置く。
3. アクション経路（`SelectAction` / `ForwardForAction`）に `sink` を通し、`Actor::MakeAction` / `DefaultDQNAgent::MakeAction` で trace を AuxData に積む。
4. `Conv2dVisualizationObserver` / `Conv2dPanel` を **trace 読み取り方式**に変更（再 Forward 呼び出しを撤去）。
5. `GetConv2dOutputs` 系と `ExtractConv2dOutputs` を**撤去**。
6. `NetworkModel::GetTensorDictFunction` を `Network::GetTensorDictFunction`（head ベースの汎用版）へ委譲変更。

### やらないこと（スコープ外）
- `DefaultDQNAgent::GetTensorDictFunction` 本体の整理（偽スタック `expand` の撤去含む）→ **Sweeped V2 対応時に実施**。今回は触らない（呼び出し元が消えて休眠するだけでよい）。
- `SweepedHeatMapObserver` の `GetTensorFunction`→`GetTensorDictFunction` 切替 → 別途（Sweeped V2）。
- target-net の可視化（後述）。trace は **行動選択に使った policy net** の内容のみ。
- Transformer / attention 等への一般化（次フェーズ）。

---

## 4. 全体データフロー（変更後）
```
Actor::MakeAction (rollout, env=N バッチ)
  obs       = context_->PushObservation(state)   // 実スタック
  norm_obs  = obs_norm_->Normalize(obs)
  trace(空の TensorDict) を用意
  sink = [&trace](key, act){ trace.Set(key, act[env0].detach().to(fp32).clone()); }
  act_info = policy_->SelectAction(norm_obs, false, network_, rnd, sink)
        └ ForwardForAction(obs, network, sink)
              └ network->Forward(obs, sink)
                    └ body_->Forward(input, sink)
                          └ branch->Execute(state, sink_with_branch_prefix)
                                └ struct->Forward(x, sink)   ★唯一の捕捉点
  act_info.aux["nn_trace/<branch>/<NN>"] = trace の各要素
  → trainer.cpp が aux をそのまま TrainEvent.action_info へ引き継ぐ
  → Observer/Panel が aux の "nn_trace/" を束ね直して Visualize
```

確認済みの要点：
- `trainer.cpp:235-239` は `action_info_raw->GetAuxData()` を**そのままコピー**して `TrainEvent` の `action_info` を作る（`trainer.cpp:267`）。replay を経由しない on-policy 直送なので **aux の trace は Observer まで生存する**。→ **trainer.cpp は無改修**。

---

## 5. 変更詳細

### 5.1 【新規】`TraceSink` 型
**`core/anet-core/include/anet/common.hpp`**（`TensorDictFunction` 定義の直後、`:99-100` 付近）に追加：
```cpp
// NN 内部 activation を本番 Forward 中に横取りするための sink。
// 第1引数: レイヤキー（"00_Input" や "01_Conv2d" 等）、第2引数: そのレイヤ出力(フルバッチ)。
// 空(=未設定)なら捕捉しない。何を保存するか(env0抽出/clone/dtype)は呼び出し側ラムダの責務。
using TraceSink = std::function<void(std::string_view, const torch::Tensor&)>;
```
- 仮想インターフェース（`struct NnTraceSink{...}`）は**作らない**。`std::function` で十分。

### 5.2 【変更】Forward チェーンに `sink` を通す（`core/anet-core/src/nn_impl.cpp` + 対応ヘッダ宣言）
4関数すべてに `const anet::TraceSink& sink = {}` を末尾追加（デフォルト空なので既存呼び出しは無改修）。**対応するヘッダ側の宣言も必ず同じシグネチャに更新**すること。

#### (a) `Network::Forward`（`nn_impl.cpp:822`）
body にだけ sink を渡す。Head は conv2d 後段なので不要。
```cpp
anet::TensorDict Network::Forward(const anet::TensorDict& input, const anet::TraceSink& sink)
{
    anet::ProfileRange r("Network::Forward");
    auto features = body_->Forward(input, sink);   // ★ sink を伝播
    if (head_) {
        anet::Autocast disable_amp(torch::kCUDA, false, torch::kFloat32);
        return head_->Forward(features.To(torch::kFloat32));
    } else {
        return features;
    }
}
```

#### (b) `NetworkBody::Forward`（`nn_impl.cpp:647`）
各 branch に sink を渡す。
```cpp
anet::TensorDict NetworkBody::Forward(const anet::TensorDict& input, const anet::TraceSink& sink)
{
    anet::ProfileRange r("NetworkBody::Forward");
    auto state = preprocessor_.Format(input);
    for (const auto& branch : branches_) {
        branch->Execute(state, sink);   // ★
    }
    anet::TensorDict out;
    for (const auto& [head_key, branch_key] : output_keys_) {
        if (auto t = state.Get(branch_key)) out.Set(head_key, *t);
        else ANET_SYSTEM_ERROR("NetworkBody output mapping failed: branch '" << branch_key << "' not found in DAG state.");
    }
    return out;
}
```

#### (c) `NetworkBranch::Execute`（`nn_impl.cpp:578`）
**branch 名プレフィックスをラムダ合成で付与**してから struct に渡す（旧 `ExtractConv2dOutputs` の `name_ + "/" + key` 相当）。
```cpp
void NetworkBranch::Execute(anet::TensorDict& current_state, const anet::TraceSink& sink)
{
    torch::Tensor block_input;
    if (bind_keys_.empty()) {
        block_input = torch::empty({ 0 }, current_state.device());
    } else {
        std::vector<torch::Tensor> inputs;
        for (const auto& key : bind_keys_) {
            auto t_opt = current_state.Get(key);
            if (!t_opt.has_value())
                ANET_SYSTEM_ERROR("NetworkBranch '" << name_ << "' failed to execute: Input key '" << key << "' not found in TensorDict.");
            inputs.push_back(*t_opt);
        }
        block_input = (inputs.size() == 1) ? inputs[0] : torch::cat(inputs, 1);
    }

    // sink があれば branch 名プレフィックスを足したラムダを下層へ
    anet::TraceSink branch_sink;
    if (sink) {
        const std::string prefix = name_ + "/";
        branch_sink = [&sink, prefix](std::string_view k, const torch::Tensor& a) {
            sink(prefix + std::string(k), a);
        };
    }

    torch::Tensor output = network_struct_->Forward(block_input, branch_sink);
    current_state.Set(name_, output);
}
```

#### (d) `NetworkStruct::Forward`（`nn_impl.cpp:451`）★唯一の捕捉点
旧 `GetConv2dOutputs`（`:462-478`）と完全に同じ捕捉条件・キー形式・`"00_Input"` を踏襲する。
```cpp
torch::Tensor NetworkStruct::Forward(torch::Tensor input, const anet::TraceSink& sink)
{
    anet::ProfileRange r("NetworkStruct::Forward");

    // 入力そのものを "00_Input" として横取り（旧 GetConv2dOutputs と同じ・無条件）
    if (sink) sink("00_Input", input);

    torch::Tensor x = input;
    int index = 1;
    for (const auto& block : blocks_) {
        x = block->Forward(x);
        if (sink && block->IsConv2dVisualizable()
                 && x.dim() == 4 && x.size(2) >= 2 && x.size(3) >= 2 && x.is_floating_point()) {
            sink(std::format("{:02d}_{}", index++, block->GetName().c_str()), x);
        }
    }
    return x;
}
```
- 注意：`index++` は**捕捉した時だけ**進む（旧実装と同じ。捕捉条件を満たさない block では進めない）。

### 5.3 【撤去】`GetConv2dOutputs` 系 / `ExtractConv2dOutputs`
本書 5.2 で同等機能が `Forward(sink)` に統合されるため、以下を**定義・宣言とも削除**：
- `NetworkStruct::GetConv2dOutputs`（`nn_impl.cpp:462-478`）
- `NetworkBranch::ExtractConv2dOutputs`（`nn_impl.cpp:600-626`）
- `NetworkBody::GetConv2dOutputs`（`nn_impl.cpp:671-687`）
- `Network::GetConv2dOutputs`（`nn_impl.cpp:842-850`）
- 上記4つに対応する**ヘッダ宣言**（`include/anet/nn.hpp` 等。`grep -n GetConv2dOutputs` / `ExtractConv2dOutputs` で全件確認して削除）

### 5.4 【変更】`NetworkModel::GetTensorDictFunction`（`core/anet-core/src/dqn_based_agent.cpp:196-216`）
`net->GetConv2dOutputs(...)` 依存を断ち、`net->GetTensorDictFunction(key)`（`nn_impl.cpp:852` の head ベース汎用版）へ委譲する形に変更する。
```cpp
std::optional<anet::TensorDictFunction> NetworkModel::GetTensorDictFunction(const std::string& key, const torch::Device& device)
{
    std::shared_ptr<anet::nn::Network> net = nullptr;
    if (key == "policy-net.conv2d")      net = policy_net_;
    else if (key == "target-net.conv2d") net = target_net_;
    if (!net) return std::nullopt;

    // 汎用 head ベース抽出へ委譲（GetConv2dOutputs 依存を撤去）。
    // 正確なキー意味づけ（"conv2d" 等の解決）は Sweeped V2 対応時に詰める＝今は対象外。
    return net->GetTensorDictFunction(key);
}
```
- この変更後、`policy-net.conv2d` キーは head 側で解決されず `nullopt` になり得るが、**conv2d 可視化は trace 方式に移行して GetTensorDictFunction を呼ばなくなる**ため問題ない（休眠シーム）。
- `Network::GetTensorDictFunction`（`nn_impl.cpp:852`）は**残す**（Sweeped V2 / Transformer 用の汎用窓口）。

### 5.5 【据置】`DefaultDQNAgent::GetTensorDictFunction`（`default_dqn_agent.cpp:315-365`）
- **今回は触らない**。偽スタック `expand`（`:346-353`）の撤去・正規化整理は **Sweeped V2 対応時**にまとめて実施する。
- 本変更後は呼び出し元（Conv2dPanel / `MakeConv2dVisualizationObserver`）が消えるため、当面は未使用（休眠）になるが意図どおり。

### 5.6 【変更】アクション経路で trace を AuxData に積む

#### (a) `ActionPolicy::ForwardForAction`（`dqn_based_agent.cpp:261-266`）
```cpp
anet::TensorDict anet::rl::dqn::ActionPolicy::ForwardForAction(
    const anet::TensorDict& obs,
    std::shared_ptr<anet::nn::Network> network,
    const anet::TraceSink& sink) const
{
    return network->Forward(obs, sink);
}
```
- ヘッダ宣言（`dqn_based_agent.hpp` の `ForwardForAction`）も更新。

#### (b) `SelectAction`（基底 + 3オーバーライド）に `sink` 引数を追加
- 基底 純粋仮想宣言：`dqn_based_agent.hpp:310`、派生宣言：`:345`(Epsilon) / `:364`(UQE) / `:384`(Thompson)。
- 定義：`EpsilonGreedy`(`dqn_based_agent.cpp:451`) / `UQE`(`:674`) / `Thompson`(`:703`)。
- テストモック：`core/anet-core/src/dqn_based_agent_test.cpp:31` も同シグネチャに更新。
- シグネチャ末尾に `const anet::TraceSink& sink = {}` を追加し、各実装内の `ForwardForAction(obs, network)` を `ForwardForAction(obs, network, sink)` に変更。

```cpp
// 例: EpsilonGreedyActionPolicy::SelectAction
anet::rl::BatchActionInfo EpsilonGreedyActionPolicy::SelectAction(
    const anet::TensorDict& obs, bool greedy_only,
    std::shared_ptr<anet::nn::Network> network,
    std::shared_ptr<anet::RandomGenerator> rnd,
    const anet::TraceSink& sink) const
{
    ...
    auto out = ForwardForAction(obs, network, sink);   // ★ sink を渡す
    ...
}
```
> **デフォルト引数の注意**：仮想関数のデフォルト引数は静的束縛。**デフォルト `= {}` は基底の純粋仮想宣言にのみ書き**、オーバーライド側では書かない（重複定義による混乱回避）。呼び出しは `shared_ptr<ActionPolicy>` 経由なので基底のデフォルトが効く。

- **sink を渡さない既存呼び出しは無改修でよい**：
  - learner の target 選択 `dqn_based_agent.cpp:1241` / `:1433`（`greedy_only=true`）
  - `rainbow_agent.cpp:198`
  - 各テスト
  これらはデフォルト空 sink で**捕捉しない**＝従来どおり。

#### (c) `Actor::MakeAction`（`dqn_based_agent.cpp:737-778`）★主経路
`SelectAction` 呼び出しの前に trace と sink を用意し、後で aux に積む。`network_ != src_network_` 分岐の**両方**に sink を渡すこと。
```cpp
// 行動選択
auto rnd = context_->GetRandomGenerator();

anet::TensorDict trace;
anet::TraceSink sink = [&trace](std::string_view k, const torch::Tensor& a) {
    // env0 のみ・AMP対策でfp32化・cloneで独立。NoGradGuard 下なのでdetachは保険。
    trace.Set(std::string(k), a.slice(0, 0, 1).detach().to(torch::kFloat32).clone());
};

anet::rl::BatchActionInfo act_info;
if (network_ != src_network_) {
    act_info = policy_->SelectAction(norm_obs, false, network_, rnd, sink);
} else {
    std::shared_lock<std::shared_mutex> lock(*mutex_);
    act_info = policy_->SelectAction(norm_obs, false, network_, rnd, sink);
}

// AuxData の詰め込み（既存）
act_info.GetAuxData()["raw_obs"] = anet::rl::ToUnifiedObservation(obs);
if (obs_norm_ != nullptr) {
    act_info.GetAuxData()["norm_obs"] = anet::rl::ToUnifiedObservation(norm_obs);
}
// ★ trace を flat-prefix で aux へ
for (const auto& kv : trace) {
    act_info.GetAuxData()[std::string("nn_trace/") + kv.first] = kv.second;
}
```

#### (d) `DefaultDQNAgent::MakeAction`（`default_dqn_agent.cpp:460-500`）
`Actor::MakeAction` と同様に、`eval_policy_->SelectAction` / `train_policy_->SelectAction` の**両方**に sink を渡し、末尾で trace を `nn_trace/` プレフィックスで aux に積む（コードは (c) と同形）。

### 5.7 【変更】Observer / Panel を trace 読み取りへ

#### (a) `Conv2dVisualizationObserver`（`core/anet-core/src/observers.cpp:566-641`、宣言 `include/anet/observers.hpp:270-292`）
- コンストラクタから `anet::TensorDictFunction dict_func` を**削除**（メンバ `dict_func_` も削除）。
- `OnTrain`（`:577`）の描画部（`:610-632`）を、`state.obs` からの再 Forward ではなく **event の AuxData から trace を束ねて Visualize** に変更：
```cpp
// 旧:
//   anet::TensorDict single_obs = state.obs[0].Unsqueeze(0);
//   auto dict = dict_func_(single_obs);
// 新:
anet::TensorDict dict;
const auto& aux = event.action_info->GetAuxData();   // ※ const アクセス可否は実装に合わせる
static constexpr std::string_view kPrefix = "nn_trace/";
for (const auto& [k, t] : aux) {
    if (k.rfind(kPrefix.data(), 0) == 0) {
        dict.Set(k.substr(kPrefix.size()), t);   // プレフィックスを剥がして元のレイヤキーに戻す
    }
}
if (!dict.empty()) {
    auto vis_result = visualizer_.Visualize(step, dict);
    ... // 以降の画像保存/ログは既存のまま
}
```
- 録画開始/終了の `is_recording_` ロジック（`:585-640`）は**そのまま維持**。
- `event.action_info` から取れない場合は `event.experience.action`（同一インスタンス、`trainer.cpp:263/267` で同じ `action_info` を共有）からでも可。**const アクセサの有無に合わせてどちらか**を使う。

#### (b) `Conv2dPanel`（`apps/runner/src/Conv2dPanel.cpp`）
- `CreateVisualizer`（`:69-83`）：`agent->GetTensorDictFunction(...)` の取得・`ANET_CHECK_MSG`・`vis_dict_fn_` 代入（`:76-79`）を**削除**。Visualizer 生成（`:82`）だけ残す。
- メンバ `vis_dict_fn_`（`apps/runner/src/Conv2dPanel.hpp:60`）を削除。
- `CreateObserver`（`:85-122`）のラムダ（`:91-117`）：`single_obs = state.obs[0]...` → `vis_dict_fn_(single_obs)` をやめ、(a) と同じく `event` の AuxData の `nn_trace/` を束ねた `dict` を `visualizer_->Visualize(...)` に渡す。
- `state.obs` の Defined チェックは trace 有無チェックに置換（trace 空ならスキップ）。

#### (c) `MakeConv2dVisualizationObserver`（`core/anet-core/src/image.cpp:327-346`）
- `agent->GetTensorDictFunction(config.conv2d.network_key)` の取得（`:334`）と `dict_func` 引数の受け渡しを**削除**。
- `Conv2dVisualizationObserver` の新コンストラクタ（dict_func 無し）に合わせて生成（`:342`）を更新。
- `config.conv2d.network_key` は当面**参照されなくなる**（target 切替は後述スコープ外）。Config 自体は残してよい。

---

## 6. 命名規約
- trace 関連の識別子は conv2d 限定にせず**中立名**にする：型は `TraceSink`、aux キーのプレフィックスは `nn_trace/`。
- レイヤキー本体は従来形式を踏襲：`"00_Input"`、`"<branch名>/<NN名>"`（例 `"q_branch/01_Conv2d"`）。

---

## 7. 落とし穴・注意点（必読）
1. **AMP/autocast**：本番アクション Forward は `SelectAction` 内で `Autocast`（BF16/FP16）下に走る（`dqn_based_agent.cpp:456-457`）。横取りした activation は **half/bf16 になり得る**ので、sink で `.to(torch::kFloat32)` する（5.6c に反映済み）。旧再 Forward は FP32 だったため、ここは新たに必要な処理。
2. **env0 のみ・clone 必須**：sink は `a.slice(0,0,1)` で先頭 env だけ抽出し `clone()` で独立させる。view のまま aux に載せるとフルバッチ storage を保持してしまう。
3. **aux は非永続**：`AuxData` は replay buffer に保存されない transient 情報（`info_` は永続なので trace を入れてはいけない）。今回は on-policy 直送（`trainer.cpp:238`）なので Observer まで届く。
4. **trainer.cpp は無改修**：`action_info_raw->GetAuxData()` がそのまま `TrainEvent.action_info` に渡る（`trainer.cpp:235-239, 263, 267`）。
5. **デフォルト引数 × 仮想関数**：`SelectAction` のデフォルト `= {}` は基底宣言だけに置く（5.6b）。
6. **`"00_Input"` の無条件捕捉**：旧 `GetConv2dOutputs` は入力を無条件で `"00_Input"` 登録していた。Visualizer 側は 3次元(squeeze後)以外を弾くため、非画像 branch の `"00_Input"` は描画段で無視される＝従来挙動と同じ。
7. **target-net 可視化はスコープ外**：trace は行動選択に使った policy net の内容のみ。target の可視化は別アプローチ（eval Runner trace / オンデマンド再 Forward）で後日。trace 機構は net 非依存に保つこと。
8. **常時捕捉のコスト**：本実装では `MakeAction`（=毎 rollout ステップ）が常に capturing sink を渡す＝env0 分だけ毎ステップ slice+clone が走る。env0 限定なので軽微だが、もし rollout スループット低下が計測されたら、**Agent 側に「可視化アクティブ時のみ sink を渡す」フラグ**（panel/observer の attach 時に立てる `std::atomic<bool>`）を足す。**今は実装しない**（必要になってからでよい）。

---

## 8. 検証手順
1. ビルドが通ること（`GetConv2dOutputs`/`ExtractConv2dOutputs` 撤去に伴う未解決参照が無いこと）。
2. `grep -rn "GetConv2dOutputs\|ExtractConv2dOutputs"` で**残骸ゼロ**を確認。
3. runner GUI で `Conv2dPanel` を開き、**FrameStack 枚数ぶんの異なるフレーム**が並んで表示されること（従来は同一フレームの複製だった）。
4. `Conv2dVisualizationObserver` の episode_interval 録画が従来どおり開始/終了し、画像が TensorBoard に出ること。
5. 既存の学習が回帰しないこと（target 選択・learner・rainbow・テストは sink 未指定＝従来どおり）。
6. （AMP 有効時）可視化画像が NaN/真っ黒にならないこと（fp32 化の確認）。
