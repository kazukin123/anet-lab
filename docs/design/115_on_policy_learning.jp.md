# 方策ベースの学習の共通部品

> 主たる観点: 機能単位（方策ベースのAgentが使うcoreの部品。行動の分布、advantageの推定、Headの合成）

## 1. はじめに

### 1.1 目的と範囲

本書は、方策ベース（方策を直接学ぶ）のAgentが使う、coreの共通部品を説明する。部品は特定のAgentの名前を持たず、`anet-core-base`の公開ヘッダから使う。

| 部品 | 公開ヘッダ | 役割 |
|---|---|---|
| `CategoricalDistribution` | [distribution.hpp](../../src/core/base/include/anet/distribution.hpp) | 離散行動の分布。抽選、log確率、entropy |
| `ComputeGae` | [advantage.hpp](../../src/core/base/include/anet/advantage.hpp) | advantageとvalueの目標の推定（GAE） |
| `CompositeHeadFactory` | [nn_head.hpp](../../src/core/base/include/anet/nn_head.hpp) | 複数のHeadを1つのNetworkに並べる |
| `LaneTimeTensorStore::GetAll` | [tensor_store.hpp](../../src/core/base/include/anet/tensor_store.hpp) | ためた経験の全体を、時刻×laneの形で読む |

これらを使う具象のAgentは[PPO系Agent](230_ppo_agents.jp.md)を参照する。Agent・Actor・Learnerの共通の契約は[Agentと学習](110_agents_and_learning.jp.md)、Networkの組み立ては[ニューラルネットワーク](130_neural_networks.jp.md)、ためる器`LaneTimeTensorStore`は[ReplayBufferのコンポーネント定義](150_replay_buffer.jp.md#3-コンポーネント定義)を参照する。

### 1.2 責任の境界

- 部品は計算だけを受け持つ。損失の式、rolloutをいつ更新に使うか、optimizerは、使う側のAgentが持つ。
- 部品は入力と同じdeviceで計算し、結果も同じdeviceに返す。値をCPUへ読み戻さない。CPUへ読むかどうかは、使う側が決める。
- 部品は設定キーを読まない。設定の読み取りと値域の検証は、使う側のConfigが行う。

## 2. 行動の分布

### 2.1 CategoricalDistribution

`CategoricalDistribution`は、行ごとのlogits（`[行, 行動数]`）から作る、離散行動の分布である。行は、laneやミニバッチの1件に当たる。logitsに行ごとの定数を足しても、分布は変わらない。

```cpp
const anet::rl::CategoricalDistribution distribution(logits);            // [行, 行動数]
const auto actions = distribution.Sample(random.GetTorchGenerator(device)); // [行]。int64
const auto log_probs = distribution.ComputeLogProb(actions);             // [行]
const auto entropy = distribution.ComputeEntropy();                      // [行]
```

| 口 | 返す値 | 形 |
|---|---|---|
| `Sample(generator)` | 分布から抽選した行動。生成器は、logitsと同じdeviceのものを渡す | `[行]`（int64） |
| `GetMode()` | 確率が最大の行動 | `[行]`（int64） |
| `ComputeLogProb(actions)` | 行ごとに、渡した行動のlog確率 `log π(a|s)` | `[行]` |
| `ComputeEntropy()` | 行ごとのentropy `H = −Σ p log p` | `[行]` |
| `GetProbs()`・`GetLogProbs()` | 全行動の確率とlog確率 | `[行, 行動数]` |
| `GetActionCount()` | 行動数 | スカラー |

- logitsはfloat32にしてから`log_softmax`で正規化する。半精度のlogitsでも、確率の和が1からずれない。
- 確率が0の行動（logitsが負の無限大）は、entropyの項を0として数える。
- logitsが勾配を持てば、log確率とentropyも勾配を持つ。行動選択のように勾配が要らない呼び出しは、使う側が勾配を切る。

### 2.2 エラー

| 条件 | 止まる場所 |
|---|---|
| logitsが未定義、浮動小数でない、rankが2でない、行動数が2未満 | 構築 |
| 行動が未定義、整数の型でない、形が`[行]`でない | `ComputeLogProb` |

## 3. advantageの推定

### 3.1 ComputeGae

`ComputeGae`は、GAE（Generalized Advantage Estimation）で、advantageとvalueの目標を作る。advantageは「その行動が、その状態の平均よりどれだけ良かったか」の推定で、方策の更新の向きと大きさを決める。

入力はどれも`[時刻, lane]`の形で、時刻0が最も古い。

| 入力 | 中身 |
|---|---|
| `rewards` | その時刻のstepの報酬 |
| `values` | その時刻の観測のvalue `V(s_t)` |
| `next_values` | その時刻のstepの、次の観測のvalue `V(s_{t+1})` |
| `terminated` | 真なら、次の観測のvalueを使わない（その先の収益が無い） |
| `episode_end` | 真なら、あとの時刻のadvantageを引き継がない（episodeがそこで切れる） |
| `gamma`・`lambda` | 割引率γと、GAEのλ。どちらも`[0, 1]` |

```text
delta[t]        = rewards[t] + γ · next_values[t] · (1 − terminated[t]) − values[t]
advantage[t]    = delta[t] + γ · λ · (1 − episode_end[t]) · advantage[t+1]      （最後の時刻の次は0）
value_target[t] = advantage[t] + values[t]
```

- `delta`は1 stepのTD誤差である。λが0なら、advantageは`delta`そのもの（偏りが大きく、ばらつきが小さい）になる。λが1なら、割引した報酬の和からvalueを引いた値（偏りが小さく、ばらつきが大きい）になる。λはその間を選ぶ。
- 戻り値`GaeResult`は`advantages`と`value_targets`を持ち、どちらも入力と同じ形である。勾配は作らない。

### 3.2 終端とepisodeの終わりを分ける理由

2つの印を分けて受け取る。打ち切り（時間切れなど）で終わったepisodeは、本当は続くはずだったので、次の観測のvalueで先を見積もりたい。一方、あとの時刻は別のepisodeなので、advantageは引き継がない。

| 局面 | `terminated` | `episode_end` | `next_values`に入れる値 |
|---|---|---|---|
| 続いている | 偽 | 偽 | 次の観測のvalue |
| 終端（失敗・成功で終わった） | 真 | 真 | 使われない |
| 打ち切り。先を見積もる | 偽 | 真 | Resetの前の観測のvalue |
| 打ち切り。終端と同じに扱う | 真 | 真 | 使われない |

次の観測のvalueをどこから取るか（次の時刻のvalueか、Resetの前の観測か、まとまりの最後の次の観測か）は、使う側が決めて`next_values`に入れる。`ComputeGae`は、Envの印の意味を知らない。

### 3.3 エラー

入力のどれかが未定義、rankが2でない、形かdeviceが互いに違う、報酬とvalueが浮動小数でない、印がboolでない、γかλが`[0, 1]`に無いときは、計算の前に止まる。

## 4. Headの合成

### 4.1 CompositeHeadFactory

`CompositeHeadFactory`は、複数のHeadを1つのNetworkのHeadとして並べる。方策のlogitsと状態のvalueのように、1つのNetworkから複数の出力を取るときに使う。

```cpp
auto head_factory = std::make_shared<anet::nn::CompositeHeadFactory>(std::vector<anet::nn::CompositeHeadEntry>{
    { .name = "policy", .feature_key = "policy_feature",
      .factory = std::make_shared<anet::nn::LinearHeadFactory>(action_count, "logits", policy_init) },
    { .name = "value", .feature_key = "value_feature",
      .factory = std::make_shared<anet::nn::LinearHeadFactory>(1, "value", value_init) } });
auto network = anet::nn::NetworkBuilder::BuildNetwork(network_config, input_specs, head_factory, seed, device);
```

| 組（`CompositeHeadEntry`）の項目 | 意味 |
|---|---|
| `name` | 中のHeadを登録する名前。parameterの名前は`head.<name>.…`になる |
| `feature_key` | 中のHeadが受け取る、Bodyの出力のキー（`net.body.output.[キー]`） |
| `factory` | 中のHeadのfactory |

- 作ったHeadは、組ごとに、`feature_key`の値を`features`として中のHeadへ渡し、出力を1つの`TensorDict`にまとめる。
- どのbranchの値を中のHeadへ渡すかは、設定の`net.body.output.[キー] = <branch名>`で決まる。2つのキーを同じbranchへ向ければBodyを共有し、別のbranchへ向ければ別のBodyになる。コードは変わらない。
- 中のHeadを名前つきで登録するので、parameterを名前の接頭辞で分けられる（勾配の内訳を測るときなど）。
- forwardの実行器、Noisyのparameterの一覧、図の情報は、中のHeadのものを順に集める。

### 4.2 エラー

| 条件 | 止まる場所 |
|---|---|
| 組が0個、名前が空、名前が`.`を含む、`feature_key`が空、factoryが無い、名前の重複 | factoryの構築 |
| `feature_key`がBodyの出力に無い | `CreateHead`（Networkの組み立て） |
| 中のHeadどうしで、出力のキーが重なる | `CreateHead`（Networkの組み立て） |

出力のキーの重なりは、仮の特徴量で中のHeadを1回通して確かめる。中のHeadは、仮の特徴量と同じdeviceへ移してから通す。

## 5. ためた経験の全体の読み出し

`LaneTimeTensorStore::GetAll`は、たまっている範囲の全体を、キーごとに`[ためた時刻の数, lane数, …]`の形で返す。複製は作らず、器の記憶域を指す。空の器で呼ぶと止まる。

器は、lane一式のtensorを`Append`で時刻の順にためる。ためた経験を1回の更新で使い切る学習（rollout）は、器が埋まったら`GetAll`で全体を読み、更新のあとに`Clear`で空にする。

## 6. コードマップ

| 領域 | 主なファイル |
|---|---|
| 行動の分布 | [distribution.hpp](../../src/core/base/include/anet/distribution.hpp)、[distribution.cpp](../../src/core/base/src/distribution.cpp) |
| advantageの推定 | [advantage.hpp](../../src/core/base/include/anet/advantage.hpp)、[advantage.cpp](../../src/core/base/src/advantage.cpp) |
| Headの合成 | [nn_head.hpp](../../src/core/base/include/anet/nn_head.hpp)、[nn_head.cpp](../../src/core/base/src/nn_head.cpp) |
| 器 | [tensor_store.hpp](../../src/core/base/include/anet/tensor_store.hpp)、[tensor_store.cpp](../../src/core/base/src/tensor_store.cpp) |

## 7. テスト

`anet-core-base-test`のタグで確かめる。deviceを受けるテストは、CPUとCUDAの両方で走る（CUDAが使えない環境では、CUDAの分を飛ばす）。

| タグ | 確かめること |
|---|---|
| `[distribution]` | log確率とentropyが手計算と合う。抽選の頻度が確率に合う。不正な形で止まる |
| `[advantage]` | 手計算の例、終端、episodeの終わり、λ = 0、λ = 1で値が合う。不正な形と範囲で止まる |
| `[composite_head]` | 出力が1つにまとまり、キーごとに別のbranchの値が届く。重複と不足で止まる |
| `[tensor_store]` | `GetAll`が、ためた値を時刻×laneの形で返す |

```powershell
out\test\Debug\anet-core-base-test.exe "[distribution],[advantage],[composite_head],[tensor_store]"
```

## 8. 関連文書

- [Agentと学習](110_agents_and_learning.jp.md)
- [ニューラルネットワーク](130_neural_networks.jp.md)
- [ReplayBuffer](150_replay_buffer.jp.md)
- [PPO系Agent](230_ppo_agents.jp.md)
- [用語集](../../CONTEXT.md)
