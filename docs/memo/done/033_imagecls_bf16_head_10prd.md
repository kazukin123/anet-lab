# ImageCls BF16(AMP)対応 + 汎用 LinearHead による Head 化

> 設計分担: Claude=設計/PRD、実装=Codex。本書は self-contained。
> 前提の議論: BF16 autocast の適用箇所（learner/actor）と、Head/loss の精度保護をどう担保するか。
> 結論として **Head 化で「Head=FP32」を構造で保証**し、BF16 設定は「有効無効＋適用箇所」だけに絞る。

## Context（背景・目的）

ImageCls（Food-101 / 101 クラス / 224×224、ResNet18ish 系で 6〜8h）は **Learn が GPU compute-bound**
（cf. 作業メモ: Learn GPU 稼働率 ~87%）。ここに **BF16 の autocast(AMP)** を入れて学習 forward を高速化したい。
FP16 は非対応（BF16 固定）。

BF16 を素直に入れる上での構造的な障害が 1 点ある:

- フレームワークは [`Network::Forward`](../../../core/anet-core/src/nn_impl.cpp:1177) で
  **body(autocast 対象) → head(autocast 強制 OFF=FP32) → 出力を FP32 化して head へ** という
  「Head=FP32」境界を既に持つ。backend も [`use_tf32_cublas/cudnn=true`](../../../core/anet-core/include/anet/init.hpp:13)（既定）で、
  FP32 に残った matmul は実効 TF32。
- **ところがこの保護は head_factory を渡す DQN/MuZero 系だけ**。ImageCls は
  [`BuildNetwork(..., nullptr, ...)`](../../../core/anet-core/src/image_cls_agent.cpp:474) で **head_factory=nullptr →
  head=nullptr（body only）**（[`head = head_factory ? ... : nullptr`](../../../core/anet-core/src/nn_impl.cpp:1511)）。
  よって Forward は [else 経路「Head が無い場合は Body の出力をそのまま返す」](../../../core/anet-core/src/nn_impl.cpp:1183)
  を通り、**最終段 `LinearOut`（logits 生成）まで autocast スコープ内なら BF16 化**してしまう。

目的: **ImageCls を head 化**（`LinearOut` を Head へ移す）して「Head=FP32/TF32」を **設定ではなく構造で担保**し、
その上に BF16 autocast を薄く乗せる。loss(cross_entropy/softmax) は autocast の fp32 policy＋スコープ外実行で自動 FP32。

## 確定した設計判断

1. **汎用 `LinearHead` / `LinearHeadFactory` を `nn_heads`（`anet::nn`）へ追加**。
   出力キーをコンストラクタ引数で可変にする（ImageCls は `"logits"`）。既存 `PassThroughHead` の隣に同スタイルで置く。
2. **DQN 側 `LinearHeadFactory`（`anet::rl::dqn`, `dqn_based_heads.*`）は一切触らない**。namespace が異なり衝突しない。
   DQN 用実装として現状のままがスマート（ユーザー確定）。将来の寄せ替えは後続（本 PRD 非対象）。
3. **ImageCls を head 化**: 全 branch 末尾の `> LinearOut` を除去、`net.body.output.[features] = main_feature`
   （`features` = [`kKey_DefaultOutput`](../../../core/anet-core/src/nn_heads.hpp:10)）。`ImageClsAgent` が
   `LinearHeadFactory(num_classes, "logits", init)` を `BuildNetwork` に渡す。`num_classes` は **env_spec のクラス数**
   （DQN の `n_actions` と同経路）から取得＝config の `101` 二重管理を解消。
4. **BF16 は「有効無効＋適用箇所」の 3 設定のみ**。Head FP32・loss FP32 は構造/autocast policy が保証するので、
   精度境界フラグ（head_fp32 等）は **持たせない**。
   ```
   ImageClsAgent.bf16.enabled = false  # マスタースイッチ（既定 false=全 FP32）
   ImageClsAgent.bf16.learner = true   # 学習 forward を BF16 autocast（body=特徴抽出）
   ImageClsAgent.bf16.actor   = false  # eval/GUI 推論 forward を BF16 autocast
   ```
5. **GradScaler 不要**（BF16 固定）。DQN の BF16 経路と同じく
   [`backward → grad clip → step`](../../../core/anet-core/src/dqn_based_agent.cpp:988) をそのまま。現行 ImageCls の
   backward/clip/step 構造（[image_cls_agent.cpp:360-372](../../../core/anet-core/src/image_cls_agent.cpp:360)）は不変。
6. **2 フェーズで実装・検証**（1 個ずつ入れて評価）。フェーズ1=head 化（FP32 のまま等価）、フェーズ2=BF16。

## 前提事実（実コード確認済み）

> 基準コミット: HEAD `4b10b0f`（working tree に未コミット変更あり。行番号は現 working tree 基準）。

- **共通部品は既存**:
  - [`anet::Autocast(device_type, enabled, dtype=kHalf)`](../../../core/anet-core/include/anet/nn_util.hpp:56) …
    `at::autocast::set_autocast_enabled/dtype` の RAII ガード。`~Autocast` で前値へ復元。
  - `anet::GradScaler`（[nn_util.hpp:116](../../../core/anet-core/include/anet/nn_util.hpp:116)）… FP16 用。**本 PRD では使わない**。
- **DQN の BF16 適用の型**（流用の手本）:
  - [`amp_dtype = use_amp_bf16 ? kBFloat16 : kHalf`](../../../core/anet-core/src/dqn_based_agent.cpp:1542) →
    [`Autocast amp_guard(kCUDA, use_amp, amp_dtype)`](../../../core/anet-core/src/dqn_based_agent.cpp:1561) で
    **forward+loss だけを囲み**、[スコープを閉じてから Optimize](../../../core/anet-core/src/dqn_based_agent.cpp:1657)。
  - BF16 時は [scaler を使わず backward→clip→step](../../../core/anet-core/src/dqn_based_agent.cpp:988)。
- **Head 機構**:
  - [`NetworkHead`](../../../core/anet-core/include/anet/nn.hpp:98) は `Forward(feature_dict)` のみの単純 IF。
  - `nn_heads`（`anet::nn`）は現在 [`PassThroughHead` + `kKey_DefaultOutput="features"`](../../../core/anet-core/src/nn_heads.hpp:10) のみ。
  - DQN 版 `LinearHead`（[dqn_based_heads.cpp:34-82](../../../core/anet-core/src/dqn_based_heads.cpp:34)）は
    `feature_dict.At(kKey_DefaultOutput)` を `Linear` に通し `out.Set("q", ...)`。**出力キー "q" 固定**・GraphViz あり。
    汎用版はこれを出力キー可変にした等価物。
  - [`Network::Forward`](../../../core/anet-core/src/nn_impl.cpp:1177): head ありなら `body → Autocast(OFF) →
    head->Forward(features.To(kFloat32))`。**head 化するだけで Head=FP32 が自動**。
  - [`NetworkBody::Forward`](../../../core/anet-core/src/nn_impl.cpp:863): branch DAG を実行し `output_keys_` で head 用
    TensorDict を組む。→ head 期待キーは `net.body.output.[<key>]` で供給。
  - [`Network::Clone`](../../../core/anet-core/src/nn_impl.cpp:1367) は `head_factory_` を保持し clone 時に head を再構築。
    → head 化しても `clone_model`（Actor 複製）は動く。
- **ImageCls 現状**:
  - Learner: [obs→Forward→loss→backward→clip→step](../../../core/anet-core/src/image_cls_agent.cpp:337) が 1 つの
    `unique_lock` 内。`network_->Forward(obs)` は 1 回（body only）。
  - Actor: [`network_->Forward(obs, sink)` → `outputs.At("logits")`](../../../core/anet-core/src/image_cls_agent.cpp:58) →
    softmax/argmax。actor は eval/GUI 推論に使われ、device は `actor_device`（CPU eval あり得る）。
  - config: 全 branch 末尾が `> LinearOut`、[`net.body.output.[logits]=main_feature`](../../../apps/runner/config/ImageCls.txt:420)、
    [`[LinearOut] out_features=101, init.mode=he`](../../../apps/runner/config/ImageCls.txt:151)。
  - loss は [Learner 側で cross_entropy](../../../core/anet-core/src/image_cls_agent.cpp:352)（Forward の外）。

## 設計方針

### フェーズ1: 汎用 LinearHead 追加 ＋ ImageCls head 化（FP32 のまま）

**A. `nn_heads`（`anet::nn`）に汎用 head を追加**
- `nn_heads.hpp`: `class LinearHead : public NetworkHead`（`in_features, out_features, output_key, WeightInitConfig`）、
  `class LinearHeadFactory : public NetworkHeadFactory`（`out_features, output_key, WeightInitConfig`）。
- `nn_heads.cpp`: DQN 版 `LinearHead` と等価。差分は **出力キーを `output_key_` に**（DQN は "q" 固定）。
  - `Forward`: `x = feature_dict.At(kKey_DefaultOutput)` → `out.Set(output_key_, linear_->forward(x))`。
  - `GetTensorDictFunction`: key ∈ {`"forward"`, `output_key_`} でクロージャ返却（`PassThroughHead` と同作法）。
  - `GetGraphVizInfo`: type `"LinearHead"`、outputs `{output_key_, {out_features_}}`、details `{"out_features", N}`。
  - 重み初期化は `WeightInitializer::Initialize(linear_, init_config)`。
  - `LinearHeadFactory::CreateHead`: `in = GetFeature(dummy, kKey_DefaultOutput).size(-1)` →
    `make_shared<LinearHead>(in, out_features_, output_key_, init_config_)`。

**B. ImageCls を head 化（`image_cls_agent.cpp`）**
- `ImageClsAgent` 構築で `num_classes` を env_spec のクラス数から取得（DQN の n_actions と同経路。
  `env_spec.action_spec` の離散候補数 = クラス数）。`WeightInitConfig` は he（現 LinearOut 踏襲）。
- [`BuildNetwork(network_config, obs_spec, nullptr, device_)`](../../../core/anet-core/src/image_cls_agent.cpp:474) を
  `BuildNetwork(network_config, obs_spec, std::make_shared<anet::nn::LinearHeadFactory>(num_classes, "logits", he_init), device_)` へ。
- Actor の `outputs.At("logits")` は head 出力キー `"logits"` で維持（変更不要）。

**C. config（`apps/runner/config/ImageCls.txt`）**
- 全 branch の `structure` 末尾 `> LinearOut` を除去（例: `... > GAP2D > DropHead > LinearOut` →
  `... > GAP2D > DropHead`）。`DropHead`/`Drop`（Dropout）は **body 末尾に残す**（BF16 でも Dropout は精度非依存）。
- `net.body.output.[logits] = main_feature` → **`net.body.output.[features] = main_feature`**（head 期待キー）。
- `net.block.[LinearOut]`（out_features=101 / init.he）は不要化 → 削除（out_features は env 由来、init は head_factory）。
- BF16 セクション（現「★未対応」）に上記 3 設定を記載（既定は全 FP32）。

### フェーズ2: BF16 autocast を乗せる

**D. ImageClsAgentConfig（`image_cls_agent.hpp`）**
- `struct { bool enabled=false; bool learner=true; bool actor=false; } bf16;` を追加、`ANET_READ_CONFIG` 3 本。

**E. Learner（`UpdateFromBatch`）**
- `network_->Forward(obs)`（[image_cls_agent.cpp:344](../../../core/anet-core/src/image_cls_agent.cpp:344)）だけを
  `{ anet::Autocast g(device_.type(), config_.bf16.enabled && config_.bf16.learner, torch::kBFloat16); logits = Forward(obs); }`
  で囲む。**mix/loss/backward/clip/step はスコープ外**（logits は head 化で既に FP32）。
- GradScaler 不要。既存の backward→clip_grad_norm_→step のまま。

**F. Actor（`MakeAction`）**
- `network_->Forward(obs, sink)`（[image_cls_agent.cpp:58](../../../core/anet-core/src/image_cls_agent.cpp:58)）を
  `anet::Autocast g(device_.type(), config_.bf16.enabled && config_.bf16.actor, torch::kBFloat16)` で囲む。
  device は actor の `device_`（CPU eval 対応）。softmax/argmax は head 後 FP32 logits で従来どおり。

## 非対象（Out of Scope）

- **DQN `LinearHeadFactory`（`anet::rl::dqn`）の変更・汎用版への寄せ替え**（温存）。
- **FP16 AMP / GradScaler 経路**（BF16 固定）。
- 他 env（LunarLander/DropMerge 等）の head 化・BF16 化。
- per-block autocast 除外機構（body 途中で切る案は不採用）。
- **checkpoint 後方互換**（旧 body-only `.anet` の load 保証）。ImageCls は実験中で `auto_load` は基本 OFF。
- eval を BF16 にした際の accuracy の FP32 完全一致。

## 影響ファイル

| ファイル | 変更 |
|---|---|
| `core/anet-core/src/nn_heads.hpp` | 汎用 `LinearHead` / `LinearHeadFactory` 宣言（`anet::nn`、出力キー可変） |
| `core/anet-core/src/nn_heads.cpp` | 同実装（DQN 版と等価＋出力キー可変・he 初期化・GraphViz） |
| `core/anet-core/src/nn_test.cpp` | 汎用 `LinearHead` 単体テスト（`features[B,in]`→`out_key[B,out]`、キー可変を確認） |
| `core/anet-core/include/anet/image_cls_agent.hpp` | `ImageClsAgentConfig.bf16.{enabled,learner,actor}` 追加 |
| `core/anet-core/src/image_cls_agent.cpp` | `LinearHeadFactory` 配線＋`num_classes` 取得、Learner/Actor forward を `Autocast` 化 |
| `core/anet-core/src/image_cls_agent_test.cpp` | net 構造を head 化（`Flatten > LinearOut`→`Flatten`＋head_factory）、`bf16` config round-trip |
| `apps/runner/config/ImageCls.txt` | 全 branch の `LinearOut` 除去、`net.body.output.[features]`、`[LinearOut]` 削除、`bf16` 3 設定 |

## 受け入れ基準

**フェーズ1（head 化・FP32）**
1. ビルド緑（x64-Debug）、既存テスト緑。
2. 汎用 `LinearHead` 単体テスト: 入力 `features [B,in]` → `out_key [B,out]`、出力キーがコンストラクタ引数どおり、
   `he` 初期化で weight が生成されること。
3. ImageCls 起動の MODEL SHAPE DUMP（[image_cls_agent.cpp:481](../../../core/anet-core/src/image_cls_agent.cpp:481)）に
   `head.linear.weight [101, in]` が出る。Actor/Learner の forward が通り logits `[B,101]`。GraphViz に head 情報が出る
   （[`nn_viz.show_head_info`](../../../apps/runner/config/ImageCls.txt:130) が効くようになる）。
4. **FP32 等価**: head 化前後で、同 seed・同設定の学習曲線が終盤ブレ幅内で一致（構造変更が挙動を変えないこと）。

**フェーズ2（BF16）**
5. `bf16.enabled=true, bf16.learner=true` で学習が回り、`90_perf/12_exp_step_per_sec`
   （[ImageCls.txt:465](../../../apps/runner/config/ImageCls.txt:465)）が FP32 比で改善（compute-bound ゆえ効く見込み）。
6. **精度保護の確認**: logits/loss が FP32（autocast スコープ外＋head=FP32）であること。eval accuracy が FP32 比で
   終盤ブレ幅内。学習劣化が無い（あれば下記リスク参照）。
7. `bf16.enabled=false` で完全に従来経路（全 FP32）に戻ること。

## 正直なリスク

- **BF16 が割に合わない可能性**: 作業メモ（LunarLander の BF16 調査）では「CPU 律速で 1.2x どまり＋学習劣化」だった。
  ただし ImageCls は Learn が GPU compute-bound で文脈が異なり、効く見込み。**seed 違い複数 run の終盤平均ブレ幅**で判断し、
  劣化なら `bf16.enabled=false` で即戻し（構成比較はブレ幅基準）。
- **autocast スコープに backward を含める実装ミス**は勾配 BF16 化＝劣化に直結。E/F で **forward だけ**を囲むこと。
- **head 化で checkpoint 非互換**（body-only → body+head）。旧 `.anet` の load は壊れる。`auto_load` は基本 OFF なので許容。
- **`num_classes` の env 由来配線**が現 `out_features=101` と一致すること（過渡期は DUMP で二重確認）。
- BF16 は非決定性を増やすが、ImageCls は既に `deterministic_algorithms=false` 運用（ブレ幅基準）なので方針と整合。

## 検証

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test'
core\anet-core\bin\Debug\anet-core-test.exe "[nn]"
core\anet-core\bin\Debug\anet-core-test.exe "[image_cls]"
core\anet-core\bin\Debug\anet-core-test.exe
git diff --check
```

- 機能確認は runner を online モードで起動し、MODEL SHAPE DUMP で `head.linear [101,in]` と logits `[B,101]` を確認。
- perf/精度（exp_step_per_sec / eval accuracy）はユーザーが seed 違い複数 run の終盤平均で評価。

## 後続

1. フェーズ1 → FP32 等価を確認 → フェーズ2（BF16）→ ユーザー perf/精度評価。必要なら `023..._20impl.md` 相当の実装メモ。
2. 効果が確認できたら、DQN 側 `LinearHeadFactory` を汎用 `anet::nn` 版へ寄せる検討（別 PRD・本書非対象）。
3. `bf16.actor=true`（eval/GUI も BF16）の採否は、eval accuracy のブレを見てから判断。
