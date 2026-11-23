# Mixup / CutMix 導入（ImageCls）

> 設計分担: Claude=設計/PRD、実装=Codex、Run/commit=ユーザー。本書は self-contained。
> 基準コミット: HEAD `420cdb8`（`net.config_profileを追加`）。行番号は同 HEAD 基準（実装時は近傍のシンボル名で再検索すること）。
> 動機: ImageCls（Food-101, スクラッチ）で ConvNeXt(cn-nano/femto) が eval ~57% 頭打ち・train ~96%（gap ~39pt＝全run最大）の過学習。ResNet18ish(70-77%)に 13-20pt 負ける主因は「ConvNeXt が BatchNorm を持たず、Mixup/CutMix 級の強拡張を前提に設計されている」点。現状の拡張は hflip + RandomResizedCrop のみ。**Mixup/CutMix は過学習 gap を直接叩く最大レバー**であり、ConvNeXt/ViT に実質必須、ResNet 系にも上乗せが見込める（共通投資）。

## Context（背景・目的）

`022_imagecls_augmentation_10prd.md`（hflip + RRC、`ImageClsEnv::FetchRandomImageState` の train 経路に per-sample 適用）で eval 0.14→0.45 と過学習を大きく削ったが、ConvNeXt では容量に対して拡張が弱く再び train 96% まで暗記する。Mixup（2枚を線形ブレンド）と CutMix（矩形パッチ貼り替え）は **バッチ内のサンプル対を混合し、ラベルも混合する**強拡張で、暗記を構造的に不能にする。

**配置の判断（重要）**: 022 の per-sample 拡張は Env 側だが、Mixup/CutMix は
1. **バッチ単位**（バッチ内でペアを作る）で、
2. **loss を混合ラベルにする**（ソフトターゲット）
必要があり、Env（1サンプルずつ experience を吐く層）では扱えない。バッチが揃い loss を計算する **`ImageClsLearner::UpdateFromBatch`（`core/anet-core/src/image_cls_agent.cpp:76-133`）に実装する**（timm/torchvision も dataset でなく訓練ループ側に置く流儀）。

処理パイプラインは `ImageDataSource` の読み込み/resize -> Env 側 train-only RandomResizedCrop/horizontal flip -> Runner が作る mini-batch -> Learner 側 MixUp/CutMix -> NN 境界の `uint8 / 255` と整理する。
MixUp/CutMix は Agent/Learner 側の学習入力変換であり、Env の observation contract や `ImageClsView` の表示対象へ逆流させない。

## 確定した設計判断

1. **実装点は Learner の `UpdateFromBatch`**（Env は無改造）。per-sample 拡張(022)と直交し、両方同時に効く。
2. **loss は「2つの CE の凸結合」方式**（timm 標準）: one-hot ソフトラベル CE を新設せず、既存の
   `cross_entropy(logits, targets, label_smoothing)` を **2回**呼んで重み付け合成する。label_smoothing はそのまま両項に効く。
   ```
   loss = lam * CE(logits, targets) + (1 - lam) * CE(logits, targets_perm)
   ```
3. **画像の混合は uint8 の grid テンソル上で行う**。obs の grid は `uint8 [B,3,224,224]`（`net.detail.dot` の input grid=`[3,224,224] Byte`、NN境界の `NetworkBoundaryPreprocessor` が `÷255` する）。
   - **CutMix**: `image[i]` の矩形領域に `image[perm[i]]` の同座標パッチを貼る（uint8 で完全可逆、精度劣化なし）。`lam` は実際のパッチ面積比で補正する。
   - **Mixup**: `out = lam*a + (1-lam)*b` を float で計算し uint8 へ round/clamp/cast。`÷255` 前後で `lam*(a/255)+(1-lam)*(b/255)` と数学的に一致（丸め誤差のみ、無害）。
   - **grid の uint8→network `÷255` 契約は不変**（float を network に流し込む改造はしない）。
4. **1バッチにつき Mixup か CutMix のどちらか一方を確率的に選ぶ**（timm 準拠）。`prob` で「適用するか（=素通しか）」、`switch_prob` で「適用する場合に CutMix を選ぶ確率」を制御。両方の alpha は独立。
5. **mini-batch 内 permutation で全サンプルを混合する**。`B` 件の入力から `B` 件の mixed batch を作る。`B=128` なら 64 ペアへ減らさず、各 `i` に `perm[i]` を対応させて `mixed[i]` を作る。partner は current mini-batch 内から選び、全 dataset から追加ロードしない。
6. **self-pair は許容する**。`perm[i] == i` の場合はそのサンプルだけ実質的に素通しになるが、発生頻度は低く通常の unaugmented sample と同等に扱う。`B < 2` では MixUp/CutMix を bypass する。
7. **再現性**: `lam`・コイン投げ・permutation・CutMix bbox はすべて **Learner の `RandomHolder` が持つ seed 付き乱数**から生成する（[[project_repro_determinism]] の通り本プロジェクトは同seed再現を要件とする）。Learner seed は `ImageClsAgent` の実効 seed（`GetSeed()`）から `SeedMaker(...).MakeNamedSeed("learner")` で派生する。`prob`/`switch_prob` 判定、`lam` の Beta sampling、CutMix bbox 座標計算は `anet::RandomGenerator` の CPU 側乱数で行い、permutation は同 generator の device 別 `torch::Generator` で `torch::randperm` する。**torch のグローバル RNG は使わない**（他スレッドと干渉し再現性を壊す）。
8. **重い batch tensor 演算は GPU 側で行う**。`torch::randperm`、`index_select`、MixUp の blend/round/clamp/cast、CutMix の patch copy、Forward/loss/backward は `device_` 上で実行する。実時間差が小さい scalar/bbox 計算は、可読性と再現性を優先して CPU に残す。GPU-only augmentation path は v1 では採用しない。
9. **常設メトリクスは `target_prob_mix` だけ追加する**。`target_prob_mix = lam * p[target_a] + (1 - lam) * p[target_b]` とし、Mix 無効時は既存 hard target の確率と同義にする。`mix.lambda` や `mix.mode` は分布・設定から自明、またはデバッグ用途なので常設 scalar にはしない。
10. **Verbose ログは Learner 全体のログとして設定時だけ出す**。`ImageClsAgent.learn_log_interval` で `LOG::verbose()` の出力間隔を制御し、`0` は無効にする。`mixup.enabled=false` でも、loss / accuracy / target_prob_mix / step など Mix 以外の learner 情報は出す。Mix 関連項目は有効時または bypass 時の付加情報として扱う。
11. **accuracy メトリクスは混合前の元 `targets` に対して計算**（混合後だと train accuracy が無意味になるため、`targets` のコピーを混合前に退避）。混合ONでも train accuracy は解釈可能な値のまま（暗記が効かないので下がるが、それが正しい挙動）。
12. **`ImageClsView` は pre-mix 表示のまま維持する**。現行 View は Env 由来の observation、hard label、Actor 推論結果を表示する。Learner 内で作る mixed image / lambda / paired target の表示は本 PRD の対象外とし、`999_agent_update_result_view_10prd.md` の後続候補に分離する。
13. **デフォルトは無効（`enabled=false`）で後方互換**。既存 branch/run の挙動を一切変えない。ConvNeXt run でのみ `E.` オーバーレイ等で ON にする（設計方針D）。
14. **ADR は不要**。強化学習ドメイン用語の追加でなく、既存教師あり学習ループへの局所的な拡張。

## 前提事実（実コード確認済み）

- **`ImageClsLearner::UpdateFromBatch`**（`image_cls_agent.cpp:76-133`）現状:
  ```cpp
  auto vector  = experiences.state.obs.At(anet::rl::ObsKeys::kVector);
  auto targets = vector.to(device_).squeeze(-1).to(torch::kInt64);       // [B] クラスindex
  ...
  auto obs     = experiences.state.obs.To(device_);
  auto outputs = network_->Forward(obs);
  logits = outputs.At("logits");                                          // [B,101]
  auto loss_opts = torch::nn::functional::CrossEntropyFuncOptions().label_smoothing(config_.label_smoothing);
  loss = torch::nn::functional::cross_entropy(logits, targets, loss_opts);
  loss.backward();
  ```
- **obs の grid キー**: `anet::rl::ObsKeys::kGrid = "grid"`、`kVector = "vector"`（`core/anet-core/include/anet/rl.hpp:187-188`）。grid の取り出しは 022 実装や当ファイルの commented `experiences.state.obs.At(anet::rl::ObsKeys::kGrid)`（`image_cls_agent.cpp:84`）に前例。
- **obs は TensorDict**。grid を混合したものへ差し替えて Forward する必要がある（`obs.Set(kGrid, mixed_grid)` 等、TensorDict の可変API は既存の `info.Set(...)`（`image_cls_agent.cpp:51`）に倣う）。
- **seed の供給**: `ImageClsAgent` は `AgentBase` / `RandomHolder` 経由で実効 seed を持つ。Learner 用には `CreateLearner()` で `GetSeed()` を取得し、`SeedMaker(GetSeed()).MakeNamedSeed("learner")` で派生 seed を作って渡す。現状 Learner は `RandomHolder` 未継承なので**引数追加と継承追加が必要**。
- **`ImageClsAgentConfig`**（`image_cls_agent.hpp:18-44`）は `ANET_READ_CONFIG` でフラット/ドット階層のフィールドを読む。`nn_viz.*` のように**サブ構造体をぶら下げる前例あり**（`mixup.*` も同型で追加可能）。
- label_smoothing は既に `config_.label_smoothing=0.1`（判断2の2項 CE 双方に効く）。

## 設計方針

### A. Config 追加（`image_cls_agent.hpp` の `ImageClsAgentConfig`）

```cpp
struct MixupConfig {                    // ImageClsAgentConfig 内にネスト（nn_viz と同じ流儀）
    bool   enabled      = false;        ///< 既定OFF（後方互換）
    double mixup_alpha  = 0.2;          ///< Mixup の Beta(α,α)。0で Mixup 無効
    double cutmix_alpha = 1.0;          ///< CutMix の Beta(α,α)。0で CutMix 無効
    double prob         = 1.0;          ///< バッチ単位で混合を適用する確率（残りは素通し）
    double switch_prob  = 0.5;          ///< 混合時に CutMix を選ぶ確率（1-switch_prob で Mixup）
};

int learn_log_interval = 0;             ///< Verbose learner log の learn 間隔。0で無効
```
`ImageClsAgentConfig` に `MixupConfig mixup;` と top-level の `int learn_log_interval` を追加し、ctor で
```cpp
ANET_READ_CONFIG(config_data, mixup.enabled);
ANET_READ_CONFIG(config_data, mixup.mixup_alpha);
ANET_READ_CONFIG(config_data, mixup.cutmix_alpha);
ANET_READ_CONFIG(config_data, mixup.prob);
ANET_READ_CONFIG(config_data, mixup.switch_prob);
ANET_READ_CONFIG(config_data, learn_log_interval);
```
を読む。範囲検証: `prob`/`switch_prob` は `[0,1]`、`*_alpha` は `>=0`、`learn_log_interval` は `>=0`。範囲外は `ANET_SYSTEM_ERROR`。

### B. Learner への seed 供給

`ImageClsLearner` は `anet::RandomHolder` を継承し、ctor に `std::optional<anet::seed_t> seed` 引数を追加して `RandomHolder(seed)` へ渡す。Learner 自身は `RandomGenerator` メンバを別途持たず、`UpdateFromBatch` で `GetRandomGenerator()` を取得して `prob` / `switch_prob` / `lam` / bbox / permutation に使う。

`ImageClsAgent::CreateLearner()` は Agent の seed を別メンバとして保存せず、`GetSeed()` で Agent の実効 seed を取得し、`anet::SeedMaker(GetSeed()).MakeNamedSeed("learner")` で Learner seed を派生して渡す。Agent seed 未指定時は `AgentBase` / `RandomHolder` の既存 auto-seed に従う。同じ Agent seed を指定した run では同じ Learner seed になり、seed 未指定 run は自動 seed として扱う。

乱数と実行場所の分担:

| 処理 | 実行場所 | 理由 |
|---|---|---|
| `prob` 判定 / `switch_prob` 判定 | CPU | scalar 制御で、GPU 化しても実時間差がない |
| Beta sampling による `lam` | CPU | `std::gamma_distribution` で実装しやすく、seed 再現性を追いやすい |
| CutMix bbox 座標計算 | CPU | 座標 4 個だけなので可読性を優先する |
| `torch::randperm` | GPU | `RandomGenerator` の device 別 `torch::Generator` を使い、batch tensor の index としてそのまま `device_` 上で使う |
| `index_select` / MixUp blend / CutMix patch copy | GPU | 画像 batch 本体の重い tensor 演算 |

### C. 混合本体（`UpdateFromBatch` 内、Forward 前に挿入）

擬似コード（Codex が TensorDict/torch API に合わせて実装）:
```cpp
auto obs      = experiences.state.obs.To(device_);
auto grid     = obs.At(ObsKeys::kGrid);          // uint8 [B,3,H,W]
auto targets  = vector.to(device_).squeeze(-1).to(torch::kInt64);  // [B]
auto acc_targets = targets;                       // accuracy 用に元ラベルを退避（判断6）
auto rnd = GetRandomGenerator();

double lam = 1.0;
torch::Tensor targets_b = targets;                // 既定は自分自身（混合なし）

const int64_t B = grid.size(0);
if (config_.mixup.enabled && B >= 2 && Bernoulli(config_.mixup.prob)) {
    // current mini-batch 内で partner を作る。全 dataset から追加ロードしない。
    // self-pair は許容し、batch size は B のまま維持する。
    auto gen = rnd->GetTorchGenerator(device_);
    auto perm = torch::randperm(B, gen, torch::TensorOptions().device(device_).dtype(torch::kInt64));
    targets_b = targets.index_select(0, perm);

    bool use_cutmix = Bernoulli(config_.mixup.switch_prob) && config_.mixup.cutmix_alpha > 0.0;

    if (use_cutmix) {
        lam = SampleBeta(config_.mixup.cutmix_alpha);           // CPU scalar
        const auto box = SampleCutMixBox(H, W, lam);            // CPU で λに応じた矩形を作る
        grid.index({Slice(), Slice(), Slice(box.y1, box.y2), Slice(box.x1, box.x2)})
            = grid.index_select(0, perm).index({Slice(), Slice(), Slice(box.y1, box.y2), Slice(box.x1, box.x2)});
        lam = 1.0 - double((box.x2 - box.x1) * (box.y2 - box.y1)) / double(H*W); // 実面積比で補正
    } else if (config_.mixup.mixup_alpha > 0.0) {
        lam = SampleBeta(config_.mixup.mixup_alpha);            // CPU scalar
        auto a = grid.to(torch::kFloat32);
        auto b = grid.index_select(0, perm).to(torch::kFloat32);
        grid = (a.mul(lam) + b.mul(1.0 - lam)).round_().clamp_(0,255).to(torch::kUInt8);
    }
    obs.Set(ObsKeys::kGrid, grid);                 // 混合後 grid で Forward
}
```
補助関数は `ImageClsLearner` の private member / private static member に置く。RNG を使う補助関数は `RandomHolder` 経由で乱数を取る private member とし、純粋計算だけ static member に分ける。混合本体は `ApplyMix` private member とする。無名名前空間は使わない（[[feedback_avoid_anonymous_namespace]]）。
- `SampleBeta(a)`: `RandomHolder` の RNG から gamma sample を2回引き `g1/(g1+g2)`。α=α の対称 Beta。
- `SampleCutMixBox(H,W,lam)`: `cut_ratio = sqrt(1-lam)`、中心一様、`[0,H]/[0,W]` にクランプ（timm `rand_bbox` と同一）。
- `Bernoulli(p)`: `RandomHolder` の RNG から一様乱数を引き `uniform(0,1) < p`。

`B < 2`、`mixup.enabled=false`、`prob` 判定で不適用、または該当 alpha が 0 の場合は、従来と同じ `obs` / `targets` のまま Forward する。

### D. Loss と accuracy（`UpdateFromBatch` の loss 計算差し替え）

```cpp
auto ce = [&](const torch::Tensor& t){
    auto o = torch::nn::functional::CrossEntropyFuncOptions().label_smoothing(config_.label_smoothing);
    return torch::nn::functional::cross_entropy(logits, t, o);
};
loss = (lam >= 1.0) ? ce(targets)                          // 混合なしは従来通り1回
                    : ce(targets) * lam + ce(targets_b) * (1.0 - lam);
...
// accuracy は退避した元ラベルに対して（判断6）
auto preds = logits.argmax(1);
float accuracy = (preds == acc_targets).to(torch::kFloat32).mean().item<float>();

// Mix soft target に対してモデルが置いた確率
auto probs = torch::softmax(logits, 1);
auto prob_a = probs.gather(1, targets.unsqueeze(1)).squeeze(1);
auto prob_b = probs.gather(1, targets_b.unsqueeze(1)).squeeze(1);
float target_prob_mix = (prob_a * lam + prob_b * (1.0 - lam)).mean().item<float>();
```

`ImageClsUpdateResult` には既存の `loss` / `accuracy` に加えて `target_prob_mix` だけを追加し、`GetScalar("target_prob_mix")` で返す。
`mix.lambda`、`mix.mode`、`loss_a`、`loss_b`、`same_class_pair_ratio` などは、通常運用での意思決定に対して情報量が薄いかデバッグ用途なので常設メトリクスにはしない。

### E. Verbose learner log

`config_.learn_log_interval > 0` のときだけ、`step.learn_step % learn_log_interval == 0` で `LOG::verbose()` を 1 行出す。
`learn_log_interval = 1` なら 1 learn 1 行、`0` なら無効。
このログは MixUp 専用ではなく Learner 全体の状態確認ログなので、`mixup.enabled=false` でも出力対象にする。

共通ログ項目:

- `exp_step`: `step.exp_step`。
- `learn_step`: `step.learn_step`。
- `B`: mini-batch size。
- `loss`: 最終 loss。
- `accuracy`: 元ラベル基準の train accuracy。
- `target_prob_mix`: mixed target 確率。Mix 無効時は元ラベル確率。
- `lr`: 現在の learning rate。

Mix 関連の付加項目:

- `mix`: `disabled` / `none` / `mixup` / `cutmix`。
- `lambda`: Mix が実際に適用された場合だけ。CutMix は面積補正後。
- `self_pairs`: Mix が実際に適用された場合だけ、`perm[i] == i` の件数。
- `same_class_pairs`: Mix が実際に適用された場合だけ、`target_a == target_b` の件数。
- `bbox`: CutMix の場合だけ `(x1,y1)-(x2,y2)`。
- `reason`: `mix=none` の場合だけ `prob` / `batch_size` / `alpha` など。

例:

```text
ImageClsLearner learn: exp_step=123456 learn_step=1200 B=128 loss=3.421 accuracy=0.188 target_prob_mix=0.092 lr=0.00096 mix=cutmix lambda=0.684 self_pairs=1 same_class_pairs=2 bbox=(34,18)-(146,130)
ImageClsLearner learn: exp_step=123584 learn_step=1201 B=128 loss=3.508 accuracy=0.164 target_prob_mix=0.087 lr=0.00096 mix=mixup lambda=0.421 self_pairs=0 same_class_pairs=1
ImageClsLearner learn: exp_step=123712 learn_step=1202 B=128 loss=3.337 accuracy=0.211 target_prob_mix=0.103 lr=0.00096 mix=none reason=prob
ImageClsLearner learn: exp_step=123840 learn_step=1203 B=128 loss=3.291 accuracy=0.219 target_prob_mix=0.106 lr=0.00096 mix=disabled
```

このログは実装確認用であり、統計分析は `target_prob_mix`、既存 `loss`、既存 `accuracy`、eval accuracy で行う。
### F. ImageCls.txt での有効化（`apps/runner/config/ImageCls.txt`）

ConvNeXt run 用に、`# 実験` セクション（`ImageCls.txt:35-43` 付近、`ImageClsEnv.$ = ImageClsEnv.baseline > E` のオーバーレイと同様に）Agent 側も直接指定で ON にする。**A/B は 1 個ずつ**（[[feedback_compare_with_run_variance]]）:
```
ImageClsAgent.mixup.enabled      = true
ImageClsAgent.mixup.mixup_alpha  = 0.2
ImageClsAgent.mixup.cutmix_alpha = 1.0
ImageClsAgent.mixup.prob         = 1.0
ImageClsAgent.mixup.switch_prob  = 0.5
#ImageClsAgent.learn_log_interval = 1,000
```
（論文 ConvNeXt は mixup 0.8 / cutmix 1.0。小データでは過強の恐れがあり、まず mixup_alpha 0.2 の弱めから。効けば 0.8 へ上げて再A/B。）

### G. ImageClsView との関係

現行の `ImageClsView` は Env 側 View として、`TrainEvent.experience.state.obs` と `action_info` を表示する。
この PRD では `ImageClsView` を変更しない。表示される画像は pre-mix の Env observation であり、loss に使った mixed image ではない。

MixUp/CutMix の mixed image、lambda、paired target、CutMix bbox を確認したい場合は、Agent/Learner 側の update result 由来データを扱う別経路が必要である。
この論点は `999_agent_update_result_view_10prd.md` に暫定 PRD として分離済み。

## 影響ファイル

| ファイル | 変更 |
|---|---|
| `core/anet-core/include/anet/image_cls_agent.hpp` | `MixupConfig` ネスト構造体追加、`ImageClsAgentConfig` に `mixup` / `learn_log_interval` フィールド＋`ANET_READ_CONFIG`、`ImageClsUpdateResult` に `target_prob_mix`、`ImageClsLearner` ctor に seed 引数＋`RandomHolder` 継承 |
| `core/anet-core/src/image_cls_agent.cpp` | `UpdateFromBatch` に混合ロジック（C）＋loss/accuracy/target_prob_mix 差し替え（D）、Verbose learner log（E）、`CreateLearner()` で Agent 実効 seed 由来の Learner seed を派生、`ImageClsLearner` private member/static member の補助関数 |
| `core/anet-core/src/image_cls_agent_test.cpp`（無ければ新設 or 既存テスト場所） | 単体テスト（下記） |
| `apps/runner/config/ImageCls.txt` | 有効化キー（E、既定はコメントアウトで維持） |

## 受け入れ基準

1. **ビルド緑**（x64-Debug）。
2. **後方互換**: `mixup.enabled=false`（既定）で loss/accuracy/挙動が現行と bit 一致（混合コードを完全バイパス）。既存 ResNet/ConvNeXt run の config dump に `mixup.*` が既定値で載るのみ。
3. **再現性**: 同 seed の 2 run で、混合ON時も学習曲線が一致（`lam`/perm/bbox がグローバル RNG 非依存）。[[project_repro_determinism]] の `setDeterministicAlgorithms` 設定と両立。
4. **batch size 不変**: `B >= 2` で MixUp/CutMix を適用しても、Forward に渡す grid と targets は入力と同じ batch size を維持する。
5. **self-pair 許容**: `perm[i] == i` が発生しても失敗せず、そのサンプルは実質素通しとして扱われる。
6. **`B < 2` bypass**: `B < 2` では mix を bypass し、従来 CE と一致する。
7. **Mixup 数値**: `switch_prob=0`（Mixup固定）・既知 seed で、GPU 上の混合後 grid が `round(lam*a+(1-lam)*b)` と一致。`lam∈[0,1]`。
8. **CutMix 数値**: `switch_prob=1`（CutMix固定）で、GPU 上の貼替え領域外が元画像と一致・領域内が perm 画像と一致し、`lam == 1 - area/(H*W)`。
9. **Loss 合成**: `lam=1`（prob=0）で従来 CE と一致。`0<lam<1` で `lam*CE_a+(1-lam)*CE_b` に一致。label_smoothing が両項に効く。
10. **target_prob_mix**: `target_prob_mix` が `lam * p[target_a] + (1 - lam) * p[target_b]` の batch mean と一致する。Mix 無効時は元ラベル確率と一致する。
11. **accuracy**: 混合ON時も元ラベル基準で計算される（`acc_targets` 使用）。
12. **Verbose log**: `learn_log_interval=0` で出力されず、`learn_log_interval>0` で指定間隔の `LOG::verbose()` が出る。`mixup.enabled=false` でも Learner 全体の情報は出る。
13. **ImageClsView**: `ImageClsView` は pre-mix の Env observation 表示として維持され、mixed image 表示は本 PRD の受け入れ基準に含めない。
14. **Config 検証**: `prob`/`switch_prob` が `[0,1]` 外、`*_alpha<0`、`learn_log_interval<0` で `ANET_SYSTEM_ERROR`。
15. **実 run 起動**: `ImageClsAgent.mixup.enabled=true` で ConvNeXtNano runner がクラッシュせず数百 step 回り、train accuracy が混合なし比で低下（暗記抑制の兆候）することを確認。

## テスト項目リスト

1. enabled=false で混合完全バイパス（bit 一致）
2. `B < 2` で混合 bypass
3. `B >= 2` で output batch size が入力 batch size と一致
4. self-pair を許容しても失敗しない
5. Mixup ブレンド数値（既知 seed）
6. CutMix パッチ貼替え＋λ面積補正
7. loss 凸結合（lam=1 / 0<lam<1）
8. `target_prob_mix` が mixed target 確率の batch mean
9. accuracy が元ラベル基準
10. Verbose learner log の interval / disabled 動作。`mixup.enabled=false` でも Learner 情報が出ること
11. 同 seed 再現性（混合ON、2 run 一致）
12. config 範囲外 fail-fast
13. permutation が seed 付き generator 由来（グローバル RNG 不変を確認）
14. `ImageClsView` が mixed image ではなく pre-mix observation を表示する前提を維持

## 正直なリスク / 注意

- **partner は mini-batch 内に限定**: MixUp/CutMix の相方は current mini-batch 内から選ぶ。train mini-batch は Env/Runner 側で十分ランダムに構成される前提なので、一般的な batch-level MixUp/CutMix と同等の運用とみなす。batch size が極端に小さい場合は partner 多様性が落ちる。
- **self-pair**: self-pair は一部サンプルの素通しとして扱う。厳密に避ける derangement は v1 では採用しない。
- **uint8 Mixup の丸め**: float ブレンドを uint8 へ丸めるため 0.5LSB 程度の誤差。学習に無害だが、厳密一致テストは float 中間値で比較すること。
- **小データでの過強**: ConvNeXt 論文値（mixup 0.8/cutmix 1.0, prob 1.0）は ImageNet 前提。Food-101 75k では強すぎて収束が遅くなる/eval が伸び切らない可能性。**弱め（mixup_alpha 0.2）から A/B**。
- **train accuracy の解釈変化**: 混合ONで train accuracy は必ず下がる（混合画像に元ラベルを当てるタスクになるため）。gap の縮小と **eval の上昇**で判断すること（train 単独で回帰と誤認しない）。
- **target_prob_mix の解釈**: `target_prob_mix` は mixed target に対するモデル確率を見る補助指標であり、eval accuracy の代替ではない。Mix が強い場合の train accuracy 低下を読むための補助線として使う。
- **Verbose log の量**: `learn_log_interval=1` は 1 learn 1 行になるため長時間 run では重い。通常は初期確認時だけ小さくし、通常 run では `0` または大きめの interval にする。MixUp 専用ではなく Learner 全体ログなので、MixUp 無効 run でも出力量に注意する。
- **MixUp/CutMix 可視化**: `ImageClsView` は mixed image を表示しない。混合結果の可視確認は `999_agent_update_result_view_10prd.md` の Agent update result 可視化で扱う。
- **step/sec への影響**: バッチ単位の index_select/コピーは軽微。env 律速（[[project_device_transfer_prd]]）に対して無視できるが、実測で確認。
- **cosine LR（031）との併用**: 独立に A/B 可能だが、強拡張は収束を遅らせるため、Mixup 有効時は 031 の総 step を伸ばすと効きやすい（本書単体では step 延長は必須でない）。

## 検証

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test'
core\anet-core\bin\Debug\anet-core-test.exe "[imagecls][mixup]"
core\anet-core\bin\Debug\anet-core-test.exe
git diff --check
```
- 機能確認: `ImageClsAgent.mixup.enabled=true` で ConvNeXtNano runner を起動し、`runs/<name>/config/config_data.txt` に `mixup.*` が ground truth として載ること、train accuracy が混合なし比で低下し数百 step クラッシュしないことを確認。精度評価（eval 上昇・gap 縮小）はユーザーが別途実施（[[feedback_compare_with_run_variance]]、終盤 EMA で判断）。

## 後続

1. 効果確認後、`mixup_alpha` を 0.2→0.8 へ上げて再 A/B、ResNet18ish_hr にも適用して上乗せを確認。
2. RandAugment / Random Erasing の追加（本書スコープ外、別 PRD 候補）。
3. `031_imagecls_cosine_lr_10prd.md`（cosine LR + warmup）と組み合わせた本命レシピでの再評価。
