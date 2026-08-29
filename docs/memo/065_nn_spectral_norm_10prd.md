# NN Spectral Normalization（weight_norm_mode）PRD

> 起点: 2026-08-28〜29、高 `replay_ratio` Breakout 崩壊の機序確定（①重み成長→②活性成長→③ReLU恒久死→④表現痩せ）と
> weight_decay の限界実証（[探索ブロック 04 / 05](../experiments/default-dqn/atari/2026-08-28_plasticity.md)）。
> 裁定: 2026-08-29 グリル（D1〜D10 + 簡素化監査 6 項目）で全決定済み。設計分担: Claude=設計/PRD、実装=Codex。本書は self-contained。
> 関連: [ADR 0032](../adr/0032-spectral-norm-self-impl-buffer-semantics.md)（本 PRD の決定記録）、
> [062](done/062_plasticity_metrics_10prd.md)（可塑性メトリクス。本 PRD は「保護機構の効果測定器」の最初の利用者）、
> [063](done/063_plasticity_weight_norm_10prd.md)（weight norm 2 群。本 PRD のメトリクス増分は 63 の同型拡張）、
> [2026-08-17_baseline.md](../experiments/default-dqn/atari/2026-08-17_baseline.md)（探索ブロック 15: BTR / BBF / 本記録の保護機構比較表）、
> [999_MunchausenRL_10prd.md](999_MunchausenRL_10prd.md)（同じ「コストだけ採って補償器を採っていない」系列の別件）。

## Context（背景・目的）

### 機序と weight_decay の限界

高 `replay_ratio`（RR≥4 目安）の Breakout で起きる崩壊の機序は次の一本道に絞れている
（数値の正本は [2026-08-28_plasticity.md](../experiments/default-dqn/atari/2026-08-28_plasticity.md)）:

```
① 重みが止まらず育つ → ② 活性が育つ → ③ ReLU の負側で恒久的に死ぬ → ④ 表現が痩せる → スコア低下
```

weight_decay は①のレバーとして正しいことが実証済み — `61_weight_norm_feature` を**用量順に完全制御**した
（1.47 → 0.94、複製 0.79 / 0.80。探索ブロック 04）。しかし 10M では **V 字**（45.8 → 30.5 → 43.4 → 41.4。探索ブロック 05）。
λ·w は w が縮むほど弱まる**均衡機構**なので、勾配側の押し上げと釣り合う点で止まり、早期に稼いだ分の大半を押し戻される。

Spectral Normalization（SN）はペナルティではなく**毎 step の射影**（forward が常に W/σ を使う）なので、この均衡が存在しない。
BTR は SN を全 conv residual layer に適用して RR=4 で走れており、本コードベースは「BTR の γ 0.997 を Munchausen なしで、
高 RR を SN なしで」回している構図が記録済み（[2026-08-17_baseline.md](../experiments/default-dqn/atari/2026-08-17_baseline.md) 探索ブロック 15）。

### 部分適用の罠（D1 の背景）

2026-08-28 の GroupNorm 試験は「ResBlock 内部のみ適用で、ダウンサンプリング Conv 3 本と最終 Linear512 が素のまま」
だったため公正な検証にならなかった（探索ブロック 04 考察）。‖w‖ 成長を無害化する機序が、測っている特徴（`main_feature` =
`AtariLinear512 > ReLU` の出力）の直前で成立していなかった。本 PRD は適用範囲を config の判断にし、この罠を設計から消す。

### 本 PRD の中核 = メトリクスの帰結

SN の標準実装は生の W をパラメータとして保持し、forward で W/σ を使う。したがって SN 下では:

- 生パラメータのノルム（現行 `61_weight_norm_feature` の測定対象）→ **制約されず伸び続ける**
- forward が実際に使う実効重み → σ で固定される

つまり「①のブレーキが効いたか」を測るはずの 61 が何も語らなくなる。この帰結への裁定（D8）が本 PRD の中核。

## 数理と適用の定義

### weight_norm_mode

重み所有ブロックの**重み正規化モード**を表す文字列 config。`none`（既定 = 現行動作、正規化なし）/ `spectral`（SN）。
未知値は fail-fast。モード空間は将来の拡張（例: Salimans & Kingma の Weight Normalization → `direction` 等）を想定した
命名で、bool flag にしない。

SN は「活性を変換する層」ではなく「**重みの再パラメータ化**」なので、構造チェーン上の独立ブロック
（`Conv2d > SpectralNorm` のような後置）では原理的に実現できない — 後段に届くのは conv の出力テンソルであり、
W/σ での計算は W を所有するモジュール自身の forward の中でしか起きられない（libtorch C++ には Python の
parametrization フックが無い。§事実 1）。よって mode は各ブロックの config になる。

### SN の数理契約（Miyato / PyTorch / BTR 準拠の標準一式）

対象重み W（conv は `(c_out, c_in·kh·kw)` へ view して行列扱い。bias は対象外）に対し:

```
power iteration（training mode の forward ごとに 1 回、NoGrad、eps=1e-12）:
    v ← normalize(Wᵀ u)
    u ← normalize(W v)
σ の計算（train / eval を問わず毎 forward、その場の W で）:
    σ = normalize(u) · (W normalize(v))        ※ u/v は detach 済み buffer、W は勾配経路あり
実効重み:
    W_eff = W / σ                              ※ 常時除算。forward は W_eff で計算する
```

- **勾配は σ 経由でも流す**（∂σ/∂W = uvᵀ。detach しない）。この項が支配的特異方向への成長を打ち消す実質の正則化で、
  PyTorch / BTR と同じ学習動態になる。
- **常時除算**なので σ<1 の層では W が「拡大」される side effect がある（成長を止めるだけでなく、常に σ=1 へ射影する）。
- power iteration の回数は 1 回/training forward の**固定**（config 化しない。§複雑度監査）。
- 1 update に複数の training mode forward が走る構成では u/v がその回数だけ進む。決定的であり問題ない
  （現行 learner で training mode に入る forward は `ForwardOnlineWithTrain` の 1 系統。§事実 4）。

### u/v の保持と σ の再計算（PyTorch parity + 本コードベース固有の 2 逸脱）

- buffer は **u / v の 2 本のみ**を named buffer として register する（σ は buffer にしない）。
  `requires_grad == false` なので既存の `ComputeParameterNormSplit` 集計（61/62）に混入せず（§事実 6）、
  CopyTo / SoftCopyTo / シリアライズは既存の named_buffers 走査が無償で運ぶ（§事実 2）。
- **power iteration（u/v の更新）は training mode の forward のみ**。target net は常時 eval（§事実 3）、
  actor snapshot・probe 部分 forward も eval なので、**u/v を変異させる経路は learner の online 学習 forward だけ**。
  eval forward は buffer を一切変更しない（062 の「測定が学習系列を変えない」契約と整合）。
- **逸脱 1: 使用時 normalize**。`SoftCopyTo` は float buffer を lerp するため（§事実 2）、soft update（Atari 実構成
  tau=0.001。§事実 5）で target 側の u/v が非単位ベクトル化しうる。σ 計算のたびに u/v を normalize してから使うことで
  任意の tau で頑健になる（tau=0.001 での偏差は ~1e-3·θ² と実質ゼロだが、契約として明示する）。
  soft update による u/v の lerp 継承自体は許容する — 一般実装も Polyak で buffer を放置 or lerp しており、
  「u/v は近似でよい、σ は使う場で W と突き合わせて再計算する」という PyTorch の意味論が受け皿になる。
- **逸脱 2: warm-start 初期化**。PyTorch は randn 初期化のみで、最初の training forward まで σ 推定がゴミになる
  （ランダム u₀ᵀWv₀ は真の σ を大幅過小評価 → W_eff が過大）。教師あり学習では顕在化しないが、RL は
  learning_starts 前に actor forward が走る構造なので当たる。対策: **weight 実体化と同時に u/v を
  randn-normalize で生成し、その場で NoGrad の power iteration を k 回（実装定数、k=3 目安）回す**。
  乱数は global RNG（`MasterSeedManager` が manual_seed 済み。§事実 7）で、構築・lazy init の順序は構成で
  固定されるため決定的。lazy init（Linear の in_features 自動推論等）の場合は weight 実体化時点で同時に行う —
  初期化は eval forward 中であっても 1 回だけ許される変異とし、以後の非 training forward は buffer を変更しない。
- actor snapshot は CopyTo の正確コピー。snapshot 間は W が凍結なので σ 再計算は同値を返すだけ（無駄だが無害。
  クローン時に σ を確定して再計算を skip する最適化は将来の余地として注記に留める）。
- 精度: power iteration・σ・除算は **FP32**（Autocast 局所 OFF + FP32 cast。`force_fp32` イディオム。§事実 8）。
  W_eff を使う conv / linear 演算自体は autocast に任せる（bf16 構成では従来どおり bf16 で走る）。

### 適用範囲（D1）

`weight_norm_mode` は**重み行列を所有する全登録ブロック型**が持つ:
**Linear / Conv1d / Conv2d / ResBlock / CNBlock / TransformerEncoder**（共通部品化。実際にどれで使うかは config の判断）。

一律規則（D11）: **粒度は 1 ブロック 1 mode** で、ON はそのブロックが所有する全**重み行列**（乗算パイプラインの行列）に
適用する。**bias・normalization affine・layerscale γ・embedding テーブル・cls token は対象外**（SN の対象は乗算
パイプラインの重み行列であり、文献も embedding / スケール系には掛けない）。各重み行列が独立の u/v を持つ。
ネットワーク内の空間的な選択はカタログの**ブロック・インスタンス単位**（`[ResA]` と `[ResB]` に別 mode）で行う。

| ブロック | config キー | ON 時の適用対象 |
|---|---|---|
| Linear | `linear.weight_norm_mode` | 自身の linear 重み |
| Conv1d / Conv2d | `conv.weight_norm_mode` | 自身の conv 重み（ConvConfig 共有のため両型同時配線。片方だけでは黙殺キーが生まれる） |
| ResBlock | `res.weight_norm_mode` | 内部の conv1 / conv2 / downsample **全部** |
| CNBlock | `cn.weight_norm_mode` | 内部の dwconv / pwconv1 / pwconv2 **全部**（layerscale γ・内部 norm は対象外） |
| TransformerEncoder | `tf.weight_norm_mode` | 全 layer の **Q / K / V**（packed `in_proj_weight` のスライスごとに独立 σ）+ out_proj + linear1 / linear2（norm affine は対象外）。**`spectral` × `use_sdpa=false` は fail-fast** |

depthwise conv（CNBlock の dwconv、`groups=channels`）も Miyato の reshape 規約 `(size(0), -1)` を一律適用する —
この行列化は dense conv でも畳み込みの真の作用素ノルムではなく代理であり、規約として一律にする。

Atari Impala backbone は standalone Conv2d（3 本）+ ResBlock 群 + Linear512 なので、
「BTR と同じ範囲（ResBlock のみ）」も「全層」も config の選択になる。

## 決定事項（2026-08-29 グリル + 簡素化監査 6 項目）

| # | 論点 | 裁定 |
|---|---|---|
| D1 | 適用範囲 | **重み行列を所有する全登録ブロック = 6 型**（Linear / Conv1d / Conv2d / ResBlock / CNBlock / TransformerEncoder）。当初は Atari baseline 使用の 4 型 + CNBlock/Transformer deferred としたが、2026-08-29 追加グリルで**共通部品化**（配線は全部・実使用は config の判断、型レベルの部分適用罠を残さない）へ改訂。独立 SpectralNorm ブロック（チェーン後置）は原理的に不成立で棄却、builder の wrap 機構（前ブロックを食う DSL）は 1関心2機構 + 新DSL で棄却。ConvConfig は Conv1d/Conv2d 共有 — 片方だけの配線は黙殺キーを生む（spec drift 監査で問題視した黙殺 fallback の再生産）ため両型同時 |
| D2 | config 形式 | bool flag でなく**文字列モード `weight_norm_mode = none \| spectral`**（既定 `none`）。拡張性（将来 `direction` 等）と house style（`norm_type` / `activation` / weight init mode が全て文字列モード）。値は綴り出し `spectral`（`sn` は将来 Salimans WN 追加時に値名が衝突気味になる） |
| D3 | 数理 | **標準一式**（常時除算 / 勾配は σ 経由でも流す / power iteration 1 回・NoGrad・eps=1e-12 / conv reshape / bias 対象外）。cap 型 `W/max(1,σ)` は文献非標準で BTR 再現の参照点を失うため棄却。σ detach は方向正則化項が消え学習動態が別物になるため棄却 |
| D4 | σ の意味論 | **PyTorch parity**: σ は buffer にせず**毎 forward その場の W で再計算**。buffer は u/v の 2 本のみ。対抗案の「σ を buffer に焼き非 training forward は読むだけ」（追加計算ゼロ）は、PyTorch と異なる独自意味論の記述コストが残るため棄却（[ADR 0032](../adr/0032-spectral-norm-self-impl-buffer-semantics.md)） |
| D5 | u/v 更新点 | **training mode の forward のみ**（= learner online 学習 forward だけ。target 常時 eval / actor / probe は非変異）。soft update の u/v lerp 継承は許容し、**使用時 normalize** で頑健化 |
| D6 | 初期化 | **warm-start**: weight 実体化時に randn-normalize + NoGrad power iteration k 回（実装定数 k=3 目安）。global RNG・構築順固定で決定的。ModuleContext の拡張はしない（空構造体のまま） |
| D7 | 精度 | power iteration / σ / 除算は **FP32 固定**（Autocast 局所 OFF。`force_fp32` イディオム踏襲）。conv/linear 演算は autocast 任せ |
| D8 | メトリクス | **4 本追加・既定コメントアウト**（§メトリクス拡張）。61/62 は無改修（生ノルムを測り続ける）。「61 を実効重みで計算し直す」案は既存 Run との互換破壊（61 の意味が Run 世代で変わる）で棄却 |
| D9 | 収集の口 | **interface 方式**: SN 保持モジュールが名前付き interface を実装し (weight, u, v) 列を返す。walk は dynamic_cast。buffer 命名規約のパターンマッチは文字列規約の暗黙契約化で棄却 |
| D10 | 受入 | §受入基準の 5 項目（OFF 完全不変 / ON 決定性 / smoke / 単体テスト / throughput 目視） |
| D11 | 粒度と境界（2026-08-29 追加グリル） | **1 ブロック 1 mode**（ON = 内部全重み行列。BTR 準拠のブロック一様。per-weight override キーは Gogianu 型の層選択実験 — 1 層だけ SN が全層適用を上回るケースの追試 — が pin されたら追加 = deferred gate）。対象は乗算パイプラインの**重み行列のみ**: bias・normalization affine・layerscale γ・embedding テーブル・cls token は対象外。TransformerEncoder は packed `in_proj_weight` を **Q/K/V 別 σ**（スライスごとに独立 state、実効 = cat(Wq/σq, Wk/σk, Wv/σv)。「1 重み行列 = 1 σ」の契約を全ブロックで揃える。packed のまま 1 σ は最大射影の σ で他 2 つも割る非標準結合で棄却）。**`spectral` × `use_sdpa=false` は fail-fast**（旧 MHA 経路は SDPA 等価性確認用の互換参照。functional 書き換え対応は必要が pin されたら） |

## 実装仕様

### 1. 共有ヘルパ（nn 機能グループ同居、名前付き namespace）

```cpp
namespace anet::nn {   // 実名は実装裁量（無名 namespace は使わない）

struct SpectralNormState {
    torch::Tensor u;   // (rows)  named buffer として所有モジュールに register
    torch::Tensor v;   // (cols)  同上
};

// weight を (weight.size(0), -1) へ view した行列 W について:
//   do_power_iteration=true なら NoGrad で v←normalize(Wᵀu), u←normalize(Wv) を 1 回実行（in-place）
//   戻り値 σ = normalize(u)·(W normalize(v))（0-dim FP32。u/v は detach、W には勾配経路を残す）
// 全体を Autocast 局所 OFF + FP32 で計算する（W が FP32 パラメータならゼロコスト cast）
torch::Tensor ComputeSpectralSigma(const torch::Tensor& weight, SpectralNormState& state, bool do_power_iteration);

// warm-start: u/v を randn-normalize で生成し、NoGrad で power iteration を k 回実行して返す
SpectralNormState MakeSpectralNormState(const torch::Tensor& weight);

}
```

- `do_power_iteration` は呼び出し側が `is_training()` を渡す（モジュールの mode がそのまま契約になる。§事実 3）。
- eps=1e-12、k は実装定数（`inline constexpr int kSpectralNormWarmStartIters = 3;` 等）。
- **使用時 normalize は out-of-place 必須**: normalize が新 tensor を作ることが、PyTorch 実装の `u.clone()`
  保護を兼ねる — power iteration は buffer を in-place 更新するため、buffer を直接 autograd graph に載せると
  次の forward の in-place 更新が version counter と衝突する。「既に単位ベクトルなら normalize を省く」
  最適化は**禁止**（保護が消える）。

### 2. メトリクス収集 interface（D9）

```cpp
struct SpectralNormEntry {
    const torch::Tensor* weight;
    const SpectralNormState* state;
};
class SpectralNormedModule {   // 実名は実装裁量
public:
    virtual ~SpectralNormedModule() = default;
    virtual std::vector<SpectralNormEntry> GetSpectralNormEntries() const = 0;   // mode=none なら空
};
```

SN を配線した 6 モジュール（LinearModule / Conv1dModule / Conv2dModule / ResBlockModule / CNBlockModule /
TransformerEncoderModule）がこれを実装する（Transformer は全 layer 分を集約。packed `in_proj_weight` は
Q / K / V のスライス 3 本として返す）。

### 3. ブロック配線

- **ResBlockModule**（config 構造体渡し。§事実 9）: `ResBlockConfig` に `std::string weight_norm_mode = "none";` を追加
  （パース = `ANET_READ_CONFIG(config_data, res.weight_norm_mode)`、`GetCurrentConfigData()` へ dump、
  `CreateModule` で値検証 fail-fast）。ctor シグネチャは不変。`spectral` 時は conv1 / conv2 / downsample の各 weight に
  独立の `SpectralNormState` を持ち（buffer 登録名は実装裁量。例 `sn_u_conv1` 等）、forward で
  `torch::nn::functional::conv2d(x, W/σ, bias, opts)` へ分岐する（opts = stride/padding 等は保持済み conv module から取得）。
  `none` 時は既存経路そのまま（分岐 1 個のみ）。
- **Conv2dModule / Conv1dModule / LinearModule**（引数バラ渡し。§事実 9）: `ConvConfig` / `LinearConfig` に
  `std::string weight_norm_mode = "none";` を追加し、factory パース → ctor 引数追加 → メンバ保持 →
  `GetCurrentConfigData()` dump の 5 箇所。lazy init（in_features / in_channels 自動推論）のモジュールは
  weight 実体化時点で `MakeSpectralNormState` + buffer 登録を行う。forward は functional 呼び出しへ分岐。
- **CNBlockModule**（config 構造体渡し。§事実 12）: `CNBlockConfig` に `std::string weight_norm_mode = "none";` を
  追加（`cn.weight_norm_mode`。ResBlock と同じ 4 箇所パターン・ctor 不変）。`spectral` 時は dwconv / pwconv1 /
  pwconv2 の各 weight に独立の state を持ち、forward を functional conv2d 分岐（dwconv は `groups=channels` を
  opts へ渡す）。lazy init（初回 forward の重み実体化）時に state 生成 + warm-start。layerscale γ・内部 norm は対象外。
- **TransformerEncoderModule**（§事実 13）: `TransformerConfig` に `std::string weight_norm_mode = "none";` を追加
  （`tf.weight_norm_mode`。READ / dump に加え、`CreateModule` の検証で値チェックと **`spectral` かつ
  `use_sdpa=false` の fail-fast**（`ValidateDropRate` の並び）を行う）。mode は `CustomTransformerEncoderLayer` の
  ctor へ渡す。`spectral` 時は layer ごとに **6 state**（`in_proj_weight` の Q / K / V スライス各 1 + out_proj +
  linear1 + linear2）を named buffer で持つ。attention は `anet::nn::SdpaSelfAttention` に実効重み
  （`cat(Wq/σq, Wk/σk, Wv/σv)` と out_proj の W/σ）を渡すオーバーロードを追加して呼ぶ。FFN は
  functional::linear 分岐。buffer は layer（torch::nn::Module）に register するため `named_buffers(true)` が
  module ツリー経由で拾い、CopyTo / SoftCopyTo / serialize は無償のまま。`GetSpectralNormEntries()` は
  TransformerEncoderModule が全 layer 分を集約して返す。
- 既存の weight init（`init.mode` 系）はそのまま生 W に適用され、その後 warm-start が走る（直交）。
- 全配線に `ANET_PROFILE_SCOPE` は不要（forward 内のベクトル演算のみで専用 capture 相当の処理が無いため）。
  throughput への影響は受入 5 の目視で確認する。

### 4. メトリクス拡張（`ComputeParameterNormSplit` の拡張）

現行 `Network::ComputeParameterNormSplit(feature_key)`（§事実 6）を拡張し、同じ walk（branch 単位の module 走査 +
閉包で feature/readout 帰属、heads は常に readout）の中で `SpectralNormedModule` を dynamic_cast で検出して集計を足す:

| 追加 field | 定義 |
|---|---|
| `feature_effective` / `readout_effective` | 61/62 と同じ群一括 L2 だが、**SN 層の weight のみ ‖W‖_F/σ で換算**（σ は `ComputeSpectralSigma(…, do_power_iteration=false)` でその場計算。NoGrad）。bias・非 SN パラメータは生のまま。**SN 層ゼロの群では 61/62 と同値** |
| `sigma_feature_max` / `sigma_readout_max` | 群内 SN 層の **max σ**（最も強くクランプされている層）。**SN 層ゼロの群では NaN**（既知 key の「値なし」= 062 の NaN 契約） |

- 測定時点は現行どおり **update 適用直前**（cadence gate の位置も現行のまま。§事実 6）。
- 呼び出し側（DQN learner / ImageCls）の変更は購読 key の追加のみ:
  - DQN: `ConfigureScalarMetricSubscriptions` の weight_norm 分類（§事実 6）へ 4 key を追加し、既存
    `weight_norm_enabled / weight_norm_interval` に min 合成で合流（61/62 と同じ棚）。搭載 tensor
    （`plasticity_.weight_norms` の stack）を 2 → 6 要素へ拡張し、`GetPlasticityScalar` に key を足す。
  - ImageCls: recognized set へ 4 key 追加、同じ cadence gate で搭載。
- 未測定 step・購読ゼロは 062/063 の契約そのまま（NaN / 完全不活性）。

### 5. config 契約

```
# nn.txt カタログ（既定 none なのでコメントアウト行 + 説明。force_fp32 の慣習と同じ）
#net.block.[ResA].res.weight_norm_mode = spectral      # 重み正規化モード(none|spectral)。spectral=W/σ射影(BTR系可塑性保護)。default=none
#net.block.[ConvA].conv.weight_norm_mode = spectral    # 同上
#net.block.[AtariLinear512].linear.weight_norm_mode = spectral   # 同上
#net.block.[CN64].cn.weight_norm_mode = spectral       # 同上
#net.block.[TransEnc].tf.weight_norm_mode = spectral   # 同上（use_sdpa=false との併用は fail-fast）
```

- 検証規則: `none` / `spectral` 以外は fail-fast（キー・指定値・許容値一覧を含める）。
- 実効値は `GetCurrentConfigData()` 経由で config dump（`config_data.txt`）に出る（Run 検証は dump が ground truth）。

### 6. metrics 行

```
# metrics_scalar.txt @baseline（34 群 6x 帯の続き。SN 実験時に uncomment）
#metrics.scalar.@baseline.[34_agent_plasticity/63_weight_norm_feature_effective] = plasticity_weight_norm_feature_effective @learn $learn_step $update_result interval:500
#metrics.scalar.@baseline.[34_agent_plasticity/64_weight_norm_readout_effective] = plasticity_weight_norm_readout_effective @learn $learn_step $update_result interval:500
#metrics.scalar.@baseline.[34_agent_plasticity/65_spectral_sigma_feature] = plasticity_spectral_sigma_feature @learn $learn_step $update_result interval:500
#metrics.scalar.@baseline.[34_agent_plasticity/66_spectral_sigma_readout] = plasticity_spectral_sigma_readout @learn $learn_step $update_result interval:500
```

- **既定コメントアウト**（target 系 `2x` の「uncomment だけで発動」前例）。SN OFF で uncomment しても壊れない
  （63/64 は 61/62 と同値、65/66 は NaN → isfinite skip）。
- ImageCls.txt のフラット群にも同 4 行（コメントアウト）を置く。
- **規約コメントの追記**（実装時）: metrics_scalar.txt 冒頭の 34 群規約コメント（`x1`-`x5` 既定 ON / `x6`-`x9`
  既定 OFF はチャネル decade 用）へ「**6x は独立 decade: 61/62 = 生ノルム（既定 ON）、63-66 = SN 系
  （既定コメントアウト）**」を 1 行足す — 63/64 が x3/x4 位置なのに既定 OFF であることをチャネル規約と
  読み違えさせないため。
- 読み方: **生 61 ↑ + σ 65 ↑ + 実効 63 フラット** が「SN が仕事をしている」ことの三点証拠。63 は SN OFF の
  過去 Run（生 = 実効が恒等）と同じ土俵で接続でき、WD 実験の V 字（45.8 → 30.5 → 43.4）との対比が主読みになる。

### 7. 単体テスト項目

- **σ の正しさ**: 既知の小行列で `ComputeSpectralSigma` を `torch::linalg::svdvals` の最大特異値と突合
  （power iteration 収束の許容誤差つき。warm-start k 回後 + 追加 iteration 数回で十分収束すること）。
- **mode=none 恒等**: forward 出力が既存実装とビット一致。SN の数値経路（state 生成・power iteration・除算）に
  不到達（`GetSpectralNormEntries()` が空を返す walk は走ってよい）。
- **ResBlock 適用範囲**: `spectral` で conv1 / conv2 / downsample 全部の entries が `GetSpectralNormEntries()` に出る。
- **CNBlock 適用範囲**: `spectral` で dwconv / pwconv1 / pwconv2 の entries が出る（depthwise の reshape `(C, k²)` 含む）。
  layerscale γ が entries に出ない。
- **Transformer 適用範囲**: `spectral` で layer ごとに Q / K / V（`in_proj_weight` のスライス 3 本）+ out_proj +
  linear1 / linear2 の entries が出る / Q / K / V の σ が独立に計算される / **`spectral` × `use_sdpa=false` が
  fail-fast**（エラーにキーと理由を含む）。
- **非変異**: eval mode forward で u/v が不変（target / probe 相当経路）。training mode forward でのみ更新される。
- **buffer 継承**: CopyTo で u/v が正確コピー / SoftCopyTo で lerp される / serialize round-trip で復元される。
- **使用時 normalize**: u/v を人工的に非単位化しても σ が単位化済みの値と一致する。
- **FP32 経路**: bf16 autocast 下で power iteration / σ が FP32 で走り、W_eff を使う conv/linear は autocast どおり。
- **メトリクス**: SN 層ゼロ群で `*_effective` が 61/62 と同値・`sigma_*` が NaN / SN 層あり群で実効 < 生（σ>1 の場合）/
  interface 経由の収集が walk の帰属（feature/readout）と一致。
- **warm-start 決定性**: 同 seed で u/v 初期値・σ が一致（2 回構築で再現）。
- **ON/OFF 等価性**: mode=none の同 seed Run で学習系列 + `agent_close.anet` が SN コード追加前と一致（受入 1 の単体版）。

## 複雑度監査（グリル簡素化パスの記録）

| 対象 | verdict | pin / ゲート |
|---|---|---|
| power iteration ヘルパ + 6 ブロック mode | keep | 崩壊対策そのもの（探索 04/05 + BTR 実証） |
| warm-start 初期化 | keep | learning_starts 前の junk 正規化（RL 構造上の実害） |
| 使用時 normalize | keep | SoftCopyTo の buffer lerp（実装事実）への頑健化 |
| FP32 強制 | keep | learner bf16=true の実構成 |
| メトリクス 4 本 + interface 口 | keep | 61 が①のブレーキを語らなくなる実測由来の問題。readout 側 64/66 は 61/62 との対称性 + 同 walk で追加コスト微小 |
| CNBlock / TransformerEncoder への mode | keep（2026-08-29 改訂） | 当初 deferred gate としたが、共通部品化（配線は全部・実使用は config の判断、型レベルの部分適用罠を残さない）で全 6 型配線へ改訂（D1/D11） |
| per-weight override キー（`res.conv1_weight_norm_mode` 式） | **deferred gate** | Gogianu 型の層選択実験（1 層だけ SN）が pin されたら追加。空間的選択は当面ブロック・インスタンス単位で足りる |
| `spectral` × `use_sdpa=false` の旧 MHA 経路対応 | **cut**（fail-fast） | 旧経路は SDPA 等価性確認用の互換参照。functional multi_head_attention_forward への書き換えは必要が pin されたら |
| 層別 σ の詳細 | **deferred gate** | [920_nn_block_metrics](920_nn_block_metrics_10prd.md) の領分 |
| actor クローン時に σ を確定して再計算 skip | **deferred gate** | actor 側 throughput が実測で問題になったとき |
| power iteration 回数の config 化 / σ₀ 目標係数 | **cut** | 文献標準は 1 回・σ₀ 無し。必要になった実験が存在しない |
| 独立 SpectralNorm ブロック（wrap DSL） | **cut** | 原理的に不成立（重み所有 forward 内でしか実現できない）+ 1関心2機構 |
| σ buffer 継承（非 training forward は読むだけ） | **cut** | PyTorch と異なる独自意味論の記述コスト。[ADR 0032](../adr/0032-spectral-norm-self-impl-buffer-semantics.md) |
| 61 の実効重み差し替え | **cut** | 既存 Run との互換破壊（61 の意味が Run 世代で変わる） |

## 検証計画（実行は実装後・結果は実験記録側へ）

検証は **screening / confirmation の 2 段構え**とする。損傷モードが RR で異なることが実測で出ており
（下記）、アッセイ合格のみで運用点有効と判定しない。

**screening = RR8 Breakout 5M ×2**（複製。ユニット死モードの高速アッセイ。無保護の対照 =
`run_20260829-143027` / `run_20260829-153617`（plasticity_rr8_breakout）— 下表の −57% / −53% の出典）。
判定は単一 Run の last 値でなく**複数 Run の終盤平均ブレ幅基準**（eval ピーク高の既知変動 ±26% を明記した上で読む）:

| 観測 | 判定 |
|---|---|
| `63_effective` がフラット | ①のブレーキが均衡でなく射影として効いた（WD の V 字 45.8→30.5→43.4 との対比が決定打） |
| `02_dead_ratio` の谷後増加の抑制 | ③への波及が切れた（現行は谷から 6.7 倍増） |
| eval ピーク後落差の縮小 | 崩壊の緩和（現行 −57% / −53%） |
| `65_sigma` の単調成長 | クランプ量が実在した証拠（生 61 の成長と対で読む） |

**confirmation = RR1 @ 50M（運用点）**: **主読みは、終盤窓の eval1 が無保護の新アンカー
（2026-08-29 実測 ≈427〜454 帯、`run_20260829-163959_a5_breakout_apex`）を超えるか**。
srank 浸食の抑制はその機序説明として併読する — RR8 アッセイが再現しない**第 2 の損傷モード = ランク浸食**
（同 Run 実測: probe srank 440→354 の単調 −20%。RR8 アッセイでは −4〜6% で不再現）が
`45_probe_srank_ratio` の低下抑制として出るか。主従をこの順に固定するのは、機序指標だけ改善してスコアが
動かない形（WD 実測: `wn_feat` は用量順に完全制御・スコアは悪化）を再び踏まないため。
SN は σ（支配特異方向）を直接クランプする機構であり、ユニット死よりランク浸食側にこそ効く可能性がある —
screening だけで判定すると SN の一番の得意分野を測り損ねうる。

出典 Run はいずれも workspace `atari-2nd`。実験記録（2026-08-29 campaign）は別セッションで起票予定で、
それまでの検証の正は Run フォルダ側にある（Run フォルダ = 真実）。

適用範囲（BTR 忠実 = ResBlock のみ vs 全層）は実験側の判断だが、**初手は全層**（standalone Conv2d 3 本 +
ResBlock 全部 + Linear512）を推奨 — GroupNorm 試験の部分適用の罠を踏まないため。

## 測定上の注意

- **σ<1 の層は拡大される**（常時射影）。「成長を止める」だけの機構ではない。
- **ON 腕は初期スケールから別物**: He init の σ_max は本構成の形状（conv 64×576 / Linear 512×3136）でいずれも
  概算 ≈1.9〜2.0。`spectral` ON の瞬間に全 SN 層の実効重みが約半分になり、立ち上がりの学習曲線・q_max スケール
  （BF16 ULP 余裕の文脈含む）は OFF 腕と初手から別物になる。**立ち上がりの差を SN の保護効果と誤読しない**。
- **weight_decay との併用**: SN 下の生 W への WD は σ を縮めるだけで実効重みに直接効かず、勾配スケール（実効学習率）を
  変える別経路になる。ベースライン採用構成は WD=0 であり、SN 実験も WD=0 で開始する。
- 61/62 は生ノルム（SN 下では制約されない）、63/64 は実効ノルム。**混同しない**（62 と同じくノルム絶対値は
  同構成 Run の時系列・同構成 Run 間の比較専用）。
- u/v の進みは training mode forward の回数に従う。update 構造を変える改修（forward 回数が変わる）をしたら
  σ 推定の追従性が変わりうる（決定性は保たれる）。

## スコープ外

- reset 系（Shrink-and-Perturb / ReDo）— 抑制系と別系統の保護機構。探索ブロック 05 の「次の検証」に残置。
- Munchausen（[999_MunchausenRL_10prd.md](999_MunchausenRL_10prd.md)）、eval N 本平均（[060](060_eval_batch_episodes_10prd.md)）。
- ResBlock への leaky ReLU / LayerNorm 追加など③への直接介入。
- `direction`（Salimans WN）等の追加モード実装（モード空間の予約のみ）。
- Rainbow への配線（062 D8 と同じ理由: 実行 smoke 不能）。MuZero はスコープ外。

## 受入基準

1. **OFF 完全不変**: `weight_norm_mode = none`（既定）の同 seed Run で学習系列 + `agent_close.anet` が
   本改修前と一致（063 受入 3 と同型）。**SN の数値経路（state 生成・power iteration・除算）に不到達**
   （`GetSpectralNormEntries()` が空を返す interface walk 自体は走ってよい）。
2. **ON 決定性**: `spectral` の同 seed 2 Run で学習系列一致（determinism 既定 ON 前提）。
3. **smoke**: Atari 構成 mode=spectral（全層）+ 63〜66 uncomment → `inspect_run.py tags` で
   `34_agent_plasticity/6x` 全 6 本が status=ok・count>0。61/62 は従来どおり生ノルム。
4. **単体テスト**: §実装仕様 7 の全項目が緑。
5. **throughput**: 学習 forward への追加はベクトル演算 × SN 層数のみ — interval 既定で既存比有意差なしの目視確認。

## 実装フェーズ（Codex 向け）

- **Phase 1 = SN 本体**: 共有ヘルパ + 6 ブロック配線 + warm-start + 受入 1/2/4（メトリクス以外）。
  この時点で eval score ベースの効果実験は開始可能。
- **Phase 2 = メトリクス**: interface 口 + `ComputeParameterNormSplit` 拡張 + 4 key 配線 + metrics 行 + 受入 3。

Phase 1 で止めても悪化しない（既定 none なので存在自体が無害）。逆順は無意味（測る対象が無い）。

## 現行コードで確定している事実（実装の下地）

1. **libtorch C++ に `spectral_norm` は無い**。`libtorch/include/torch/csrc/api/include/torch/nn/utils/` にあるのは
   `clip_grad.h` / `convert_parameters.h` / `rnn.h` のみ。Python の parametrization 機構は C++ API 非公開。
2. **buffer は clone / soft update / シリアライズで既に全対応**。`Network::CopyTo`（`nn_impl.cpp:1789-1809`）は
   named_parameters / named_buffers を key 一致で copy_、`SoftCopyTo`（`:1811-1840`）はパラメータを
   `_foreach_lerp_`、**float buffer も lerp**（int buffer は copy_）。u/v を register_buffer するだけで
   actor snapshot の clone も target 同期も無償で乗る。
3. **target net は常時 eval mode**（生成 `dqn_based_agent.cpp:369-370`、別 ctor `:397-398`、Load 後 `:519-520`）。
   `ForwardTarget`（`:412-415`）はモード切替をしない。learner の学習 forward だけが
   `ForwardOnlineWithTrain`（`:406-410`）の `TrainingModeGuard(*online_net_, true)` で train に入る。
4. `ForwardOnlineWithTrain` の呼び出しは TD / QR / IQN の 3 箇所（062 §事実 3）。probe 部分 forward
   （`ForwardOnlineUpTo`）は NoGrad + eval 固定。
5. **Atari 実構成は soft update**: `A1.model.soft_update_tau = 0.001`（`Atari.txt:441`）、hard は `@nature` のみ
   （`soft_update_tau = 0` + `hard_update_interval = 10,000`、`Atari.txt:388-389`）。
6. **`Network::ComputeParameterNormSplit`**（`nn_impl.cpp:1553-1591`）: branch 単位の module 走査で
   `requires_grad() == true` のみ累積（`:1573`）→ buffer は混入しない。閉包は
   `ComputeDependencyClosure`（`:1173-1207`）、heads は常に readout（`:1585`）。呼び出しは cadence gate
   （`dqn_based_agent.cpp:2518-2524`、update 適用直前）、購読分類は `ConfigureScalarMetricSubscriptions`
   （`:1878-1898`、`weight_norm_interval` min 合成 `:1881-1884`）、搭載 `:2371-2373`、取り出し
   `dqn_based_agent.hpp:355-366`（未測定 NaN）。ImageCls 同型（`image_cls_agent.cpp:362-366`, `:490`）。
7. **ModuleContext は空構造体**（`nn_impl.hpp:246-248`）で RNG の受け渡し口が無い。重み初期化は
   global RNG（`torch::nn::init::*` 直呼び）で、seed は `MasterSeedManager::ApplyTorchSeed`
   （`random.cpp:147-157`、`torch::manual_seed` / `cuda::manual_seed_all`）が反映済み。
8. **FP32 強制イディオム**: `LayerNorm2dModule`（`nn_modules.cpp:635-720`）— パラメータ生成時 dtype 固定
   （`:656-660`）+ forward で `anet::Autocast disable_amp(device, false, kFloat32)` + 入力 cast（`:662-666`）+
   `GetCurrentConfigData` へ dump（`:674`）。`BatchNorm2dModule`（`:575-607`）も同型。
9. **ブロック config パターン**: ResBlock は config 構造体渡し（`ResBlockConfig` `nn_modules.cpp:790-804`、
   factory パース `:1041-1085`、検証 + ctor `:1088-1100`、dump `:966-981`）で flag 追加は 4 箇所・ctor 不変。
   Conv2d / Conv1d / Linear は引数バラ渡し（ctor `:187-190` / `:69-72`、`LinearConfig` `:2195-2198`、
   `ConvConfig` `:2200-2206`、factory `:2220-2245` / `:2247-2277` / `:2279-2309`）で 5 箇所・ctor 変更。
   **ConvConfig は Conv1d / Conv2d 共有**。パース prefix は `res.` / `conv.` / `linear.`。
10. **config の enum 検証**: bool/値パースは fail-fast（`config.cpp:315-321`）。既定 OFF の flag は
    nn.txt でコメントアウト行 + `default=...` 注記が慣習（`nn.txt:54, 152, 196`）。
11. ResBlock の `activation` は relu / silu の 2 択、`norm_type` は none / batch / group のみ（③直接対策を
    ResBlock 内部へ入れるなら別 PRD）。`LayerNorm`（1D）と `LeakyReLU` は登録済みブロックで config のみで試せる。
12. **CNBlockModule**（`nn_modules.cpp:1108-1223`）: 重み所有は dwconv（depthwise, `groups=channels`）/ pwconv1 /
    pwconv2 の conv 3 本。layerscale γ（`gamma_`、1D）と内部 norm（LayerNorm2d）は重み行列でない。config 構造体
    渡し（`cn.` prefix、factory `:1225-`）+ 初回 forward の lazy init（`:1135-1181`）で、flag 追加は ResBlock と
    同じ 4 箇所パターン・ctor 不変。
13. **TransformerEncoderModule**（`nn_modules.cpp:1845-1956`）: `CustomTransformerEncoderLayer`（`:1702-1829`）×
    num_layers。layer の重み所有は MHA 保持器（packed `in_proj_weight` (3E,E) + `out_proj`）と linear1 / linear2、
    norm1 / norm2 / 最終 norm は affine。SDPA 経路は `anet::nn::SdpaSelfAttention(mha_, x)` の free 関数
    （`:1751, 1785`。PRD 012 で MHA を保持器に残し forward だけ関数化済み）なので実効重みを渡すオーバーロードが
    素直。旧経路（`use_sdpa=false`）は `mha_->forward()` が内部パラメータを直参照（`:1755, 1789`）し、注入には
    functional 書き換えが要る（→ fail-fast で回避、D11）。config は `tf.` prefix の構造体渡し
    （`TransformerConfig` `:1831-1842`、factory `:1929-1956`、`ValidateDropRate` の並びに検証追加可）。
