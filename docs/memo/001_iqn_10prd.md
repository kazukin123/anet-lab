# IQN 対応改修（quantile_mode 導入・bind `*` 演算子・IQNHead/IQNLearner）

- 起票: 2026-08（旧 001_iqn_10prd.txt を詳細化して .md へ移行）
- 実装: Codex（本 PRD 単体で 20impl を生成できる粒度を目標とする）
- 関連: ADR 0018（IQN を bind `*` DAG 汎用化と最小 Head で表現する）、CONTEXT.md「taus」「tau配置方式」

## Context（背景・目的）

既存の QR-DQN（固定分位・`QuantileHead`/`QRLearner`）に加えて IQN（Implicit Quantile Networks, Dabney et al. 2018）をサポートする。IQN は τ∈[0,1] を forward ごとにサンプリングし、cos 基底埋め込みで特徴量と融合して分位関数 Z(s,a;τ) を暗黙的に学習する。

方針は「専用ネットワーククラスを作らない」こと。既存の branch DAG 設定（`bind`）へ要素積演算子 `*` を追加し、taus ブランチ・fusion ブランチを**設定で**構成する。Head は最終射影（Linear）だけを受け持つ（ADR 0018）。

**用語注記**: taus は環境が渡す Observation ではない。Agent（ActionPolicy / IQNLearner）が forward 直前に **NN 入力 TensorDict** へ注入する評価点である。本 PRD では「obs へマージ」ではなく「NN 入力 TensorDict へマージ」と表記する（CONTEXT.md 参照）。

## 確定した設計判断（グリル済み）

| # | 決定 |
|---|------|
| D1 | 旧キー `use_qr`/`num_quantiles` は互換処理・マイグレーション・WARN 一切なし。コード参照ごと削除（設定に残っていても単に読まれない）。`quantile_mode` デフォルトは **qr**、`qr.num_quantiles` デフォルト 51 |
| D2 | IQN 対応・quantile_mode 化は **DefaultDQNAgent のみ**。RainbowAgent は共有クラスのシグネチャ変更へのコンパイル追従だけ行い、use_qr 設定体系を温存 |
| D3 | policy 側 tau_rule デフォルト: train=random×32 / eval=fixed_midpoint×32 / target=fixed_midpoint×32 |
| D4 | loss 用 current_taus/target_taus は **Learner 専用**につき `learner.iqn.current_taus.*` / `learner.iqn.target_taus.*`。デフォルト random×64（両方）、range 固定 [0,1]。`qr.num_quantiles` は Head 構造（出力幅）を決めるため agent 直下のまま |
| D5 | UQE+IQN は全対応: 下限＝減衰後の実効 tau（spatial 時は per-env tensor）、`uqe_use_tail_mean=false` は全点をその tau に固定して mean（=Z_τ 推定） |
| D6 | Thompson+IQN の tau_rule デフォルトは一律 32。「古典 Thompson 相当は num_taus=1 を明示設定」と本 PRD に注記（policy_type によるデフォルト分岐はしない） |
| D7 | iqn なのにどの branch も taus を bind していない場合は**構築時 fail-fast**。dummy build 用の taus 本数は train_policy.tau_rule.num_taus を名目値に使用 |
| D8 | bind `*` 細則: (raw) は factor 単位で付与可 / 3 項以上の積可（左結合）/ 明示検証は batch 次元サイズ一致のみ（他は torch の broadcast エラーに委ねる）/ 積 factor はトポソート・循環検出・GraphViz・ToJson で通常 bind key と同等扱い / `bind_concat_dim` は `,` 結合にのみ適用 |
| D9 | CosineEmbedding は基底展開のみ担当（(B,K)→(B,K,n)、cos(πiτ)、i=0..n−1）。後段の Linear+ReLU は既存ブロックで構成（次元一致は設定者責任） |
| D10 | 本 PRD にテスト要件（受け入れ基準）を含める。ファイルは .md へ移行 |
| D11 | CONTEXT.md へ「taus」「tau配置方式」の 2 語を追加。fusion/taus ブランチは実装寄りのため glossary 対象外 |
| D12 | ADR 0018 を起票 |

## 前提事実（実コード確認済み）

- `bind` パース: `ReadBranchConfig`（nn_impl.cpp）は `,` split のフラットな `bind_keys` と `(raw)` サフィクスを解釈する。
- 実行: `NetworkBranch::Execute`（nn_impl.cpp）は bind_keys の tensor を集め `inputs.size()==1 ? inputs[0] : torch::cat(inputs, 1)` 固定。
- 依存解決: `NetworkBodyBuilder::Build`（nn_impl.cpp）が bind_keys から input/branch の依存グラフを構築しトポソート・循環検出。**未 bind の input_specs キーは無警告で無視される**（D7 の fail-fast を足す理由）。bind 先が input にも branch にも無ければ既存の unknown input key エラー。
- GraphViz: `Network::MakeGraphViz` が `branch->GetBindKeys()` でエッジ描画。`NetworkConfig::ToJson` が bind_keys を dump。
- `LinearModule`（nn_modules.cpp）は lazy 構築で**最終次元に適用** → (B,K,n) 入力にそのまま使える。
- `QuantileLearnerBase::ComputeQuantileHuberLoss(current, target, taus, kappa)` は current (B,N) × target (B,M) の N≠M と、taus の broadcast（QRLearner は (1,N,1) を渡している）に対応済み → IQN は per-sample の (B,N,1) を渡すだけでよい。
- `QuantileLearnerBase` の `GatherActionQuantiles` / `SelectTargetActions` / `CalcTargetQuantiles` / `MakeQuantileMetrics` は q_dist (B,A,N) 前提でそのまま再利用可。
- Stacker: `DefaultDQNAgent` コンストラクタは stack_keys 空のとき**全 obs_spec の先頭次元を stack_count 倍**する → taus の spec 追加は必ずこのループの**後**に行う。
- 全 ActionPolicy は `ForwardForAction(obs, network, sink)` → `network->Forward(obs)` を通る。`rnd` は `SelectAction` 引数で全 policy に供給済み（Actor が ActionContext の RandomGenerator を渡す）。
- Learner は RandomHolder（`GetRandomGenerator()`、target_seed 系列）。
- `NetworkModel::num_quantiles_` の用途は `IsDistributional()` のみ（本番参照なし・テスト参照のみ）。
- QValuePanel は `aux_data["q_quantiles"]`（(B,A,K)）を読むだけ → IQN の q_dist と shape 互換で**変更不要**。
- 稼働中設定: agent.txt baseline に `use_qr=true` / `num_quantiles=51`、LunarLander.txt に `R.num_quantiles=32`（active はこの 2 ファイルのみ。DropMerge/GridMaze はコメントアウトのみ）。

## 設計方針

### 1. Config

設定キー全体像（DefaultDQNAgent スコープ）:

```
quantile_mode = qr                              # none / qr / iqn （デフォルト: qr）
qr.num_quantiles = 51                           # QR 専用（Head 出力幅）


train_policy.tau_rule.num_taus = 32             # デフォルト 32
train_policy.tau_rule.sample_mode = random      # random / fixed_midpoint（train デフォルト: random）

eval_policy.tau_rule.num_taus = 32
eval_policy.tau_rule.sample_mode = fixed_midpoint   # eval デフォルト: fixed_midpoint（決定的評価）

target_policy.tau_rule.num_taus = 32
target_policy.tau_rule.sample_mode = fixed_midpoint # target デフォルト: fixed_midpoint


learner.iqn.current_taus.num_taus = 64          # loss の current 分布用（勾配側）
learner.iqn.current_taus.sample_mode = random

learner.iqn.target_taus.num_taus = 64           # loss の target 分布用（N≠M 可）
learner.iqn.target_taus.sample_mode = random
```

実装:

- `TauRuleConfig` struct を agent.hpp の ActionPolicyConfig 近傍に追加: `{ int num_taus = 32; std::string sample_mode = "random"; }`。**tau_min/tau_max は持たない**（範囲は呼び出し側が決める。UQE は実効 tau から、その他 policy と learner は [0,1] 固定）。
- `ActionPolicyConfig` へ追加: `TauRuleConfig tau_rule;` と `std::string quantile_mode = "none";`。quantile_mode は**設定ファイルから直接読まない**（Agent 直下の値を自動引継ぎ。個別設定キーは作らない）。
- `LearnerConfig` へ追加: `struct IqnConfig { TauRuleConfig current_taus{ .num_taus = 64 }; TauRuleConfig target_taus{ .num_taus = 64 }; } iqn;` と `std::string quantile_mode = "none";`（同じく自動引継ぎ）。
- `DefaultDQNAgentConfig` へ追加: `std::string quantile_mode = "qr";`、`struct QrConfig { int num_quantiles = 51; } qr;`。`ANET_READ_CONFIG(config_data, quantile_mode)` / `ANET_READ_CONFIG(config_data, qr.num_quantiles)` / 各 `*.tau_rule.num_taus` / `*.tau_rule.sample_mode` / `learner.iqn.*` を読む。eval の sample_mode デフォルト差し替え（fixed_midpoint）は、既存の「ANET_READ_CONFIG 前にデフォルト値を代入する」パターン（`eval_policy.policy_type = "Greedy"` と同じ位置）で行う。
- **target_policy.tau_rule の既定（fixed_midpoint×32）の代入位置は `use_optimistic_target` 分岐の後・target 系 ANET_READ_CONFIG の直前**とする。`use_optimistic_target=true` は既存実装で `target_policy = train_policy;` を丸ごとコピーするため、分岐前に代入すると train の tau_rule（random×32）で上書きされてしまう。楽観性は uqe_tau 範囲が担う設定であり、tau_rule（推定品質）は楽観コピーに追随させず既定 fixed_midpoint を維持する（明示設定があれば READ_CONFIG が優先）。
- **旧キー削除**: `DefaultDQNAgentConfig::use_qr` / `::num_quantiles` メンバと ANET_READ_CONFIG、および参照箇所（コンストラクタの整合性チェック・Head factory 分岐・NetworkModel 引数・Learner 分岐・CreateActionPolicy の ANET_CHECK）を全て置換する。互換処理・WARN は書かない。`LearnerConfig::num_quantiles` メンバは QRLearner が使うため**温存**し、`learner.num_quantiles = qr.num_quantiles;` を既存代入と同位置（コンストラクタ末尾）で行う。
- **自動引継ぎ**: コンストラクタ末尾で `learner.quantile_mode = quantile_mode; train_policy.quantile_mode = quantile_mode; eval_policy.quantile_mode = quantile_mode; target_policy.quantile_mode = quantile_mode;`。
- **検証（fail-fast、ANET_SYSTEM_ERROR、キー名・指定値・期待値を含める）**:
  - `quantile_mode` ∉ {none, qr, iqn}
  - `quantile_mode==qr && qr.num_quantiles <= 1`
  - `tau_rule.num_taus <= 0`（5 箇所全て）、`sample_mode` ∉ {random, fixed_midpoint}
  - UQE/ThompsonSampling 選択で `quantile_mode == none`（既存 `ANET_CHECK(config_.use_qr)` の置換）

### 2. NN — bind `*` 演算子

文法（既存 `bind = a, b` の拡張）:

```
net.branch.[fusion].bind = main_feature * tau_embedding
net.branch.[x].bind = a * b, c          # 「*」は「,」より優先。項1 = a*b、項2 = c
net.branch.[y].bind = a(raw) * b        # (raw) は factor 単位
```

- パース: `,` で**項（term）**に分割 → 各項を `*` で**因子（factor）**に分割 → 各 factor を Trim し `(raw)` サフィクスを解釈（既存の raw_keys へ追加）。3 項以上の積可（左結合）。
- 項の評価（`NetworkBranch::Execute`）:
  - 単一 factor の項 → tensor をそのまま使用（既存挙動）。
  - 積の項 → **feature-last elementwise product**: batch 次元を dim 0、feature 次元を最終次元とみなす。rank が異なる場合、低 rank 側に batch 直後（dim 1）へ singleton 次元を挿入して rank を揃え、batch 次元以外は通常の broadcast 規則で乗算する。左結合で順次適用。
  - 例: features (B,D) × tau_embedding (B,K,D) → features を (B,1,D) に拡張 → 結果 (B,K,D)。
  - **明示検証は batch サイズ（size(0)）一致のみ**。不一致は branch 名・factor 名・両 shape 入りの ANET_SYSTEM_ERROR。それ以外の shape 不整合は torch の broadcast エラーに委ねる（過剰検証しない）。
- 項が複数のとき、項評価結果を `bind_concat_dim`（branch 単位の設定キー `net.branch.[name].bind_concat_dim`、int、デフォルト 1）で `torch::cat`。shape から concat 次元を推測しない。積の項も cat 対象として同列に扱う。
- **内部データモデル**: `NetworkBranchConfig::bind_keys`（フラット vector）を項リストへ変更する（例: `std::vector<std::vector<std::string>> bind_terms` — 各項 = factor 列、単一 factor 項は size 1。加えて `int64_t bind_concat_dim = 1;`）。依存解決・GraphViz・ToJson が使う「全 bind キー集合」は全 factor の平坦化で得る（`NetworkBranch::GetBindKeys()` 相当は平坦化版を返す形でよい）。`raw_keys` / `auto_format` の意味は不変（auto_format=false は全 factor を raw 扱いにする既存ロジックを factor 平坦化に対して適用）。
- **波及先（全て追従必須）**: `ReadBranchConfig`（パース。branch プロパティ正規表現 `re_branch` の `(bind|structure|auto_format)` へ `bind_concat_dim` を追加）/ `NetworkBranchConfig`（構造）/ `NetworkBranch::Execute`（評価）/ `NetworkBodyBuilder::Build`（依存 edge を全 factor から張る・循環検出・unknown key 検出）/ `Network::MakeGraphViz`（factor ごとにエッジ描画）/ `NetworkConfig::ToJson`（bind の項構造が復元できる表現で dump）。
- スコープ注記: `*` の主用途は IQN の fusion ブランチ。CNN 出力 (B,C,H,W) 等、IQN 想定外の rank との組合せは**動作保証外**（broadcast が通れば動くが仕様として保証しない）。

### 3. NN — CosineEmbedding モジュール

- nn_modules.cpp に `NetworkModule` 実装 + Factory を追加し、`InitNN()` の登録ブロックへ `repo.Register("CosineEmbedding", ...)` を追加。
- 設定: `cos.num_basis`（int、デフォルト 64。<=0 は ANET_SYSTEM_ERROR）。
- 変換: 入力 τ (B,K) → 出力 (B,K,n)。`out[b,k,i] = cos(π · i · τ[b,k])`、i = 0..n−1（論文準拠。i=0 は定数 1 列）。
- 実装: `torch::arange(n)` × π を τ.unsqueeze(-1) と broadcast 乗算 → `torch::cos`。学習パラメータ無し・stateless。device/dtype は入力に従う。入力 rank が 2 以外は ANET_SYSTEM_ERROR（設定ミスの早期検出）。`ANET_PROFILE_FUNC()` を付与。
- 後段の射影 φ(τ)=ReLU(Linear(cos基底)) は**既存の Linear / ReLU ブロックを設定で並べて**構成する（本モジュールには内包しない）。Linear の out_features を features ブランチの最終次元と一致させるのは設定者責任。

### 4. NN — IQNHead / IQNDuelingHead

dqn_based_heads.hpp/.cpp に既存 Quantile 系と同居で追加する。

- **IQNHead**:
  - Factory `IQNHeadFactory(action_dim, init_config)`（num_quantiles は持たない）。`CreateHead` は dummy_features の `kKey_DefaultOutput`（"features"）tensor (B,K,D) の size(-1)=D から `Linear(D, A)` を構築。
  - `Forward`: x (B,K,D) → linear → (B,K,A) → `permute({0,2,1})` + `.contiguous()` → `q_dist` (B,A,K)。`q` = `q_dist.mean(2)` (B,A)。出力 TensorDict は既存 QuantileHead と同じく `q_dist` と `q`。入力 rank 3 以外は ANET_SYSTEM_ERROR。
  - `GetTensorDictFunction`: QuantileHead と同じキー群（"forward"/"forward.q"/"q_values"、"forward.dist"/"distributions"）を提供。
  - `GetGraphVizInfo`: type="IQNHead"、outputs に q (A) / q_dist (A, -1)（tau 本数は動的なので -1 か省略）、details に action_dim。
- **IQNDuelingHead**:
  - Factory は既存 `QuantileDuelingHeadFactory` と同じキー規約（`value_feature`/`adv_feature` が両方あれば branched、両方無ければ `features` 共有、片方のみはエラー）。
  - `Forward`: value_x (B,K,Dv) → Linear(Dv,1) → v (B,K,1)。adv_x (B,K,Da) → Linear(Da,A) → a (B,K,A)。`a_mean = a.mean(2, /*keepdim=*/true)`（action 次元で平均）。`q_kta = v + (a − a_mean)` (B,K,A) → permute+contiguous → `q_dist` (B,A,K)、`q = q_dist.mean(2)`。`v_dist` (B,1,K)・`a_dist` (B,A,K) も既存に倣い permute して返す。
- **Head factory 選択**（default_dqn_agent.cpp の既存 is_distributional 分岐を置換）:
  - none: LinearHeadFactory / DuelingHeadFactory
  - qr: QuantileHeadFactory / QuantileDuelingHeadFactory（qr.num_quantiles を渡す）
  - iqn: IQNHeadFactory / IQNDuelingHeadFactory
  - use_dueling_net との直積 6 分岐。LOG::info のヘッド名も追従。

### 5. taus 配線（固定キー・spec・fail-fast）

- 固定キー: `static constexpr const char* kKey_Taus = "taus";` を **nn_heads.hpp の kKey_DefaultOutput 隣**に定義（ActionPolicy/Learner/NN 実装のいずれからも include 可能な内部ヘッダ）。
- **fusion / taus ブランチは IQN 有効時にユーザが NN 設定（net.branch.*）で記述する。フレームワークによる自動挿入は行わない**（書き忘れは下記の構築時 fail-fast で検出する）。
- input_specs 追加: `DefaultDQNAgent` コンストラクタ内、**Stacker 調整ループの後・NetworkModel 生成の直前**で、`quantile_mode=="iqn"` のとき `network_obs_spec[kKey_Taus] = TensorSpec{ type=Vector, shape={train_policy.tau_rule.num_taus}, dtype=float32, num_classes=0 }` を追加する。`DefaultDQNAgentConfig`（設定読み込み側）では追加しない。
  - 名目 shape の意味: ネットは tau 本数非依存（lazy Linear は最終次元適用、積は broadcast）。spec の本数はダミー forward による lazy 初期化にのみ使われ、実行時は任意本数を流せる。
- **構築時 fail-fast（D7）**: iqn のとき、`net_config.branches` のいずれかの bind（factor 平坦化後）に "taus" が含まれることを NetworkModel 生成前に検証。無ければ ANET_SYSTEM_ERROR（"quantile_mode=iqn requires a branch binding input 'taus'. Add net.branch.[<name>].bind = taus ..." の形で誘導）。
- 逆方向（非 iqn で taus を bind）: input_specs に taus が無いため `NetworkBodyBuilder` の既存 unknown input key エラーが発火する（追加実装不要）。
- 実行時に taus 未マージで forward した場合: `NetworkBranch::Execute` の既存入力欠落エラー（"Input key 'taus' not found ..."）が発火する（追加実装不要。taus は実装が生成するのでバグケースのみ）。
- **マージ規約**: taus は呼び出し側（ActionPolicy / IQNLearner）が forward 前に NN 入力 TensorDict へマージする。**受け取った obs TensorDict を直接変更せず、shallow copy（`anet::TensorDict` のコピー）に `Set(kKey_Taus, taus)` する**。他所有者（ReplayBuffer / ActionContext / 呼び出し元）の dict を汚さない。
- `NetworkModel` の Forward 系メソッドシグネチャは変えない（TensorDict キー渡し）。

### 6. TauGenerator

- 新規ファイル `core/anet-core/src/tau_generator.hpp` / `.cpp`（AGENTS.md 規約: 機能グループ単位の新設。名前付き namespace `anet::rl::dqn`）。
- 責務: **tau 配置方式（random / fixed_midpoint）だけ**を担当する。範囲・本数は引数で受ける。tail という policy 概念は持たせない。stateless（メンバ状態なし。free 関数群または static のみのユーティリティ）。
- API（シグネチャは 20impl で最終化してよいが、意味論は固定）:

```cpp
// 全バッチ共通範囲: (B, K) float32 を device 上に生成
torch::Tensor GenerateTaus(
    int64_t batch_size, int64_t num_taus, const std::string& sample_mode,
    float tau_min, float tau_max,
    const torch::Device& device, anet::RandomGenerator& rnd);

// per-env 下限（UQE spatial 用）: lower (B,) tensor → (B, K)
torch::Tensor GenerateTaus(
    const torch::Tensor& tau_min_per_env, float tau_max,
    int64_t num_taus, const std::string& sample_mode,
    anet::RandomGenerator& rnd);
```

- `random`: u ~ U[0,1) を `rnd.GetTorchGenerator(device)` で (B,K) 一括生成し、`tau = tau_min + u·(tau_max−tau_min)`（per-env 版は `tau = l.unsqueeze(1) + u·(tau_max − l.unsqueeze(1))`）。**ソートしない**（非ソートのまま使う）。
- `fixed_midpoint`: `tau_i = tau_min + (i+0.5)/K·(tau_max−tau_min)`、i=0..K−1。RNG を消費しない。**全バッチ共通範囲版**では全バッチ同一（(1,K) 生成 → (B,K) へ expand で可）。**per-env 下限 overload では `tau[b,i] = l[b] + (i+0.5)/K·(tau_max − l[b])` とバッチごとに異なる**（expand 不可。sample_mode × overload の 4 通り全てを定義する）。
- 不正 sample_mode は ANET_SYSTEM_ERROR。GPU フル対応（指定 device 上で生成、同期を挟まない）。`ANET_PROFILE_SCOPE` を付与。
- `TauRuleConfig`（num_taus / sample_mode）を受けて範囲引数と組み合わせる convenience overload を設けてよい（意味論は上記と同一。§8 の擬似コードはこの形）。

### 7. ActionPolicy（全 policy の iqn 経路）

- **クラス構成は既存から増やさない**（EpsilonGreedy / UQE / ThompsonSampling の 3 クラス + Greedy 縮退のまま）。既存 policy 内で quantile_mode に応じて action score 計算を切り替える。
- 共通: `quantile_mode=="iqn"` のとき、各 `SelectAction` は forward 前に taus を生成し、NN 入力 copy へ `kKey_Taus` をマージしてから `ForwardForAction` する（共通ヘルパ化は 20impl 裁量。責務は「Policy がマージ」で固定）。RNG は `SelectAction` の `rnd` 引数を使う。
- **EpsilonGreedy / Greedy**: 範囲 [0,1] × 自分の `tau_rule` で生成 → forward → 既存どおり `out.At("q")`（=mean 済み）で greedy/ε 選択。それ以外の変更なし。
- **UQE**（iqn 経路。既存の vectorized UQE＝QR 用 sort/mask 経路は QR 専用に残す）:
  - 下限 = **減衰後の実効 tau**（非 spatial: `current_uqe_tau_`。spatial: `GetSpatialTauTensor` の per-env tensor → per-env 下限 overload を使用）。
  - `uqe_use_tail_mean=true`: 範囲 [下限, 1.0] × tau_rule で生成 → forward → action score = `q_dist.mean(2)`。**sort/gather はしない**。
  - `uqe_use_tail_mean=false`: taus を**全点その下限値に固定**（per-env の場合は各 env の値を K 本複製）→ mean(2)（= Z_τ の推定として機能維持）。
  - aux の `uqe_values` には mean 結果を格納（既存キー維持）。greedy_only の ε ゼロ化は既存踏襲（UQE の楽観基準は greedy_only でも維持、という既存セマンティクスも踏襲）。
  - **仕様注記（E[Z] は得られない）**: IQN+UQE の forward は tail 範囲の taus しか流さないため、出力 `q`（= q_dist.mean(2)）は tail mean そのものになり、全分布平均 E[Z] はこの forward から得られない（aux の q_values と uqe_values は同値になる）。actor_q_hint（近似Actor初期優先度）や `episode_start_action_q_margin` 等の平均 Q 系メトリクスも tail mean ベースになる（QR-UQE との挙動差）。E[Z] 用に [0,1] の追加 forward を行う案は計算 2 倍のため不採用。ADR 0010 が許容する「UQE 構成の系統差」の延長として扱う。
- **ThompsonSampling**（iqn 経路）: 非 spatial は範囲 [0,1] × tau_rule（sample_mode は設定に従う。通常 random）で生成 → mean(2) を action score。spatial は UQE と同じ per-env 下限挙動（既存の spatial 分岐位置を踏襲）。
  - **注記（D6）**: num_taus=1 で古典 Thompson（1 サンプル分位で選択）相当。デフォルト 32 では E[Z] に近づき楽観化はマイルドになる。古典挙動が欲しい構成は `train_policy.tau_rule.num_taus = 1` を明示設定する。
- 有効判定: `CreateActionPolicy` の `ANET_CHECK(config_.use_qr)` を「`quantile_mode != "none"` でなければ ANET_SYSTEM_ERROR（policy_type と quantile_mode をメッセージに含める）」へ置換。
- QR 経路（quantile_mode==qr）は既存実装のまま一切変更しない。

### 8. IQNLearner

- dqn_based_agent.hpp/.cpp へ `QuantileLearnerBase` 継承で追加（QRLearner と同居）。コンストラクタは QRLearner と同型（SetupReplayBuffer + SetupOptimizer）。tau_i_ の事前計算は持たない。
- `UpdateFromSamples` は QRLearner::UpdateFromSamples をベースに以下の差分:
  1. `current_taus = GenerateTaus(B, learner.iqn.current_taus, [0,1], device, GetRandomGenerator())` — (B,N)
  2. 正規化済み obs の copy へ `kKey_Taus` をマージ → `ForwardOnlineWithTrain` → `q_dist` (B,A,N) → `GatherActionQuantiles` → current_dist (B,N)
  3. `MakeQuantileMetrics(current_dist, q)`（既存再利用）
  4. NoGrad 内: `SelectTargetActions(next_obs)`（既存再利用）— target_policy が**自分の tau_rule**（D3: fixed_midpoint×32）で taus を注入する。Learner はこの forward の taus に関与しない
  5. `target_taus = GenerateTaus(B, learner.iqn.target_taus, [0,1], ...)` — (B,M)。next_obs の copy へマージ → `ForwardTarget` → (B,A,M) → `GatherActionQuantiles` → next_dist (B,M) → `CalcTargetQuantiles`（既存再利用）→ target_dist (B,M)
  6. `loss = ComputeQuantileHuberLoss(current_dist, target_dist, current_taus.view({B,N,1}))` — per-sample の τ が quantile weight |τ − I(δ<0)| に入る（既存関数の broadcast がそのまま効く）
  7. td_error / PER / Optimize / MakeBatchUpdateResult は QRLearner と同一
- 1 回の更新に tau は 3 系統現れることに注意（混同しない）: ①target_policy.tau_rule（next-action 選択 forward、手順 4）②current_taus（勾配側、手順 1-2, 6）③target_taus（target 分布、手順 5）。
- Learner factory（default_dqn_agent.cpp）: `quantile_mode` 3 分岐 — none→TDLearner / qr→QRLearner / iqn→IQNLearner。LOG::info も追従。
- shape アサート（ANET_ASSERT_SHAPE）は N=current num_taus / M=target num_taus で張り直す。

### 9. NetworkModel

- コンストラクタ引数 `int64_t num_quantiles` を `bool distributional` へ変更（メンバ `num_quantiles_` は削除、`IsDistributional()` は `distributional_` を返す形で温存）。
- DefaultDQNAgent 側は `quantile_mode != "none"` を渡す。rainbow_agent.cpp は `config_.use_qr` を渡す**機械的追従のみ**（Rainbow の設定体系・分岐は変えない）。

### 10. 設定ファイル更新（本改修内で実施）

- `apps/runner/config/agent.txt`:
  - `DefaultDQNAgent.baseline.use_qr` / `.num_quantiles` 行を `quantile_mode = qr` / `qr.num_quantiles = 51` へ置換。tau_rule / learner.iqn.* のコメント付き既定値サンプルを追記。
  - `RainbowAgent.baseline.*` は現状維持（use_qr 温存）。
- `apps/runner/config/LunarLander.txt`: `R.num_quantiles = 32` → `R.qr.num_quantiles = 32`。**怠ると新デフォルト 51 に化けて挙動が変わる**（旧キーは読まれないため）。近傍のコメント行も揃える。
- `apps/runner/config/nn.txt`: 冒頭の「Supported Module Types」コメント一覧へ CosineEmbedding を追記（docs/design/130 の「設定例と名称を一致させる」慣行に合わせる）。
- IQN 用サンプル NN 設定（参考例。実ファイルへの追加はコメントアウトで任意）:

```
# --- IQN taus / fusion ブランチ例（MLP 系 backbone の場合） ---
net.block.[CosEmb].type = CosineEmbedding
net.block.[CosEmb].cos.num_basis = 64

net.block.[TauProj].type = Linear
net.block.[TauProj].linear.out_features = 84        # features ブランチ最終次元と一致させる

net.block.[FusionFC].type = Linear
net.block.[FusionFC].linear.out_features = 84

net.branch.[main_feature].bind = vector
net.branch.[main_feature].structure = Flatten > MLP_FC1 > ReLU > MLP_FC2 > ReLU   # (B,84)

net.branch.[tau_embedding].bind = taus
net.branch.[tau_embedding].structure = CosEmb > TauProj > ReLU                    # (B,K,84)

net.branch.[fusion].bind = main_feature * tau_embedding                           # (B,1,84)×(B,K,84)→(B,K,84)
net.branch.[fusion].structure = FusionFC > ReLU

net.body.output.[features] = fusion
```

Dueling 構成では fusion の後に `bind = fusion` の value/adv stream ブランチを作り、`net.body.output.[value_feature]` / `[adv_feature]` を割り当てる（既存 Dueling 記法踏襲。各 stream 出力は (B,K,D')）。

### 11. メトリクス・可視化

- QValuePanel: **変更不要**（q_quantiles aux は (B,A,K) 互換。IQN では τ が毎 forward 変わるため「固定分位点の分布」ではない旨だけ留意）。
- BatchUpdateResult（q_std/q_gap 等）: QR と同一経路で算出、変更不要。
- GraphViz: 積 factor のエッジ描画（bind 拡張の一部として §2 に含む）。

## 非対象（Out of Scope）

- RainbowAgent の IQN 対応・quantile_mode 化（機械的コンパイル追従のみ）
- risk-sensitive な tau 歪み（CVaR 制御等）・tail policy 概念（持たせない）
- probe / `GetTensorDictFunction` 経路（SweepHeatMap 等）への taus 自動補完（下記リスク参照。後続候補）
- CNN 出力等、IQN 想定外 rank との bind `*` 組合せの動作保証

## 影響ファイル

- `core/anet-core/include/anet/nn.hpp` — NetworkBranchConfig の項リスト化・bind_concat_dim
- `core/anet-core/src/nn_impl.hpp` / `nn_impl.cpp` — ReadBranchConfig / NetworkBranch::Execute / NetworkBodyBuilder / MakeGraphViz / ToJson
- `core/anet-core/src/nn_modules.cpp` — CosineEmbedding + InitNN 登録
- `core/anet-core/src/nn_heads.hpp` — kKey_Taus 追加
- `core/anet-core/src/dqn_based_heads.hpp` / `.cpp` — IQNHead / IQNDuelingHead + Factory
- `core/anet-core/include/anet/agent.hpp` — TauRuleConfig、ActionPolicyConfig / LearnerConfig 拡張
- `core/anet-core/include/anet/default_dqn_agent.hpp` — quantile_mode / qr.num_quantiles / tau_rule 読み込み・引継ぎ・旧キー削除
- `core/anet-core/src/default_dqn_agent.cpp` — Head factory 分岐・taus spec 追加・fail-fast・Learner factory・CreateActionPolicy 判定
- `core/anet-core/src/dqn_based_agent.hpp` / `.cpp` — ActionPolicy iqn 経路・IQNLearner・NetworkModel 変更
- `core/anet-core/src/tau_generator.hpp` / `.cpp` — 新規
- `core/anet-core/src/rainbow_agent.cpp` — 機械的追従
- `apps/runner/config/agent.txt` / `LunarLander.txt` / `nn.txt` — 設定書き換え・コメント一覧追記
- `docs/design/130_neural_networks.jp.md` — bind 記法へ `*` 演算子・bind_concat_dim・構築時失敗条件（taus 未 bind）を追記、CosineEmbedding を module type として反映
- `docs/design/200_dqn_agents.jp.md` — アルゴリズム表（TD/QR → TD/QR/IQN）・Config 一覧（quantile_mode / tau_rule / learner.iqn.*）・Learner 一覧（IQNLearner）を反映
- テスト: `core/anet-core/src/nn_test.cpp` / `dqn_based_agent_test.cpp` / `dqn_based_test.cpp`

## 受け入れ基準（テスト要件）

1. **bind `*`**: パース（`a * b, c` / `a*b*c` / `a(raw) * b` / 空白 Trim）、実行の数値検証（(B,D)×(B,K,D)→(B,K,D) が手計算と一致）、batch サイズ不一致エラー、積 factor 経由の依存解決とトポソート順、積 factor での循環検出、`bind_concat_dim` の反映、GraphViz に積エッジが出る、ToJson に項構造が出る
2. **TauGenerator**: random の range・shape・同 seed 再現、fixed_midpoint の値位置（(i+0.5)/K の lerp）と RNG 非消費、per-env 下限 overload、不正 sample_mode エラー（CUDA 環境では device 生成も）
3. **CosineEmbedding**: (B,K)→(B,K,n) と cos(πiτ) の値検証（τ=0 で全 1、i=0 列は常に 1）、rank≠2 エラー、`cos.num_basis<=0` エラー
4. **IQNHead / IQNDuelingHead**: (B,K,D) 入力 → `q_dist` (B,A,K)・`q`=mean(2) の一致、Dueling の v+(a−ā) 合成、features 共有/branched 両モード、GetGraphVizInfo
5. **ActionPolicy**: iqn で全 policy_type（EpsilonGreedy/Greedy/UQE/Thompson）の SelectAction が成功し、**入力 obs dict に taus が残らない**（非汚染）。UQE の実効 tau 下限（decay 進行後の値が使われる）、spatial per-env、tail_mean=false の全点固定、Thompson の random 生成
6. **IQNLearner**: 小規模ネットでの e2e（UpdateFromSamples が正常完了、N≠M 設定、loss 有限、PER 優先度更新）、`ComputeQuantileHuberLoss` への (B,N,1) taus 供給
7. **fail-fast**: quantile_mode 不正値 / iqn で taus 未 bind / qr で num_quantiles<=1 / UQE・Thompson × none
8. **Config**: デフォルト値（qr / 51 / 32×3 policy / 64×2 learner / sample_mode 既定）、quantile_mode の policy・learner への自動引継ぎ
9. **回帰**: 既存テスト全緑（QR / none 経路の挙動不変）。既存テストの機械的追従を含む — dqn_based_agent_test.cpp の `config.use_qr = false;`（2 箇所）→ `config.quantile_mode = "none";`、TestNetworkModel が NetworkModel ctor へ渡す引数（num_quantiles → distributional bool）、IsDistributional テストの構築引数。nn_test.cpp の QuantileDuelingHead GraphViz テスト（num_quantiles 表示検証）は QR Head のままなので変更不要

## 正直なリスク

- **probe 経路**: `GetTensorDictFunction`（SweepHeatMapPanel の状態スイープ等）は obs だけを渡すため、IQN では branch 入力欠落エラーになる（無症状ではなく明示エラーで止まる）。恒久対応（eval tau_rule の fixed_midpoint を wrapper で自動補完する案）は後続。
- **IQN+UQE の q 系値は tail mean**: §7 の仕様注記のとおり、aux q_values / actor_q_hint（近似Actor初期優先度）/ 平均 Q 系メトリクスが QR-UQE と系統差を持つ（ADR 0010 の許容範囲として扱う）。
- **計算量**: IQN は fusion 以降と Head の中間テンソルが (B,K,·) になり、K 倍のメモリ・演算になる。既定 K/N（32/64）は小規模 env 前提の選定。
- **bind 構造変更の回帰リスク**: パース・実行・依存解決という全設定共通経路に触れる。受け入れ基準 9（既存テスト緑）で担保する。
- **非 contiguous q_dist**: permute 由来。Head で `.contiguous()` を明示して下流（gather/sort/CPU 転送）の想定外を避ける。

## 検証

- Debug ビルド（VsDevCmd 経由の PowerShell-safe コマンド）+ `anet-core-test` 全緑。
- LunarLander: 書き換え後設定（quantile_mode=qr / R.qr.num_quantiles=32）の短時間 Run が従来と同等に動く（挙動不変確認。config dump = runs/<name>/config/config_data.txt で quantile_mode / qr.num_quantiles を確認）。
- LunarLander: quantile_mode=iqn + サンプル NN 設定で短時間 Run が学習ループを回る（クラッシュ・NaN なし。eval が動く）。

## 後続

- probe / TensorDictFunction 経路への taus 自動補完（fixed_midpoint）
- risk-sensitive tau 歪み（CVaR 等）の tau_rule 拡張
- Rainbow の quantile_mode 統一
