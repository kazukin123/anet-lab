# 可塑性 weight norm（feature/readout 分割）PRD

> 設計分担: Claude=設計/PRD、実装=Codex。本書は self-contained。
> 親 PRD: `062_plasticity_metrics_10prd.md`（同じ測定タップへの増分。062 スコープ外節の「重みノルム・層別勾配などの追加指標。欲しくなったら同じタップに足せる」の実行）。
> 関連バックログ: `920_nn_block_metrics_10prd.md`（層別・ブロック別の汎用 weight/grad メトリクス、暫定）。**本 PRD は 920 の代替ではない** — Q = w・φ の帰属という一問に答える 2 スカラーの最小追加であり、920 の層別汎用系は据え置き。

## Context（背景・目的）

062 実装後の初回検証 Run（RR8 breakout 5M、`run_20260828-000954_plasticity_rr8_breakout`）で次が実測された:

- 崩壊（eval ピーク 2.66M exp → 低下、q_gap −57%）と同期する srank 低下・dormant 増加は**無し**（方向系の損耗はシロ）。
- 一方で activation 側 `feature_norm`（E[‖φ‖₂]）は**無制限成長**し、後半（3.5M→5M exp）は q_max がほぼ頭打ち（+4%）なのに feature_norm だけ加速（+26%）。同じ後半窓で `dead_ratio` だけが単調 3 倍（0.011→0.033）。

Q(s,a) ≈ w・φ(s) の観点で「Q 横ばい + ‖φ‖ 成長」は、**w が縮んでいる**のか **φ が Q に寄与しない方向へ成長している**のかの二択だが、現メトリクスはパラメータ側を測っておらず分離できない。パラメータノルムを φ を作る側（feature）と読み出す側（readout）の 2 群で出し、この帰属を閉じる。

パラメータノルム成長自体が可塑性喪失の独立機序候補（ランク崩壊・休眠と並ぶスケール系。BBF/BTR が Spectral Norm / reset で抑えている対象）であり、保護機構の導入判断材料になる。

## 計測の定義

### 群分割 = feature_key の依存閉包（D1）

- **feature 群** = `feature_key`（既定 `main_feature`）の**依存閉包に属する branch** の全学習パラメータ。閉包規則は `ForwardUpTo` と同一（bind 項を逆向きに辿り、入力 key を同名 branch より優先。`nn_impl.cpp:1100-1112`）。
- **readout 群** = **閉包外の全 branch + 全 heads** の全学習パラメータ。

現行構成での帰属:

| 構成 | feature 群 | readout 群 |
|---|---|---|
| Atari IQN（現行 baseline） | `main_feature`（Impala backbone、bind=grid） | `tau_embedding` / `value_stream` / `adv_stream` / heads（`iqn_fusion` は structure 空 = パラメータ無し） |
| Atari QR / Nature | `main_feature` | `value_stream` / `adv_stream` / heads |
| ImageCls | backbone（`main_feature` 閉包） | LinearHead ほか閉包外 |

モジュール境界（body/head）分割は不採用: 現行 Atari は dueling を**明示 V/A ストリーム = body 側 branch** で組む（`Atari.txt:585-610`）ため、w 成分が φ 側に混ざり帰属に答えられない。

### ノルム定義（D2）

群内**全学習パラメータ**（`requires_grad == true`。weight・bias・normalization affine を区別しない）の一括 L2:

```
weight_norm_group = sqrt( Σ_p ‖p‖₂² )    集計は FP32（BF16 パラメータは cast して二乗和）
```

- 測定時点 = **update 適用前**（loss / q_gap / 062 @learn capture と同一重み）。
- データ非依存（バッチ・RB と無関係）・RNG 非消費・決定的。forward 不要。
- チャネルは 1 本のみ。**probe 変種なし**（バッチ非依存のため無意味）、**target 変種なし**（online の EMA でほぼ冗長。必要になれば同型で追加可能 = スコープ外）。

### 出力スカラー

| tag | source key | 内容 |
|---|---|---|
| `34_agent_plasticity/61_weight_norm_feature` | `plasticity_weight_norm_feature` | feature 群の一括 L2 |
| `34_agent_plasticity/62_weight_norm_readout` | `plasticity_weight_norm_readout` | readout 群の一括 L2 |

decade `6x` は activation の 3 チャネル（`0x` / `2x` / `4x`）と別枠。weight norm はデータに依存せず
probe / target 変種を持たないため、チャネルの番号体系に属さない。

EMA 行は作らない（点のばらつきが原理的に無い量のため）。

## 決定事項（2026-08-28 グリル）

| # | 決定 |
|---|---|
| D1 | 群分割 = **依存閉包 split**。モジュール境界 body/head は不採用（上記）。層別 config 指定式は 920 の領分で過剰 |
| D2 | ノルム対象 = **全学習パラメータの一括 L2**（bias 経由の値スケール成長も漏らさない。文献の parameter norm と同定義） |
| D3 | 文書 = 新規 063 単独 PRD（062 はレビュー完了・コミット待ちのまま凍結） |
| D4 | target / probe 変種なし・EMA 行なし（Claude 提案・異議なし） |
| D5 | **新 config キーなし**（`learner.plasticity.feature_key` を再利用）。cadence = 該当 2 行 interval の min（062 と同じ規則、購読ヒント経由） |

## 実装仕様

### 1. 閉包計算の共通化（NetworkBody）

`ForwardUpTo`（`nn_impl.cpp:1100-1112`）にインラインの閉包計算を `NetworkBody::ComputeDependencyClosure(const std::string& branch_key) -> std::set<std::string>` として抽出し、`ForwardUpTo` はこれを呼ぶ形へ（挙動不変のリファクタ。回帰は既存 ForwardUpTo テストが担保）。未知 key は既存 fail-fast（`GetBranchNames()` 一覧付き、`nn_impl.cpp:1096-1097`）。

### 2. パラメータノルム分割（Network）

`NetworkParameterNormSplit { feature, readout }` と const な
`Network::ComputeParameterNormSplit(const std::string& feature_key)` を追加。両fieldは 0-dim FP32 Tensor（device 上。呼び出し側で 1 回にまとめて D2H してよい）:

- branch のパラメータは閉包所属で feature / readout に分割。heads のパラメータは常に readout（heads は `NetworkBuilder::BuildNetwork(…, head_factory, …)` で Network 内に構築される。`dqn_based_agent.cpp:346,355`）。
- パラメータ列挙は各 module の `parameters()`（recurse）から`requires_grad == true`だけを対象にする。v1ではparameter集合をcacheせず、購読cadenceごとに所属を解決する。列挙costが実測で問題になった場合だけcacheを後続検討する。
- 群にパラメータが 1 つも無い場合は 0.0（正常値。例: feature_key が入力直結の無パラメータ branch）。

### 3. DQN 系 Learner の配線（dqn_based_agent.cpp）

- `PlasticityState` に `weight_norm_enabled` / `weight_norm_interval` を追加。
- `ConfigureScalarMetricSubscriptions`（`:1869-1916`）の分類へ 2 key を追加。`feature_key` 必須化・branch 存在検証の発動条件（`:1900-1901`）に `weight_norm_enabled` を含める。
- learner ループの cadence gate（062 online capture と同じ位置 = update 適用前）で `ComputeParameterNormSplit` を呼び、結果を pending へ → `MakeBatchUpdateResult`（`:2354-2368`）で搭載。既知 key・非測定 step は **NaN**（062 の契約そのまま）。購読ゼロなら計算コード完全不到達。

### 4. ImageCls の配線（image_cls_agent.cpp）

- recognized set（`:472-474`）へ 2 key 追加、同じ cadence gate（`:341`）で `network_->ComputeParameterNormSplit` を実行し result へ搭載（`:423-428` の隣）。

### 5. metrics 行

```
# metrics_scalar.txt @baseline（34 群、probe 系の後）
metrics.scalar.@baseline.[34_agent_plasticity/61_weight_norm_feature] = plasticity_weight_norm_feature @learn $learn_step $update_result interval:500
metrics.scalar.@baseline.[34_agent_plasticity/62_weight_norm_readout] = plasticity_weight_norm_readout @learn $learn_step $update_result interval:500
```

ImageCls.txt のフラット群にも同 2 行。**$update_result を明示**する（062 残課題の「三者不一致は明示側へ統一」に合わせる）。

### 6. その他

- Rainbow は見送り（062 D8 と同じ理由: 実行 smoke 不能）。
- `CONTEXT.md`「可塑性・表現統計」節へ用語 1 件追加: **weight norm 分割（feature/readout）** — feature_key の依存閉包で分けたパラメータ 2 群の一括 L2。readout = 閉包外 branch + heads。
- 実装後、`docs/design/030_user_guide_analysis.jp.md` §4.7 へ 06/07 の読み方 1 項目を追記（q_max・05/45 と重ねる三点読み）。

## 予測（この計測で何が判定できるか）

RR8 後半（learn 110k〜155.6k = exp 3.5M〜5M。q_max +4% / activation feature_norm +26% / dead 単調増）の再 Run で:

| 観測 | 判定 |
|---|---|
| readout norm が低下 | w 縮小で Q 横ばい = 値スケールの担い手が φ 側へ移動（スケールの押し付け合い。homeostasis 型） |
| readout norm 平坦・feature norm（06）上昇 | φ が Q に寄与しない方向へ成長（実効的な cos 低下）。backbone のパラメータスケール暴走が activation norm 成長（45）の源泉 |
| 両群とも上昇 | 全体スケール暴走。Spectral Norm / L2-init / reset 系（BBF/BTR 装備）の導入判断材料 |

`weight_decay = 0` 採用構成ではノルム成長は素で出る — WD 実験（0.1、30M で交差し −43.4%）の「機構は効くが成績は落ちる」との対比材料にもなる。

## 測定上の注意

- readout は非線形（V/A stream + SiLU + heads）なので、厳密な Q = w・φ の代数分解ではなく**群スケールの代理**。cos そのものは測らない（スコープ外）。
- ノルム絶対値はパラメータ数に依存する。**同構成 Run の時系列・同構成 Run 間の比較専用**。アーキ・Scale をまたぐ絶対値比較は不成立。
- activation 側 `03/43_feature_norm`（E[‖φ(s)‖₂]、データ依存）と `61_weight_norm_feature`（‖θ‖、データ非依存）は**別物**。前者は「特徴ベクトルの大きさ」、後者は「それを作るパラメータの大きさ」。
- 群分割は起動時の構造で固定（学習中不変）。

## 現行コードで確定している事実（実装の下地）

1. 閉包計算は `ForwardUpTo` 内のインラインラムダ（`nn_impl.cpp:1100-1112`。bind 逆辿り・入力 key 優先・再帰 insert）。`GetBranchNames()` は `:1124-1132`。
2. heads は `NetworkBuilder::BuildNetwork(network_config, obs_spec, head_factory, device)` で Network 内に構築される（`dqn_based_agent.cpp:346,355`）— heads のパラメータへ Network 経由で到達できる。
3. Learner の購読分類は `ConfigureScalarMetricSubscriptions`（`dqn_based_agent.cpp:1869-1916`）: probe / target / online の 3 分類 + min interval、`feature_key` 必須化と branch 存在検証は `:1900-1913`。
4. @learn 系の搭載点は `MakeBatchUpdateResult`（`:2354-2368`、rank-2 fail-fast 同居）。既知 key・非測定 step = NaN、未知 key = nullopt の契約は 062 実装で確立済み（観測側は `UpdateResultMetricsLookup.recognized` で探索停止）。
5. ImageCls の分類は flat（`image_cls_agent.cpp:467-483`）、capture gate は `:338-341`、搭載は `:423-428`。
6. Atari の V/A ストリームは body 側 branch（`Atari.txt:585-610`）。`iqn_fusion` は structure 空 = パラメータ無し（`:605`）。IQN の `tau_embedding` は `taus` 起点で `main_feature` の閉包外。
7. 現行 baseline の `feature_key = main_feature`（`agent.txt` の `DefaultDQNAgent.@baseline.learner.plasticity.feature_key` / `ImageCls.txt` の `ImageClsAgent.plasticity.feature_key`）。

## テスト項目

1. **閉包分割の帰属**: IQN 風ミニ構成（backbone / tau_embedding / fusion(bind 積・structure 空) / V/A / heads）で feature 群 = backbone のみ、readout 群 = 残り全部になること。`ComputeDependencyClosure` と `ForwardUpTo` の閉包が一致すること（共通化の回帰）。
2. **値検証**: 既知パラメータの小ネットで sqrt(Σ‖p‖²) の手計算と一致。パラメータ無し群 = 0.0。BF16 パラメータ構成で FP32 集計・入力非破壊。
3. **不活性と独立性**: 購読ゼロで計算コード不到達。06/07 の 2 行だけ購読したとき 062 系（capture/probe）は不活性のまま、weight norm だけ独立 cadence で動く。
4. **NaN 契約**: 非測定 learn_step で既知 key → NaN（nullopt でない）。
5. **ON/OFF 等価性**: RNG 非消費・同 seed で学習系列不変（062 受入 5 と同型）。
6. **ImageCls 経路**: 搭載と cadence が DQN 系と同契約。

## スコープ外

- 層別・ブロック別のノルム/勾配/update 比率（`920_nn_block_metrics`）。
- target net 側のノルム、cos(w, φ) の直接測定、Δw 系。
- 保護機構そのもの（Spectral Norm / L2-init / reset）— 本 PRD は判断材料を出すまで。

## 受入基準

1. smoke Run（DQN 系いずれか + ImageCls）で `06/07` が `inspect_run.py tags` で `status=ok`・`count>0`。
2. 購読ゼロ（06/07 行なし）で完全不活性。
3. ON/OFF 等価性: 同 seed で学習系列 + `agent_close.anet` 一致。
4. throughput: interval:100 で既存比有意差なしの目視確認（厳密測定は不要 — forward を伴わない 2 ノルムのため）。
