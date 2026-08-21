# V2: 可視化層（metrics_image / HeatMap / 状態スイープ）の TensorDict 復旧 実装指示書

## Context（なぜやるか）

V2 対応として Observation を `torch::Tensor` から `anet::TensorDict`（複数の観測キー: `vector` / `grid` / `action_mask`）へ移行してきた。コア層（`ExperienceSamples`、`ReplayBuffer` storage、`Network` / `DefaultDQNAgent::GetTensorDictFunction`、`DictFrameStacker`）は移行済みだが、**可視化層が複数の暫定スタブで辛うじて成立**しており、`apps/runner/config/metrics_image.txt` の各 image type が動かない。これがV2移行のほぼ最後の残件。

設計判断は本書冒頭の「確定方針」で確定済み。**本書は実装手順書であり、設計の再検討は不要**。記載どおり実装してほしい。言語・コメントは日本語、ログは英語（`AGENTS.md` のルールに従う）。

### 関連ドキュメント
- `docs/v2_remaining.md` — 本書はこの「可視化」節を実施するもの。
- `docs/memo/006_conv2d_visual_trace_10prd.md` — conv2d 可視化を trace 化した先行作業。その中で**本作業（Sweeped V2）へ明示的に先送りした項目**がある（`SweepedHeatMapObserver` の `GetTensorFunction`→`GetTensorDictFunction` 切替、`DefaultDQNAgent::GetTensorDictFunction` 本体整理、`NetworkModel::GetTensorDictFunction` のキー意味づけ）。本書で回収する。
- `docs/memo/002_vec_exp_10prd.md` — Observation の TensorDict 化（vector experience）の起点。
- `CONTEXT.md` — 用語（Observation / 観測キー / probe / 状態スイープ）。本作業に合わせて追加済み。
- `docs/adr/0002-tensordict-function-unify.md` — 本書の根幹の API 方針決定（`TensorDictFunction` 一本化）。

---

## 1. 現状（調査結果）

可視化のデータ取得は **3 つの source** に分かれ、残件はここに対応する。可視化 Observer 本体は `metrics.image.[tag]` → `ImageProviderManager` 経由で**生きた登録経路**にある（`core/anet-core/src/image.cpp:572-605`、設定パース `image.cpp:631-691`）。壊れているのは Observer が呼ぶ**データ層**だけ。

| source | 経路 | 対応 image type | 壊れている箇所 |
|---|---|---|---|
| experience | バッチの obs/next_obs/action aux | `HeatMap visit-maxq` | `probe.cpp` の vector 前提暫定処理 |
| agent | ReplayBuffer storage / PER | `HeatMap per-prio`, `TimeHistgram per-prio` | `ReplayBuffer::GetTensorVector` が nullopt |
| network(sweep) | state 格子を NN forward | `SweepStateHeatMap qmax`, **HeatMapPanel** | `GetTensorFunction` スタブ / NetworkModel head 未対応 / Dialog vector 前提 |

確認済みの要点（file:line は実装時に grep 再確認のこと）:
- `anet::rl::ToUnifiedObservation`（`core/anet-core/src/rl.cpp:260-303`）は **multi-key TensorDict → 単一 Tensor** を「vector 束（辞書順）→ grid 束（辞書順）」で concat する。**vector が先頭**に来るため、既存の `index = 0/1` 指定は vector-only でも DropMerge（vector+grid）でも vector 部分を指し続ける。→ **互換ブリッジとして使える**。
- `action.max_q` は V2 でも供給されている（`core/anet-core/src/dqn_based_agent.cpp:505` `aux["max_q"] = max_q`、aux 取り出しは `dqn_based_agent.hpp:213`）。experience source の visit-maxq は復旧可能。
- `DefaultReplayBuffer::GetScalar` / `GetTensor` / `GetTensorVector` は**素の nullopt スタブ**（`core/anet-core/src/replay_buffer_impl.cpp:1087-1100`）。storage は TensorDict（`obs_storage_`、`replay_buffer_impl.hpp:148-189`）。
- `NetworkModel::GetTensorDictFunction`（`core/anet-core/src/dqn_based_agent.cpp:196-214`）は現状 `policy-net.conv2d` / `target-net.conv2d` のみルーティング。sweep が使う `policy-net.forward(.q/.dist/.v/.a)` / `target-net.*` は**未解決**。
- `DefaultDQNAgent::GetTensorFunction`（`default_dqn_agent.cpp:278-314`）と `NetworkModel::GetTensorFunction`（`dqn_based_agent.cpp:172-194`）は**全 Agent で nullopt スタブ**。`TensorFunction` の利用箇所は sweep 経路のみ（`HeatMapPanel.cpp:303`、`image.cpp:257`、`SweepedHeatMap::EvaluateTensorFunction`、`SweepedHeatMapObserver::tensor_fn_`）。
- env の観測構成: LunarLander / CartPole / GridMaze = `vector` のみ。DropMerge / ImageCls = `vector` + `grid` の **multi-key が実在**。

---

## 2. 確定方針（決定事項）

- **D1（sweep API）**: NN 呼び出しを **`TensorDictFunction` に一本化**する。旧 `TensorFunction(torch::Tensor)` 系（`TensorFunctionProvider`、各 Agent の `GetTensorFunction` オーバーライド、`NetworkModel::GetTensorFunction`）を**撤去**する。→ ADR-0002。
- **D2（観測キーのスコープ）**: `SweepStateHeatMap` は `sweep_obs.obs_key` で指定された `SpaceType::Vector` の観測キーを sweep 対象にする。未指定時だけ後方互換として `vector` を使う。grid / image sweep は延期するが、複数の vector-type 観測キーを持つ env でも対象を明示できる形にする。
- **probe 文法**: ドット接尾 `experience.next_state.obs.<subkey>`。無接尾時は unified（vector 先頭）。**既存パーサを流用、設定スキーマは変えない**。
- **ReplayBuffer**: storage-level のみ復旧。sampled-batch view は別 view として延期。
- **multi-key sweep**: sweep 対象以外の観測キー（grid 等）は `StateSpec` の shape / dtype に合わせた zeros 埋めで同梱する（DropMerge も crash させない）。意味的限界はコメント／ドキュメントに明記。
- **sweep 出力選択**: `sweep_obs.network_key` は `policy-net.` / `target-net.` と `Network::GetTensorDictFunction()` の key を表す。`sweep_obs.output_key` は戻り `TensorDict` から描画対象 tensor を選ぶ key とする。未指定時は `q`。

---

## 3. フェーズ計画（依存順 easy→hard・各フェーズ独立検証可能）

> SweepStateHeatMap 群（Phase 3-4）を最後に置く。各フェーズ後に Debug ビルドを通し、該当 image type の出力を確認してから次へ進む。

### Phase 1 — experience source 復旧 + probe.cpp 整地（基盤）

**目的**: `HeatMap visit-maxq` を復旧し、以降が乗る probe データ層を整える。

**作業**:
1. `core/anet-core/src/probe.cpp` の「default vector observation key 前提」暫定処理を整理する。obs 抽出は `anet::rl::ToUnifiedObservation()` 経由に統一する。
   - 対象: `BatchExperienceStateProbe`、`BatchExperienceVectorProbe::GetVector`（宣言 `core/anet-core/include/anet/probe.hpp:180-202`）。`obs->flatten(1)` 直書きや「唯一キー前提」のコメント付き箇所を `ToUnifiedObservation` 呼び出しに置換する。
2. `BatchExperience::GetTensor()`（`core/anet-core/src/rl.cpp:459-497`）のキー解決を `obs.<subkey>` パース対応化する。
   - `experience.next_state.obs` / `experience.state.obs`（無接尾）→ 現状どおり `ToUnifiedObservation`（vector 先頭）。
   - `...obs.<subkey>` 接尾 → TensorDict から該当キーを抽出する経路を**用意**する。ただし**実配線・テストは `vector` のみ**でよい（他キーは将来）。未知 subkey は明示エラー（`ANET_SYSTEM_ERROR`）。
3. `action.max_q`（aux）経路が通ることを確認する（データは供給済み。`probe.cpp` の `action.` プレフィックス解決 → `DQNActionInfo` aux dict）。

**変更ファイル**: `core/anet-core/src/probe.cpp`、`core/anet-core/include/anet/probe.hpp`、`core/anet-core/src/rl.cpp`（`BatchExperience::GetTensor` のキー解決）。

**検証**: LunarLander で短時間 run → `runs/.../` の image 出力に `image.phm.visit-maxq` の heatmap PNG が出ること。

### Phase 2 — agent source 復旧（ReplayBuffer 可視化）

**目的**: `HeatMap per-prio` と `TimeHistgram per-prio` を復旧する。

**作業**:
1. `DefaultReplayBuffer::GetTensorVector()`（`core/anet-core/src/replay_buffer_impl.cpp:1097-1100`）を実装する。必要に応じ `GetScalar` / `GetTensor`（同 1087-1095）も。
   - `replaybuffer.storage.state` / `.next_state`（および `.target_return` / `.action` / `.terminal` / `.n_step`）→ storage を直接 gather して `[N, D]` で返す（`storage.reward` は実態が N-step target return のため `storage.target_return` に改称）。
   - `replaybuffer.per.values` / `.total` → 生 PER priority。`replaybuffer.per.distribution` → 正規化サンプリング確率 `p/total`。
   - キー定数は `ReplayBuffer`（`core/anet-core/include/anet/rl.hpp:736-748`）に既存。それを参照する。
2. 委譲経路は `Learner::GetTensorVector`（`core/anet-core/src/dqn_based_agent.cpp`、`replaybuffer.` プレフィックスで ReplayBuffer へ委譲。grep で位置確認）→ `AgentTensorVectorProbe`（`probe.cpp`）。既存ディスパッチを使う。
3. スコープは **storage-level のみ**。sampled-batch view は実装しない。

**変更ファイル**: `core/anet-core/src/replay_buffer_impl.cpp`（必要なら `.hpp`）。経路確認のため `dqn_based_agent.cpp` の `Learner::GetTensorVector` を参照。

**検証**: run → `image.phm.per-prio`（heatmap）と `image.thg.per-prio`（time-histgram）が出ること。

### Phase 3-4 — network sweep 基盤 + SweepStateHeatMap observer / UI 復旧

**目的**: API を `TensorDictFunction` に一本化し、`SweepStateHeatMap qmax`（file 出力）と runner の対話 Sweep HeatMap パネルを同時に復旧する。

**作業**:
1. **旧 API 撤去 [D1]**:
    - `Agent` の基底から `TensorFunctionProvider` を外す（`core/anet-core/include/anet/rl.hpp:677`）。
    - `TensorFunction` typedef / `TensorFunctionProvider`（`core/anet-core/include/anet/common.hpp:101-108`）を撤去。
    - 各 `GetTensorFunction` 宣言・定義を削除: `default_dqn_agent.{hpp:243, cpp:278-314}`、`dqn_based_agent.cpp:172-194`（NetworkModel）、`rainbow_agent.{hpp:83, cpp:118}`、`image_cls_agent.hpp:127`、`muzero_proto_agent.hpp:159`、`episode_end_test.cpp:248`。`grep -rn "GetTensorFunction\|TensorFunctionProvider"` で残骸ゼロを確認。
2. **NetworkModel head routing**: `NetworkModel::GetTensorDictFunction`（`dqn_based_agent.cpp:196-214`、現状 `*.conv2d` のみ）を拡張し、`policy-net.` / `target-net.` プレフィックスを対象 net に振り分けてから、プレフィックスを外した key を `net->GetTensorDictFunction(key)`（head ベース汎用版、`nn_impl.cpp` 内。`006` 5.4 参照）へ委譲する。`policy-net.forward` / `policy-net.forward.q` / `policy-net.forward.dist` / `target-net.forward.*` 等が解決できること。
3. **sweep 経路を dict 化**:
    - `SweepedHeatMap::EvaluateTensorFunction`（`core/anet-core/src/heat_map.cpp:838`、宣言 `heat_map.hpp:216`）の引数を `TensorDictFunction` に。
    - `SweepedHeatMapObserver`（`observers.hpp:164-212`、`tensor_fn_` は :187,203、impl `observers.cpp:332-474`、ctor :336）の `tensor_fn_` を `TensorDictFunction` に。
    - `MakeSweepedHeatMapObserver`（`image.cpp:192-266`、特に `image.cpp:257` の `agent->GetTensorFunction(...)`）を `GetTensorDictFunction` に。
4. **StateSweepProcessor**: `BuildInputTensor`（`probe.cpp:619-665`、ctor :552-602、宣言 `probe.hpp:388-449`）の出力を `TensorDict` に。`sweep_obs.obs_key` の flat-vector 計算（linspace / index_put）は内部に維持し、env の `obs_spec` にある**他キーは zeros 埋め**で同梱する（multi-key env で network forward が落ちないように）。`x_index == y_index` は許容し、現行順序に近い Y 値上書きとする。
5. **出力 tensor 選択**: `SweepedHeatMapObserver` は `TensorDictFunction` の戻り `TensorDict` から `sweep_obs.output_key` の tensor を取得し、既存 extractor に渡す。missing output key は明示エラーにする。
6. **`DefaultDQNAgent::GetTensorDictFunction` / `RainbowAgent::GetTensorDictFunction` の見直し**: sweep 入力は実フレーム履歴でない合成 state なので、stacker 有効時の偽スタック `expand`（1 フレーム → stack_count 複製）の意味づけをコメントで明記する。**挙動を壊さない範囲の最小整理**に留める。
7. **HeatMapPanel UI 復旧**:
   - `apps/runner/src/HeatMapPanel.cpp:303-305` の `agent->GetTensorFunction(...)` を `GetTensorDictFunction` に変更する。
   - `SweepHeatMapDialog` は obs key、network side（policy-net / target-net）、function key（forward / forward.q / forward.dist / forward.v / forward.a）、output key、extractor、X/Y index を分離表示する。
   - `obs_key` は `StateSpec::obs_spec` の `SpaceType::Vector` のみ候補にする。候補が 0 件なら OK を無効化し、理由を表示する。
   - function key 変更時は output key を `forward` / `forward.q` → `q`、`forward.dist` → `q_dist`、`forward.v` → `v`、`forward.a` → `a` に初期化する。
   - `CreateObserver()`（`HeatMapPanel.cpp:236-308`）の `StateSweepProcessor` 構築を dict 経路に合わせる。

**変更ファイル**: `rl.hpp`、`common.hpp`、`default_dqn_agent.{hpp,cpp}`、`dqn_based_agent.{hpp,cpp}`、`rainbow_agent.{hpp,cpp}`、`image_cls_agent.hpp`、`muzero_proto_agent.hpp`、`episode_end_test.cpp`、`heat_map.{hpp,cpp}`、`observers.{hpp,cpp}`、`image.{hpp,cpp}`、`probe.{hpp,cpp}`、`apps/runner/src/HeatMapPanel.{hpp,cpp}`、`apps/runner/config/metrics_image.txt`。

**検証**: LunarLander で `image.shm.qmax` の sweep heatmap PNG が出ること。DropMerge（vector+grid）でも起動し crash しない（grid=zeros）こと。runner GUI で Sweep HeatMap パネルを開き、obs key 選択と OK 無効化が機能すること。

---

## 4. 後方互換・エッジケース

- **既存 metrics_image.txt は新仕様へ更新**する。旧 `sweep_obs.network_key = policy-net.forward` は `sweep_obs.output_key = q` の既定でも動くが、tracked config には `obs_key` / `output_key` を明示する。外部旧 config 互換パースは入れない。
- **`obs_key` 未指定時は `vector`** を使う。`vector` が無くても他の vector-type key を推定しない。metrics 側は `ANET_SYSTEM_ERROR`、UI 側は dialog の OK 無効化でユーザーに分かる失敗にする。
- **multi-key sweep の意味的限界**: sweep 対象以外の観測キーを zeros 固定にするため、grid を主観測とする env（DropMerge 等）の sweep 結果は「grid=blank 時の選択 vector 感度」に限られる。コメントで明記。
- **学習回帰なし**: 旧 `GetTensorFunction` 撤去は sweep 経路のみに影響。learner / target 選択 / rollout には波及しない（利用箇所限定を確認済み）。
- `ProfileRange`: 新規に重い処理（ReplayBuffer flatten、sweep の forward）を足す境界には `anet::ProfileRange` を入れる（`AGENTS.md`）。

---

## 5. 検証

ビルドは `AGENTS.md` の MSVC 初期化形式を厳守:
```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug'
```
1. 各フェーズ後に Debug ビルドが通ること。Phase 3 では `grep -rn "GetTensorFunction\|TensorFunctionProvider"` で残骸ゼロ。
2. Phase 1-3: runner を該当 env（LunarLander / DropMerge）で短時間 run し、`runs/.../` の image 出力 PNG を目視確認:
   - Phase 1 → `image.phm.visit-maxq`
   - Phase 2 → `image.phm.per-prio`, `image.thg.per-prio`
   - Phase 3 → `image.shm.qmax`（LunarLander）、DropMerge は起動 crash なし
3. Phase 4: runner GUI を起動し Sweep HeatMap パネルを実操作。
4. 既存テストがあれば `anet-core-test` を実行（`core\anet-core\bin\Debug\anet-core-test.exe`）。probe / ReplayBuffer アクセサのテストは必要に応じ public 経路で追加（本体に test-only API を増やさない）。

---

## 6. スコープ外 / 延期

- grid / image key の sweep。`obs_key` は今回 `SpaceType::Vector` のみを対象にする。
- ReplayBuffer の sampled-batch view（storage-level と別 view）。
- `observers.cpp:1336` の `@todo HeatMap系Observerに対応`（`metrics.[tag]` 別経路。既存 `image.*` 復旧には非クリティカル。余力があれば Phase 3-4 末で対応、なければ後回し）。
- `#if 0` デバッグ断片の整理（mainline 戻し前に削除／正式化）。
- target-net 可視化の拡充、Transformer/attention 系の sweep 一般化。
