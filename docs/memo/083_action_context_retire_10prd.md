# ActionContext の廃止（Actor が RNG と frame stacker を直接持つ）

> 本書は self-contained。実装時は行番号ではなく、近傍のシンボル名で再検索する。
> 挙動を変えない等価リファクタ。
> 改訂: 2026-09-27 再グリル（精査、Claude、`/grill-with-docs`）。起票版が導入予定だった `ObservationPipeline` クラス・
> `ObservationPipelineOutput`・新ファイル対・新テスト・用語「観測パイプライン」・ADR 0048 を後続の観測パイプライン PRD へ送り、
> Actor が RNG と frame stacker を直接持つ最小形へ縮小した（§複雑さ監査）。設計文書に残る旧 `CreateActor` シグネチャ 3 箇所の修正と
> `agent.txt` コメントの移行を追加。

> 実装時の追加合意（2026-09-27）：全件テストで発見したthread pool SIGSEGVの診断・修正もユーザー指示により範囲へ追加。
> 原因はTracy静的リンク時のWindows allocatorのthread終了回収無効化であり、詳細と検証結果は[実装メモ](083_action_context_retire_20impl.md)に記録する。

## Context（背景・目的）

DQN の Actor は `ActionContext` を介して 2 つの無関係な関心を 1 つの束で受け取っている。

- **観測加工**: `ActionContext::PushObservation(state)`。`StackerActionContext` は `DictFrameStacker::Stack(obs, episode_start)` で
  stack と device 転送を行い、`DefaultActionContext` は `obs.To(device)` だけを行う。ところが正規化は Actor 本体が別メンバ
  `obs_norm_` で行うため、「行動選択前の観測加工」が context と Actor 本体の 2 箇所に割れている。
- **RNG**: `ActionContext` は `anet::RandomHolder` を継承し、Actor は `context_->GetRandomGenerator()` で方策の乱数を得る。
  `MuZeroActor` は `RandomHolder` を直接継承しており、context 経由で RNG を受け取るのは DQN の Actor だけである。

加えて `ActionContext::Reset()` は呼び出し元が無く、stacker.cpp には
`@todo 推論と学習における FrameStacker の重複解消と ActionContext 不要論` と `@todo ActionContext依存のStacking再考` が残っている。
`Actor::MakeAction` は `context_ != nullptr` を確認して `PushObservation` を呼ぶ一方、直後の `GetRandomGenerator()` は無条件に参照しており、
null 許容の契約も曖昧である。

後続 PRD では観測加工を「フレーム段 → スタック段 → バッチ段」の観測パイプラインへ一般化し、spec 変換の所有をステージ側へ移す。
その容器（クラス）・用語・ADR は段の要件が揃う後続 PRD で導入する。本 PRD は **ActionContext を引退させ、行動選択前の観測加工を
Actor 本体の 1 箇所に置く**だけに留める。データの演算列・乱数列・aux の集合は一切変えない。

## 確定した設計判断

1. **`ActionContext` / `DefaultActionContext` / `StackerActionContext` を削除する。** クリーンブレークで、alias・互換 overload は残さない。
2. **RNG は Actor が持つ。** `dqn::Actor` が `anet::RandomHolder` を継承し `RandomHolder(seed)` で初期化する。seed は
   `ActorRequest::seed` をそのまま渡す（ctor 引数は `MuZeroActor` と同じ `std::optional<seed_t>`）ので、乱数の派生と消費順は現行と同一。
   `MakeAction() const` の中では `MuZeroActor` と同じく protected の `rnd_` を直接 `SelectAction` へ渡す
   （`RandomHolder::GetRandomGenerator()` は非 const）。これは ownership_guideline.md が private Resource の例に挙げる
   「`ActorRequest.seed` から生成し、その Actor だけが消費する private RNG」に当たる。
3. **観測加工は Actor 本体に置く。** `dqn::Actor` が `std::unique_ptr<FrameStacker> stacker_`（nullptr 可）と `torch::Device device_` を
   直接持ち、既存の `obs_norm_` と合わせて stack → device 転送 → 正規化を `MakeAction` の中で現行の並びのまま行う。
   クラス化（`ObservationPipeline`）はしない。
4. **device 転送の所在は現行どおり。** stacker があるときは `DictFrameStacker` が自身の device 上で stack し（`non_blocking=true`）、
   無いときは Actor が `state.obs.To(device_)` する（blocking。現行 `DefaultActionContext` と同じ）。二重転送を足さない。
   `device_` は optional にしない（production は常に `request.device` を渡す。テストは `torch::kCPU` を渡す）。
5. **所有。** stacker は per-Actor の ring buffer なので Actor が `unique_ptr` で私有する（現行の `shared_ptr<FrameStacker>` から変更。
   共有者は無い）。normalizer は Agent 所有の共有 Resource を `shared_ptr` で参照する（現行 `obs_norm_` のまま）。
6. **生成。** `DefaultDQNAgent::CreateActionContext` を `CreateFrameStacker(const ActorRequest&) const` へ置き換える
   （戻り値 `std::unique_ptr<FrameStacker>`。`use_stacker` でなければ nullptr）。Rainbow は stacker 無し（nullptr）で Actor を直接構築する。
7. **ファイル配置。** 新規ファイル無し。`agent.hpp` から ActionContext 群を削除し、`stacker.hpp` / `stacker.cpp` は
   `FrameStacker` / `DictFrameStacker` だけを残して `agent.hpp` の include を外す（`rl.hpp` / `tensor_util.hpp` は既に include 済み。
   `stacker.hpp` の include 元は `default_dqn_agent.cpp` / `stacker.cpp` / `stacker_test.cpp` の 3 つ）。CMake はソースを glob しているので列挙の編集は不要。
8. **名前。** 公開クラス名は増えないので CONTEXT.md への用語追加は無し。CONTEXT.md「Actor 生成要求」の `_Avoid_` にある
   「actor context（Observation 加工の部品）」だけ、廃止済みを示す注記へ改める。
9. **`Reset()` は復活させない。** lane 単位の初期化は `episode_start` マスクが担っており、全体リセットの呼び出し元は無い。
   `FrameStacker::Reset()` / `DictFrameStacker::Reset()` は本 PRD 前から production 呼び出し元が無い既存 dead code として残し、削除しない（報告のみ）。
10. **直さないもの。** Actor が `obs_norm_->Normalize` を Agent mutex の外で呼び、統計更新は `DefaultDQNAgent::UpdateFromBatch` が
    unique lock 内で行う潜在競合（dynamic scaling 時のみ顕在化。現行設定は全て `pass_through = true`）。挙動を変えないため本 PRD では触らず、
    後続 PRD の共有 / 私有境界の設計で扱う。設計文書には既知事項として残す。
11. **等価性の合否は metrics checksum。** [065](done/065_nn_spectral_norm_10prd.md) の手順に従う。checkpoint の raw checksum は
    現行 serialize が非決定のためゲートにしない（[930](930_serialize_10prd.md)）。
12. **隣接ドリフトの修正。** 設計文書 110 §6.1 / 200 §6.1 の sequence 図と 110 §7.1 本文に残る旧
    `CreateActor(batch_env_spec, env_spec, run_mode, override, device)`（[ADR 0038](../adr/0038-actor-config-catalog-without-runmode.md) で
    `ActorRequest` へ置き換え済み）を、書き換える図と同じ変更内で現行契約へ直す。
13. **ADR は作らない。** RNG の Actor 所有は `MuZeroActor` と ownership_guideline の前例踏襲で、後から読んで驚く判断が無い。
    クラス化を見送った取捨は §複雑さ監査に残す。

## 仕様

### dqn::Actor

- 宣言: `class Actor : public anet::rl::Actor, public anet::RandomHolder`。
- メンバ: `obs_norm_` は不変。`std::unique_ptr<FrameStacker> stacker_` と `torch::Device device_` を追加し、`context_` を削除する。
- コンストラクタ: `Actor(std::shared_ptr<ActionPolicy> policy, std::shared_ptr<ObservationNormalizer> obs_norm,
  std::unique_ptr<FrameStacker> stacker, torch::Device device, std::optional<seed_t> seed, std::shared_ptr<std::shared_mutex> mutex,
  network, src_network, emit_actor_q_hint, snapshot_sync_interval, emit_snapshot_metrics, actor_q_hint_config)`。
  旧 `context` の位置に stacker / device / seed を挿入し、`obs_norm` の位置は変えない（テスト差分を最小にする）。初期化子で `RandomHolder(seed)`。
- `MakeAction`: 現行の

  ```cpp
  auto obs = state.obs;
  if (context_ != nullptr) {
      obs = context_->PushObservation(state);
  }
  ```

  を

  ```cpp
  // frame stack と device 転送（stacker 無しなら device 転送のみ）
  auto obs = stacker_ ? stacker_->Stack(state.obs, state.episode_start) : state.obs.To(device_);
  ```

  へ置き換え、`auto rnd = context_->GetRandomGenerator();` を削除して `policy_->SelectAction(norm_obs, false, network_, rnd_, callback)` に
  `rnd_` を直接渡す（`std::shared_ptr` の値渡し。現行と同じ）。正規化、aux（`raw_obs` は常時、`norm_obs` は `obs_norm_` 有のときだけ）、
  `ANET_LOG_DEBUG` の obs / norm_obs 出力は無変更。
- `Sync` / `GetScalar` は不変。

### DefaultDQNAgent / RainbowAgent

- DefaultDQN: `CreateFrameStacker(request)` は `config_.stucker.use_stacker` のとき
  `std::make_unique<DictFrameStacker>(stack_count, request.batch_env_spec.num_envs, request.device, keys)`（`keys` は現行どおり `stack_keys` が空なら nullopt）、
  そうでなければ nullptr を返す。`CreateActor` は
  `std::make_shared<Actor>(policy, obs_norm_, CreateFrameStacker(request), request.device, request.seed, mutex_, network, source, ...)`。
- Rainbow: `std::make_shared<Actor>(policy, nullptr, nullptr, request.device, request.seed, mutex_, network, source, ...)`。

### 変わらないもの

- RNG の seed 派生と消費順、stack / normalize の演算列、device 転送の回数と blocking / non_blocking の別、aux キーの集合。
- Learner 側（`NormalizeSampleObservations`、ReplayBuffer の stack 再構成）と `GetTensorDictFunction` の可視化経路。
- `rl::Actor` の公開インタフェース（`MakeAction(step, state) const` と `Sync()`）。
- `stacker_test.cpp`（`DictFrameStacker` のみを対象にしている）。

### 設計文書と用語の更新（同じ変更内で行う）

| 文書 | 箇所 | 変更 |
|---|---|---|
| `docs/design/110_agents_and_learning.jp.md` | §3 コンポーネント定義 | `ActionContext` 行を削除する（置換無し。§6.1 末尾が既に「Actor内部のObservation加工、Network forward、Policy、snapshot同期は具象実装の責務」と述べている） |
| 同 | §6.1 sequence 図 | `R->>G: CreateActor(batch_env_spec, env_spec, run_mode, override, device)` → `R->>G: CreateActor(actor_request)`（判断 12） |
| 同 | §7.1 構築設定 | 「ActorのRunModeとmodel複製有無は、Runnerのoverrideと具象Agentの既定をAgent生成境界で解決する。」→「Actorの方策・network選択・model複製有無は、`ActorRequest::actor_key`で参照するActorカタログ（`<Agent>.actor.[key]`）から具象Agentが解決する。RunModeは受け取らない。」（ADR 0038、同文書 §2.1・§2.2 と整合。判断 12） |
| `docs/design/200_dqn_agents.jp.md` | §2.3 本文 | 「ActionContextによるframe stackとdevice転送を行い、Observation正規化」→「`DictFrameStacker`によるframe stackとdevice転送（stacker無しではdevice転送のみ）を行い、Observation正規化」 |
| 同 | §3 コンポーネント定義 `dqn::Actor` 行 | 「ActionContext、正規化、Policy、Network、同期State」→「frame stacker、正規化、Policy、Network、RNG、同期State」 |
| 同 | §6.1 sequence 図 | participant `C as ActionContext` → `S as DictFrameStacker`。`A->>C: PushObservation(batch_state)` / `C-->>A: 加工済みObservation` → `opt use_stacker` 内の `A->>S: Stack(obs, episode_start)` / `S-->>A: stack済みObservation`、続けて `A->>A: device転送（stacker無し）と正規化`。`R->>G: CreateActor(...)` は `CreateActor(actor_request)` へ（判断 12） |
| 同 | §7.1 表 Rainbow 前処理 | 「共通ActionContext。専用scaler/normalizer設定なし」→「frame stackなし（device転送のみ）。専用scaler/normalizer設定なし」 |
| 同 | §9.3 エラー境界 | 判断 10 の潜在競合を既知事項として 1 行追記 |
| `docs/design/150_replay_buffer.jp.md` | §2.3 | 「Actor側の`StackerActionContext`」→「Actor側の`DictFrameStacker`」 |
| `CONTEXT.md` | 「Actor 生成要求」の `_Avoid_` | 「actor context（Observation 加工の部品）」→「actor context（旧 ActionContext の連想。廃止済み）」。用語の追加は無し |
| `apps/runner/config/agent.txt` | Action Policy 設定ガイドラインのコメント 2 行 | 「CreateActionContext/MakeAction呼出」→「CreateActor/MakeAction呼出」（受入 1 の rg 0 件のため） |

`.en.md` は翻訳成果物なので本 PRD では触らず、`anet-translate-docs` に追従を任せる。ADR 0044 は当時の記録として変更しない。
[999_episode_start_stack_contract_10prd.md](999_episode_start_stack_contract_10prd.md) は暫定メモで、既に別点（ReplayBuffer の
`episode_start` 扱い）が ADR 0044 と食い違っているため触らない。150 §3 の `DictFrameStacker` 行と 200 §8.2 の「Actor RNG」（非保存 State）は
現行のまま正しい。

## 非対象（Non-goals）

- spec 所有の移動（`DefaultDQNAgent` ctor の `network_obs_spec` 手畳み）、`stack_keys` 判定 4 箇所の統合、学習側のフレーム段、
  prev_action、`MakeAction` への forced action 入力。すべて後続 PRD。
- 観測パイプラインの容器（クラス）、用語「観測パイプライン」、その ADR。段の要件が揃う後続 PRD で導入する。
- 判断 10 の mutex 競合の修正。
- `RandomHolder::GetRandomGenerator()` の const 化など、共通ヘッダの整備。
- `FrameStacker::Reset()` の削除。
- `.en.md` の更新。

## 受け入れ基準

1. 現用コード・テスト・設定から `ActionContext` / `DefaultActionContext` / `StackerActionContext` / `PushObservation` / `CreateActionContext` が消える
   （`core/` と `apps/` で `rg` 0 件。`apps/runner/config/agent.txt` のコメント 2 行を含む。`docs/memo` の過去 PRD と `docs/adr` は対象外）。
2. **等価性**: 改修前ビルドと改修後ビルドで、同 seed・同 config・同 `exp_exit_step` の短 Run（DefaultDQN、LunarLander、stack4）の
   主要 scalar 系列（loss、q_max 系、`61_eval1` / `62_eval2` の episode return、action 系）が一致する。Rainbow はユーザー指定（2026-09-27）により改修前後 Run 比較の対象外。
   コード移行と既存テストによる回帰確認は行う。checkpoint の raw checksum はゲートにしない。
3. **aux 契約**: DefaultDQN の `MakeAction` は `raw_obs` と `norm_obs` の両方を、Rainbow は `raw_obs` だけを aux に持つ。
4. `MakeAction` は `const` のまま、`dqn::Actor` が `RandomHolder` を継承し、同 seed の action 列が改修前と一致する（2 で担保）。
5. `anet-core-test` 全緑、`git diff --check` 空。
6. 上表の設計文書、`CONTEXT.md` の注記、`agent.txt` のコメントが新契約に同期している。

## テスト項目

新規テストケースは無い。既存テストへ aux の値・shape・キー有無の assertion を追加する（stack の意味論は `stacker_test.cpp`、Actor の aux / snapshot / Q ヒントは既存テストが持つ）。

既存テストの移行（`dqn::Actor` の直接構築 8 箇所）:

1. `dqn_based_agent_test.cpp` の "Actor sync leaves cloned network in eval mode"（context 無しで `Sync()` だけを呼ぶ）:
   `Actor(nullptr, nullptr, nullptr, mutex, clone_network, src_network)` →
   `Actor(nullptr, nullptr, nullptr, torch::kCPU, std::nullopt, mutex, clone_network, src_network)`。
2. 同ファイルで `std::make_shared<rl::DefaultActionContext>(123)` を使う 6 箇所（snapshot 同期系、`emit_hint` 系）:
   `Actor(policy, nullptr, context, mutex, ...)` → `Actor(policy, nullptr, nullptr, torch::kCPU, 123, mutex, ...)`。
3. `dqn_munchausen_test.cpp` の `DefaultActionContext(67021)` 1 箇所: 同様に `nullptr, torch::kCPU, 67021`（同テストの device は `torch::kCPU`）。
   ループ内で毎回 Actor を作り直す構造は維持する（同じ seed から同じ hint が出ることを見るテスト）。
4. 期待値は不変。`DefaultActionContext(seed)` は device 無しで `state.obs` をそのまま返していたが、`state.obs.To(torch::kCPU)` は
   CPU 上の tensor をそのまま返す（`anet::To` は `t.to(device)`）ので、値も実体も同じ。
5. 既存の aux（`raw_obs` / `norm_obs`）と probe を見るテスト、`[native_stack]` 5 件（`CreateActor` 経由）が緑。
6. `stacker_test.cpp` は無変更で緑。

統合（ユーザー実施）:

7. 受入 2 の等価 Run。

## 実装対象

- `core/anet-core/include/anet/agent.hpp` — `ActionContext` / `DefaultActionContext` の削除
- `core/anet-core/include/anet/stacker.hpp`、`core/anet-core/src/stacker.cpp` — `StackerActionContext` の削除（@todo 2 件ごと）、`agent.hpp` include の除去
- `core/anet-core/src/dqn_based_agent.hpp`、`core/anet-core/src/dqn_based_agent.cpp` — Actor の基底、メンバ、ctor、`MakeAction`
- `core/anet-core/include/anet/default_dqn_agent.hpp`、`core/anet-core/src/default_dqn_agent.cpp` — `CreateFrameStacker`、`CreateActor`
- `core/anet-core/src/rainbow_agent.cpp` — `CreateActor`
- `core/anet-core/src/dqn_based_agent_test.cpp`、`core/anet-core/src/dqn_munchausen_test.cpp` — ctor 移行 8 箇所
- `apps/runner/config/agent.txt` — コメント 2 行
- `docs/design/110_agents_and_learning.jp.md`、`docs/design/200_dqn_agents.jp.md`、`docs/design/150_replay_buffer.jp.md`、`CONTEXT.md`

## 検証

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test'
core\anet-core\bin\Debug\anet-core-test.exe "[stacker],[dqn]"
core\anet-core\bin\Debug\anet-core-test.exe
git diff --check
```

等価 Run（ユーザー実施）は 065 の手順に従う。改修前ビルドで同 seed・同 config・`exp_exit_step` 固定の短 Run を採取し、改修後ビルドで
同一コマンドを再実行し、`viewers/metrics-tools/inspect_run.py metrics` に 2 つの Run 名と主要 tag を渡して全点一致を確認する。
Rainbow の改修前後 Run 比較はユーザー指定（2026-09-27）により対象外とする。

## 複雑さ監査（2026-09-27 再グリル）

起票版からの取捨と、残した機構ごとの「切ると戻る痛み」。

| 項目 | verdict | 理由 |
|---|---|---|
| `ObservationPipeline` クラス + `ObservationPipelineOutput` + 新ファイル対 + 新テスト 3 本 + `Process` の profile | **cut（後続 PRD へ）** | Context の 3 ゴール（ActionContext 削除 / 観測加工を 1 箇所へ / 等価）は Actor 本体への inline で満たせる。固定 3 段のクラスは後続の「段の容器」の形ではなく、段の要件が揃う前に形を決めると作り直しになる |
| CONTEXT.md 用語「観測パイプライン」 | **defer** | 公開クラス名が増えないので不要。容器が生まれる後続 PRD で確定する |
| ADR 0048 | **cut** | RNG の Actor 所有は `MuZeroActor` と ownership_guideline の前例踏襲で驚きが無い。A / B の取捨はこの表に残す |
| `dqn::Actor : RandomHolder` | keep | 切ると RNG の置き場が無い |
| Actor 直持ちの `stacker_` / `device_` | keep | 切ると stack と device 転送ができない |
| `CreateFrameStacker` | keep | `CreateActionContext` の機械的置換 |
| 設計文書ドリフト 3 箇所（判断 12） | keep（追加） | ADR 0038 との実矛盾。書き換える図の中に既知の誤りを残さない |
| `agent.txt` コメント 2 行 | keep（追加） | 受入 1 の rg 0 件 |
| 200 §9.3 の既知競合 1 行 | keep | 「直さない」判断の記録 |

## 関連

- [921_agent_prev_action_obs_10prd.md](921_agent_prev_action_obs_10prd.md) — 後続。ActionContext 前提の記述は本 PRD 後に書き直す。
  順序（本 PRD → 観測パイプライン PRD → 921 改訂）は不変。
- [999_episode_start_stack_contract_10prd.md](999_episode_start_stack_contract_10prd.md) — `StackerActionContext` を引用するが暫定メモで既に陳腐化しているため触らない。
- [ADR 0013](../adr/0013-actor-network-resource-policy.md) — Actor private Resource の前例。
- [ADR 0038](../adr/0038-actor-config-catalog-without-runmode.md) — `ActorRequest` と Actor カタログ。判断 12 の修正先の契約。
- [ADR 0044](../adr/0044-replay-frame-history-start-from-episode-start-at-push.md) — Actor 側 stack が `episode_start` で初期化する事実は不変。
- [065](done/065_nn_spectral_norm_10prd.md) — metrics checksum による等価性手順。
- [930_serialize_10prd.md](930_serialize_10prd.md) — checkpoint checksum をゲートにしない根拠。
