# エージェント選択 action の Observation 注入（共通部品・暫定）

> 設計分担: Claude/Codex=設計/PRD、実装=Codex、Run/commit=ユーザー。
> 本書は self-contained。実装時は行番号ではなく、近傍のシンボル名で再検索する。
> ステータス: バックログ（暫定設計）。先行検証は env ローカルの 032 で行う。

## Context（背景・目的）

風など**非観測の外乱**を持つ POMDP では、エージェントは frame stack の速度差分から外乱を
推定する必要がある。しかしその差分には自機エンジンの推力が混入するため、観測だけからの
推定は「まず行動を物理署名から間接デコードし、残差を外乱に帰属する」2 段階の暗黙学習を
要求する。直前 action a_{t-1} を観測に含めれば `Δv − 既知のエンジン効果 = 外乱` が
well-posed になり、外乱推定の学習が速く安定する（R2D2 / Agent57 が prev action を入力に含め、
RMA が state-action 履歴から環境パラメータを推定するのと同型）。

「自分の方策だから行動は観測から復元できる」という仮定は成立しない：
①学習時の行動は確率的（探索）、②replay は過去方策のデータ、③a_{t-1} は stack 窓の外の
入力 o_{t-4} から選ばれており窓内から復元不能。

[032_lunarlander_obs_include_action_10prd.md](done/032_lunarlander_obs_include_action_10prd.md) は
LunarLanderEnv ローカルの暫定実装（env の 8 次元 vector に one-hot(4) を連結）。本 PRD は
その**恒久・env 非依存の共通部品版**の設計を記録する。032 で効果を確認してから本 PRD の
実装に進む段取り。

## 確定した設計判断

1. **表現は別 obs キー**。`ObsKeys::kPrevAction = "prev_action"` を新設し、one-hot float32
   shape `[action_count]` を独立キーとして観測 dict に足す。**kVector への連結はしない**。
   - 理由: env 由来の観測キーを汚さない（関心の分離）。`kActionMask` という「メタ観測キー」の
     前例がある。ReplayBuffer は任意 obs キーを end-to-end で汎用サポートする（後述）。
   - NN は `NetworkBranch` の複数キー bind（内部 `torch::cat`）で合流する。Concat ブロックの
     新設は不要。
2. **配置は 2 案を併記し、実装時に最終決定する**。
   - **案A（本命）: エージェント内部注入**。「action を observation の一部として扱うのは
     agent 側の都合」なので責務上は本筋。ただし act 経路と learn 経路に個別の注入が要り実装が重い。
   - **案B（単純代替）: BatchEnv デコレータ**。ステートレスで実装が軽く、act/learn 整合が
     自動で取れる。ただし action 認識が env 層に漏れる。
3. **意味論**: `obs_t` には「その obs に至らせた行動 a_{t-1}」を入れる。episode 先頭
   （`episode_start=true`）は one-hot 全ゼロ（未行動）。
4. **スコープは discrete one-hot のみ**。連続 action は将来スコープ。`ActionSpec::is_discrete`
   を構築時ガードにする。
5. **default OFF**。flag OFF 時は現行と完全に同一挙動（キーを一切足さない）。

## 案A vs 案B（配置の比較）

| 観点 | 案A: エージェント内部注入（本命） | 案B: BatchEnv デコレータ（単純代替） |
|---|---|---|
| 責務 | ◎ action は agent の産物。env は純粋なまま | △ env 層が action を意識する |
| ステート | act 側 `ActionContext` と learn 側 `Learner` に per-env 直前 action バッファ**二重** | ◎ ステートレス（`Step` 引数 a_t のみ） |
| act/learn 整合 | 各経路で個別注入（learn の next_state は同 exp の action で自動） | ◎ runner が `continue_state` を両用途に使い自動整合 |
| spec 消費者 | obs_norm（env_spec 由来）と network_obs_spec（折込コピー）の**両方**に注入が要る | ◎ `GetSpec` 拡張 1 点で全消費者が追従 |
| wiring | agent 内に閉じる。env 生成点は無関係 | factory + `RunManager` の**2 点** wrap（FLAG1） |
| コード量 | 多い（2 注入点 + 2 spec + リセット意味論の一致） | 少ない（1 デコレータ + 2 wrap 点） |

両案とも観測 dict に `prev_action` キーを生じさせるので、後述の共通設計（spec / NN / viewer /
obs_norm / stacking / edge）は**両案に等しく適用**される。差分は「どこでキーを足すか」だけ。

### 案A: エージェント内部注入（詳細）

`prev_action` を agent が自分の産物として観測へ足す。act と learn の 2 経路で個別に注入する。

- **act 経路**: `MakeAction` の観測前処理で注入する。`ActionContext`（`StackerActionContext`）が
  **stacker の前**で `one_hot(直前 a)` を `obs[prev_action]` として差し込み、`state.episode_start`
  マスクでゼロ化 → stacker が他キー同様に stack → obs_norm → 行動選択 → 選択した a_t を per-env
  バッファへ保存。次 step の注入に使う。per-env バッファは context 保持（runner 毎に独立）。
- **learn 経路**: `Learner::UpdateFromBatch` が Push 前に experience の観測へ注入する。
  - `next_state.obs[prev_action] = one_hot(experience.action)`。next_state に至らせた行動は
    その experience の `action`（a_t）そのもの。**per-env メモリ不要**。
  - `state.obs[prev_action] = one_hot(Learner の per-env 直前 action バッファ)`。`state.episode_start`
    でゼロ化。バッファは毎回 `experience.action` で更新。episode 境界のゼロ化は次 state の
    `episode_start` が担うので、バッファのリセットは不要（マスクで吸収）。
- **spec**: env spec は不変。agent が
  - `network_obs_spec`（`default_dqn_agent.cpp` の折り畳みローカルコピー。`ObservationNormalizer`
    が参照する `env_spec.state_spec` とは別物）**の両方**に `prev_action` を追加する必要がある。
    この二重の spec 追従が案A の実装コストの主因。
- **注意**: act 側と learn 側の per-env バッファは別実体だが、リセット意味論（episode_start で
  ゼロ）を**一致**させること。ここがズレると train と replay で観測分布が食い違う。

### 案B: BatchEnv デコレータ（詳細）

`PrevActionObsBatchEnv`（`BatchEnv` の decorator）を factory の choke point で config flag により
挟む。前例は `PrefetchingReplayBuffer`（`replay_buffer.hpp`、`dqn_based_agent.cpp` で
`use_rb_prefetch` により wrap）。

- **ステートレス**: `BatchEnv::Step(action_info)` は既に全 env の a_t を引数で受け取る。
  デコレータは `next_state.obs` と `continue_state.obs` に `one_hot(a_t)` を足すだけ。per-env の
  履歴を持たない。auto-reset は `VectorizedDiscreteBatchEnv::Step` / `ThreadPoolDiscreteEnv::Step`
  の**内側**で起き、`continue_state` はリセット env に対しリセット obs + `episode_start=true` を
  運ぶ。`Reset()` は全ゼロ。
- **整合が自動**: runner は `state_ = result->continue_state` を次の `MakeAction` にも Push する
  experience の state にも使う（`trainer.cpp` の `DoStep`）。よって act 経路と learn 経路が同じ
  `continue_state`/`next_state` を消費し、agent 側の変更ゼロで整合する（032 の env 側実装と同型）。
- **注入規則（不変条件）**: 全ての発行 `BatchState` について
  `obs[prev_action][i] = one_hot(a_i)` を `episode_start[i]` で 0 マスク。`next_state.episode_start`
  は契約上 true にならないので next_state は常に `one_hot(a_t)`（終端 obs も a_t で到達したので正）。
- **result 生成**: in-place 変異は不可（`getResetResult` は同一 `reset_result_` を毎回返す再利用が
  あり、Pipeline runner が `prev_result_` を 1 step 保持するため）。**delegating wrapper result
  クラス**（inner を shared_ptr で保持、aux は委譲）を**毎回 fresh 確保**する。`BatchState` の
  shallow copy（obs map をコピー、tensor はハンドル共有）に `prev_action` を足すだけなのでコストは
  `(N,A)` float ×2 と小さい。
- **override 一覧**: `BatchEnv` の 6 virtual（`GetSpec` / `GetBatchSpec` / `GetDevice` / `Reset` /
  `Step` / `Shutdown`）+ `Module` 由来の 3 pure virtual（`GetScalar` / `GetTensor` /
  `GetTensorVector`）。`Shutdown` は ThreadPool 停止を含むので**必ず forward**（握り潰さない）。
- **構築時ガード**: `is_discrete` 必須、inner spec が既に `prev_action` を持つ場合はエラー、
  tensor は `inner_->GetDevice()` 上に作る（runner は CPU obs を assert）。

## 共通 spec 仕様

観測 dict に足す `prev_action` の `TensorSpec`（`common.hpp` の TensorSpec）:

| field | 値 | 備考 |
|---|---|---|
| `type` | `SpaceType::Vector` | Vector ⇒ 1-D shape（AssertSanity） |
| `shape` | `{ action_count }` | `action_count = ActionSpec::GetNumActions()`（= `value_labels.size()`、agent の n_actions と同源） |
| `dtype` | `torch::kFloat32` | |
| `num_classes` | `0` | **0 必須**。float×discrete は AssertSanity で弾かれる。obs_norm 回避目的で A を入れてはいけない |
| `labels` | `action_spec.value_labels` をそのまま | 例 LunarLander: "do nothing" / "fire left" / … |
| `min_values` / `max_values` | `{0.0}` / `{1.0}` | broadcast size-1 可 |

`ObsKeys`（`rl.hpp`）に `static constexpr const char* kPrevAction = "prev_action";` を
`kActionMask` の隣に追加。`ValidateObservation`（`rl.cpp`）は**spec 側キーを走査**し obs の
過剰キーは無視するので、拡張 spec で Reset/Step とも通る。

## NN 統合（FLAG2・重要）

`NetworkBranch::Execute` の bind 合流は **`torch::cat(inputs, dim=1)`**（bind 順・決定的）。

**罠**: `bind = vector prev_action` の 1 行変更は stack>1 で**実 forward で落ちる**。stacker/replay は
stack 軸を dim1 に持つため実データは `vector (N,4,8)` と `prev_action (N,4,4)`。`cat(dim=1)` は
後続次元 8≠4 で不可。dummy forward（2D `(1,32)+(1,16)→(1,48)`）だけ通るので初回実 forward まで
露見しない。関連: [057_native_stack_dim_handling_10prd.md](done/057_native_stack_dim_handling_10prd.md)。

### Pattern A（MLP 既定・既存機能で動く）

per-key に Flatten してから合流する。

```txt
net.branch.[vec_flat].bind = vector
net.branch.[vec_flat].structure = Flatten
net.branch.[pa_flat].bind = prev_action
net.branch.[pa_flat].structure = Flatten
net.branch.[main_feature].bind = vec_flat pa_flat       # 共に 2D → cat(dim=1) 可
net.branch.[main_feature].structure = MLP_FC1 > LN256 > SiLU > MLP_FC2 > LN128 > SiLU
```

dummy `(1,32)+(1,16)→(1,48)`、実 `(N,32)+(N,16)→(N,48)`。Lazy Linear の in_features は両経路で
48 に一致。999 の spec 変更後（`[4,8]`→dummy `(1,4,8)`）も Flatten が橋渡しするので堅牢。

### Conv1D 時間軸（per-frame に action を混ぜる）

Pattern A を Conv1D 前段に流用すると崩れる：Flatten 後の cat は**key-major**
（`[v_t0..v_t3 | a_t0..a_t3]`）で、`Reshape "4 -1"` が frame と action を混線させる（`[v_t | a_t]`
にならない）。現行 032 が kVector 連結でうまくいくのは、stack **前**に per-frame 12 次元が
揃っているため `Flatten→(N,48)→reshape(4,12)` が `[8 obs | 4 action]` 行を復元できるから。

- **Pattern C（late fusion・既存機能）**: temporal conv は `bind = vector` のまま
  `Flatten > ReS4F8 > Conv1D_Permute > TConv64 > ...` を維持し、`pa_flat`（bind=prev_action,
  Flatten→`(N,16)`）を dueling stream 側で合流（`value_stream.bind = main_feature pa_flat`、
  adv も同様。共に 2D）。action 条件付けが temporal conv の**後**に入る。
- **Pattern B（要小改修・optional だが本 PRD に含める作業項目）**: `net.branch.[x].bind_cat_dim`
  を追加（default 1 = 現行）。`bind_cat_dim = -1` なら実 `cat((N,4,8),(N,4,4),dim=-1)=(N,4,12)` の
  per-frame `[v_t|a_t]`（正しい時間整列）、dummy `(1,48)`。branch 構造 `ReS4F8 > Conv1D_Permute >
  TConv64 > ...` が dummy `(1,48)→(1,4,12)` / 実 `(N,4,12)→(N,4,12)`（no-op）でランク一致。999
  redesign 後は Reshape が不要化してさらに素直。**cut する場合は Conv1D×action は 032 の env
  ローカル flag に委ねる**と注記する。

## viewer 互換（FLAG3）

2 個目の vector-bucket キーが出ると `ToUnifiedObservation`（`rl.cpp`）が単一キー高速路から
辞書順連結に切替わり、`"prev_action" < "vector"` で prev_action が**先頭**に来る。これは
`raw_obs`/`norm_obs` aux や un-suffixed obs probe の「vector 先頭互換」（`probe.cpp`）を壊す。

→ 随伴変更として **vector bucket の先頭に `kVector` を hoist**（`ToUnifiedObservation` の連結順）。
現状 2+ vector-bucket キーを持つ env は無い（grid は別 bucket、action_mask は除外）ので
behavior-neutral。回帰テストで固定する。

## obs_norm / stacking

- **obs_norm**: `ObservationNormalizerFactory` は float32 かつ `num_classes=0` のキーに per-key
  normalizer を付ける（除外 config は現状無し）。**document-as-acceptable**（032 判断#10 と同じ）。
  - LunarLander は `obs_norm.pass_through = true` で恒等 → 無影響。
  - 一般 dynamic（`use_centering=false, epsilon=1`, SymLog）でも one-hot std ≤0.5、分母 ≥epsilon で
    出力は概ね [0,1]、SymLog は near-identity。良性の軽い再スケールのみ。
  - `use_robust_update=true` は稀 action の `1.0` を outlier 検出しログを出す（機能は良性、ログ noise
    のみ）。follow-up 候補として `obs_norm.exclude_keys`（per-key opt-out）を挙げるが **v1 非スコープ**。
- **stacking 既定**: `stack_keys` 空 = 全キー stack（直近 4 action 履歴 = R2D2 風）を**既定**とする。
  `stack_keys = vector` にすると prev_action は非 stack キー扱いで**最新フレームのみ**（a_{t-1} のみ）に
  なる（`replay_buffer_impl.cpp` の非 stack キー意味論）。両方を document、既定は「全 stack」。

## config surface と wiring

- **案B**: `env.obs_include_prev_action`（bool, default false）を `DefaultBatchEnvFactoryConfig`
  （`env.hpp`、既に `env` prefix を所有）に追加、`ANET_READ_CONFIG` で読み config dump へ自動 log。
  - **FLAG1（choke point は 1 箇所ではない）**: BatchEnv 生成は 3 箇所。
    1. `DefaultBatchEnvFactory::CreateBatchEnv`（`env.cpp`）→ train（`trainer.cpp` の
       `env_factory_->CreateBatchEnv(...)`）と app eval panel（`RunManager::CreateEvalRunner` →
       `RunnerFrame.cpp`）を**カバー**。
    2. `train.eval.[tag]` の eval env は `RunManager` 内で `VectorizedDiscreteBatchEnv` を**直接**
       構築し factory を**バイパス**（`CreateBatchEnv(seed,num_envs)` に config_prefix 引数が無いため）。
       ここを wrap しないと eval だけ `prev_action` を欠き、train 由来の spec で組んだ NN が
       `Input key 'prev_action' not found` で落ちる。
    3. テストの直接構築（`*_test.cpp`）は out of scope。
    - 対応: helper `MaybeWrapPrevActionObs(env, factory_config)`（flag OFF や nullptr は素通り）を
      **factory 内**と **`RunManager` の eval env 直接構築直後**の 2 箇所で呼ぶ。flag は base `env`
      scope なので train/eval 全 variant が一致（032 の variant 分岐リスクは起きない）。
- **案A**: agent config 側（例 `<Agent>.obs_include_prev_action`）。env 生成点には触れない。
- **032 との併用禁止**: env flag（032）と本 flag を同時 ON にすると vector が 12 次元化 **かつ**
  prev_action キーも生じ二重条件付けになる（エラーではないが無意味）。PRD に明記、併用しない。

## 非対象（Non-goals）

- 連続 action の注入（`is_discrete` ガードで弾く。将来は raw action ベクトルを同キーで注入する変種）。
- `Concat` NN ブロックの新設（複数キー bind の内部 cat で足りる）。
- 032（env ローカル flag）との同時使用サポート。
- `obs_norm.exclude_keys` の実装（follow-up、v1 非スコープ。Pattern B の `bind_cat_dim` は
  optional 作業項目として本 PRD に含める）。

## 受け入れ基準

1. flag OFF で現行と完全に同一挙動（キーを足さない、flag OFF は wrapper/注入が構築されず
   ビット一致）。
2. flag ON で観測 dict に `prev_action`（`[action_count]` float32）が現れ、`GetSpec`/agent spec に
   labels=value_labels・min0/max1 で反映、`ValidateObservation` を通過。
3. `episode_start=true` の全 state で `prev_action` が全ゼロ（不変条件）。`Step(a)` 後の
   next_state は該当 index のみ 1.0。
4. Pattern A の NN config で train/eval/replay の全経路が dummy と実データでランク一致し学習が回る。
5. 案B: train / app-eval / `train.eval.[tag]` の**全 eval 経路**で spec と実 obs が一致
   （FLAG1 回帰）。
6. `ToUnifiedObservation` が `vector` を先頭に保つ（FLAG3 回帰）。
7. run の config dump に flag が出力される。
8. MuZero など未 bind の agent で `prev_action` キーが無害に素通りする。

## テスト項目

**Unit（案B, 新規 `core/anet-core/src/env_prev_action_test.cpp`, scripted fake `SingleDiscreteEnv`,
`VectorizedDiscreteBatchEnv` を wrap）**
1. spec: `prev_action` が `{A}`/float32/labels=value_labels/min0/max1 で追加、他キーはバイト一致、
   `AssertSanity` 通過、`GetNumActions` 不変。
2. Reset: `prev_action` 全ゼロ `(N,A)`、拡張 spec で `ValidateObservation` true。
3. Step: 各 action 値で `next_state.obs[prev_action][i] == one_hot(a_i)`、非終端 env の
   continue_state も同様。
4. auto-reset マスク: env を done / truncated に強制 → `continue_state` はゼロ、`next_state` は
   `one_hot(a_i)` 維持。不変条件 `episode_start[i] ⇒ prev_action[i]==0` を全発行 state で検証。
5. pass-through 純度: 同 seed・固定 action 列で wrap 有無を比較し、env 由来 obs・reward・
   done/truncated/episode_start・n_transitions・n_episode_end が一致。
6. forwarding: `GetAuxDataList`/`GetScalar`/`GetTensor`/`GetTensorVector`/`GetBatchSpec`/
   `GetDevice`/`Shutdown` が inner に届く。
7. ガード: 連続 action で構築エラー、inner spec が既に `prev_action` を持つとエラー。
8. `ThreadPoolDiscreteEnv`（num_envs>1）を wrap して不変条件 2–4。
9. result 独立: step t の result を保持し step t+1 実行後に t の `prev_action` tensor が不変
   （fresh 確保回帰、pipeline runner hazard）。
10. `ToUnifiedObservation` 順序: `{vector, prev_action}` dict で vector ブロックが先頭。

**Integration**
11. runner smoke（fake discrete env + `DefaultDQNAgent`, stack4, Pattern A）: Reset 検証通過、
    複数 `DoStep`（stacker→obs_norm→dummy/実 forward→push）。
12. replay 往復: 境界（truncation 含む）跨ぎで push → `Sample` が `obs[prev_action]` `[B,4,A]`、
    境界隣接サンプルはゼロフレーム（padding + terminal dummy）。
13. `stack_keys = vector`: dummy `[A]`/act `(N,A)`/sample `(B,A)` で Pattern A forward 一貫。
14. eval 経路（FLAG1 回帰）: `train.eval.[x]` 1 件 + flag ON で eval env `GetSpec()` == train env
    `GetSpec()`、eval `DoStep` 1 回成功。
15. （Pattern B 採用時）`bind_cat_dim=-1`: dummy `(1,48)`・実 `(N,4,12)` が
    `ReS4F8 > Conv1D_Permute > TConv64` を通過、実 cat 行が `[v_t|a_t]`。

**案A 固有**
16. ActionContext 注入の per-env リセット（episode_start でゼロ）と選択後バッファ更新。
17. Learner の `state`（per-env バッファ, episode_start ゼロ）/ `next_state`（同 exp action）注入整合。
18. network_obs_spec と obs_norm spec の**両方**が `prev_action` を追従。

**手動（ユーザー）**
19. LunarLander flag ON + Pattern A: nn_viz に `prev_action` 入力ノード（4 ラベル）、config dump に
    flag、`E.obs_include_action`（032）を外して併用しないこと。

## 実装対象

- `core/anet-core/include/anet/rl.hpp` — `ObsKeys::kPrevAction`。
- 案B:
  - `core/anet-core/include/anet/env.hpp` / `src/env.cpp` — `PrevActionObsBatchEnv`（delegating
    result + 毎回 fresh 確保）、`MaybeWrapPrevActionObs`、`DefaultBatchEnvFactoryConfig` の flag。
  - `core/anet-core/src/trainer.cpp` — factory 内 wrap と `RunManager` の eval env 直接構築後 wrap
    の 2 点（FLAG1）。
- 案A:
  - `core/anet-core/src/default_dqn_agent.cpp` / `dqn_based_agent.cpp` — ActionContext 注入、
    `Learner::UpdateFromBatch` 注入、`network_obs_spec` 拡張、obs_norm spec 追従、agent config flag。
- 共通:
  - `core/anet-core/src/nn_impl.cpp` — （Pattern B 採用時）`net.branch.[x].bind_cat_dim` 解析と
    cat dim 反映。dummy build は変更不要。
  - `core/anet-core/src/rl.cpp` — `ToUnifiedObservation` の `kVector` hoist（FLAG3）。
  - `core/anet-core/src/env_prev_action_test.cpp`（新規, 案B）。
  - `apps/runner/config/LunarLander.txt` — コメント例（flag + Pattern A/C の branch 行、032 の
    `E.obs_include_action` を外す注記）。

## 検証

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test'
core\anet-core\bin\Debug\anet-core-test.exe "[prev_action],[env],[nn],[replay_buffer]"
core\anet-core\bin\Debug\anet-core-test.exe
git diff --check
```

runner smoke（ユーザー実施）は「テスト項目 19」。

## 関連

- [032_lunarlander_obs_include_action_10prd.md](done/032_lunarlander_obs_include_action_10prd.md) —
  env ローカル暫定（先行検証）。
- [057_native_stack_dim_handling_10prd.md](done/057_native_stack_dim_handling_10prd.md) —
  stack 軸ネイティブ化。本 PRD の Pattern B/Conv1D 経路と相互作用（Reshape 不要化）。
