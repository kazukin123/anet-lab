# Push 経路の割当削減によるオフ GPU 高速化（挙動不変・スレッド化なし）仕様書

## Context（背景・目的）

学習ループは env と learner を overlap 済みで **learner（GPU）が long pole**。`PrefetchingReplayBuffer`（[ADR 0005](../adr/0005-sample-prefetch-stale-per.md) / memo [013](013_sample_prefetch_10prd.md)・[014](014_async_loader_prefetch_10prd.md)）により **Sample + H2D は既に GPU 更新の裏へ隠蔽**され、prefetch スレッドは idle 余裕がある（Tracy 実測）。したがって RB 側で wall-clock に効くのは、**long-pole の LearnThread 上に裸の同期 CPU コストとして残る `DefaultReplayBuffer::Push`**（`core/anet-core/src/replay_buffer_impl.cpp:1153`、self-time ~5.22ms/step・全体 5.20%）だけである。

Push self-time のサンプリング内訳:

- **alloc churn** — `operator new` / `std::vector::{ctor}` / `std::map::_Try_emplace` / `std::_Tree::_Find_lower_bound` / `std::basic_string::{ctor}`。主因は ①per-env `UpdatePriorities({i},{p})` の vector 確保、②per-env `obs[b]`/`info[b]` の TensorDict 再構築。
- Tracy 計装 overhead は capture 時のみで本番に乗らない（`ANET_ENABLE_PROFILE=OFF` と実時間ほぼ不変を実測済み）→ 計測上の水増しであり最適化対象外。

目的: **挙動を変えず（学習等価。実際には結果 bit 一致）・スレッド化もせず**、①②の alloc churn を一掃し Push の CPU 時間を縮めて LearnThread の wall-clock を削る。

## 確定した設計判断（グリリング済み）

1. 等価性契約＝**学習等価**（同等設定の複数 run のブレ幅内）。ただし本最適化は **sampled index 列が bit 一致**（結果不変）に作るので、回帰は契約より厳しく抑える。
2. スコープ＝**①UpdatePriorities/metadata の Push 内バッチ化 + ②TensorDict スクラッチ再利用**。両方 `DefaultReplayBuffer::Push` に閉じる。
3. **スレッド化・非同期 Push は対象外**（ADR 0005 の順序制約。将来の最終手段）。**SumTree 内部・Sample/Extract・GPU storage 化（推論 H2D 再利用）も対象外**。
4. **config フラグ無し・常時有効**（挙動不変の内部最適化）。旧経路は残さず置換。A/B 実測は最適化前後の build（git）で比較。
5. 受け入れ＝**決定性 UT（golden 一致, UNIFORM/PER）+ 実 FPS をブレ幅基準で別途実測** + 既存テスト緑。

## 前提事実（調査済み・再調査不要）

- `DefaultReplayBuffer::Push`（`replay_buffer_impl.cpp:1153`）は `storage_mutex_` を exclusive 取得し num_envs ループ。per-env で: `storage_->Push` → `metadata_lock{MarkWritten/AdvanceWriteCursor/UpdatePriorities(0.0)}`（`:1172-1176`）→ queue push →（truncated 時）`PushTerminalDummy` + `metadata_lock{MarkDummy/AdvanceWriteCursor/UpdatePriorities(0.0)}`（`:1195-1200`）→ `ProcessQueue(b)`。
- `ProcessQueue`（`:1219`）は内部で `metadata_lock` を取り、完了系列に `MarkValid` + `UpdatePriorities(newly_valid, init_prios=-1.0)`（`:1260`）。
- `PrioritizedSampler::UpdatePriorities`（`:878`）のフラグ意味: `p<0`→初期優先度（`initial_priority_<0` なら `max_prio_`、else 固定値）、`p==0`→無効化、`p>0`→通常更新（**ここだけ `max_prio_` を更新**, `:903`）。→ **Push が渡すのは 0.0 と -1.0 のみ**なので **Push 内では `max_prio_` 不変 → 優先度更新は適用順序に非依存**（バッチ化が bit 一致になる根拠）。
- 各 `UpdatePriorities({i},{p})` は `std::vector` を2本 heap 確保（per-env で ~512 回/step）。
- `TensorDict` は `std::map<std::string,Tensor>` 実装。`Set` は `dict_[key]=tensor`（既存キーは**ノード再確保なし**, `tensor_util.hpp:176`）。`operator[](int64_t)`（`tensor_util.hpp:413`）は**毎回新 map を構築**。Push の `single_obs = batch.state.obs[b]`（`:1166`）が per-env に map node + string を量産。
- `storage_->Push` は `obs_storage_.At(key)[env][t].copy_(src.At(key))`（`:300-307`）で**即時 copy_**（source を保持しない）→ source dict は逐次再利用して安全。
- Push 区間は storage exclusive のため Sample/accessor は走らない。`Size()`（`:1287`）のみ `metadata_mutex_` 単独取得あり。

## 設計方針

### A. 優先度更新の Push 内バッチ化（①）

- `DefaultReplayBuffer` に再利用メンバ `pending_prio_indices_`(`std::vector<int64_t>`) / `pending_prio_values_`(`std::vector<float>`) を持つ（Push 冒頭で `clear()`、capacity 保持＝再確保なし）。
- per-env `UpdatePriorities(...)`（push-zero `:1176` / dummy-zero `:1200`）を **accumulator への append** に置換。
- `ProcessQueue` を、内部で `UpdatePriorities` を呼ばず **accumulator へ append** する形にリファクタ（初期優先度 -1.0 を `newly_valid` と共に積む）。`MarkValid` は従来どおり `metadata_lock` 内。
- Push 末尾で **`prio_controller_->UpdatePriorities(pending_prio_indices_, pending_prio_values_)` を1回**呼ぶ（`metadata_lock` 内）。append 順＝逐次呼び出し順を保持。`max_prio_` 不変ゆえ結果は bit 一致。
- `metadata_lock` 粒度は据え置き（index_manager は per-env）。**big-lock 化（Push 全体で1ロック）は `Size()` の待ちを延ばすため行わない**。

### B. TensorDict スクラッチ再利用（②）

- `single_obs` / `single_info` を**ループ外**で1度宣言し、各 env では `obs[b]` で新規生成せず **スクラッチへ `Set` 上書き**（既存キーはノード再確保なし）して `storage_->Push` に渡す。`scratch_obs.Set(key, batch.state.obs.At(key)[b])`（tensor view, コピーなし）。
- `storage_->Push` の copy_ は即時なので逐次上書きで安全。キー集合は全 env 共通ゆえ初回確立後はノード再確保ゼロ。
- `storage_->Push` のシグネチャは**不変**（interface 非変更）。

## 非対象（Out of Scope）

- SumTree 内部の高速化（O(log N) 直列・疎更新で割に合わない）。
- Push・優先度更新のスレッド化・非同期化（ADR 0005 の順序制約。将来の最終手段）。
- Sample / ExtractSamples / H2D 経路（prefetch で隠蔽済み）。
- GPU 常駐 storage・推論 H2D 再利用（B案）。Sample が隠れている現状は wall-clock ゼロ。
- config フラグ追加（挙動不変ゆえ不要）。

## 影響ファイル

| ファイル | 変更 |
|---|---|
| `core/anet-core/src/replay_buffer_impl.hpp` | `DefaultReplayBuffer` に pending 優先度 accumulator メンバ追加。`ProcessQueue` シグネチャ（accumulator 利用へ）変更 |
| `core/anet-core/src/replay_buffer_impl.cpp` | `Push` / `ProcessQueue` の優先度更新バッチ化（A）、TensorDict スクラッチ再利用（B） |
| `core/anet-core/src/replay_buffer_test.cpp` | 決定性 UT 追加（受け入れ基準 §1） |

## 受け入れ基準

1. **結果不変（主ガード・安価）**: CPU `DefaultReplayBuffer` で固定 seed・固定 Push 系列 → Sample の sampled index 列が **UNIFORM と PRIORITIZED の両方で、最適化前に採取した golden 列と bit 一致**。PER は `UpdatePriorities` を挟む系列で初期優先度・無効化経路を通す。既存 `MakeBuffer(...seed)`（`replay_buffer_test.cpp:191`）と index 収集パターン（`:813`）を流用。
2. **既存テスト緑**: `[replay_buffer]` 全 case（prefetch 決定性 `:803` 含む）。
3. **高速化（実測・Tracy 非依存）**: DropMerge で steps/sec を最適化前後 build で比較し、**同等設定の複数 run のブレ幅を超える改善**を確認（ブレ内は効果なしと判断）。Push self-time 低下は Tracy で補助確認可だが判定は実 FPS。

## 後続

`017_replay_push_alloc_20impl.md`（実装メモ）→ 実装（Codex 担当）→ §1 決定性 UT 緑 → 実 FPS 比較、の順で進む。
