# Phase 2.5: ReplayBuffer 可視化アクセサの高速化（ExtractSamples 撤去）

## Context（なぜやるか）

Phase 2 で `DefaultReplayBuffer::GetTensorVector` が復旧したが、実 run（DropMerge, capacity 524,288, stack_count=4）で HeatMap tick のたびに TrainThread が**目に見えて固まる**。Tracy 計測の結果:

- `DefaultSampleExtractor::ExtractSamples` が **zone time の 91.92%（合計 3:27.8、平均 299ms × 695回）**
- 呼び出し経路: `HeatMapVectorObserver::OnTrain` → `AgentTensorVectorProbe::GetVector` → `DefaultReplayBuffer::GetTensorVector` → `ExtractSamples`（TrainThread 上で同期実行）

根本原因: `ExtractSamples`（[replay_buffer_impl.cpp:823-893](core/anet-core/src/replay_buffer_impl.cpp:823)）は**学習 minibatch（B≈256）向けの per-row C++ ループ**で、1行ごとに `.item()` 同期・`RingSlice`・`torch::stack`・terminal 境界スキャンを行う。可視化はこれを **B=全 sampleable 行（≈524k）**で、しかも per-prio の x/y probe が**同一キーを同 tick に2回**呼ぶ（299ms×2≈600ms/tick）。既存コード自体に「Advanced Indexing でベクトル化すべき」と `@todo [Performance]`（:844-846）が明記されている。V1 は raw storage（`next_states_` バッファ）をほぼゼロコピーで返していたため、この痛みが無かった。

付随バグ: stack_count>1 のとき `samples.obs` は `[N, stack, ...]` で、`ToUnifiedRows` の flatten(1) 後に probe index 0/1 が**スタック最古フレーム**の値を指す（V1 の単一フレーム意味論と相違）。

## 修正方針

可視化（storage-level view）は**スタック復元も境界スキャンも不要**（単一フレームの値で良い）。`ExtractSamples` を使わず、ベクトル化した直接 gather に置き換える。期待効果: **299ms → 数ms（50〜100倍）**、tick合計 ~600ms → <10ms。

## 変更内容（すべて `core/anet-core/src/replay_buffer_impl.cpp` 内・公開ヘッダ不変）

### 1. ベクトル化直接 gather（中核）

`DefaultReplayBuffer::GetTensorVector` の storage 系キーを、`ExtractSamples` 経由から純テンソル演算へ:

- 物理レイアウトは `[num_envs, cap, ...]`、`valid_1d = env*cap + time`（IndexManager のエンコードと一致を確認済み、[replay_buffer_impl.hpp:122](core/anet-core/src/replay_buffer_impl.hpp:122)）。各 storage tensor を `[num_envs*cap, ...]` に view → `index_select(0, valid_1d)`。
- `STATE_OBS`: obs_storage_ の各キーを valid_1d で gather（**単一フレーム・スタックなし**）→ `ToUnifiedObservation` → `[N, D]`。stack>1 時の最古フレーム問題も同時解消。
- `NEXT_STATE_OBS`: `actual_n_steps_` を gather → `next_phys = env*cap + (time + actual_n) % cap` をテンソル演算で構築 → obs を gather。意味論は既存テストが固定（t=0, n=3 → t=3 の obs = time + actual_n）。terminal 境界スキャン不要（単一フレーム、validity は `GetValidIndices1D(n_step)` が保証、dummy 行は IndexManager が除外済み）。
- `ACTION` / `REWARD`(=target_returns) / `NEXT_STATE_TERMINAL` / `N_STEP`: flat view + index_select → `[N,1]`。
- 新 gather 経路に `anet::ProfileRange` を付与（AGENTS.md 規約。Tracy で before/after 比較可能に）。

### 2. 同 tick 重複排除（バージョン付き memo）

- storage push カウンタ（version）を導入し、`(key, version) → 結果` の単純 memo を `DefaultReplayBuffer` に持つ（mutable・1〜数エントリで十分）。
- x/y probe の同一キー2回呼び＋ `GetValidIndices1D` の3回再計算（x/y/value）を1回に。PER 系（`GatherPriorityRows` の 524k ループ）も恩恵。

### 3. subkey 対応（Phase 1 文法との対称性）

- `replaybuffer.storage.state.vector` / `replaybuffer.storage.next_state.vector` を受け、該当 obs キーのみ gather（unified を作らない）。
- DropMerge（vector+grid）では grid の gather/concat（~数百MB/ tick のメモリトラフィック）を丸ごと回避できる。config 側の切替は任意（無接尾の既存 config は従来どおり unified で動く）。
- 未知 subkey は `ANET_SYSTEM_ERROR`（Phase 1 と同じ流儀）。

### 4. テスト（`core/anet-core/src/replay_buffer_test.cpp`）

- 既存の Phase 2 可視化テスト3本は**そのまま green を維持**（意味論ロックとして機能）。
- 追加: stack_count=4 のバッファで `STATE_OBS`/`NEXT_STATE_OBS` が**単一フレーム値**（`[N, D]`、スタック次元なし）を返すこと。
- 追加: `.vector` subkey が vector キーの生値を返すこと。

## やらないこと（延期）

- 列 pushdown（probe の index を RB まで通して1列だけ gather）— forwarding 変更が広がるため今回見送り。
- Observer 描画の別スレッド化（アーキテクチャ変更）。
- interval / max_points の config チューニング（ユーザ側でいつでも可能。本修正後は不要のはず）。

## 検証

1. ビルド（AGENTS.md の VsDevCmd 形式）＋ `anet-core-test.exe "[replay_buffer]"` 全green（既存3本＋新規）。
2. DropMerge を短時間 run し Tracy で確認: 可視化経路から `ExtractSamples` が消え、`HeatMapVectorObserver::OnTrain.get_vector` が ~10ms 以下、TrainThread のスタッターが解消。
3. per-prio heatmap PNG が修正前と同等の見た目で出力されること（stack>1 環境では最古→現在フレームに変わるため厳密一致はしない。意図どおりの差）。

## 備考

- 実装を Codex さんへ渡す場合は、本プランを `docs/memo/010_image_metrics_v2_23impl_ph2x.md` 等の実装指示書として書き出して引き継ぐ（PRD 本体 `010_image_metrics_v2_10prd.md` の Phase 2 への追補に当たる）。
