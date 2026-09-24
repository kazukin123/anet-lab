# PRD 077: sampleable index 列の確保を持続バッファへ移す

## Context

`ValidIndexManager::GetValidIndices1D` は `Sample()` のたびに replay capacity と同じ長さの
配列を 2 本確保している。`replay_buffer_impl.cpp:269`。

```cpp
torch::Tensor ValidIndexManager::GetValidIndices1D(int stack_count, int unroll_steps, int n_step) const
{
    std::vector<int64_t> valid_list;
    valid_list.reserve(num_envs_ * capacity_per_env_);   // ← 1 本目
    for (int64_t env = 0; env < num_envs_; ++env) {
        ForEachSampleableIndex(env, stack_count, unroll_steps, n_step, [&](int64_t idx1d) {
            valid_list.push_back(idx1d);
        });
    }
    if (valid_list.empty()) return torch::empty({ 0 }, torch::kInt64);
    return torch::tensor(valid_list, torch::kInt64);                 // ← 2 本目 + コピー
}
```

`reserve` は `num_envs_ * capacity_per_env_` = replay capacity ちょうどを要求する。

| replay_capacity | 1 本 | vector + tensor (同時生存の一時ピーク) |
|---|---|---|
| 1,048,576 | 8.4 MB | 17 MB |
| 1,572,864 | 12.6 MB | 25 MB |
| 2,097,152 | 16.8 MB | 34 MB |
| 4,194,304 | 33.6 MB | 67 MB |

これが学習ステップごとに確保・解放される。RR4 の 50M Run なら数十万回で、
アロケータへ流れる総量は TB 級になる。

### 実害

2026-09-19 の Run `run_20260919-190647_rr4_btrsn12_envs128_cap1536k_50m` が
exp_step 27,092,992 / 経過 4.318h で `bad_alloc` により停止した。checkpoint は残らず、
後続の評価専用 Run も checkpoint 不在で fail-fast した。

`_set_new_handler` 診断が残したスタックを PDB で解決した結果、確保地点は本関数だった。

```
[ 3] operator new
[ 4] std::_Allocate_manually_vector_aligned
[ 5] std::vector<__int64>::reserve                   vector:1730
[ 6] anet::rl::ValidIndexManager::GetValidIndices1D  replay_buffer_impl.cpp:276
[ 7] anet::rl::DefaultReplayBuffer::Sample           replay_buffer_impl.cpp:1515
[ 8] anet::rl::PrefetchingReplayBuffer::State::Fetch replay_buffer_impl.cpp:2033
[11] anet::PinnedThreadPool::WorkerLoop              thread.cpp:203
```

要求サイズも一致する。

```
requested_bytes = 12,582,951
12,582,951 - 39 = 12,582,912 = 1,572,864 x 8
1,572,864 = num_envs_(128) x capacity_per_env_(12,288) = 当該 Run の replay_capacity
39 = MSVC 大確保パスの上乗せ (_Big_allocation_alignment 32 + sizeof(void*) 8 - 1)
```

同種の停止は他にも起きている。全 workspace の走査で `bad_alloc` により落ちた Run は 9 本、
うち Atari が 4 本 (`a5_phoenix_apex` 2026-08-27 / `z4_rr1_va_btrflat_breakout` 2026-09-03 /
`k2_rr4_hard500_munch_breakout` 2026-09-06 / 上記 2026-09-19)。
`requested_bytes` とスタックが残っているのは診断導入後の最新 1 本だけで、
残り 8 本は確保地点が不明のまま閉じている。

### 位置づけ

失敗した確保は 12.6 MB である。64bit プロセスではアドレス空間の断片化は事実上起きないので、
この `bad_alloc` は「システムの commit 上限に達した瞬間に、最初に大きく要求した側が引いた」
ものと読む。引き金は外部のメモリ圧 (同居プロセスなど) で構わない。修正後も同じ外圧なら、
次に大きく要求する側 (torch の batch 確保など) が同様に失敗する。
**本 PRD は `bad_alloc` の再発防止を約束しない** (NG4)。

本 PRD が確実に消すものは次の 3 つで、位置づけは hardening と性能改善である。

- 毎ステップの一時ピーク。vector と tensor が同時に生存する瞬間に上表の右列ぶんが積み増される
  (cap1.5M で 25 MB、cap4M で 67 MB)。
- 新規コミットページのゼロ埋めとページフォルト。毎ステップ同量が発生し、O(capacity) 走査
  (bool 配列のタイトループ) より重い可能性が高い。cap4M の throughput 低下 (NG1) の一因として有力。
- アロケータへ流れる総量 (TB 級) の churn。

## ゴール

- `GetValidIndices1D` の呼び出しあたりの確保回数を 0 にする。capacity サイズの確保は
  `ValidIndexManager` の構築時 1 回だけにする。
- capacity サイズのコピー (`torch::tensor(valid_list)`) をなくす。
- 既存の列挙結果と完全に同一の内容・順序を返す。

## 非ゴール

- **NG1: O(capacity) 走査の除去。** 本関数は毎回 capacity 個の slot を走査している。
  これも容量に比例するコストで、容量を上げたときの throughput 低下 (cap4M は cap2M 比
  clean throughput -16.3%) の一因と考えられるが、除去には sampleable 集合を
  「env ごとの物理連続 run + dummy 除外」として持ち直す設計変更が要る。段階 2 として分離する。
  **RB 整合性アッセイが検出した不整合の修復 PRD である
  [PRD 078](078_replay_frame_history_integrity_10prd.md) が `done/` へ移るまで着手しない。**
  index 列挙の内部を同時に触ると、アッセイの陽性がどちらに由来するか切り分けられなくなる。
- **NG2: dummy 機構の撤廃。** truncation 対策のダミー挿入 (`replay_buffer_impl.cpp:1356`) は
  PRD 018 の対象であり、本 PRD は現行の dummy 除外をそのまま維持する。
- **NG3: `GetSampleableCount` の変更。** 集計のみで Tensor を作らない現行の設計は正しい。
- **NG4: `bad_alloc` 一般の対策。** 他 8 本の Run の確保地点は不明のままで、本 PRD は
  それらを閉じない。診断 (`_set_new_handler`) が次の実例を捕らえるのを待つ。

## 確定事項

### D1: 持続バッファは `ValidIndexManager` が `torch::Tensor` で所有する

構築時に `torch::empty({ num_envs * capacity_per_env }, torch::kInt64)` を 1 回確保し、
以後は再確保しない。`std::vector<int64_t>` ではなく `torch::Tensor` にするのは、
返り値を同じストレージのビューにして 2 本目の確保とコピーを同時に消すため。
本書ではこのメンバを「持続バッファ」と呼ぶ (ファイル名の `scratch` は同じものを指す)。

追加の常駐メモリは capacity x 8 バイト。replay storage 本体 (uint8 単フレームで
capacity x 約 7KB) に対して 0.1% 程度で、実質無視できる。

`torch::Tensor` メンバの暗黙コピーはストレージを共有する (参照カウント) ため、
`ValidIndexManager` をコピーすると 2 つの instance が同じ持続バッファを共用する。
現用は `std::unique_ptr` 保持とテストでの直接構築だけでコピー箇所が無いため、
コピー禁止などの対処は加えない。

### D2: 返り値は `narrow` したビュー。内容の有効期間は次の呼び出しまで

```cpp
return valid_buf_.narrow(0, 0, count);
```

`torch::Tensor` のストレージは参照カウントで保たれるため、呼び出し側が保持しても
**dangling は起こらない**。起こりうるのは内容の上書きだけである。契約は
「返り値の内容は、同じ `ValidIndexManager` に対する次の `GetValidIndices1D` 呼び出しまで有効。
それ以降の内容は未規定」とし、宣言のコメントへ明記する。
「次回呼び出しで必ず上書きされる」ことは契約に含めない (ダブルバッファ化や段階 2 を縛らない)。

`torch::from_blob` は採らない。dangling を呼び出し側の規律に依存させる形になるため。

### D3: ロックを跨いで保持する 2 箇所は `clone()` する

現行の呼び出し 4 箇所のうち、2 箇所は `metadata_mutex_` を解放したあとも `valid_1d` を使う。

| 箇所 | ロック | 扱い |
|---|---|---|
| `DefaultReplayBuffer::Sample` `:1515` | 保持中に使い切る | ビューのまま |
| `DefaultReplayBuffer::SampleUniqueUniform` `:1547` | 保持中に使い切る | ビューのまま |
| `DefaultReplayBuffer::GetTensorVector` `:1888` | 解放後も使う | `clone()` |
| `DefaultReplayBuffer::DumpToLog` `:1957` | 解放後も使う | `clone()` |

`clone()` は **`metadata_mutex_` の lock スコープ内**で行う。解放後に clone すると、
別スレッドの `Sample` が同じ持続バッファを書き換えている最中に読むことになる。
現行コードでは `valid_1d` への代入が lock スコープ内にあるので、代入式へ `.clone()` を付けるだけで満たす。

後者 2 つは可視化のキャッシュミス時とデバッグダンプで、いずれもコールドパス。
`clone()` の確保は現行と同じコストなので退行しない。

### D4: `valid_buf_` は `mutable`、`GetValidIndices1D` は const を維持する

`DefaultReplayBuffer` は `sampled_once_` を `mutable` で持ち、const な `Sample()` から
`metadata_mutex_` 保持下で `MarkSampledOnce` により更新している (`replay_buffer_impl.hpp:493`)。
持続バッファも同じ「const な照会経路が owner の metadata 排他下で更新する scratch」であり、
この流儀に揃える。`GetSampleableCount` が const であることとも対称になる。

非 const にする案は採らない。const な呼び出し元 4 箇所が、`std::unique_ptr` の `operator->` が
pointee の const を伝播しないことに依存して非 const を呼ぶ形になり、将来 `propagate_const` 等で
塞がれると通らなくなる。

宣言コメントへ次の 2 点を明記する。

- owner (`DefaultReplayBuffer`) の `metadata_mutex_` 排他下でのみ呼ぶ。`ValidIndexManager` 自身は同期しない。
- 返り値の内容は次回呼び出しまで有効 (D2)。

### D5: 昇順契約を維持する

PER の `SampleIndices` は `std::binary_search(valid_ptr, valid_ptr + valid_count, idx)` で
所属判定しており、`ForEachSampleableIndex` のコメントどおり物理インデックス昇順が前提である
(`replay_buffer_impl.hpp:176`)。書き込み先がバッファへ変わっても順序は変えない。

### D6: 空集合の表現を変えない

現行は `valid_list.empty()` のとき `torch::empty({ 0 }, torch::kInt64)` を返す。
`narrow(0, 0, 0)` は `size(0) == 0` の Tensor になるので呼び出し側の分岐は変わらないが、
`numel() == 0` / `size(0) < batch_size` の両方の判定が現行どおり成立することを確認する。

## 実装契約

### `core/anet-core/src/replay_buffer_impl.hpp`

- `ValidIndexManager` へ `mutable torch::Tensor valid_buf_;` を追加する。コンストラクタで確保する。
- `GetValidIndices1D` の宣言は const のまま。コメントを D2 の有効期間と D4 の排他前提へ書き換える。

### `core/anet-core/src/replay_buffer_impl.cpp`

- `ValidIndexManager::ValidIndexManager` で `valid_buf_` を確保する。
- `GetValidIndices1D` を次の形にする。

```cpp
torch::Tensor ValidIndexManager::GetValidIndices1D(int stack_count, int unroll_steps, int n_step) const
{
    ANET_PROFILE_FUNC();

    int64_t* out = valid_buf_.data_ptr<int64_t>();  // mutable なので const 関数内で書ける
    int64_t count = 0;
    for (int64_t env = 0; env < num_envs_; ++env) {
        ForEachSampleableIndex(env, stack_count, unroll_steps, n_step, [&](int64_t idx1d) {
            out[count++] = idx1d;
        });
    }
    return valid_buf_.narrow(0, 0, count);
}
```

- `GetTensorVector` (`:1888`) と `DumpToLog` (`:1957`) の代入へ `.clone()` を足す (lock スコープ内、D3)。

`ANET_PROFILE_SCOPE(valid_indices)` (`:1514`, `:1884`) は既存のまま残す。段階 2 の判断材料になる。

## テスト

既存の `core/anet-core/src/replay_buffer_test.cpp` へ追加する。

- **T1 再確保しない**: 同一状態で `GetValidIndices1D` を 2 回呼び、返り値の `data_ptr()` が
  2 回とも同じ値であること。バッファが作り直されていないことの直接確認。
- **T4 有効期間契約**: (a) 1 回目の返り値を `clone()` して保持し、状態を進めて 2 回目を呼んでも
  clone の内容が不変であること。(b) 同一状態で 2 回呼んだ返り値の内容が一致すること
  (次回呼び出しまで有効)。古いビューの内容が書き換わることは検査しない。
  次回呼び出し以降の内容は未規定であり、テストで仕様に昇格させない。
- **T5 空集合**: sampleable が 0 本の状態で `size(0) == 0` かつ `numel() == 0` になること。
- **T6 回帰**: `replay_buffer_test.cpp` の既存の wrap / stack / n-step / PER 系が全緑であること。
  列挙内容 (値と順序、ring 跨ぎの物理昇順、dummy 除外、history margin) の oracle は
  次の値固定テストが既に担っており、本 PRD で列挙内容のテストは新設しない。
  - 「ValidIndexManager sampleability consumers agree before and after wrap」
  - 「ValidIndexManager keeps dummy filtering outside the shared logical range」
  - 「ValidIndexManager applies frame stack history margin per env lane」
  - 「ValidIndexManager filters dummy slots after applying frame stack history margin」
    (期待列 `{ 0, 5 }` が ring 跨ぎの物理昇順を固定している)
  - 「ReplayBuffer PER samples only safe wrapped frame-stack indices」

## 受入基準

1. T1・T4・T5・T6 が緑で、`anet-core-test` 全体が既知の失敗以外で緑であること。
2. `GetValidIndices1D` の呼び出しあたりの capacity サイズの確保が 0 であること (T1)。
3. 等価性: 同 seed の短尺 Run (10M / RR4 / cap2M) で metrics checksum が実装前と一致すること。
   列挙内容を変えない改修なので、ここは厳密一致を要求する。
   checkpoint の raw SHA は再実行で一致しないため受入ゲートにしない。
4. clean throughput が退行しないこと。cap2M / RR4 で計測する。
   計測はラウンドロビン配置 (実装前・実装後を交互) で行う。実験機は 1 時間で最大 8% の
   throughput ドリフトがあり、ブロック配置は符号を誤らせる。
   一時ピークとページフォルトの除去ぶんは改善するはずだが、O(capacity) 走査は残るので
   **改善幅は受入条件にしない**。測定値は実装計画側に記録し、段階 2 の判断材料にする。

## 影響・移行

- 設定キーの追加・変更なし。メトリクスの追加・変更なし。シリアライズ形式の変更なし。
- 公開ヘッダ (`core/anet-core/include/anet/replay_buffer.hpp`) は変更しない。
  `ValidIndexManager` は `src/` 内の実装クラスである。
- ADR は作らない。持続バッファは実装内の最適化で、領域の概念を変えないため。
  D2 の有効期間契約は宣言のコメントで持つ。
- `CONTEXT.md` への用語追加もしない。本 PRD が扱う列は「sampleable range」の _Avoid_ が
  述べるとおり実装上の列挙結果であって概念ではない。

## 段階 2 への申し送り

sampleable 集合は、env ごとに高々 2 本の物理連続区間から dummy を除いたものである
(`ForEachSampleableIndex`, `replay_buffer_impl.hpp:153`)。消費側が必要としているのは
次の 3 つだけで、いずれも配列の実体を要求しない。

| 用途 | 現行 | 区間表現でのコスト |
|---|---|---|
| `valid_count` | `size(0)` | O(num_envs) |
| k 番目の取得 (一様抽選) | `index_select` | O(log num_envs + log D) |
| 所属判定 (PER 棄却) | `std::binary_search` O(log V) | **O(1)〜O(log D)** |

D は ready 範囲内の dummy 本数で、truncation 1 回につき 1 つしか増えないため実測上は
lane あたり数個である。区間表現にすれば確保も走査も capacity 非依存になり、
PER の所属判定はむしろ速くなる。

ただし dummy を正確に扱うために、env ごとの dummy 物理インデックスを
ソート済みで保持し `MarkDummy` / `MarkWritten` で増減させる必要がある。
設計と実装量は本 PRD の数倍になる。
**[PRD 078](078_replay_frame_history_integrity_10prd.md) が `done/` へ移ってから別 PRD で起こす。**

実装順は本 PRD (077) を PRD 078 より先にすることを推奨する。077 は小さく等価性 Run が安価で、
078 の 384 条件 matrix の完走待ちで遅らせる理由がない。078 は `ValidIndexManager` を契約維持と
しており、コード領域は重ならない。受入 3 の等価性 Run は、実装前・実装後を同じ base で揃える。
