# anet-lab ハンドオフ: ReplayBuffer の t=0 FrameStack で未初期化 grid を読む問題

## 推奨スキル

- `diagnose`: 再現 -> 仮説 -> 計測 -> 修正 -> 回帰確認の流れを継続する。
- `tdd`: 修正に進む場合、ReplayBuffer の境界条件を小さい回帰テストで固定する。

## 現在の目的

次の設定で起動した run のクラッシュ原因を調査。

```text
DefaultDQNAgent.auto_load_file = ../../runs/run_20260512-171248_gs_5846_cy11/agent_close.anet
```

クラッシュした run:

```text
apps\runner\runs\run_20260531-004204_gs_5846_cy12_apx1
```

最初は `Learner::UpdatePerPriorities()` の `samples.indices.cpu()` で落ちているように見えたが、これは CUDA エラーが CPU 同期境界で表面化しただけだった。環境変数CUDA_LAUNCH_BLOCKING=1を設定した上でデバッガで確認した真のスタックは以下。

```text
at::one_hot
anet::nn::NetworkBoundaryPreprocessor::Format
anet::nn::NetworkBody::Forward
anet::nn::Network::Forward
anet::rl::dqn::NetworkModel::Forward
anet::rl::dqn::QRLearner::UpdateFromSamples
```

つまり直接原因は、離散 `grid` 観測に不正値が入り、`torch::one_hot` が失敗していること。

## 確認済み事実

`DropMergeEnv::GetSpec()` では `grid` は `Int8` の離散値で、`num_classes = kFruitTypeCount + 2 = 13`。有効値は `0..12`。今回の `direct_noop` 設定では Dropper grid 描画が無効なので、実質的には通常 `0..11` のはず。

`replay_buffer_->Sample()` 後、`.To(device_)` 前に観測をログした結果:

```text
g.shape=[128, 4, 1, 46, 58] dtype=Char min=-128 max=127 bad_count=168185
ng.shape=[128, 4, 1, 46, 58] dtype=Char min=-128 max=127 bad_count=112112
samples.n_steps all 1
samples.terminals all 0
sample[0] idx=73728 env=36 t=0 n=1 terminal=0 g.min=-128 g.max=127 ng.min=-128 ng.max=127
sample[1] idx=260096 env=127 t=0 ... g.min=0 g.max=0 ng.min=0 ng.max=5
```

重要な点は、sample index が `2048` の倍数に偏っていること。今回の `capacity_per_env` は `524288 / 256 = 2048` なので、これらは `time_idx=0` を指す。

続けて `DefaultSampleExtractor::ExtractSamples` の `b=0, env=36, time_idx=0` 付近に probe を入れた結果:

```text
DBG_RB b=0 idx=73728 env=36 t=0 actual_n=1
DBG_RB tail terminals 0 0 0
DBG_RB grid[2045] min=-128 max=127
DBG_RB grid[2046] min=-128 max=127
DBG_RB grid[2047] min=-128 max=127
DBG_RB grid[0] min=0 max=0
DBG_RB grid[1] min=0 max=2
```

これで、実際に保存された `grid[0]` と `grid[1]` は正常だが、FrameStack 抽出が未書き込みのリング末尾 `2045..2047` を読んでいることが確認できた。

## 根本原因

`DefaultSampleExtractor::ExtractSamples` は `time_idx=0`、`stack_count=4` で以下のように過去フレーム範囲を計算する。

```cpp
int64_t obs_start = time_idx - stack_count + 1; // -3
for (int64_t k = time_idx - 1; k >= obs_start; --k) {
    int64_t phys_k = (k % cap + cap) % cap;
    if (terminals_tensor[env_idx][phys_k].item<bool>()) {
        obs_valid_start = k + 1;
        break;
    }
}
```

初期 `t=0` の場合、論理インデックス `-3..-1` が物理インデックス `2045..2047` に折り返される。この領域はまだ一度も書かれていない。

ところが `ReplayExperienceStorage` は `target_returns_`、`terminals_`、`actual_n_steps_` を `torch::empty` で確保している。したがって `terminals_[env][2045..2047]` は未初期化で、今回のログでは偶然 `false` になっていた。

`terminals_` が false だと extractor は「エピソード境界がない」と判断し、本来 `grid[0]` を複製 padding すべきところで、リング末尾を含む `[2045, 2046, 2047, 0]` を stack する。その結果、未初期化 `grid` の `-128..127` が `one_hot` に渡って落ちる。

`g.bad_count` が `ng.bad_count` より多いこともこの説明と一致する。`n=1` では、`g` は最大 3 枚の未初期化 tail frame を含み、`ng` は最大 2 枚になりやすい。

## 関連ファイル

- `core\anet-core\src\replay_buffer_impl.cpp`
  - `ReplayExperienceStorage::ReplayExperienceStorage`: metadata tensor を `torch::empty` で確保している箇所。
  - `DefaultSampleExtractor::ExtractSamples`: `obs_start` / `next_obs_start` から FrameStack を抽出する箇所。
- `core\anet-core\src\nn_impl.cpp`
  - `NetworkBoundaryPreprocessor::Format`: 離散入力を `torch::one_hot` する箇所。
- `core\envs\dropmerge1\src\DropMergeEnv.cpp`
  - `DropMergeEnv::GetSpec`: `grid` の `num_classes` と値範囲。

## 推奨修正方針

まずは `ReplayExperienceStorage` コンストラクタで、未書き込み metadata を安全側へ初期化するのが局所的でよい。

```cpp
target_returns_ = torch::empty({ num_envs_, capacity_per_env_ }, options.dtype(torch::kFloat32));
terminals_ = torch::empty({ num_envs_, capacity_per_env_ }, options.dtype(torch::kBool));
actual_n_steps_ = torch::empty({ num_envs_, capacity_per_env_ }, options.dtype(torch::kInt64));

target_returns_.fill_(0.0f);
terminals_.fill_(true);
actual_n_steps_.fill_(0);
```

今回の直接原因に効く最小修正は `terminals_.fill_(true)`。未書き込みリングスロットは FrameStack の境界判定ではエピソード境界として扱うべきだから。

`target_returns_` と `actual_n_steps_` も未初期化読みの余地を残さないため、同時に初期化した方がよい。

別案として extractor 側で「初期書き込み前の負論理 index は必ず padding」と明示処理する方法もある。ただし現在の extractor は各 env の絶対 write cursor を持っておらず、リング wrap 済みの `t=0` と初期 `t=0` を区別しにくい。現時点では metadata 初期化の方が小さい修正。

## 検証案

1. 修正後、同じ probe を再実行する。期待値:

```text
DBG_RB tail terminals 1 1 1
```

2. 元の `auto_load_file` 付き run を再実行し、`replay_buffer_->Sample()` 後の `g/ng` に `[0, 12]` 範囲外の値が出ないことを確認する。

3. 実装する場合は ReplayBuffer の回帰テストを追加する。

- `stack_count=4`, `n_step=1`, 小さめの `capacity_per_env` を使う。
- 1 env に正常な `grid` frame を 2 つ push する。
- `time_idx=0` を sample/extract する。
- `obs["grid"]` がリング末尾ではなく frame 0 の複製で padding されることを確認する。
- 全 `grid` 値が有効範囲内であることを確認する。

4. 実装後は少なくとも `anet-core-test` を実行する。AGENTS.md の指示通り、素の PowerShell ではなく `VsDevCmd.bat` を `cmd /s /c` 内で call してからビルドする。

## 次の担当者への注意

ユーザーは `dqn_based_agent.cpp` と `replay_buffer_impl.cpp` に一時デバッグログを入れている。これはユーザー変更として扱い、明示依頼なしに戻さないこと。

`auto_load_file` が絡んで見えるのは、読み込んだ policy により初期の行動分布や sample 分布が変わり、ReplayBuffer の境界バグが露出した可能性が高い。静的確認では action 数、grid shape、QR/dueling/head size は大きく矛盾していなかったため、checkpoint 互換性そのものは第一原因ではなさそう。

