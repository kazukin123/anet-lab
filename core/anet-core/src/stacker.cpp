// stacker.cpp

#include "anet/stacker.hpp"
#include "anet/log.hpp" 


using namespace anet::rl;


// =============================================================
// DictFrameStacker
// =============================================================

DictFrameStacker::DictFrameStacker(int stack_count, int num_envs, torch::Device device, std::optional<std::vector<std::string>> stack_keys)
    : stack_count_(stack_count), num_envs_(num_envs), stack_keys_(stack_keys), device_(device)
{
    ANET_ASSERT(stack_count_ > 0);
    ANET_ASSERT(num_envs_ > 0);
}

anet::TensorDict DictFrameStacker::Stack(const anet::TensorDict& data, const torch::Tensor& resets)
{
    ANET_PROFILE_FUNC();

    if (data.empty()) return data;

    // 入力チェック (resetsの環境数だけ確認)
    if (resets.size(0) != num_envs_) {
        ANET_SYSTEM_ERROR("DictFrameStacker dimension mismatch. Expected num_envs="
            << num_envs_ << ", Got resets=" << resets.size(0));
    }

    // ----------------------------------------------------------------
    // リセット情報はキー間で共通なのでループ外で1回だけ処理する。
    // resets は通常CPUテンソルなので any() の判定にGPU同期は発生しない
    // (CUDAで来た場合も同期はこの1回で、キーごとに nonzero() で同期するより軽い)
    // ----------------------------------------------------------------
    const bool has_reset = resets.any().item<bool>();
    torch::Tensor reset_indices;   // device_上 (リセット発生時のみ作る)
    if (has_reset && stack_count_ > 1) {
        reset_indices = resets.to(torch::kBool).nonzero().squeeze(1).to(device_);
    }

    // リングバッファの書き込み位置 (全キー共通)
    // head_ は最古フレームのスロット。ここを最新フレームで上書きすると (head_+1)%S が最古になる
    const int write_pos = head_;
    const int oldest = (head_ + 1) % stack_count_;

    anet::TensorDict result;

    for (const auto& kv : data) {
        const auto& key = kv.first;
        const auto& tensor = kv.second;

        // スタック対象判定 (nulloptなら全対象)
		bool should_stack = !stack_keys_.has_value() || // Stack対象キーが指定されていないならスタック必要
			std::find(stack_keys_->begin(), stack_keys_->end(), key) != stack_keys_->end(); // キーがStack対象に含まれている

        // スタック対象外は Pass-through
        if (!should_stack) {
            // copy=true で同一デバイスでも必ず新規テンソルになるため、to().clone() の二重コピーを避けられる
            result.Set(key, tensor.to(device_, /*non_blocking=*/true, /*copy=*/true));
            continue;
        }

        // --- ここからスタック処理 ---

        // ターゲットデバイス上でStack処理
        auto data_dev = tensor.to(device_, /*non_blocking=*/true);

        auto buffer_itr = buffers_.find(key);
        if (buffer_itr == buffers_.end()) {
            // バッファが未定義（初回）の場合: 全スロットを初期フレームで充填 (ブロードキャストコピー1回)
            // 全スロット同値なので head_ がどこを指していても時系列順は崩れない
            auto shape = data_dev.sizes().vec();
            shape.insert(shape.begin() + 1, stack_count_);
            auto buffer = torch::empty(shape, data_dev.options());
            buffer.copy_(data_dev.unsqueeze(1).expand(shape));
            buffer_itr = buffers_.emplace(key, std::move(buffer)).first;
        } else {
            auto& buffer = buffer_itr->second;

            // リングバッファ: 最古スロットを最新フレームで上書きするだけ (毎ステップの履歴シフトを排除)
            buffer.select(1, write_pos).copy_(data_dev);

            // リセットされた環境は全スロットを最新フレームで塗りつぶす (過去フレームのデータリーク防止)
            if (has_reset && stack_count_ > 1) {
                // dim0 の index_copy_ 1回で全スロットをまとめて書き換える
                auto src = data_dev.index_select(0, reset_indices).unsqueeze(1);   // (R, 1, ...)
                auto expand_shape = src.sizes().vec();
                expand_shape[1] = stack_count_;
                buffer.index_copy_(0, reset_indices, src.expand(expand_shape).contiguous());
            }
        }

        // 出力は時系列順 [最古..最新] に並べ替えた新規テンソル
        // (cat が新規テンソルを返すため、外部でのテンソル破壊を防ぐ clone を兼ねる)
        const auto& buffer = buffer_itr->second;
        if (oldest == 0) {
            result.Set(key, buffer.clone());
        } else {
            result.Set(key, torch::cat({ buffer.slice(1, oldest, stack_count_),
                                         buffer.slice(1, 0, oldest) }, 1));
        }
    }

    // 書き込み位置を進める (キー間で共通なのでループ後に1回)
    head_ = oldest;

    return result;
}

void DictFrameStacker::Reset()
{
    // 全キーのバッファを破棄
    buffers_.clear();
    head_ = 0;
}


// =============================================================
// StackerActionContext
// =============================================================

StackerActionContext::StackerActionContext(RunMode run_mode, std::shared_ptr<FrameStacker> stacker, std::optional<seed_t> seed)
    : ActionContext(run_mode, seed), stacker_(std::move(stacker))
{
    /// @todo 推論と学習における FrameStacker の重複解消と ActionContext 不要論
    /// @todo ActionContext依存のStacking再考

}

anet::TensorDict StackerActionContext::PushObservation(const BatchState& state)
{
    if (stacker_) {
        // BatchState から obs と episode_start を抽出して委譲
        return stacker_->Stack(state.obs, state.episode_start);
    }
    return state.obs;
}

void StackerActionContext::Reset()
{
    if (stacker_) stacker_->Reset();
}
