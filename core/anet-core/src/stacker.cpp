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
    anet::ProfileRange r("DictFrameStacker::Stack");

    if (data.empty()) return data;

    // 入力チェック (resetsの環境数だけ確認)
    if (resets.size(0) != num_envs_) {
        ANET_SYSTEM_ERROR("DictFrameStacker dimension mismatch. Expected num_envs="
            << num_envs_ << ", Got resets=" << resets.size(0));
    }

    anet::TensorDict result;

    for (const auto& kv : data) {
        const auto& key = kv.first;
        const auto& tensor = kv.second;

        // スタック対象判定 (nulloptなら全対象)
		bool should_stack = !stack_keys_.has_value() || // Stack対象キーが指定されていないならスタック必要
			std::find(stack_keys_->begin(), stack_keys_->end(), key) != stack_keys_->end(); // キーがStack対象に含まれている

        // スタック対象外は Pass-through
        if (!should_stack) {
            result.Set(key, tensor.to(device_, /*non_blocking=*/true).clone());
            continue;
        }

        // --- ここからスタック処理 ---

        // ターゲットデバイス上でStack処理
        auto data_dev = tensor.to(device_, /*non_blocking=*/true);

        // バッファが未定義（初回）の場合
        if (buffers_.find(key) == buffers_.end()) {
            auto shape = data_dev.sizes().vec();
            shape.insert(shape.begin() + 1, stack_count_);
            buffers_[key] = torch::empty(shape, data_dev.options());

            for (int k = 0; k < stack_count_; ++k) {
                buffers_[key].select(1, k).copy_(data_dev);
            }
        } else {
            auto& buffer = buffers_[key];

            /// @todo [パフォーマンス改善] RingBuffer構造に変更

            // 履歴をインプレースでスライド (デバイス上の高速なメモリブロック転送)
            if (stack_count_ > 1) {
                // メモリ領域の重複による例外を防ぐため、コピー元を clone() で独立させる
                buffer.slice(1, 0, stack_count_ - 1)
                    .copy_(buffer.slice(1, 1, stack_count_).clone());
            }

            // 最新フレームを末尾に挿入
            buffer.select(1, -1).copy_(data_dev);

            // resets_dev の中で true (リセット発生) になっているバッチインデックスを取得
            auto resets_dev = resets.to(device_, /*non_blocking=*/true).to(torch::kBool);
            auto reset_indices = resets_dev.nonzero().squeeze(1);

            if (reset_indices.numel() > 0) {
                // リセットされた環境の最新フレームだけを抽出: [NumResets, C, H, W]
                auto reset_data = data_dev.index_select(0, reset_indices);

                for (int k = 0; k < stack_count_ - 1; ++k) {
                    // index_copy_ を使って、該当インデックスの過去フレームをすべて最新フレームで塗りつぶす (データリーク防止)
                    buffer.select(1, k).index_copy_(0, reset_indices, reset_data);
                }
            }
        }

        // バッファ自体が既に device_ 上にあるので clone() だけで済む
        result.Set(key, buffers_[key].clone());   // 外部でのテンソル破壊を防ぐため clone
    }

    return result;
}

void DictFrameStacker::Reset()
{
    // 全キーのバッファを破棄
    buffers_.clear();
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
