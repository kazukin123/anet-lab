
#include "anet/vec_env.hpp"

using namespace anet::rl;

VectorizedDiscreteBatchEnv::VectorizedDiscreteBatchEnv(
    std::shared_ptr<SingleDiscreteEnvFactory> factory, int batch_size, std::optional<seed_t> seed)
    : batch_spec_({ batch_size, 1 }), batch_size_(batch_size)
{
    ANET_ASSERT(batch_size_ > 0);

    // ベースのシードを準備
    anet::SeedMaker seed_maker(seed);

    //  batch_size 個のENVを生成
    envs_.reserve(batch_size_);
    for (int i = 0; i < batch_size; ++i) {
        anet::seed_t seed = seed_maker.MakeIndexedSeed(i);
        auto env = factory->Create(seed);
        envs_.push_back(std::move(env));
    }

    // EnvSpec取得
    spec_ = std::make_unique<EnvSpec>(envs_[0]->GetSpec());

    // EnvSpec からobsのshapeを取得
    auto shape = spec_->state_spec.shape;  // 例: {4}
    obs_dims_.push_back(batch_size_);
    obs_dims_.insert(obs_dims_.end(), shape.begin(), shape.end());

    // 初期化
    float_opts_ = torch::TensorOptions().dtype(torch::kFloat32);// .device(state_.obs.device());
    bool_opts_ = torch::TensorOptions().dtype(torch::kBool);// .device(state_.done.device());
    state_ = createEmptyState();
}

EnvSpec VectorizedDiscreteBatchEnv::GetSpec() const
{
    return *spec_;
}

BatchEnvSpec VectorizedDiscreteBatchEnv::GetBatchSpec() const
{
    return batch_spec_;
}

anet::rl::BatchState VectorizedDiscreteBatchEnv::createEmptyState() const
{
    anet::rl::BatchState batch_state {
        torch::empty(obs_dims_, float_opts_),
        torch::empty({ batch_size_ }, bool_opts_),
        torch::empty({ batch_size_ }, bool_opts_),
        torch::empty({ batch_size_ }, bool_opts_)
    };
    return batch_state;
}

anet::rl::BatchStepResult VectorizedDiscreteBatchEnv::createEmptyStepResult() const
{
    BatchStepResult result {
        torch::empty({ batch_size_ }, float_opts_),       // reward        (N) kFloat32
        {    // next_state
            torch::empty(obs_dims_, float_opts_),         // obs           (N, state_dim..) kFloat32
            torch::empty({ batch_size_ }, bool_opts_),    // done          (N) kBool
            torch::empty({ batch_size_ }, bool_opts_),    // truncated     (N) kBool
            torch::empty({ batch_size_ }, bool_opts_)     // episode_start (N) kBool
        },
        {    // continue_state
            torch::empty(obs_dims_, float_opts_),         // obs           (N, state_dim..) kFloat32
            torch::empty({ batch_size_ }, bool_opts_),    // done          (N) kBool
            torch::empty({ batch_size_ }, bool_opts_),    // truncated     (N) kBool
            torch::empty({ batch_size_ }, bool_opts_)     // episode_start (N) kBool
        },
    };
    return result;
};

BatchState VectorizedDiscreteBatchEnv::Reset(RunMode mode)
{
    // 全環境を初期化し、state_ バッファに書き込む（バッファは constructor 確保済み）
    auto state = createEmptyState();
    for (int i = 0; i < batch_size_; ++i) {
        auto reset_state = envs_[i]->Reset(mode);
        state.obs[i].copy_(reset_state.obs);
        state.done[i] = reset_state.done;
        state.truncated[i] = reset_state.truncated;
        state.episode_start[i] = reset_state.episode_start;
    }

    state_ = state.Clone();

    return state;
}

BatchStepResult VectorizedDiscreteBatchEnv::Step(const torch::Tensor& batch_action, RunMode mode)
{
    const int64_t N = batch_spec_.batch_size;

    ANET_CHECK_DTYPE_MSG(batch_action, torch::kInt64,
        "VectorizedDiscreteBatchEnv supports discrete action only. actions should be kInt64.");
    ANET_CHECK_SHAPE(batch_action, { N });

    // 戻りの枠生成
    BatchStepResult batch_result = createEmptyStepResult();

    // ----- 環境を順次実行して埋める -----
    for (int i = 0; i < N; ++i) {
        auto a = batch_action[i].item<int64_t>();
        SingleStepResult single_result = envs_[i]->Step(a, mode);

        batch_result.next_state.obs.index_put_({ i }, single_result.next_state.obs);
        batch_result.next_state.done.index_put_({ i }, single_result.next_state.done);
        batch_result.next_state.truncated.index_put_({ i }, single_result.next_state.truncated);
        batch_result.next_state.episode_start.index_put_({ i }, single_result.next_state.episode_start);

        batch_result.reward.index_put_({ i }, single_result.reward);

        // Auto reset
        if (single_result.next_state.done || single_result.next_state.truncated) {
            SingleState reset_state = envs_[i]->Reset(mode);
            batch_result.continue_state.obs.index_put_({ i }, reset_state.obs);
            batch_result.continue_state.done.index_put_({ i }, reset_state.done);
            batch_result.continue_state.truncated.index_put_({ i }, reset_state.truncated);
            batch_result.continue_state.episode_start.index_put_({ i }, reset_state.episode_start);
        } else {
            batch_result.continue_state.obs.index_put_({ i }, single_result.next_state.obs);
            batch_result.continue_state.done.index_put_({ i }, false);
            batch_result.continue_state.truncated.index_put_({ i }, false);
            batch_result.continue_state.episode_start.index_put_({ i }, single_result.next_state.episode_start);
        }
    }

    // 継続用Stateを保存
    state_ = batch_result.continue_state.Clone();

    return batch_result;
}

BatchState VectorizedDiscreteBatchEnv::GetState() const
{
    return state_.Clone();
}

