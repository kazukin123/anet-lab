
#include "anet/env.hpp"
#include <functional>
#include <deque>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <thread>
#include "anet/profile.hpp"
#include "anet/thread.hpp"
#include "anet/tensor_utils.hpp"

using namespace anet::rl;

// ---- DiscreteBatchEnvBase

DiscreteBatchEnvBase::DiscreteBatchEnvBase(
    const ConfigData& config_data,
    std::shared_ptr<SingleDiscreteEnvFactory> factory, int batch_size,
    const torch::Device& device, std::optional<seed_t> seed)
    : batch_spec_({ batch_size, 1 }), batch_size_(batch_size), device_(device)
{
    ANET_ASSERT(batch_size_ > 0);

    // ベースのシードを準備
    anet::SeedMaker seed_maker(seed);

    //  batch_size 個のENVを生成
    envs_.reserve(batch_size_);
    for (int i = 0; i < batch_size; ++i) {
        anet::seed_t env_seed = seed_maker.MakeIndexedSeed(i);
        auto env = factory->CreateSingleEnv(config_data, device, env_seed);
        envs_.push_back(std::move(env));
    }

    // EnvSpec取得
    spec_ = std::make_unique<EnvSpec>(envs_[0]->GetSpec());

    // EnvSpec からobsのshapeを取得
    auto shape = spec_->state_spec.shape;  // 例: {4}
    obs_dims_.push_back(batch_size_);
    obs_dims_.insert(obs_dims_.end(), shape.begin(), shape.end());

    // 初期化
    float_opt_ = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    bool_opt_ = torch::TensorOptions().dtype(torch::kBool).device(device);
}

EnvSpec DiscreteBatchEnvBase::GetSpec() const
{
    return *spec_;
}

BatchEnvSpec DiscreteBatchEnvBase::GetBatchSpec() const
{
    return batch_spec_;
}

anet::rl::BatchState DiscreteBatchEnvBase::createEmptyState() const
{
    anet::rl::BatchState batch_state {
        torch::empty(obs_dims_, float_opt_),
        torch::empty({ batch_size_ }, bool_opt_),
        torch::empty({ batch_size_ }, bool_opt_),
        torch::empty({ batch_size_ }, bool_opt_)
    };
    return batch_state;
}

anet::rl::BatchStepResult DiscreteBatchEnvBase::createEmptyStepResult() const
{
    BatchStepResult result {
        torch::empty({ batch_size_ }, float_opt_),       // reward        (N) kFloat32
        {    // next_state
            torch::empty(obs_dims_, float_opt_),         // obs           (N, state_dim..) kFloat32
            torch::empty({ batch_size_ }, bool_opt_),    // done          (N) kBool
            torch::empty({ batch_size_ }, bool_opt_),    // truncated     (N) kBool
            torch::empty({ batch_size_ }, bool_opt_)     // episode_start (N) kBool
        },
        {    // continue_state
            torch::empty(obs_dims_, float_opt_),         // obs           (N, state_dim..) kFloat32
            torch::empty({ batch_size_ }, bool_opt_),    // done          (N) kBool
            torch::empty({ batch_size_ }, bool_opt_),    // truncated     (N) kBool
            torch::empty({ batch_size_ }, bool_opt_)     // episode_start (N) kBool
        },
    };
    return result;
};

// ---- VectorizedDiscreteBatchEnv

VectorizedDiscreteBatchEnv::VectorizedDiscreteBatchEnv(
    const ConfigData& configData,
    std::shared_ptr<SingleDiscreteEnvFactory> factory,
    int batch_size,
    const torch::Device& device,
    std::optional<seed_t> seed)
    : DiscreteBatchEnvBase(configData, factory, batch_size, device, seed)
{
    ;
}

BatchState VectorizedDiscreteBatchEnv::Reset(RunMode mode)
{
    ProfileRange r("VectorizedDiscreteBatchEnv::Reset");

    // 戻り用に空のBatchState枠を作る
    auto state = createEmptyState();

    // 全環境を初期化し、state_ バッファに書き込む（バッファは constructor 確保済み）
    for (int i = 0; i < batch_size_; ++i) {
        auto reset_state = envs_[i]->Reset(mode);
        ANET_CHECK_DEVICE(reset_state.obs, device_);
        state.obs[i].copy_(reset_state.obs);
        state.done[i] = reset_state.done;
        state.truncated[i] = reset_state.truncated;
        state.episode_start[i] = reset_state.episode_start;
    }

    return state;
}

BatchStepResult VectorizedDiscreteBatchEnv::Step(const torch::Tensor& batch_action, RunMode mode)
{
    ProfileRange r("VectorizedDiscreteBatchEnv::Step");

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
        ANET_CHECK_DEVICE(single_result.next_state.obs, device_);

        batch_result.next_state.obs.index_put_({ i }, single_result.next_state.obs);
        batch_result.next_state.done.index_put_({ i }, single_result.next_state.done);
        batch_result.next_state.truncated.index_put_({ i }, single_result.next_state.truncated);
        batch_result.next_state.episode_start.index_put_({ i }, single_result.next_state.episode_start);

        batch_result.reward.index_put_({ i }, single_result.reward);

        // Auto reset
        if (single_result.next_state.done || single_result.next_state.truncated) {
            SingleState reset_state = envs_[i]->Reset(mode);
            ANET_CHECK_DEVICE(reset_state.obs, device_);
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

    return batch_result;
}

// ---- ThreadPoolDiscreteEnv

ThreadPoolDiscreteEnv::ThreadPoolDiscreteEnv(
    const ConfigData& configData,
    std::shared_ptr<SingleDiscreteEnvFactory> factory,
    int batch_size,
    const torch::Device& device,
    std::shared_ptr<ThreadPool> pool,
    std::optional<seed_t> seed)
    : DiscreteBatchEnvBase(configData, factory, batch_size, device, seed)
    , pool_(std::move(pool))
{
    ANET_ASSERT(pool_ != nullptr);
    this->batch_spec_.num_threads = pool_->GetWorkerCount();
}

BatchState ThreadPoolDiscreteEnv::Reset(RunMode mode)
{
    ProfileRange r("ThreadPoolDiscreteEnv::Reset");

    ANET_ASSERT(pool_ != nullptr);
    const int worker_count = pool_->GetWorkerCount();
    ANET_ASSERT(worker_count > 0);

    const int N = batch_size_;

    // 全ENV分の Reset結果を一時保存するバッファ（スレッド毎に index 固定なので race なし）
    BatchState state = createEmptyState();

    // 全ENV分のResetタスクをキューに積む
    for (int i = 0; i < N; ++i) {
        const int worker_id = i % worker_count;

        pool_->Enqueue(worker_id, [this, &state, i, mode]()
            {
                // ENV Reset 実行
                SingleState s = envs_[i]->Reset(mode);
                ANET_CHECK_DEVICE(s.obs, device_);

                // 結果書き込み(i番目の行だけを書くので他 Worker と race しない)
                state.obs[i].copy_(s.obs);
                state.done[i] = s.done;
                state.truncated[i] = s.truncated;
                state.episode_start[i] = s.episode_start;
            });
    }

    // 全タスク終了待ち
    pool_->WaitAll();

    return state;
}

BatchStepResult ThreadPoolDiscreteEnv::Step(const torch::Tensor& actions, RunMode mode)
{
    ProfileRange r("ThreadPoolDiscreteEnv::Step");

    const int N = batch_size_;
    ANET_CHECK_DTYPE_MSG(actions, torch::kInt64,
        "ThreadPoolDiscreteEnv supports discrete action only. actions should be kInt64.");
    ANET_CHECK_SHAPE(actions, { N });
    const int worker_count = pool_->GetWorkerCount();
    ANET_ASSERT(worker_count > 0);

    // --- 返却バッファ（メインスレッドで一度だけ確保） ---

    BatchStepResult result {
        torch::empty({ N }, float_opt_),
        BatchState {
            torch::empty(obs_dims_, float_opt_),
            torch::empty({ N }, bool_opt_),
            torch::empty({ N }, bool_opt_),
            torch::empty({ N }, bool_opt_),
        },
        BatchState {
            torch::empty(obs_dims_, float_opt_),
            torch::empty({ N }, bool_opt_),
            torch::empty({ N }, bool_opt_),
            torch::empty({ N }, bool_opt_),
        },
    };

    // --- 並列に Step + 結果書き込み ---
    for (int i = 0; i < N; ++i) {
        const int worker_id = i % worker_count;
        int64_t action_i = actions[i].item<int64_t>();

        pool_->Enqueue(worker_id,
            [this, &result, i, action_i, mode]()
            {
                // --- SingleEnv の Step 実行 ---
                SingleStepResult r = envs_[i]->Step(action_i, mode);
                ANET_CHECK_DEVICE(r.next_state.obs, device_);

                // --- next_state ---
                result.next_state.obs[i].copy_(r.next_state.obs);
                result.next_state.done[i] = r.next_state.done;
                result.next_state.truncated[i] = r.next_state.truncated;
                result.next_state.episode_start[i] = r.next_state.episode_start;

                // --- reward ---
                result.reward[i] = r.reward;

                // --- continue_state ---
                if (r.next_state.done || r.next_state.truncated) {
                    // Reset_required
                    SingleState reset_state = envs_[i]->Reset(mode);
                    ANET_CHECK_DEVICE(reset_state.obs, device_);
                    result.continue_state.obs[i].copy_(reset_state.obs);
                    result.continue_state.done[i] = reset_state.done;
                    result.continue_state.truncated[i] = reset_state.truncated;
                    result.continue_state.episode_start[i] = reset_state.episode_start;
                } else {
                    // Continue as-is
                    result.continue_state.obs[i].copy_(r.next_state.obs);
                    result.continue_state.done[i] = false;
                    result.continue_state.truncated[i] = false;
                    result.continue_state.episode_start[i] = r.next_state.episode_start;
                }
            });
    }

    // --- すべてのタスク完了を待つ ---
    pool_->WaitAll();

    return result;
}

// ---- DefaultBatchEnvFactory

DefaultBatchEnvFactory::DefaultBatchEnvFactory(
    const DefaultBatchEnvFactoryConfig& config,
    const ConfigData& config_data,
    int batch_size,
    std::optional<seed_t> seed,
    std::optional<const torch::Device> device)
	: config_data_(config_data)
    , config_(config)
    , batch_size_(batch_size)
    , seed_(seed)
    , device_(device.value_or(anet::MakeDevice(config_.device_type, config_.device_index)))
{
}

int DefaultBatchEnvFactory::getLogicalCores() const {
    unsigned int n = std::thread::hardware_concurrency();
    return (n == 0 ? 4 : (int)n);
}

int DefaultBatchEnvFactory::resolveWorkerThreads(int batch) const
{
    int wt = config_.worker_threads;

    if (wt > 0) return wt;  // 明示指定

    const int logical = getLogicalCores();

    switch (wt) {
    case WorkerThreadAuto::AUTO: {
        int safe = std::max(1, logical - 2);
        return std::min(batch, safe);
    }
    case WorkerThreadAuto::ENV_COUNT:
        return batch;

    case WorkerThreadAuto::LOGICAL_CORES: {
        int safe = std::max(1, logical - 2);
        return safe;
    }
    case WorkerThreadAuto::LOGICAL_CORES_EXACT:
        return logical;

    default:
        ANET_ASSERT(false && "Invalid worker_threads mode");
        return 1;
    }
}

std::shared_ptr<anet::ThreadPool> DefaultBatchEnvFactory::createPool(int worker_threads) const
{
    return std::make_shared<PinnedThreadPool>(worker_threads);
}

std::shared_ptr<BatchEnv> DefaultBatchEnvFactory::CreateBatchEnv()
{
    ANET_ASSERT(config_.batch_size > 0);

    auto factory = GetSingleFactory();
    if (factory == nullptr)
        return nullptr;
    auto env_class_id = factory->GetTargetEnvClassId();

    // batch_size == 1 は VectorizedDiscreteBatchEnv の方が有利
    if (batch_size_ == 1) {
        return std::make_shared<VectorizedDiscreteBatchEnv>(
            config_data_, factory, 1, device_, seed_);
    }

    int workers = resolveWorkerThreads(batch_size_);

    // workers == 0 の理論的ケース防止
    if (workers <= 0) workers = 1;

    // ThreadPool の生成
    auto pool = createPool(workers);

    // ThreadPoolDiscreteEnv の生成
    return std::make_shared<ThreadPoolDiscreteEnv>(
        config_data_, factory, batch_size_, device_, pool, seed_);
}

std::shared_ptr<SingleDiscreteEnvFactory> DefaultBatchEnvFactory::GetSingleFactory() const
{
    auto factory = EnvRepository::Instance().GetSingleDiscreteEnvFactory(config_.env_class_id);
    if (factory == nullptr)
        return nullptr;
    return factory;
}

void EnvRepository::Regist(std::shared_ptr<SingleDiscreteEnvFactory> factory)
{
    std::lock_guard<std::mutex> lock(mtx_);
    auto class_id = factory->GetTargetEnvClassId();
    factories_[class_id] = factory;
}

std::shared_ptr<SingleDiscreteEnvFactory> EnvRepository::GetSingleDiscreteEnvFactory(const std::string& id) const
{
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = factories_.find(id);
    if (it == factories_.end()) return nullptr;
    return it->second;
}
