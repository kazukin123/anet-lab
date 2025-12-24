
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
#include "anet/tensor_util.hpp"
#include "anet/log.hpp"

using namespace anet::rl;
namespace LOG = anet::log;


// ---- DiscreteBatchEnvBase::Result

class DiscreteBatchEnvBase::Result : virtual public BatchEnvResult {
public:
    Result(int batch_size)
    {
        single_results.resize(batch_size);
    }

    std::vector<AuxData> GetAuxDataList(int env_index = -1) const override
    {
        std::vector<AuxData> auxs;
        if (env_index >= 0) {
            auto aux = single_results[env_index]->GetAuxData();
            auxs.push_back(aux);
        } else {
            for (auto result : single_results) {
                auto aux = result->GetAuxData();
                auxs.push_back(aux);
            }
        }

        return auxs;
    }
public:
    std::vector<std::shared_ptr<const SingleEnvResult>> single_results;

};

class DiscreteBatchEnvBase::ResetResult : virtual public DiscreteBatchEnvBase::Result, public anet::rl::BatchResetResult {
public:
    ResetResult(int batch_size, anet::rl::BatchState state)
        : Result(batch_size), BatchResetResult(std::move(state))
    {
        ;
    }
};

class DiscreteBatchEnvBase::StepResult : virtual public DiscreteBatchEnvBase::Result, public anet::rl::BatchStepResult {
public:
    StepResult(int batch_size,
        torch::Tensor reward, BatchState next_state, BatchState continue_state, uint32_t n_transitions, uint32_t n_done)
        : Result(batch_size), BatchStepResult(std::move(reward), std::move(next_state), std::move(continue_state), n_transitions, n_done)
    {
        ;
    }
};


// ---- DiscreteBatchEnvBase

DiscreteBatchEnvBase::DiscreteBatchEnvBase(
    const ConfigData& config_data,
    std::shared_ptr<SingleDiscreteEnvFactory> factory, int batch_size,
    const torch::Device& device, std::optional<seed_t> seed)
    : RandomHolder(seed), batch_spec_({ batch_size, 1 }), batch_size_(batch_size), device_(device)
{
    ANET_ASSERT(batch_size_ > 0);
    ANET_LOG_DEBUG("seed=" << this->GetSeed());

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
    if (device_.is_cpu()) {
        //float_opt_ = float_opt_.pinned_memory(true);
        //bool_opt_ = bool_opt_.pinned_memory(true);
    }
    reset_result_ = createEmptyResetResult();
    step_result_ = createEmptyStepResult();
}

EnvSpec DiscreteBatchEnvBase::GetSpec() const
{
    return *spec_;
}

BatchEnvSpec DiscreteBatchEnvBase::GetBatchSpec() const
{
    return batch_spec_;
}

std::shared_ptr<DiscreteBatchEnvBase::ResetResult> DiscreteBatchEnvBase::createEmptyResetResult() const
{
    auto result = std::make_shared<DiscreteBatchEnvBase::ResetResult>(
        batch_size_,
        anet::rl::BatchState {
            torch::empty(obs_dims_, float_opt_),
            torch::empty({ batch_size_ }, bool_opt_),
            torch::empty({ batch_size_ }, bool_opt_),
            torch::empty({ batch_size_ }, bool_opt_)
        });
    return result;
}

std::shared_ptr<DiscreteBatchEnvBase::StepResult> DiscreteBatchEnvBase::createEmptyStepResult() const
{
    auto result = std::make_shared<DiscreteBatchEnvBase::StepResult>(
        batch_size_,

        torch::empty({ batch_size_ }, float_opt_),       // reward        (N) kFloat32
        BatchState{    // next_state
            torch::empty(obs_dims_, float_opt_),         // obs           (N, state_dim..) kFloat32
            torch::empty({ batch_size_ }, bool_opt_),    // done          (N) kBool
            torch::empty({ batch_size_ }, bool_opt_),    // truncated     (N) kBool
            torch::empty({ batch_size_ }, bool_opt_)     // episode_start (N) kBool
        },
        BatchState{    // continue_state
            torch::empty(obs_dims_, float_opt_),         // obs           (N, state_dim..) kFloat32
            torch::empty({ batch_size_ }, bool_opt_),    // done          (N) kBool
            torch::empty({ batch_size_ }, bool_opt_),    // truncated     (N) kBool
            torch::empty({ batch_size_ }, bool_opt_)     // episode_start (N) kBool
        },
        0,  // n_transitions
        0   // n_done
    );

    return result;
}

std::shared_ptr<DiscreteBatchEnvBase::ResetResult> DiscreteBatchEnvBase::getResetResult() const
{
    /// @todo どこかでDeepCopyしてない処理があるみたいので特定＆修正し、使いまわし&pinnedで高速化
    
    std::shared_ptr<DiscreteBatchEnvBase::ResetResult> result = this->reset_result_;    // 使い回す
    //std::shared_ptr<DiscreteBatchEnvBase::ResetResult> result = createEmptyResetResult();
    return result;
}

std::shared_ptr<DiscreteBatchEnvBase::StepResult> DiscreteBatchEnvBase::getStepResult() const
{
    //std::shared_ptr<DiscreteBatchEnvBase::StepResult> result = this->step_result_;    // 使い回す
    std::shared_ptr<DiscreteBatchEnvBase::StepResult> result = createEmptyStepResult();
    return result;
}


std::optional<float> DiscreteBatchEnvBase::GetScalar(const std::string& key, int index) const
{
    ANET_ASSERT(index >= 0 && index < envs_.size());
    return envs_[index]->GetScalar(key, -1);
    /// @todo vec対応
}

std::optional<torch::Tensor> DiscreteBatchEnvBase::GetTensor(const std::string& key, int index) const
{
    ANET_ASSERT(index >= 0 && index < envs_.size());
    return envs_[index]->GetTensor(key, -1);
    /// @todo vec対応
}

std::optional<std::vector<torch::Tensor>> DiscreteBatchEnvBase::GetTensorVector(const std::string& key, int index) const
{
    ANET_ASSERT(index >= 0 && index < envs_.size());
    return envs_[index]->GetTensorVector(key, -1);
    /// @todo vec対応
}

// ---- VectorizedDiscreteBatchEnv

VectorizedDiscreteBatchEnv::VectorizedDiscreteBatchEnv(
    const ConfigData& configData,
    std::shared_ptr<SingleDiscreteEnvFactory> factory,
    int batch_size,
    const torch::Device& device,
    std::optional<seed_t> seed)
    : DiscreteBatchEnvBase(configData, factory, batch_size, device, seed)
{
    ANET_LOG_DEBUG("seed=" << this->GetSeed());
}

std::shared_ptr<const BatchResetResult> VectorizedDiscreteBatchEnv::Reset(RunMode mode)
{
    ProfileRange r("VectorizedDiscreteBatchEnv::Reset");

    // 戻りの枠生成
    auto result = getResetResult();

    // 全環境を初期化し、state_ バッファに書き込む（バッファは constructor 確保済み）
    for (int i = 0; i < batch_size_; ++i) {
        auto reset_result = envs_[i]->Reset(mode);
        ANET_CHECK_DEVICE(reset_result->state.obs, device_);
        result->state.obs[i].copy_(reset_result->state.obs);
        result->state.done[i] = reset_result->state.done;
        result->state.truncated[i] = reset_result->state.truncated;
        result->state.episode_start[i] = reset_result->state.episode_start;
        result->single_results[i] = reset_result;
    }

    return result;
}

std::shared_ptr<const BatchStepResult> VectorizedDiscreteBatchEnv::Step(const BatchActionInfo& batch_action, RunMode mode)
{
    ProfileRange r("VectorizedDiscreteBatchEnv::Step");

    const int64_t N = batch_spec_.batch_size;

    ANET_CHECK_DTYPE_MSG(batch_action.GetAction(), torch::kInt64,
        "VectorizedDiscreteBatchEnv supports discrete action only. actions should be kInt64.");
    ANET_CHECK_SHAPE(batch_action.GetAction(), {N});

    // 戻りの枠生成
    auto result = getStepResult();

    auto actions = batch_action.GetAction(device_);
    // ----- 環境を順次実行して埋める -----
    for (int i = 0; i < N; ++i) {
        auto a = actions[i].item<int64_t>();
        std::shared_ptr<const SingleStepResult> single_result = envs_[i]->Step(a, mode);
        ANET_CHECK_DEVICE(single_result->next_state.obs, device_);

        result->single_results[i] = single_result;

        result->next_state.obs.index_put_({ i }, single_result->next_state.obs);
        result->next_state.done.index_put_({ i }, single_result->next_state.done);
        result->next_state.truncated.index_put_({ i }, single_result->next_state.truncated);
        result->next_state.episode_start.index_put_({ i }, single_result->next_state.episode_start);

        result->reward.select(0, i).fill_(single_result->reward);

        // Auto reset
        if (single_result->next_state.done || single_result->next_state.truncated) {
            auto reset_result = envs_[i]->Reset(mode);
            ANET_CHECK_DEVICE(reset_result->state.obs, device_);
            result->continue_state.obs.index_put_({ i }, reset_result->state.obs);
            result->continue_state.done.index_put_({ i }, reset_result->state.done);
            result->continue_state.truncated.index_put_({ i }, reset_result->state.truncated);
            result->continue_state.episode_start.index_put_({ i }, reset_result->state.episode_start);

            if (single_result->next_state.done)
                result->n_done++;
        } else {
            result->continue_state.obs.index_put_({ i }, single_result->next_state.obs);
            result->continue_state.done.index_put_({ i }, false);
            result->continue_state.truncated.index_put_({ i }, false);
            result->continue_state.episode_start.index_put_({ i }, single_result->next_state.episode_start);
        }
    }

    result->n_transitions = N;

    return result;
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
    ANET_LOG_DEBUG("seed=" << this->GetSeed());
    this->batch_spec_.num_threads = pool_->GetWorkerCount();
}

std::shared_ptr<const BatchResetResult>  ThreadPoolDiscreteEnv::Reset(RunMode mode)
{
    ProfileRange r("ThreadPoolDiscreteEnv::Reset");

    ANET_ASSERT(pool_ != nullptr);
    const int worker_count = pool_->GetWorkerCount();
    ANET_ASSERT(worker_count > 0);

    const int N = batch_size_;

    // 戻りの枠生成
    auto result = getResetResult();

    // 全ENV分のResetタスクをキューに積む
    for (int i = 0; i < N; ++i) {
        const int worker_id = i % worker_count;

        pool_->Enqueue(worker_id, [this, &result, i, mode]()
            {
                // ENV Reset 実行
                auto single_result = envs_[i]->Reset(mode);
                ANET_CHECK_DEVICE(single_result->state.obs, device_);

                // 結果書き込み(i番目の行だけを書くので他 Worker と race しない)
                result->state.obs.select(0, i).copy_(single_result->state.obs);
                result->state.done[i] = single_result->state.done;
                result->state.truncated[i] = single_result->state.truncated;
                result->state.episode_start[i] = single_result->state.episode_start;
            });
    }

    // 全タスク終了待ち
    pool_->WaitAll();

    return result;
}

std::shared_ptr<const BatchStepResult> ThreadPoolDiscreteEnv::Step(const BatchActionInfo& batch_action, RunMode mode)
{
    ProfileRange r("ThreadPoolDiscreteEnv::Step");

    const int N = batch_size_;
    ANET_LOG_DEBUG("action=" << anet::ToString(batch_action.GetAction()));
    ANET_CHECK_DTYPE_MSG(batch_action.GetAction(), torch::kInt64,
        "ThreadPoolDiscreteEnv supports discrete action only. actions should be kInt64.");
    ANET_CHECK_SHAPE(batch_action.GetAction(), {N});
    const int worker_count = pool_->GetWorkerCount();
    ANET_ASSERT(worker_count > 0);

    // --- 返却バッファ ---
    auto result = getStepResult();

    auto actions = batch_action.GetAction(this->device_);

    // --- 並列に Step + 結果書き込み ---
    for (int i = 0; i < N; ++i) {
        const int worker_id = i % worker_count;
        const int64_t action_i = actions[i].item<int64_t>();

        pool_->Enqueue(worker_id,
            [this, &result, i, action_i, mode]()
            {
                // --- SingleEnv の Step 実行 ---
                auto r = envs_[i]->Step(action_i, mode);
                ANET_CHECK_DEVICE(r->next_state.obs, device_);

                // --- single_result ---
                result->single_results[i] = r;

                // --- next_state ---
                result->next_state.obs.select(0, i).copy_(r->next_state.obs);
                result->next_state.done[i] = r->next_state.done;
                result->next_state.truncated[i] = r->next_state.truncated;
                result->next_state.episode_start[i] = r->next_state.episode_start;

                // --- reward ---
                result->reward[i] = r->reward;

                // --- continue_state ---
                if (r->next_state.done || r->next_state.truncated) {
                    // Reset_required
                    auto reset_result = envs_[i]->Reset(mode);
                    ANET_CHECK_DEVICE(reset_result->state.obs, device_);
                    result->continue_state.obs.select(0, i).copy_(reset_result->state.obs);
                    result->continue_state.done[i] = reset_result->state.done;
                    result->continue_state.truncated[i] = reset_result->state.truncated;
                    result->continue_state.episode_start[i] = reset_result->state.episode_start;
                } else {
                    // Continue as-is
                    result->continue_state.obs.select(0, i).copy_(r->next_state.obs);
                    result->continue_state.done[i] = false;
                    result->continue_state.truncated[i] = false;
                    result->continue_state.episode_start[i] = r->next_state.episode_start;
                }
            });

    }

    // --- すべてのタスク完了を待つ ---
    pool_->WaitAll();

    // カウント
    auto done = result->next_state.done;
    if (done.device().is_cuda()) {
        done = done.to(torch::kCPU);
    }
    result->n_done = done.sum().item<int>();
    result->n_transitions = N;

    // 返す
    return result;
}

// ---- DefaultBatchEnvFactory

DefaultBatchEnvFactory::DefaultBatchEnvFactory(
    const DefaultBatchEnvFactoryConfig& config,
    const ConfigData& config_data,
    int batch_size,
    std::optional<const torch::Device> device)
	: config_data_(config_data)
    , config_(config)
    , batch_size_(batch_size)
    , device_(device.value_or(anet::MakeDevice(config_.device_type, config_.device_index)))
{
    /// @todo deviceの指定方法が設定ファイル、config、device、三箇所あるのを整理
    ANET_ASSERT(batch_size_ > 0);
}

int DefaultBatchEnvFactory::GetLogicalCores() const
{
	/// @todo 物理コア数も考慮する？
    unsigned int n = std::thread::hardware_concurrency();
    return (n == 0 ? 4 : (int)n);
}

int DefaultBatchEnvFactory::ResolveWorkerThreads(int batch) const
{
    int wt = config_.worker_threads;

    if (wt > 0) return wt;  // 明示指定

    const int logical = GetLogicalCores();

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

std::shared_ptr<anet::ThreadPool> DefaultBatchEnvFactory::CreatePool(int worker_threads) const
{
    return std::make_shared<PinnedThreadPool>(worker_threads);
}

std::shared_ptr<BatchEnv> DefaultBatchEnvFactory::CreateBatchEnv(std::optional<seed_t> seed, int batch_size_in)
{
    auto factory = GetSingleFactory();
    if (factory == nullptr) return nullptr;

    auto env_class_id = factory->GetTargetEnvClassId();

    int batch_size = batch_size_in < 0 ? batch_size_ : batch_size_in;

    // batch_size == 1 では VectorizedDiscreteBatchEnv の方が有利
    if (batch_size == 1 || config_.worker_type == WorkerType::SINGLE_THREAD) {
        return std::make_shared<VectorizedDiscreteBatchEnv>(
            config_data_, factory, batch_size, device_, seed);
    }

    int workers = ResolveWorkerThreads(batch_size);

    // workers == 0 の理論的ケース防止
    if (workers <= 0) workers = 1;

    // ThreadPool の生成
    auto pool = CreatePool(workers);

    // ThreadPoolDiscreteEnv の生成
    return std::make_shared<ThreadPoolDiscreteEnv>(
        config_data_, factory, batch_size, device_, pool, seed);
}

std::shared_ptr<SingleDiscreteEnvFactory> DefaultBatchEnvFactory::GetSingleFactory() const
{
    auto factory = EnvRepository::Instance().GetSingleDiscreteEnvFactory(config_.class_id);
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
