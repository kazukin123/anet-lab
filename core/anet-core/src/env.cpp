
#include "anet/env.hpp"
#include <functional>
#include <deque>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <thread>
#include "anet/profile.hpp"
#include "anet/metrics_logger.hpp"
#include "anet/thread.hpp"
#include "anet/tensor_util.hpp"
#include "anet/log.hpp"

using namespace anet::rl;
namespace LOG = anet::log;


//----------------------------------------------
// DiscreteBatchEnvBase Result
//----------------------------------------------

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
            auto result = single_results[env_index];
            if (result != nullptr) {
                auto aux = result->GetAuxData();
                auxs.push_back(aux);
            }
        } else {
            for (auto result : single_results) {
                if (result != nullptr) {
                    auto aux = result->GetAuxData();
                    auxs.push_back(aux);
                }
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


//----------------------------------------------
// DiscreteBatchEnvBase
//----------------------------------------------

DiscreteBatchEnvBase::DiscreteBatchEnvBase(
    const ConfigData& config_data,
    std::shared_ptr<SingleDiscreteEnvFactory> factory, int batch_size,
    const torch::Device& device, std::optional<seed_t> seed,
    const std::string& config_prefix)
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
        auto env = factory->CreateSingleEnv(config_data, device, env_seed, config_prefix);
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


std::optional<float> DiscreteBatchEnvBase::GetScalar(const std::string& key, int64_t index) const
{
    ANET_ASSERT(index < 0 || index < envs_.size());

    // 特定の環境(index)が指定されている場合は、その環境の値を直接返す
    if (index >= 0) {
        ANET_ASSERT(index < envs_.size());
        return envs_[index]->GetScalar(key, -1);
    }

    // バッチ全体(index == -1)の場合、プレフィックスを解析して集計
    enum class AggType { None, Max, Mean, Min };
    AggType agg_type = AggType::None;
    std::string subkey = key;

    if (key.rfind("max.", 0) == 0) {
        agg_type = AggType::Max;
        subkey = key.substr(4);
    } else if (key.rfind("mean.", 0) == 0) {
        agg_type = AggType::Mean;
        subkey = key.substr(5);
    } else if (key.rfind("min.", 0) == 0) {
        agg_type = AggType::Min;
        subkey = key.substr(4);
    } else {
        // プレフィックスが無い場合はデフォルトでMeanとして集計
        LOG::warn() << "DiscreteBatchEnvBase::GetScalar() Unknown prefix. assuming mean. key=" << key;
        agg_type = AggType::Mean;
        subkey = key;
    }

    float agg_val = 0.0f;
    if (agg_type == AggType::Max) agg_val = std::numeric_limits<float>::lowest();
    if (agg_type == AggType::Min) agg_val = std::numeric_limits<float>::max();
    float sum_val = 0.0f;
    int valid_count = 0;

    // 全ENVのデータを収集
    for (const auto& env : envs_) {
        auto val_opt = env->GetScalar(subkey, -1);

        // 1つでも nullopt (未対応/未取得) があれば、直ちに nullopt を返す
        if (!val_opt.has_value()) {
            return std::nullopt;
        }

        // 取れた値
        float val = val_opt.value();

        // NaN は集計から除外
        if (std::isnan(val)) continue;

        // 集計
        valid_count++;
        if (agg_type == AggType::Max) {
            agg_val = std::max(agg_val, val);
        } else if (agg_type == AggType::Min) {
            agg_val = std::min(agg_val, val);
        } else if (agg_type == AggType::Mean) {
            sum_val += val;
        }
    }

    // 全ての環境が NaN だった場合 (有効な値が1つも無い)
    if (valid_count == 0) {
        return std::numeric_limits<float>::quiet_NaN();
    }

    // 集計結果を返す
    if (agg_type == AggType::Mean) {
        return sum_val / static_cast<float>(valid_count);
    }

    return agg_val;
}

std::optional<torch::Tensor> DiscreteBatchEnvBase::GetTensor(const std::string& key, int64_t index) const
{
    ANET_ASSERT(index >= 0 && index < envs_.size());
    return envs_[index]->GetTensor(key, -1);
    /// @todo vec対応
}

std::optional<std::vector<torch::Tensor>> DiscreteBatchEnvBase::GetTensorVector(const std::string& key, int64_t index) const
{
    ANET_ASSERT(index >= 0 && index < envs_.size());
    return envs_[index]->GetTensorVector(key, -1);
    /// @todo vec対応
}


//----------------------------------------------
// VectorizedDiscreteBatchEnv
//----------------------------------------------

VectorizedDiscreteBatchEnv::VectorizedDiscreteBatchEnv(
    const ConfigData& configData,
    std::shared_ptr<SingleDiscreteEnvFactory> factory,
    int batch_size,
    const torch::Device& device,
    std::optional<seed_t> seed,
    const std::string& config_prefix)
    : DiscreteBatchEnvBase(configData, factory, batch_size, device, seed, config_prefix)
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
        ANET_ASSERT_DEVICE(reset_result->state.obs, device_);
        result->state.obs[i].copy_(reset_result->state.obs);
        result->state.done[i].fill_(reset_result->state.done);
        result->state.truncated[i].fill_(reset_result->state.truncated);
        result->state.episode_start[i].fill_(reset_result->state.episode_start);
        result->single_results[i] = reset_result;
    }

    return result;
}

std::shared_ptr<const BatchStepResult> VectorizedDiscreteBatchEnv::Step(std::shared_ptr<BatchActionInfo> action_info, RunMode mode)
{
    ProfileRange r("VectorizedDiscreteBatchEnv::Step");

    const int64_t N = batch_spec_.batch_size;

    ANET_ASSERT_DTYPE_MSG(action_info->GetAction(), torch::kInt64,
        "VectorizedDiscreteBatchEnv supports discrete action only. actions should be kInt64.");
    ANET_ASSERT_SHAPE(action_info->GetAction(), {N});

    // 戻りの枠生成
    auto result = getStepResult();

    auto actions = action_info->GetAction(device_);
    // ----- 環境を順次実行して埋める -----
    for (int i = 0; i < N; ++i) {
        auto a = actions[i].item<int64_t>();
        std::shared_ptr<const SingleStepResult> single_result = envs_[i]->Step(a, mode);
        ANET_ASSERT_DEVICE(single_result->next_state.obs, device_);

        result->single_results[i] = single_result;

        result->next_state.obs.select(0, i).copy_(single_result->next_state.obs);
        result->next_state.done[i].fill_(single_result->next_state.done);
        result->next_state.truncated[i].fill_(single_result->next_state.truncated);
        result->next_state.episode_start[i].fill_(single_result->next_state.episode_start);

        result->reward.select(0, i).fill_(single_result->reward);

        // Auto reset
        if (single_result->next_state.done || single_result->next_state.truncated) {
            auto reset_result = envs_[i]->Reset(mode);
            ANET_ASSERT_DEVICE(reset_result->state.obs, device_);

            result->continue_state.obs.select(0, i).copy_(reset_result->state.obs);
            result->continue_state.done[i].fill_(reset_result->state.done);
            result->continue_state.truncated[i].fill_(reset_result->state.truncated);
            result->continue_state.episode_start[i].fill_(reset_result->state.episode_start);
            result->n_episode_end++;
        } else {
            result->continue_state.obs.select(0, i).copy_(single_result->next_state.obs);
            result->continue_state.done[i].fill_(false);
            result->continue_state.truncated[i].fill_(false);
            result->continue_state.episode_start[i].fill_(single_result->next_state.episode_start);
        }
    }

    result->n_transitions = N;

    return result;
}


//----------------------------------------------
// ThreadPoolDiscreteEnv
//----------------------------------------------

ThreadPoolDiscreteEnv::ThreadPoolDiscreteEnv(
    const ConfigData& configData,
    std::shared_ptr<SingleDiscreteEnvFactory> factory,
    int batch_size,
    const torch::Device& device,
    std::shared_ptr<ThreadPool> pool,
    std::optional<seed_t> seed,
    const std::string& config_prefix)
    : DiscreteBatchEnvBase(configData, factory, batch_size, device, seed, config_prefix)
    , pool_(std::move(pool))
{
    ANET_ASSERT(pool_ != nullptr);
    ANET_LOG_DEBUG("seed=" << this->GetSeed());
    this->batch_spec_.num_threads = pool_->GetWorkerCount();
}

void ThreadPoolDiscreteEnv::Shutdown()
{
    if (pool_) {
        pool_->Stop();
        //pool = nullptr;
    }
}

ThreadPoolDiscreteEnv::~ThreadPoolDiscreteEnv()
{
    if (pool_) {
        pool_->Stop();
    }
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
                ANET_ASSERT_DEVICE(single_result->state.obs, device_);

                // 結果書き込み(i番目の行だけを書くので他 Worker と race しない)
                result->state.obs.select(0, i).copy_(single_result->state.obs);
                result->state.done[i].fill_(single_result->state.done);
                result->state.truncated[i].fill_(single_result->state.truncated);
                result->state.episode_start[i].fill_(single_result->state.episode_start);

                // GetAuxDataList()向けにsingle_resultを詰める
                result->single_results[i] = single_result;
            });
    }

    // 全タスク終了待ち
    pool_->WaitAll();

    return result;
}

std::shared_ptr<const BatchStepResult> ThreadPoolDiscreteEnv::Step(std::shared_ptr<BatchActionInfo> action_info, RunMode mode)
{
    ProfileRange r("ThreadPoolDiscreteEnv::Step");

    const int N = batch_size_;
    ANET_LOG_DEBUG("action=" << anet::ToString(action_info->GetAction()));
    ANET_ASSERT_DTYPE_MSG(action_info->GetAction(), torch::kInt64,
        "ThreadPoolDiscreteEnv supports discrete action only. actions should be kInt64.");
    ANET_ASSERT_SHAPE(action_info->GetAction(), {N});
    const int worker_count = pool_->GetWorkerCount();
    ANET_ASSERT(worker_count > 0);

    // --- 返却バッファ ---
    auto result = getStepResult();

    auto actions = action_info->GetAction(this->device_);

    // --- 並列に Step + 結果書き込み ---
    for (int i = 0; i < N; ++i) {
        const int worker_id = i % worker_count;
        const int64_t action_i = actions[i].item<int64_t>();

        pool_->Enqueue(worker_id,
            [this, &result, i, action_i, mode]()
            {
                // --- SingleEnv の Step 実行 ---
                auto r = envs_[i]->Step(action_i, mode);
                ANET_ASSERT_DEVICE(r->next_state.obs, device_);

                // --- single_result ---
                result->single_results[i] = r;

                // --- next_state ---
                result->next_state.obs.select(0, i).copy_(r->next_state.obs);
                result->next_state.done[i].fill_(r->next_state.done);
                result->next_state.truncated[i].fill_(r->next_state.truncated);
                result->next_state.episode_start[i].fill_(r->next_state.episode_start);

                // --- reward ---
                result->reward.select(0, i).fill_(r->reward);

                // --- continue_state ---
                if (r->next_state.done || r->next_state.truncated) {
                    // Reset_required
                    auto reset_result = envs_[i]->Reset(mode);
                    ANET_ASSERT_DEVICE(reset_result->state.obs, device_);
                    result->continue_state.obs.select(0, i).copy_(reset_result->state.obs);
                    result->continue_state.done[i].fill_(reset_result->state.done);
                    result->continue_state.truncated[i].fill_(reset_result->state.truncated);
                    result->continue_state.episode_start[i].fill_(reset_result->state.episode_start);
                } else {
                    // Continue as-is
                    result->continue_state.obs.select(0, i).copy_(r->next_state.obs);
                    result->continue_state.done[i].fill_(false);
                    result->continue_state.truncated[i].fill_(false);
                    result->continue_state.episode_start[i].fill_(r->next_state.episode_start);
                }
            });

    }

    // --- すべてのタスク完了を待つ ---
    pool_->WaitAll();

    // カウント
    auto done = result->next_state.done;
    auto truncated = result->next_state.truncated;
    auto episode_end = done.logical_or(truncated);
    if (episode_end.device().is_cuda()) {
        episode_end = episode_end.to(torch::kCPU);
    }
    result->n_episode_end = episode_end.sum().item<int>();
    result->n_transitions = N;

    // 返す
    return result;
}


//----------------------------------------------
// DefaultBatchEnvFactory
//----------------------------------------------

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

    // ログ：パラメータ記録
    LOG::info() << "DefaultBatchEnvFactory config=" << config_.ToString();
    anet::MetricsLogger::Instance()->Log(config_);
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


//----------------------------------------------
// EnvRepository
//----------------------------------------------

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
