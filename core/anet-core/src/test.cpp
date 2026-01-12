// test.cpp

#include "anet/replay_buffer.hpp"
#include "replay_buffer_impl.hpp"
#include "test.hpp"

using namespace anet::rl;
namespace LOG = anet::log;


static BatchExperience MakeStep(float value, bool is_start, bool is_done)
{
    auto dev = torch::kCPU;
    BatchExperience batch;

    // (1, 1) のState
    batch.state.obs = torch::tensor({ {value} }, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    batch.state.done = torch::tensor({ false }, torch::TensorOptions().dtype(torch::kBool).device(dev));
    batch.state.truncated = torch::tensor({ false }, torch::TensorOptions().dtype(torch::kBool).device(dev));
    batch.action = torch::tensor({ 0 }, torch::TensorOptions().dtype(torch::kInt64).device(dev));
    batch.reward = torch::tensor({ 0.0f }, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    batch.next_state.obs = torch::tensor({ {value + 1.0f} }, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    batch.next_state.done = torch::tensor({ is_done }, torch::TensorOptions().dtype(torch::kBool).device(dev));
    batch.next_state.truncated = torch::tensor({ false }, torch::TensorOptions().dtype(torch::kBool).device(dev));
    batch.next_state.episode_start = torch::tensor({ false }, torch::TensorOptions().dtype(torch::kBool).device(dev));

    // episode_start
    batch.state.episode_start = torch::tensor({ is_start }, torch::TensorOptions().dtype(torch::kBool).device(dev));

    LOG::info() << "exp=" << batch.ToString();
    return batch;
}

// helper: vector<T> -> Tensor
template <typename T>
torch::Tensor ToTensor(const std::vector<T>& data, torch::Dtype dtype, const std::vector<int64_t>& shape) {
    auto opts = torch::TensorOptions().dtype(dtype).device(torch::kCPU); // テスト用なのでCPU
    // vectorのデータをコピーしてTensor化
    return torch::from_blob(const_cast<T*>(data.data()), shape, opts).clone();
}

// N環境分のデータを一括作成
static BatchExperience MakeBatchStep(
    const std::vector<float>& values, // 環境ごとの値 (サイズN)
    const std::vector<bool>& is_starts, // 環境ごとのStartフラグ (サイズN)
    bool is_done
)
{
    int64_t n_envs = values.size(); // これがバッチサイズ(N)になる

    BatchExperience batch;

    // State: (N, 1)
    batch.state.obs = ToTensor(values, torch::kFloat32, { n_envs, 1 });

    // Action, Reward, etc: (N) 
    // ※実際はActionなどは適切な値を入れる
    std::vector<int64_t> actions(n_envs, 0);
    batch.action = ToTensor(actions, torch::kInt64, { n_envs });

    std::vector<float> rewards(n_envs, 0.0f);
    batch.reward = ToTensor(rewards, torch::kFloat32, { n_envs });

    // Meta flags: (N)
    // done, truncatedなどは全員同じ値で埋める例（必要ならvector化）
    std::vector<uint8_t> dones_vec(n_envs, (uint8_t)is_done); // boolはuint8として扱うのが無難
    std::vector<uint8_t> starts_vec(n_envs, 0);
    for (size_t i = 0; i < n_envs; ++i) starts_vec[i] = (uint8_t)is_starts[i];

    // Tensor化 (Bool)
    batch.state.done = ToTensor(dones_vec, torch::kBool, { n_envs });
    batch.state.truncated = torch::zeros({ n_envs }, torch::TensorOptions().dtype(torch::kBool));

    // 今回の肝: episode_start
    batch.state.episode_start = ToTensor(starts_vec, torch::kBool, { n_envs });

    // Next State: (N, 1) -> value + 1.0
    std::vector<float> next_values = values;
    for (auto& v : next_values) v += 1.0f;
    batch.next_state.obs = ToTensor(next_values, torch::kFloat32, { n_envs, 1 });

    // Next flags
    batch.next_state.done = batch.state.done; // 簡易
    batch.next_state.truncated = batch.state.truncated;
    batch.next_state.episode_start = torch::zeros({ n_envs }, torch::TensorOptions().dtype(torch::kBool));

    return batch;
}

static void TestStacking()
{
    LOG::info() << "=== Testing FrameStacking ReplayBuffer ===";

    // 1. 設定: Stack=3, Capacity=10, BatchSize(NumEnvs)=1
    ReplayBufferConfig config;
    config.capacity = 10;
    config.use_stacker = true;
    config.stack_count = 3; // 過去2フレーム + 現在
    config.type = ReplayBuilderType::PLAIN;
    config.sampler_type = ReplaySamplerType::UNIFORM;

    EnvSpec spec;
    spec.state_spec.shape = { 1 }; // 1次元の数値だけ
    spec.action_spec.value_labels = { "action1" };
    spec.action_spec.is_discrete = true;

    ReplayBufferFactory factory(config);
    // batch_size=1 (単一環境)
    auto rb = factory.Create(spec, torch::kCPU, 1, 12345);

    // 2. データ投入: 0〜5まで連番を入れる
    // 0: Start
    // 1, 2: Continue
    // 3: Start (ここでエピソードが切れる)
    // 4, 5: Continue

    LOG::info() << "Pushing data...";
    rb->Push(MakeStep(0.0f, true, false)); // idx=0, Start
    rb->Push(MakeStep(1.0f, false, false)); // idx=1
    rb->Push(MakeStep(2.0f, false, false)); // idx=2
    rb->Push(MakeStep(3.0f, true, false)); // idx=3, Start! (ここで境界ができるはず)
    rb->Push(MakeStep(4.0f, false, false)); // idx=4
    rb->Push(MakeStep(5.0f, false, false)); // idx=5
    LOG::info() << "Buffer Size: " << rb->Size();

    // 3. サンプリングして中身を確認
    // ランダムサンプリングだと確認しづらいため、何度か回して全パターン見る
    // 期待値:
    // idx=0 (Val=0, Start): [0, 0, 0] (StartPadding)
    // idx=1 (Val=1):        [0, 0, 1] (Padding 1つ)
    // idx=2 (Val=2):        [0, 1, 2] (Full Stack)
    // idx=3 (Val=3, Start): [3, 3, 3] (StartPadding / 2,1は前のエピソードなので壁)
    // idx=4 (Val=4):        [3, 3, 4]

    LOG::info() << "\nSampling (Batch=10)... Check the Stacked States!";

    // 多めにサンプリングして、結果を目視
    auto samples = rb->Sample(10, torch::kCPU, 0.0f);

    auto obs = samples.obs; // (10, 3, 1)

    for (int i = 0; i < 10; ++i) {
        // データを取り出す
        auto current_stack = obs[i]; // (3, 1)
        float v0 = current_stack[0].item<float>();
        float v1 = current_stack[1].item<float>();
        float v2 = current_stack[2].item<float>();

        // 元のインデックスを知りたいが、Sample結果からは推測する
        // 最新の値(v2)が何かで判定
        LOG::info() << "Sample " << i << " | LastVal=" << v2 << " | Stack: [ "
            << v0 << ", " << v1 << ", " << v2 << " ] ";

        // 簡易判定
        if (v2 == 0) {
            if (v0 == 0 && v1 == 0) LOG::info() << "OK (Start Padding)";
            else LOG::info() << "NG!";
        } else if (v2 == 1) {
            if (v0 == 0 && v1 == 0) LOG::info() << "OK (Padding)";
            else LOG::info() << "NG!";
        } else if (v2 == 2) {
            if (v0 == 0 && v1 == 1) LOG::info() << "OK (Normal)";
            else LOG::info() << "NG!";
        } else if (v2 == 3) {
            if (v0 == 3 && v1 == 3) LOG::info() << "OK (New Episode Start Padding)"; // 3はStartなので、2とかが混ざってはいけない
            else LOG::info() << "NG! (Mixed with prev episode?)";
        } else if (v2 == 4) {
            if (v0 == 3 && v1 == 3) LOG::info() << "OK (Padding with New Episode)";
            else LOG::info() << "NG!";
        }

    }
}

static void TestStackingMultiEnv()
{

    LOG::info() << "=== Testing Multi-Env (N=2) FrameStacking ===";

    // 1. 設定
    int num_envs = 2;
    int stack_count = 2;

    ReplayBufferConfig config;
    config.capacity = 20;
    config.use_stacker = true;
    config.stack_count = stack_count;
    config.type = ReplayBuilderType::PLAIN;
    config.sampler_type = ReplaySamplerType::UNIFORM;

    EnvSpec spec;
    spec.state_spec.shape = { 1 };
    spec.action_spec.value_labels = { "action1" };
    spec.action_spec.is_discrete = true;

    ReplayBufferFactory factory(config);
    auto rb = factory.Create(spec, torch::kCPU, num_envs, 12345);

    // 2. データ投入シミュレーション
    LOG::info() << "Pushing data for 3 steps...";

    // Step 1: 両方 Start (10.0, 20.0)
    rb->Push(MakeBatchStep({ 10.0f, 20.0f }, { true, true }, false));

    // Step 2: 両方 Continue (11.0, 21.0)
    rb->Push(MakeBatchStep({ 11.0f, 21.0f }, { false, false }, false));

    // Step 3: Env0だけ Reset(Start), Env1は Continue (12.0, 22.0)
    rb->Push(MakeBatchStep({ 12.0f, 22.0f }, { true, false }, false));

    LOG::info() << "Buffer Size: " << rb->Size() << " (Expected: 3 * 2 = 6)";

    // 3. 検証
    LOG::info() << "Sampling... Checking Stride & Padding Logic";
    LOG::info() << "Target Logic:";
    LOG::info() << "  - Env0 (12.0) is Start -> Stack should be [12, 12] (NOT [11, 12])";
    LOG::info() << "  - Env1 (22.0) is Cont  -> Stack should be [21, 22] (NOT [20, 22] or [12, 22])";
    LOG::info() << "---------------------------------------------------";

    auto samples = rb->Sample(20, torch::kCPU, 0.0f);
    auto obs = samples.obs; // (20, 2, 1)

    const float eps = 0.001f;

    for (int i = 0; i < 20; ++i) {
        auto current_stack = obs[i];
        float prev_val = current_stack[0].item<float>();
        float curr_val = current_stack[1].item<float>();

        LOG::info() << "Sample " << i << " | Stack: [" << prev_val << ", " << curr_val << "]";

        std::string status = "UNKNOWN";
        bool ok = false;

        if (std::abs(curr_val - 12.0f) < eps) {
            // Env0 Step 3 (Start) -> Expected [12, 12]
            if (std::abs(prev_val - 12.0f) < eps) { status = "-> OK (Env0 Step3: Start Padding)"; ok = true; } else status = "NG! Expected [12, 12]";
        } else if (std::abs(curr_val - 22.0f) < eps) {
            // Env1 Step 3 (Continue) -> Expected [21, 22]
            if (std::abs(prev_val - 21.0f) < eps) { status = "-> OK (Env1 Step3: Normal Stride)"; ok = true; } else status = "NG! Expected [21, 22]";
        } else if (std::abs(curr_val - 11.0f) < eps) {
            // Env0 Step 2 (Continue) -> Expected [10, 11]
            if (std::abs(prev_val - 10.0f) < eps) { status = "-> OK (Env0 Step2: Normal)"; ok = true; } else status = "NG! Expected [10, 11]";
        } else if (std::abs(curr_val - 21.0f) < eps) {
            // Env1 Step 2 (Continue) -> Expected [20, 21]
            if (std::abs(prev_val - 20.0f) < eps) { status = "-> OK (Env1 Step2: Normal)"; ok = true; } else status = "NG! Expected [20, 21]";
        } else if (std::abs(curr_val - 10.0f) < eps) {
            // Env0 Step 1 (Start) -> [10, 10]
            if (std::abs(prev_val - 10.0f) < eps) { status = "-> OK (Env0 Step1: Start)"; ok = true; } else status = "NG!";
        } else if (std::abs(curr_val - 20.0f) < eps) {
            // Env1 Step 1 (Start) -> [20, 20]
            if (std::abs(prev_val - 20.0f) < eps) { status = "-> OK (Env1 Step1: Start)"; ok = true; } else status = "NG!";
        }

        LOG::info() << "  " << status;
        //}
    }
}

void anet::rl::Test(const Runner& trainer)
{
    TestStacking();
    TestStackingMultiEnv();
}

