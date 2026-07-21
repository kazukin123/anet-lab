#include "anet/catch_test.hpp"

#include "ImageClsEnv.hpp"

#include "anet/env.hpp"
#include "anet/test_util.hpp"

#include <algorithm>
#include <clocale>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>

#ifdef _WIN32
#include <windows.h>
#endif

#include <wx/image.h>
#include <wx/init.h>

namespace {

void SetupUtf8Console()
{
#ifdef _WIN32
    SetConsoleCP(CP_UTF8);
    SetConsoleOutputCP(CP_UTF8);
#endif
    std::setlocale(LC_CTYPE, ".UTF-8");
}

class TinyImageClsDataset {
public:
    explicit TinyImageClsDataset(int sample_count = 1)
    {
        static int serial = 0;
        dataset_key_ = "imagecls_test_" + std::to_string(++serial);
        root_ = std::filesystem::temp_directory_path()
            / ("anet_imagecls_env_test_" + dataset_key_);
        image_root_ = root_ / "images";
        const auto class_dir = image_root_ / "class0";
        std::filesystem::create_directories(class_dir);

        classes_path_ = root_ / "classes.txt";
        train_path_ = root_ / "train.txt";
        test_path_ = root_ / "test.txt";

        {
            std::ofstream ofs(classes_path_);
            ofs << "class0\n";
        }
        {
            std::ofstream ofs(train_path_);
            for (int i = 0; i < sample_count; ++i) ofs << "class0/sample" << i << "\n";
        }
        {
            std::ofstream ofs(test_path_);
            for (int i = 0; i < sample_count; ++i) ofs << "class0/sample" << i << "\n";
        }

        wxImage image(2, 2, false);
        unsigned char* data = image.GetData();
        REQUIRE(data != nullptr);
        const unsigned char pixels[] = {
            255, 0, 0,   0, 255, 0,
            0, 0, 255,   255, 255, 255,
        };
        std::copy(std::begin(pixels), std::end(pixels), data);
        for (int i = 0; i < sample_count; ++i) {
            REQUIRE(image.SaveFile(
                (class_dir / ("sample" + std::to_string(i) + ".png")).string(), wxBITMAP_TYPE_PNG));
        }
    }

    ~TinyImageClsDataset()
    {
        std::error_code ec;
        std::filesystem::remove_all(root_, ec);
    }

    anet::ConfigData MakeNativeConfigData(int max_steps) const
    {
        anet::ConfigData config_data;
        config_data.Set("ImageDataset.root_dir", image_root_.string());
        config_data.Set("ImageDataset.classes_txt_path", classes_path_.string());
        config_data.Set("ImageDataset.suffix", ".png");
        config_data.Set("ImageDataset.image_width", "2");
        config_data.Set("ImageDataset.image_height", "2");
        config_data.Set("ImageDataset.cache.mode", "none");
        config_data.Set("ImageDataset.[" + dataset_key_ + "].list_txt_path", test_path_.string());
        config_data.Set("ImageClsEnv.train.dataset_key", dataset_key_);
        config_data.Set("ImageClsEnv.eval.dataset_key", dataset_key_);
        config_data.Set("ImageClsEnv.eval.eval_window.mode", "full");
        config_data.Set("ImageClsEnv.eval.eval_window.rotating_size", "1");
        config_data.Set("ImageClsEnv.max_steps", std::to_string(max_steps));
        return config_data;
    }

    const std::string& GetDatasetKey() const { return dataset_key_; }

    void CorruptImage(int index) const
    {
        const auto path = image_root_ / "class0" / ("sample" + std::to_string(index) + ".png");
        std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
        REQUIRE(ofs.good());
        ofs << "not-an-image";
    }

    anet::img::ImageDatasetConfig MakeDatasetConfig() const
    {
        anet::img::ImageDatasetConfig config;
        config.root_dir = std::filesystem::absolute(image_root_).lexically_normal();
        config.list_txt_path = std::filesystem::absolute(test_path_).lexically_normal();
        config.classes_txt_path = std::filesystem::absolute(classes_path_).lexically_normal();
        config.suffix = ".png";
        config.image_width = 2;
        config.image_height = 2;
        config.cache_mode = anet::img::ImageCacheMode::None;
        config.cache_max_bytes = 1024;
        return config;
    }

private:
    std::string dataset_key_;
    std::filesystem::path root_;
    std::filesystem::path image_root_;
    std::filesystem::path classes_path_;
    std::filesystem::path train_path_;
    std::filesystem::path test_path_;
};

bool TensorBoolAt(const torch::Tensor& tensor, int64_t index)
{
    return tensor[index].item<bool>();
}

void EnsureWxImageSupport()
{
    // wx の shutdown が test process 終了時に不安定なため、初期化だけを process lifetime に寄せる。
    static const bool initialized = [] {
        return wxInitialize();
    }();
    REQUIRE(initialized);

    static const bool handlers_initialized = [] {
        wxInitAllImageHandlers();
        return true;
    }();
    (void)handlers_initialized;
}

} // namespace

TEST_CASE("ImageClsEnv factory runs a native eval batch from the dataset catalog", "[image_cls_env][native_batch]")
{
    EnsureWxImageSupport();

    TinyImageClsDataset dataset;
    anet::rl::env::ImageClsEnvFactory factory;
    auto env = factory.CreateBatchEnv(
        dataset.MakeNativeConfigData(/*max_steps=*/100),
        torch::Device(torch::kCPU),
        "imagecls-native-test",
        /*seed=*/1,
        /*num_envs=*/2,
        anet::rl::RunMode::Eval1,
        /*config_prefix=*/"");

    REQUIRE(env != nullptr);
    CHECK(env->GetName() == "imagecls-native-test");
    CHECK(env->GetEnvName(0) == "imagecls-native-test[0]");
    CHECK(env->GetEnvName(1) == "imagecls-native-test[1]");
    CHECK(env->GetRunMode() == anet::rl::RunMode::Eval1);
    CHECK_FALSE(env->GetSpec().info.contains("image_dataset_key"));
    const auto env_config = env->GetConfigData();
    REQUIRE(env_config.has_value());
    CHECK(env_config->Get("ImageClsEnv.train.dataset_key") == dataset.GetDatasetKey());
    CHECK(env_config->Get("ImageClsEnv.eval.dataset_key") == dataset.GetDatasetKey());

    auto reset_result = env->Reset();
    const auto grid = reset_result->state.obs.Get(anet::rl::ObsKeys::kGrid).value();
    const auto target = reset_result->state.obs.Get(anet::rl::ObsKeys::kVector).value();
    CHECK(grid.sizes() == torch::IntArrayRef({ 2, 3, 2, 2 }));
    CHECK(target.sizes() == torch::IntArrayRef({ 2, 1 }));

    auto action_info = std::make_shared<anet::rl::BatchActionInfo>(
        torch::zeros({ 2 }, torch::TensorOptions().dtype(torch::kInt64)));
    auto step_result = env->Step(action_info);
    CHECK(step_result->reward.equal(torch::tensor({ 1.0f, 0.0f })));
    CHECK(TensorBoolAt(step_result->next_state.done, 0));
    CHECK_FALSE(TensorBoolAt(step_result->next_state.done, 1));
    CHECK(step_result->n_transitions == 1);
    CHECK(step_result->n_episode_end == 1);
    CHECK(env->GetScalar("accuracy").value() == Catch::Approx(1.0f));
}

TEST_CASE("ImageDatasetManager registers a catalog atomically and shares a DatasetKey", "[image_cls_env][dataset_manager]")
{
    EnsureWxImageSupport();

    TinyImageClsDataset dataset;
    auto& manager = anet::img::ImageDatasetManager::Instance();
    const auto config = dataset.MakeDatasetConfig();
    manager.RegisterCatalog({ { dataset.GetDatasetKey(), config } });

    const auto first = manager.Acquire(dataset.GetDatasetKey());
    const auto second = manager.Acquire(dataset.GetDatasetKey());
    CHECK(first == second);

    auto conflict = config;
    conflict.image_width = 3;
    const auto uncommitted_key = dataset.GetDatasetKey() + "_uncommitted";
    CHECK_THROWS(manager.RegisterCatalog({
        { uncommitted_key, config },
        { dataset.GetDatasetKey(), conflict },
    }));
    CHECK_THROWS(manager.Acquire(uncommitted_key));
}

TEST_CASE("ImageDatasetManager keeps manifest construction failures sticky", "[image_cls_env][dataset_manager]")
{
    TinyImageClsDataset dataset;
    auto config = dataset.MakeDatasetConfig();
    config.classes_txt_path = config.classes_txt_path.parent_path() / "missing-classes.txt";
    const auto key = dataset.GetDatasetKey() + "_sticky";
    auto& manager = anet::img::ImageDatasetManager::Instance();
    manager.RegisterCatalog({ { key, config } });

    auto acquire_error = [&] {
        try {
            (void)manager.Acquire(key);
        } catch (const std::exception& e) {
            return std::string(e.what());
        }
        return std::string{};
    };
    const auto first = acquire_error();
    const auto second = acquire_error();
    CHECK_FALSE(first.empty());
    CHECK(second == first);
}

TEST_CASE("ImageClsEnv validates both standard manifests before selecting its RunMode source", "[image_cls_env][config]")
{
    EnsureWxImageSupport();
    TinyImageClsDataset dataset;
    auto config = dataset.MakeNativeConfigData(/*max_steps=*/100);
    const auto broken_eval_key = dataset.GetDatasetKey() + "_broken_eval";
    const auto valid_dataset_config = dataset.MakeDatasetConfig();
    config.Set("ImageDataset.[" + broken_eval_key + "].list_txt_path",
        valid_dataset_config.list_txt_path.string());
    config.Set("ImageDataset.[" + broken_eval_key + "].classes_txt_path",
        (valid_dataset_config.classes_txt_path.parent_path() / "missing-eval-classes.txt").string());
    config.Set("ImageClsEnv.eval.dataset_key", broken_eval_key);

    anet::rl::env::ImageClsEnvFactory factory;
    CHECK_THROWS_WITH(
        factory.CreateBatchEnv(
            config, torch::Device(torch::kCPU), "imagecls-eager-pair-test",
            /*seed=*/1, /*num_envs=*/1, anet::rl::RunMode::Train, ""),
        Catch::Matchers::ContainsSubstring("missing-eval-classes.txt"));
}

TEST_CASE("ImageCls config validation ignores unknown keys without manifest I/O", "[image_cls_env][config]")
{
    EnsureWxImageSupport();

    TinyImageClsDataset dataset;
    anet::rl::env::ImageClsEnvFactory factory;

    SECTION("unknown key") {
        auto config = dataset.MakeNativeConfigData(/*max_steps=*/100);
        config.Set("ImageClsEnv.unknown", "ignored");
        config.Set("ImageDataset.unknown", "ignored");
        config.Set("ImageDataset.[" + dataset.GetDatasetKey() + "].unknown", "ignored");
        CHECK_NOTHROW(factory.ValidateConfig(config, anet::rl::RunMode::Train, ""));
    }

    SECTION("dormant-style schema validation remains I/O free") {
        auto config = dataset.MakeNativeConfigData(/*max_steps=*/100);
        config.Set("ImageDataset.classes_txt_path", "missing-classes.txt");
        const std::string prefix = "train.eval.[dormant].env";
        config.Set(prefix + ".eval.dataset_key", dataset.GetDatasetKey());
        config.Set(prefix + ".eval.eval_window.mode", "full");
        CHECK_NOTHROW(factory.ValidateConfig(config, anet::rl::RunMode::Eval1, prefix));
    }

    SECTION("configured eval may inherit the standard eval source") {
        auto config = dataset.MakeNativeConfigData(/*max_steps=*/100);
        const std::string prefix = "train.eval.[missing].env";
        CHECK_NOTHROW(factory.ValidateConfig(config, anet::rl::RunMode::Eval1, prefix));
    }
}

TEST_CASE("ImageDataset explicit full_ram fails when the payload exceeds its cap", "[image_cls_env][cache]")
{
    EnsureWxImageSupport();

    TinyImageClsDataset dataset;
    auto config_data = dataset.MakeNativeConfigData(/*max_steps=*/100);
    config_data.Set("ImageDataset.cache.mode", "full_ram");
    config_data.Set("ImageDataset.cache.max_bytes", "1");
    anet::rl::env::ImageClsEnvFactory factory;
    auto env = factory.CreateBatchEnv(
        config_data, torch::Device(torch::kCPU), "imagecls-cache-test",
        /*seed=*/1, /*num_envs=*/1, anet::rl::RunMode::Eval1, "");

    CHECK_THROWS(env->Reset());
}

TEST_CASE("ImageDataset cache policies preserve explicit and auto failure semantics", "[image_cls_env][cache]")
{
    EnsureWxImageSupport();
    anet::rl::env::ImageClsEnvFactory factory;

    SECTION("auto falls back when the payload exceeds its cap") {
        TinyImageClsDataset dataset;
        auto config = dataset.MakeNativeConfigData(/*max_steps=*/100);
        config.Set("ImageDataset.cache.mode", "auto");
        config.Set("ImageDataset.cache.max_bytes", "1");
        auto env = factory.CreateBatchEnv(
            config, torch::Device(torch::kCPU), "imagecls-auto-cache-test",
            /*seed=*/1, /*num_envs=*/1, anet::rl::RunMode::Eval1, "");
        CHECK_NOTHROW(env->Reset());
        CHECK_NOTHROW(env->Reset());
    }

    SECTION("full_ram succeeds within its declared cap") {
        TinyImageClsDataset dataset;
        auto config = dataset.MakeNativeConfigData(/*max_steps=*/100);
        config.Set("ImageDataset.cache.mode", "full_ram");
        config.Set("ImageDataset.cache.max_bytes", "1024");
        auto env = factory.CreateBatchEnv(
            config, torch::Device(torch::kCPU), "imagecls-full-cache-test",
            /*seed=*/1, /*num_envs=*/1, anet::rl::RunMode::Eval1, "");
        CHECK_NOTHROW(env->Reset());
        CHECK_NOTHROW(env->Reset());
    }
}

TEST_CASE("ImageCls native decode honors worker_type and worker_threads", "[image_cls_env][worker]")
{
    EnsureWxImageSupport();
    anet::rl::env::ImageClsEnvFactory factory;

    SECTION("AUTO keeps B1 synchronous") {
        TinyImageClsDataset dataset;
        auto env = factory.CreateBatchEnv(
            dataset.MakeNativeConfigData(/*max_steps=*/100), torch::Device(torch::kCPU),
            "imagecls-worker-b1", /*seed=*/1, /*num_envs=*/1, anet::rl::RunMode::Eval1, "");
        CHECK(env->GetBatchSpec().num_threads == 1);
    }

    SECTION("explicit worker count is not clamped to B") {
        TinyImageClsDataset dataset;
        auto config = dataset.MakeNativeConfigData(/*max_steps=*/100);
        config.Set("env.worker_type", std::to_string(anet::rl::WorkerType::THREAD_POOL));
        config.Set("env.worker_threads", "5");
        auto env = factory.CreateBatchEnv(
            config, torch::Device(torch::kCPU), "imagecls-worker-explicit",
            /*seed=*/1, /*num_envs=*/2, anet::rl::RunMode::Eval1, "");
        CHECK(env->GetBatchSpec().num_threads == 5);
        CHECK_NOTHROW(env->Reset());
    }

    SECTION("invalid worker mode fails during construction") {
        TinyImageClsDataset dataset;
        auto config = dataset.MakeNativeConfigData(/*max_steps=*/100);
        config.Set("env.worker_threads", "0");
        CHECK_THROWS_WITH(
            factory.CreateBatchEnv(
                config, torch::Device(torch::kCPU), "imagecls-worker-invalid",
                /*seed=*/1, /*num_envs=*/2, anet::rl::RunMode::Eval1, ""),
            Catch::Matchers::ContainsSubstring("Invalid env.worker_threads=0"));
    }
}

TEST_CASE("ImageCls train augmentation is independent of worker scheduling", "[image_cls_env][worker][augment]")
{
    EnsureWxImageSupport();

    TinyImageClsDataset dataset(/*sample_count=*/4);
    auto single_config = dataset.MakeNativeConfigData(/*max_steps=*/100);
    single_config.Set("ImageClsEnv.train.augment.enabled", "true");
    single_config.Set("ImageClsEnv.train.augment.hflip_p", "0.5");
    single_config.Set("ImageClsEnv.train.augment.rrc_scale_min", "0.5");
    single_config.Set("ImageClsEnv.train.augment.rrc_scale_max", "1.0");
    single_config.Set("ImageClsEnv.train.augment.rrc_ratio_min", "0.75");
    single_config.Set("ImageClsEnv.train.augment.rrc_ratio_max", "1.3333333");
    single_config.Set("env.worker_type", std::to_string(anet::rl::WorkerType::SINGLE_THREAD));
    single_config.Set("env.worker_threads", "1");

    auto parallel_config = single_config;
    parallel_config.Set("env.worker_type", std::to_string(anet::rl::WorkerType::THREAD_POOL));
    parallel_config.Set("env.worker_threads", "3");

    anet::rl::env::ImageClsEnvFactory factory;
    auto single_env = factory.CreateBatchEnv(
        single_config, torch::Device(torch::kCPU), "imagecls-augment-single",
        /*seed=*/42, /*num_envs=*/4, anet::rl::RunMode::Train, "");
    auto parallel_env = factory.CreateBatchEnv(
        parallel_config, torch::Device(torch::kCPU), "imagecls-augment-parallel",
        /*seed=*/42, /*num_envs=*/4, anet::rl::RunMode::Train, "");

    const auto single_result = single_env->Reset();
    const auto parallel_result = parallel_env->Reset();
    CHECK(single_result->state.obs.Get(anet::rl::ObsKeys::kGrid).value().equal(
        parallel_result->state.obs.Get(anet::rl::ObsKeys::kGrid).value()));
    CHECK(single_result->state.obs.Get(anet::rl::ObsKeys::kVector).value().equal(
        parallel_result->state.obs.Get(anet::rl::ObsKeys::kVector).value()));
}

TEST_CASE("ImageCls does not publish a partial batch after worker failure", "[image_cls_env][worker]")
{
    EnsureWxImageSupport();

    TinyImageClsDataset dataset(/*sample_count=*/4);
    dataset.CorruptImage(/*index=*/2);
    auto config = dataset.MakeNativeConfigData(/*max_steps=*/100);
    config.Set("env.worker_type", std::to_string(anet::rl::WorkerType::THREAD_POOL));
    config.Set("env.worker_threads", "3");

    anet::rl::env::ImageClsEnvFactory factory;
    auto env = factory.CreateBatchEnv(
        config, torch::Device(torch::kCPU), "imagecls-worker-failure",
        /*seed=*/42, /*num_envs=*/4, anet::rl::RunMode::Train, "");

    CHECK_THROWS(env->Reset());
}

TEST_CASE("ImageClsEnv returns fresh observations and representative eval termination", "[image_cls_env][eval_window]")
{
    EnsureWxImageSupport();

    TinyImageClsDataset dataset;
    anet::rl::env::ImageClsEnvFactory factory;
    auto env = factory.CreateBatchEnv(
        dataset.MakeNativeConfigData(/*max_steps=*/100), torch::Device(torch::kCPU),
        "imagecls-fresh-test", /*seed=*/1, /*num_envs=*/2, anet::rl::RunMode::Eval1, "");

    const auto reset_result = env->Reset();
    const auto reset_grid = reset_result->state.obs.Get(anet::rl::ObsKeys::kGrid).value();
    const auto reset_snapshot = reset_grid.clone();
    auto action_info = std::make_shared<anet::rl::BatchActionInfo>(
        torch::zeros({ 2 }, torch::TensorOptions().dtype(torch::kInt64)));
    const auto step_result = env->Step(action_info);
    const auto next_grid = step_result->next_state.obs.Get(anet::rl::ObsKeys::kGrid).value();

    CHECK(reset_grid.data_ptr() != next_grid.data_ptr());
    CHECK(reset_grid.equal(reset_snapshot));
    CHECK(TensorBoolAt(step_result->next_state.done, 0));
    CHECK_FALSE(TensorBoolAt(step_result->next_state.done, 1));
    CHECK(TensorBoolAt(step_result->continue_state.episode_start, 0));
    CHECK_FALSE(TensorBoolAt(step_result->continue_state.episode_start, 1));
    CHECK(step_result->n_transitions == 1);
}

TEST_CASE("ImageCls rotating eval window uses exact valid count and padding", "[image_cls_env][eval_window]")
{
    EnsureWxImageSupport();

    TinyImageClsDataset dataset(/*sample_count=*/4);
    auto config = dataset.MakeNativeConfigData(/*max_steps=*/100);
    config.Set("ImageClsEnv.eval.eval_window.mode", "rotating");
    config.Set("ImageClsEnv.eval.eval_window.rotating_size", "3");
    anet::rl::env::ImageClsEnvFactory factory;
    auto env = factory.CreateBatchEnv(
        config, torch::Device(torch::kCPU), "imagecls-rotating-test",
        /*seed=*/1, /*num_envs=*/2, anet::rl::RunMode::Eval1, "");

    env->Reset();
    auto action_info = std::make_shared<anet::rl::BatchActionInfo>(
        torch::zeros({ 2 }, torch::TensorOptions().dtype(torch::kInt64)));
    const auto first = env->Step(action_info);
    CHECK(first->n_transitions == 2);
    CHECK(first->n_episode_end == 0);

    const auto second = env->Step(action_info);
    CHECK(second->reward.equal(torch::tensor({ 1.0f, 0.0f })));
    CHECK(second->n_transitions == 1);
    CHECK(second->n_episode_end == 1);
    CHECK(TensorBoolAt(second->next_state.done, 0));
    CHECK_FALSE(TensorBoolAt(second->next_state.done, 1));
    CHECK(env->GetScalar("accuracy").value() == Catch::Approx(1.0f));
}

TEST_CASE("ImageClsEnv train mode terminates every lane and counts dataset cycles", "[image_cls_env][train_batch]")
{
    EnsureWxImageSupport();

    TinyImageClsDataset dataset;
    anet::rl::env::ImageClsEnvFactory factory;
    auto env = factory.CreateBatchEnv(
        dataset.MakeNativeConfigData(/*max_steps=*/1), torch::Device(torch::kCPU),
        "imagecls-train-test", /*seed=*/1, /*num_envs=*/2, anet::rl::RunMode::Train, "");

    env->Reset();
    auto action_info = std::make_shared<anet::rl::BatchActionInfo>(
        torch::zeros({ 2 }, torch::TensorOptions().dtype(torch::kInt64)));
    const auto step_result = env->Step(action_info);

    CHECK(TensorBoolAt(step_result->next_state.done, 0));
    CHECK(TensorBoolAt(step_result->next_state.done, 1));
    CHECK(step_result->n_transitions == 2);
    CHECK(step_result->n_episode_end == 2);
    CHECK(env->GetScalar("accuracy").value() == Catch::Approx(1.0f));
    CHECK(env->GetScalar("epoch_count").value() == Catch::Approx(2.0f));
}

TEST_CASE("Food101 local config runs one native train and eval batch", "[.food101_smoke]")
{
    EnsureWxImageSupport();
    const auto config_path = std::filesystem::path("apps/runner/config/ImageCls.txt");
    const auto food101_root = std::filesystem::path("C:/dev/food-101/images");
    if (!std::filesystem::exists(config_path) || !std::filesystem::exists(food101_root)) {
        SKIP("Food101 local dataset is not available.");
    }

    // 実データsmokeではcache常駐を無効化し、各Sourceから1画像だけ同期decodeする。
    anet::ConfigManager config_manager(config_path.string());
    auto config = config_manager.GetConfigData();
    config.Set("ImageDataset.cache.mode", "none");
    config.Set("env.worker_type", std::to_string(anet::rl::WorkerType::SINGLE_THREAD));
    config.Set("env.worker_threads", "1");

    anet::rl::env::ImageClsEnvFactory factory;
    auto train = factory.CreateBatchEnv(
        config, torch::Device(torch::kCPU), "food101-train-smoke",
        /*seed=*/1, /*num_envs=*/1, anet::rl::RunMode::Train, "");
    auto train_reset = train->Reset();
    CHECK(train_reset->state.obs.Get(anet::rl::ObsKeys::kGrid).value().sizes()
        == torch::IntArrayRef({ 1, 3, 224, 224 }));

    const std::string eval_prefix = "train.eval.[eval1].env";
    auto eval = factory.CreateBatchEnv(
        config, torch::Device(torch::kCPU), "food101-eval-smoke",
        /*seed=*/2, /*num_envs=*/1, anet::rl::RunMode::Eval1, eval_prefix);
    auto eval_reset = eval->Reset();
    CHECK(eval_reset->state.obs.Get(anet::rl::ObsKeys::kGrid).value().sizes()
        == torch::IntArrayRef({ 1, 3, 224, 224 }));
    const auto eval_config = eval->GetConfigData();
    REQUIRE(eval_config.has_value());
    CHECK(eval_config->Get(eval_prefix + ".eval.dataset_key") == "food101_eval");
}

int main(int argc, char* argv[])
{
    SetupUtf8Console();

    anet::test::PreparedTestArgs test_args;
    try {
        test_args = anet::test::PrepareTestArgs(argc, argv);
    } catch (const std::exception& e) {
        return anet::test::ReportTestArgsError(e);
    }
    anet::test::SetupTestFailureDialog(test_args.failure_dialog_enabled);

    Catch::Session session;
    session.configData().showDurations = Catch::ShowDurations::Always;
    return session.run(test_args.Argc(), test_args.Argv());
}
