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
    TinyImageClsDataset()
    {
        static int serial = 0;
        root_ = std::filesystem::temp_directory_path()
            / ("anet_imagecls_env_test_" + std::to_string(++serial));
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
            ofs << "class0/sample0\n";
        }
        {
            std::ofstream ofs(test_path_);
            ofs << "class0/sample0\n";
        }

        wxImage image(2, 2, false);
        unsigned char* data = image.GetData();
        REQUIRE(data != nullptr);
        const unsigned char pixels[] = {
            255, 0, 0,   0, 255, 0,
            0, 0, 255,   255, 255, 255,
        };
        std::copy(std::begin(pixels), std::end(pixels), data);
        REQUIRE(image.SaveFile((class_dir / "sample0.png").string(), wxBITMAP_TYPE_PNG));
    }

    ~TinyImageClsDataset()
    {
        std::error_code ec;
        std::filesystem::remove_all(root_, ec);
    }

    anet::rl::env::ImageClsEnvConfig MakeConfig(int max_steps) const
    {
        anet::rl::env::ImageClsEnvConfig config;
        config.root_dir = image_root_.string();
        config.train_list_txt_path = train_path_.string();
        config.eval_list_txt_path = test_path_.string();
        config.classes_txt_path = classes_path_.string();
        config.suffix = ".png";
        config.image_width = 2;
        config.image_height = 2;
        config.max_steps = max_steps;
        config.augment.enabled = false;
        return config;
    }

    anet::ConfigData MakeConfigData(int max_steps) const
    {
        const auto config = MakeConfig(max_steps);
        anet::ConfigData config_data;
        config_data.Set("ImageClsEnv.root_dir", config.root_dir);
        config_data.Set("ImageClsEnv.train_list_txt_path", config.train_list_txt_path);
        config_data.Set("ImageClsEnv.eval_list_txt_path", config.eval_list_txt_path);
        config_data.Set("ImageClsEnv.classes_txt_path", config.classes_txt_path);
        config_data.Set("ImageClsEnv.suffix", config.suffix);
        config_data.Set("ImageClsEnv.image_width", std::to_string(config.image_width));
        config_data.Set("ImageClsEnv.image_height", std::to_string(config.image_height));
        config_data.Set("ImageClsEnv.max_steps", std::to_string(config.max_steps));
        config_data.Set("ImageClsEnv.augment.enabled", "false");
        return config_data;
    }

private:
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

TEST_CASE("ImageClsEnv sets episode_start flags for Conv2d metrics image recording", "[image_cls_env][episode_start]")
{
    EnsureWxImageSupport();

    TinyImageClsDataset dataset;

    anet::rl::env::ImageClsEnv env(dataset.MakeConfig(/*max_steps=*/2), /*seed=*/1);
    auto reset_result = env.Reset(anet::rl::RunMode::Train);
    CHECK(reset_result->state.done == false);
    CHECK(reset_result->state.truncated == false);
    CHECK(reset_result->state.episode_start == true);

    auto step_result = env.Step(/*action=*/0, anet::rl::RunMode::Train);
    CHECK(step_result->next_state.done == false);
    CHECK(step_result->next_state.truncated == false);
    CHECK(step_result->next_state.episode_start == false);
}

TEST_CASE("ImageClsEnv batch auto reset exposes episode_start on continue state", "[image_cls_env][episode_start]")
{
    EnsureWxImageSupport();

    TinyImageClsDataset dataset;
    auto factory = std::make_shared<anet::rl::env::ImageClsEnvFactory>();
    anet::rl::VectorizedDiscreteBatchEnv env(
        dataset.MakeConfigData(/*max_steps=*/1),
        factory,
        /*num_envs=*/1,
        torch::Device(torch::kCPU),
        /*seed=*/1,
        /*config_prefix=*/"");

    auto reset_result = env.Reset(anet::rl::RunMode::Train);
    CHECK(TensorBoolAt(reset_result->state.done, 0) == false);
    CHECK(TensorBoolAt(reset_result->state.truncated, 0) == false);
    CHECK(TensorBoolAt(reset_result->state.episode_start, 0) == true);

    auto action_info = std::make_shared<anet::rl::BatchActionInfo>(
        torch::zeros({ 1 }, torch::TensorOptions().dtype(torch::kInt64)));
    auto step_result = env.Step(action_info, anet::rl::RunMode::Train);

    CHECK(TensorBoolAt(step_result->next_state.done, 0) == true);
    CHECK(TensorBoolAt(step_result->next_state.truncated, 0) == false);
    CHECK(TensorBoolAt(step_result->next_state.episode_start, 0) == false);
    CHECK(TensorBoolAt(step_result->continue_state.done, 0) == false);
CHECK(TensorBoolAt(step_result->continue_state.truncated, 0) == false);
CHECK(TensorBoolAt(step_result->continue_state.episode_start, 0) == true);
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
