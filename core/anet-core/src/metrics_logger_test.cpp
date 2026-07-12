#include "anet/catch_test.hpp"

#include "anet/metrics_logger.hpp"
#include "anet/test_util.hpp"
#include "metrics_logger_impl.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <wx/image.h>

namespace {

class NullBackend final : public anet::IBackend {
public:
    void Open(const std::filesystem::path&, const std::string&) override {}
    void WriteJsonl(const anet::json&) override {}
    void Flush() override {}
};

std::string ReadTextFile(const std::filesystem::path& path)
{
    std::ifstream ifs(path);
    return std::string(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
}

bool ContainsText(const std::string& text, const std::string& pattern)
{
    return text.find(pattern) != std::string::npos;
}

class FailingWriteBuffer final : public std::streambuf {
protected:
    std::streamsize xsputn(const char*, std::streamsize) override
    {
        return 0;
    }

    int_type overflow(int_type) override
    {
        return traits_type::eof();
    }
};

class FailingFlushBuffer final : public std::stringbuf {
protected:
    int sync() override
    {
        return -1;
    }
};

void LogFailure(const std::optional<anet::detail::JsonlIoFailure>& failure)
{
    if (failure) {
        anet::detail::LogJsonlIoFailure(*failure);
    }
}

} // namespace

TEST_CASE("Jsonl writer logs the first stream I/O failure only", "[metrics][io]")
{
    const std::filesystem::path path = "test-run/metrics.jsonl";
    anet::test::LogCaptureGuard logs;
    bool error_reported = false;

    SECTION("healthy stream does not log an error")
    {
        std::ostringstream stream;

        const auto write_failure = anet::detail::WriteJsonlLine(
            stream, "{\"type\":\"scalar\"}\n", path, error_reported);
        const auto flush_failure = anet::detail::FlushJsonl(stream, path, error_reported);
        LogFailure(write_failure);
        LogFailure(flush_failure);
        logs.Flush();

        CHECK_FALSE(write_failure);
        CHECK_FALSE(flush_failure);
        CHECK(stream.good());
        CHECK(anet::test::CountRecords(logs.Records(), wxLOG_Error) == 0);
    }

    SECTION("write failure logs once and leaves the stream failed")
    {
        FailingWriteBuffer buffer;
        std::ostream stream(&buffer);

        const auto first_failure = anet::detail::WriteJsonlLine(
            stream, "{\"type\":\"scalar\"}\n", path, error_reported);
        const auto suppressed_failure = anet::detail::FlushJsonl(stream, path, error_reported);
        LogFailure(first_failure);
        LogFailure(suppressed_failure);
        logs.Flush();

        REQUIRE(first_failure);
        CHECK_FALSE(suppressed_failure);
        CHECK(stream.fail());
        CHECK(anet::test::CountRecords(logs.Records(), wxLOG_Error) == 1);
        CHECK(anet::test::HasRecordContaining(
            logs.Records(),
            wxLOG_Error,
            {
                "Metrics JSONL I/O failed",
                "operation=write",
                "metrics.jsonl",
                "fail=true",
                "bad=true",
                "The run is not stopped",
                "Check free disk space and filesystem health",
                "Further errors for this file are suppressed",
            }));
    }

    SECTION("flush failure logs once after a successful write")
    {
        FailingFlushBuffer buffer;
        std::ostream stream(&buffer);

        const auto write_failure = anet::detail::WriteJsonlLine(
            stream, "{\"type\":\"scalar\"}\n", path, error_reported);
        const auto first_failure = anet::detail::FlushJsonl(stream, path, error_reported);
        const auto suppressed_failure = anet::detail::FlushJsonl(stream, path, error_reported);
        LogFailure(write_failure);
        LogFailure(first_failure);
        LogFailure(suppressed_failure);
        logs.Flush();

        CHECK_FALSE(write_failure);
        REQUIRE(first_failure);
        CHECK_FALSE(suppressed_failure);
        CHECK(stream.fail());
        CHECK(anet::test::CountRecords(logs.Records(), wxLOG_Error) == 1);
        CHECK(anet::test::HasRecordContaining(
            logs.Records(),
            wxLOG_Error,
            {
                "Metrics JSONL I/O failed",
                "operation=flush",
                "metrics.jsonl",
                "fail=true",
                "bad=true",
                "The run is not stopped",
                "Check free disk space and filesystem health",
                "Further errors for this file are suppressed",
            }));
    }
}

TEST_CASE("VideoLogger checks NVENC eligible video size", "[metrics][video]")
{
    CHECK_FALSE(anet::detail::IsNvencEligibleVideoSize(128, 128));
    CHECK(anet::detail::IsNvencEligibleVideoSize(160, 64));
    CHECK_FALSE(anet::detail::IsNvencEligibleVideoSize(159, 64));
    CHECK_FALSE(anet::detail::IsNvencEligibleVideoSize(160, 63));
    CHECK_FALSE(anet::detail::IsNvencEligibleVideoSize(161, 64));
}

TEST_CASE("VideoLogger resolves configured video codec", "[metrics][video]")
{
    auto small_auto = anet::detail::ResolveVideoCodec("auto", 128, 128, "small.mkv");
    CHECK(small_auto.codec == "libx264");
    CHECK(small_auto.requested_auto);
    CHECK_FALSE(small_auto.nvenc_eligible);

    auto large_auto = anet::detail::ResolveVideoCodec("auto", 512, 512, "large.mkv");
    CHECK(large_auto.codec == "h264_nvenc");
    CHECK(large_auto.requested_auto);
    CHECK(large_auto.nvenc_eligible);

    auto explicit_nvenc = anet::detail::ResolveVideoCodec("h264_nvenc", 160, 64, "nvenc.mkv");
    CHECK(explicit_nvenc.codec == "h264_nvenc");
    CHECK_FALSE(explicit_nvenc.requested_auto);
    CHECK(explicit_nvenc.nvenc_eligible);

    auto explicit_libx264 = anet::detail::ResolveVideoCodec("libx264", 128, 128, "cpu.mkv");
    CHECK(explicit_libx264.codec == "libx264");
    CHECK_FALSE(explicit_libx264.requested_auto);

    auto passthrough = anet::detail::ResolveVideoCodec("mjpeg", 128, 128, "mjpeg.mkv");
    CHECK(passthrough.codec == "mjpeg");
    CHECK_FALSE(passthrough.requested_auto);
}

TEST_CASE("VideoLogger rejects explicit NVENC for ineligible video size", "[metrics][video]")
{
    try {
        static_cast<void>(anet::detail::ResolveVideoCodec("h264_nvenc", 128, 128, "small.mkv"));
        FAIL("Expected explicit h264_nvenc to reject an ineligible video size");
    } catch (const std::exception& e) {
        const std::string message = e.what();
        CHECK(ContainsText(message, "metrics_logger.video_codec=h264_nvenc"));
        CHECK(ContainsText(message, "128x128"));
        CHECK(ContainsText(message, "160x64"));
        CHECK(ContainsText(message, "auto or libx264"));
        CHECK(ContainsText(message, "small.mkv"));
    }
}

TEST_CASE("MetricsLoggerConfig defaults video codec to auto", "[metrics][video]")
{
    const anet::MetricsLoggerConfig config;
    CHECK(config.video_codec == "auto");
}

TEST_CASE("VideoLogger writes large frames in chunks", "[metrics][video]")
{
    const auto case_id = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto root = std::filesystem::current_path() / "out" / "test-tmp" /
        ("anet-core-video-large-frame-test-" + std::to_string(case_id));
    std::filesystem::create_directories(root);

    wxImage image(512, 512);
    unsigned char* data = image.GetData();
    REQUIRE(data != nullptr);
    for (int y = 0; y < image.GetHeight(); ++y) {
        for (int x = 0; x < image.GetWidth(); ++x) {
            const int index = (y * image.GetWidth() + x) * 3;
            data[index + 0] = static_cast<unsigned char>(x % 256);
            data[index + 1] = static_cast<unsigned char>(y % 256);
            data[index + 2] = static_cast<unsigned char>((x + y) % 256);
        }
    }

    {
        const auto video_path = root / "large_frame.mkv";
        anet::VideoLogger logger(video_path.string(), image.GetWidth(), image.GetHeight(), "libx264", 15);
        logger.WriteFrame(image);
        logger.WriteFrame(image);
    }

    std::error_code cleanup_error;
    std::filesystem::remove_all(root, cleanup_error);
}

TEST_CASE("MetricsLogger writes ConfigData text file", "[metrics][config]")
{
    const auto root = std::filesystem::current_path() / "out" / "test-tmp" / "anet-core-config-data-test";
    std::filesystem::remove_all(root);

    anet::MetricsLogger::Reset();
    auto backend = std::make_unique<NullBackend>();
    anet::MetricsLoggerConfig logger_config;
    logger_config.run_name_tmpl = "config_data_test";
    anet::MetricsLogger::Init(std::move(backend), logger_config, root);

    anet::ConfigData config_data;
    config_data.Set("app.run_name", "run_{t}");
    config_data.Set("train.num_envs", "8");
    config_data.Set("DefaultDQNAgent.batch_size", "128");
    anet::MetricsLogger::Instance()->Log("config_data", config_data);

    const auto config_path = root / "runs" / "config_data_test" / "config" / "config_data.txt";
    REQUIRE(std::filesystem::exists(config_path));
    CHECK(ReadTextFile(config_path) ==
        "app.run_name = run_{t}\n"
        "train.num_envs = 8\n"
        "DefaultDQNAgent.batch_size = 128\n");

    anet::MetricsLogger::Reset();
    std::filesystem::remove_all(root);
}

TEST_CASE("MetricsLogger uses configured runs directory", "[metrics][config]")
{
    const auto root = std::filesystem::current_path() / "out" / "test-tmp" / "anet-core-custom-runs-dir-test";
    std::filesystem::remove_all(root);

    anet::MetricsLogger::Reset();
    auto backend = std::make_unique<anet::JsonlBackend>();
    anet::MetricsLoggerConfig logger_config;
    logger_config.runs_dir = "custom-runs";
    logger_config.run_name_tmpl = "custom_runs_dir_test";
    anet::MetricsLogger::Init(std::move(backend), logger_config, root);

    const auto run_dir = root / "custom-runs" / "custom_runs_dir_test";
    CHECK(anet::MetricsLogger::Instance()->GetRunDir() == run_dir);
    CHECK(std::filesystem::exists(run_dir / "metrics.jsonl"));

    anet::MetricsLogger::Instance()->LogScalar("test/value", 7, 1.25);
    anet::MetricsLogger::Instance()->Flush();

    anet::MetricsLogger::Reset();
    CHECK(ContainsText(ReadTextFile(run_dir / "metrics.jsonl"), "\"tag\":\"test/value\""));
    std::filesystem::remove_all(root);
}
