#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <crtdbg.h>
#include <windows.h>
#include <wx/log.h>
#include "anet/exception.hpp"
#include "anet/heat_map.hpp"

namespace {

constexpr int kWidth = 7;
constexpr int kHeight = 5;
constexpr int kChannels = 3;
constexpr int kIterations = 100;
constexpr size_t kPixelBytes = kWidth * kHeight * kChannels;
constexpr const char* kEmpty = "empty";
constexpr const char* kTime = "time";
constexpr const char* kHistogram = "histogram";
constexpr const char* kSweep = "sweep";
constexpr const char* kDump = "dump";
constexpr const char* kFail = "fail";
constexpr const char* kRender = "render";

std::unique_ptr<anet::ImageSource> MakeSource(const std::string& path)
{
    if (path == kEmpty || path == kTime) {
        auto source = std::make_unique<anet::TimeHeatMap>(
            kWidth, kHeight, 0.0f, 5.0f, anet::HM_Default, 0, anet::TimeFrameMode::Scale);
        if (path == kTime) {
            source->AddData(1.0f, 1.0f);
            source->NextFrame();
            source->AddData(3.0f, 2.0f);
            source->NextFrame();
        }
        return source;
    }
    if (path == kHistogram) {
        auto source = std::make_unique<anet::Histgram>(3, 0.0f, 3.0f, kWidth, kHeight);
        source->AddData(0.5f);
        source->AddData(2.5f);
        source->AddData(2.5f);
        return source;
    }
    if (path == kSweep) {
        auto source = std::make_unique<anet::SweepedHeatMap>(
            kWidth, kHeight, 0.0f, 1.0f, 0.0f, 1.0f);
        source->Evaluate([](float x, float y) { return x + y; });
        return source;
    }
    throw std::invalid_argument("Unknown image path: " + path);
}

// 検証専用CRTフック。要求した画素領域だけを一度失敗させ、診断の確保は妨げない。
int FailPixelAllocation(int operation, size_t size, int block_type, size_t expected)
{
    if (operation == _HOOK_ALLOC && block_type == _NORMAL_BLOCK && size == expected) {
        _CrtSetAllocHook(nullptr);
        std::fputs("INJECTED_PIXEL_ALLOCATION_FAILURE\n", stderr);
        return FALSE;
    }
    return TRUE;
}

int EmptyAllocationHook(int operation, void*, size_t size, int block_type, long,
    const unsigned char*, int)
{
    return FailPixelAllocation(operation, size, block_type, kChannels);
}

int ImageAllocationHook(int operation, void*, size_t size, int block_type, long,
    const unsigned char*, int)
{
    return FailPixelAllocation(operation, size, block_type, kPixelBytes);
}

[[noreturn]] void FinishProbe(int exit_code)
{
    std::cout.flush();
    std::cerr.flush();
    std::fflush(nullptr);
    // DLLの静的終了処理を除外する。renderでは対象画像の全破棄が済んでから呼ぶ。
    TerminateProcess(GetCurrentProcess(), static_cast<UINT>(exit_code));
    std::abort();
}

void Dump(const std::string& path)
{
    auto source = MakeSource(path);
    const auto image = source->RenderRaw();
    const auto* pixels = image.GetData();
    std::cout << image.GetWidth() << " " << image.GetHeight() << " " << source->GetImageSubType() << "\n";
    for (int i = 0; i < image.GetWidth() * image.GetHeight() * kChannels; ++i) {
        std::printf("%02x", pixels[i]);
    }
    std::printf("\n");
    // 画素の採取専用。修正前の不正な解放はrenderモードで別に検査する。
    FinishProbe(0);
}

int CheckFailure(const std::string& path)
{
    auto source = MakeSource(path);
    // 依存ライブラリの既存フックは、検証区間の終了時に戻す。
    const auto previous_hook = _CrtSetAllocHook(path == kEmpty ? EmptyAllocationHook : ImageAllocationHook);
    try {
        const auto image = source->RenderRaw();
        _CrtSetAllocHook(previous_hook);
        std::cerr << "Image allocation unexpectedly succeeded\n";
        return 2;
    } catch (const anet::AnetException& e) {
        _CrtSetAllocHook(previous_hook);
        const std::string message(e.what());
        const std::string expected_target = path == kHistogram ? "Histgram" :
            path == kSweep ? "SweepedHeatMap" : "TimeHeatMap";
        const int width = path == kEmpty ? 1 : kWidth;
        const int height = path == kEmpty ? 1 : kHeight;
        const bool valid = message.find("System error:") != std::string::npos
            && message.find(expected_target) != std::string::npos
            && message.find("width=" + std::to_string(width)) != std::string::npos
            && message.find("height=" + std::to_string(height)) != std::string::npos;
        std::cout << message << "\n";
        return valid ? 0 : 3;
    }
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cerr << "Expected image path and mode\n";
        return 2;
    }
    wxLog::SetActiveTarget(new wxLogStderr());
    int exit_code = 0;
    try {
        const std::string path(argv[1]);
        const std::string mode(argv[2]);
        if (mode == kDump) {
            Dump(path);
        } else if (mode == kFail) {
            exit_code = CheckFailure(path);
        } else if (mode == kRender) {
            // コピーだけを残して生成元と元画像を破棄し、最後の所有者も破棄する。
            for (int i = 0; i < kIterations; ++i) {
                wxImage copy;
                {
                    const auto source = MakeSource(path);
                    const auto image = source->RenderRaw();
                    copy = image;
                }
                if (!copy.IsOk() || copy.GetData() == nullptr) return 4;
            }
            std::cout << "PASS " << path << " iterations=" << kIterations << "\n";
        } else {
            throw std::invalid_argument("Unknown mode: " + mode);
        }
    } catch (const std::exception& e) {
        _CrtSetAllocHook(nullptr);
        std::cerr << e.what() << "\n";
        exit_code = 1;
    }
    delete wxLog::SetActiveTarget(nullptr);
    // 対象画像の破棄を完了してから終了する。計測・Torchの静的終了処理は検査対象外。
    FinishProbe(exit_code);
}
