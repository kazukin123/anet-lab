#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <fcntl.h>
#include <io.h>

constexpr char kCodecOption[] = "-c:v";
constexpr char kTraceEnvironment[] = "PRD117_FFMPEG_TRACE";
constexpr char kNvencCodec[] = "h264_nvenc";

// 起動したcodecだけを記録し、NVENCの起動失敗とCPUでの継続を再現する。
int main(int argc, char** argv)
{
    std::string codec;
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == kCodecOption) codec = argv[i + 1];
    }
    const auto* trace_path = std::getenv(kTraceEnvironment);
    if (trace_path == nullptr || codec.empty()) return 2;
    {
        std::ofstream trace(trace_path, std::ios::app);
        if (!trace) return 3;
        trace << codec << '\n';
    }
    if (codec == kNvencCodec) {
        std::fputs("PRD117 simulated NVENC startup failure\n", stderr);
        return 1;
    }
    _setmode(_fileno(stdin), _O_BINARY);
    char buffer[16384];
    while (_read(_fileno(stdin), buffer, sizeof(buffer)) > 0) {}
    return 0;
}
