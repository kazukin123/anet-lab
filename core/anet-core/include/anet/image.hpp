// anet/image.hpp

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <wx/image.h>
#include "anet/config.hpp"
#include "anet/rl.hpp"

namespace anet {


    // ============================================================
    // ImageSource
    // ============================================================

    class ImageSource {
    public:
        virtual void Reset() = 0;
        virtual wxImage RenderRaw() const = 0;
        virtual std::string GetImageSubType() const = 0;

        wxImage Render(int width = -1, int height = -1) const;
        void SavePng(const std::string& filename, int width = -1, int height = -1) const;
        virtual ~ImageSource() = default;
    };


    // ============================================================
    // HeatMapConfig
    // ============================================================

    //enum HeatMapFlags : uint32_t {
    //    HM_None = 0,
    //    HM_LogScaleValue = 1 << 0,  // 値(色強度)の対数圧縮（既存: HM_LogScale）
    //    HM_AutoNormValue = 1 << 1,  // 値(色強度)の正規化
    //    HM_AutoNormX = 1 << 2,  // X軸の自動スケーリング
    //    HM_AutoNormY = 1 << 3,  // Y軸の自動スケーリング
    //    HM_MeanMode = 1 << 4,  // 平均モード
    //    HM_SumMode = 1 << 5,  // 合計モード

    //    // 拡張
    //    HM_AutoScaleAxis = 1 << 6,  // 値軸(データ軸)のレンジ自動追従（NaN指定側のみEMA更新）
    //    HM_LogScaleAxis = 1 << 7,  // 値軸を対数スケール表示（log1p(|v|)の符号付）
    //    HM_ShowZeroLine = 1 << 8,  // 値0の位置に水平ラインを描画
    //    HM_FlipY = 1 << 9,  // Y軸方向を反転して描画（上が大値）

    //    HM_Default = HM_AutoNormValue | HM_SumMode
    //};

    //enum class TimeFrameMode {
    //    Scale = 0,
    //    Overwrite,
    //    Scroll
    //};

    //struct HeatMapObserverConfig {
    //    int width = 256;
    //    int height = 256;

    //    int log_interval = 100;
    //    int max_points = 30000;
    //    uint32_t flags = HM_Default;

    //    int image_width = -1;
    //    int image_height = -1;

    //    // HeatMap の軸定義（優先度：固定値 → Probe）
    //    bool override_xmin = false;
    //    bool override_xmax = false;
    //    bool override_ymin = false;
    //    bool override_ymax = false;

    //    float xmin = 0.0f;
    //    float xmax = 1.0f;
    //    float ymin = 0.0f;
    //    float ymax = 1.0f;
    //};

    //struct TimeHistogramObserverConfig {
    //    int bins = 128;
    //    int max_frames = 1000;
    //    int image_height = -1;
    //    int image_width = -1;
    //    TimeFrameMode mode = TimeFrameMode::Scale;
    //    uint32_t flags = anet::HeatMapFlags::HM_Default;
    //    int log_interval = 100;
    //    int frame_interval = 10;
    //    float base_min = std::numeric_limits<float>::quiet_NaN();
    //    float base_max = std::numeric_limits<float>::quiet_NaN();
    //    float alpha = 1.0f;
    //};

    //struct SweepedHeatMapObserverConfig {
    //    int log_interval = 100;
    //    uint64_t flags = anet::HeatMapFlags::HM_Default;
    //    int grid_width = -1;    ///< 内部で生成するGridの幅。-1で指定なし(自動)
    //    int grid_height = -1;   ///< 内部で生成するGridの高さ。-1で指定なし(自動)
    //    int image_width = -1;   ///< 出力する画像の幅。-1で指定なし(自動)
    //    int image_height = -1;  ///< 出力する画像の高さ。-1で指定なし(自動)
    //};
}

namespace anet::rl {

    // ============================================================
    // ImageProviderConfig
    // ============================================================

    //auto visit_x_probe = std::make_shared<anet::rl::BatchExperienceStateProbe>(0, &env_spec.state_spec, true);
    //auto visit_rewardp = std::make_shared<anet::rl::BatchExperienceRewardProbe>(nullptr);
    //auto visit_q_probe = std::make_shared<anet::rl::BatchActionInfoToVectorProbe>("max_q");
    //auto rep_x_probe = std::make_shared<anet::rl::AgentTensorVectorProbe>(anet::rl::ReplayBuffer::NEXT_STATE_OBS, 0, &env_spec.state_spec);
    //auto visit_q_probe = std::make_shared<anet::rl::BatchActionInfoToVectorProbe>("max_q");

    static constexpr const char* kProbeType_Experience = "experience";
    static constexpr const char* kProbeType_Agent = "agent";
    //static constexpr const char* kProbeType_Action = "action";

    //static constexpr const char* kProbeKey_StateObs = "state.obs";
    //static constexpr const char* kProbeKey_Action = "action";
    //static constexpr const char* kProbeKey_Reward = "reward";

    //enum AutoScaleMode {
    //    DISABLE,
    //    PER_STEP,
    //    GLOBAL,
    //}

    static constexpr const int kAutoScaleMode_Disable = 0;
    static constexpr const int kAutoScaleMode_PerStep = 1;
    static constexpr const int kAutoScaleMode_Global = 2;

    struct ProbeConfig {
        std::string source = "experience";     // ex. "experience" "agent"
        std::string key = "state";        // ex. "state" "action" "replaybuffer.storage.next_state"
        int index = 0;              // ex. 0=x(state) 2=y(state) 1=left(action) 
        //int auto_scale_mode = 0;
    };

    struct HeatMapConfig {
        int width = 256;
        int height = 256;
        int max_points = 30000;
        struct {
            ProbeConfig x;
            ProbeConfig y;
            ProbeConfig value;
        } probe;
    };

    struct SweepObservationHeatMapConfig {
        int width = 256;
        int height = 256;
        int x_index = 0;
        int y_index = 1;
        std::string network_key = "policy-net.forward.q";
        std::string extractor_name = "mean";
        int extractor_index = -1;
    };

    static constexpr const char* kImageFrameMode_Scale = "scale";
    static constexpr const char* kImageFrameMode_Overwrite = "overwrite";
    static constexpr const char* kImageFrameMode_Scroll = "scroll";

    struct TimeHistgramConfig {
        int bins = 128;
        int max_frames = 1000;
        std::string frame_mode;
        int frame_interval = 10;
        float alpha = 1.0f;
        struct {
            ProbeConfig value;
        } probe;
    };

    static constexpr const char* kImageType_HeatMap = "HeatMap";
    static constexpr const char* kImageType_StateSweepedHeatMap = "SweepStateHeatMap";
    static constexpr const char* kImageType_TimeHistgram = "TimeHistgram";
    static constexpr const char* kImageType_Conv2d = "Conv2d";

    struct Conv2dVisualizerConfig {
        int margin_x = 4;               ///< チャンネル(列)間の余白ピクセル
        int margin_y = 4;               ///< レイヤー(行)間の余白ピクセル
        int channels_per_row = 16;      ///< 1行に並べる最大チャンネル数
        bool flip_vertical = true;      ///< 画像の上下を反転して描画するか
        std::string network_key = "policy-net.conv2d"; ///< 抽出対象のネットワーク
        std::string colormap = "jet";	///< gray / jet / hot
        int scale_factor = 2;           ///< 画像引き伸ばしスケール。動画プレイヤー側で下手なスケーリングされても耐えるように…
        int min_block_size = 40;
    };

    struct ImageProviderConfig : public anet::Config {
        std::string type;
        int image_width = -1;
        int image_height = -1;
        std::vector<std::string> flags;
        int interval = 100;

        HeatMapConfig heatmap;
        SweepObservationHeatMapConfig sweep_obs;
        TimeHistgramConfig histgram;
        Conv2dVisualizerConfig conv2d;

        ImageProviderConfig()
            : anet::Config("")
        {
            ;
        }

        ImageProviderConfig(const anet::ConfigData& config_data, const std::string& config_prefix) : anet::Config("")
        {
            ANET_READ_CONFIG(config_data, type);
            ANET_READ_CONFIG(config_data, image_width);
            ANET_READ_CONFIG(config_data, image_height);
            ANET_READ_CONFIG(config_data, flags);
            ANET_READ_CONFIG(config_data, interval);

            if (type == kImageType_HeatMap) {
                ANET_READ_CONFIG(config_data, heatmap.width);
                ANET_READ_CONFIG(config_data, heatmap.height);
                ANET_READ_CONFIG(config_data, heatmap.max_points);

                ANET_READ_CONFIG(config_data, heatmap.probe.x.source);
                ANET_READ_CONFIG(config_data, heatmap.probe.x.key);
                ANET_READ_CONFIG(config_data, heatmap.probe.x.index);

                ANET_READ_CONFIG(config_data, heatmap.probe.y.source);
                ANET_READ_CONFIG(config_data, heatmap.probe.y.key);
                ANET_READ_CONFIG(config_data, heatmap.probe.y.index);

                ANET_READ_CONFIG(config_data, heatmap.probe.value.source);
                ANET_READ_CONFIG(config_data, heatmap.probe.value.key);
                ANET_READ_CONFIG(config_data, heatmap.probe.value.index);
            } else if (type == kImageType_StateSweepedHeatMap) {
                ANET_READ_CONFIG(config_data, sweep_obs.width);
                ANET_READ_CONFIG(config_data, sweep_obs.height);
                ANET_READ_CONFIG(config_data, sweep_obs.x_index);
                ANET_READ_CONFIG(config_data, sweep_obs.y_index);
                ANET_READ_CONFIG(config_data, sweep_obs.network_key);
                ANET_READ_CONFIG(config_data, sweep_obs.extractor_name);
                ANET_READ_CONFIG(config_data, sweep_obs.extractor_index);
            } else if (type == kImageType_TimeHistgram) {
                ANET_READ_CONFIG(config_data, histgram.bins);
                ANET_READ_CONFIG(config_data, histgram.max_frames);
                ANET_READ_CONFIG(config_data, histgram.frame_mode);
                ANET_READ_CONFIG(config_data, histgram.frame_interval);
                //ANET_READ_CONFIG(config_data, histgram.alpha);
                ANET_READ_CONFIG(config_data, histgram.probe.value.source);
                ANET_READ_CONFIG(config_data, histgram.probe.value.key);
                ANET_READ_CONFIG(config_data, histgram.probe.value.index);
            } else if (type == kImageType_Conv2d) {
                ANET_READ_CONFIG(config_data, conv2d.margin_x);
                ANET_READ_CONFIG(config_data, conv2d.margin_y);
                ANET_READ_CONFIG(config_data, conv2d.channels_per_row);
                ANET_READ_CONFIG(config_data, conv2d.flip_vertical);
                ANET_READ_CONFIG(config_data, conv2d.network_key);
                ANET_READ_CONFIG(config_data, conv2d.colormap);
                ANET_READ_CONFIG(config_data, conv2d.scale_factor);
                ANET_READ_CONFIG(config_data, conv2d.min_block_size);
            }

            default_prefix_ = config_prefix;
        }
    };


    // ============================================================
    // Conv2dVisualizer
    // ============================================================

    class Conv2dVisualizer {
    public:
        explicit Conv2dVisualizer(const Conv2dVisualizerConfig& config = Conv2dVisualizerConfig{}) : config_(config) {}

        // 画像とJSONのペアを返す
        std::pair<wxImage, anet::json> Visualize(int64_t step, const anet::TensorDict& dict) const;
    private:
        Conv2dVisualizerConfig config_;
    };


    // ============================================================
    // ImageProvider
    // ============================================================

    struct ImageData {
        wxImage image;
        anet::rl::step_t step;
    };

    class ImageProvider {
    public:
        virtual ImageData GetImageData(int width = -1, int height = -1) = 0;
        virtual ~ImageProvider() = default;
    };


    // ============================================================
    // ImageProviderFactory
    // ============================================================

    using NotifierFunction = std::function<void(std::shared_ptr<anet::rl::Notifier>)>;

    struct ImageProviderBundle {
        std::shared_ptr<ImageProvider> image_provider;
        NotifierFunction attach;
        NotifierFunction detach;
    };

    class ImageProviderFactory {
    public:
        ImageProviderBundle CreateImageProvider(const std::string& tag, ImageProviderConfig config, const EnvSpec& env_spec, std::shared_ptr<Agent> agent, std::shared_ptr<const Runner> runner);
    };


    // ============================================================
    // ImageProviderManager
    // ============================================================

    class ImageProviderManager final {
    public:
        struct Entry {
            std::string tag;
            std::shared_ptr<ImageProvider> provider;
            ImageProviderConfig config;
        };
    public:
        ImageProviderManager(const EnvSpec& env_spec, std::shared_ptr<Notifier> notifier);

        void CreateAndRegister(const ConfigData& config_data, std::shared_ptr<Agent> agent, std::shared_ptr<const Runner> runner, const std::string& key_prefix = "metrics.image.");
        void CreateAndRegister(const std::string& tag, const ImageProviderConfig& config, std::shared_ptr<Agent> agent, std::shared_ptr<const Runner> runner);
        void Remove(const std::string& tag, std::shared_ptr<Notifier> notifier);
        std::vector<Entry> List() const;
    private:
        struct ManageEntry {
            Entry public_data;
            NotifierFunction detach;
        };
    private:
        ImageProviderFactory factory_;
        const EnvSpec env_spec_;
        const std::shared_ptr<Notifier> notifier_;
        std::unordered_map<std::string, ManageEntry> registry_;
    };

} // namespace anet::rl