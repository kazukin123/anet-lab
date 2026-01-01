// image.cpp

#include "anet/image.hpp"
#include "anet/profile.hpp"
#include "anet/log.hpp"
#include "anet/observers.hpp"
#include "anet/str_util.hpp"
#include "anet/metrics_logger.hpp"

using namespace anet::rl;
namespace LOG = anet::log;


// ============================================================
// ImageSource
// ============================================================

wxImage anet::ImageSource::Render(int width, int height) const {
	ProfileRange r("ImageSource::Render");

	wxImage src = RenderRaw();
	if (width < 0 && height < 0) return src;
	if (width < 0) width = src.GetWidth();
	if (height < 0) height = src.GetHeight();
	if (width == src.GetWidth() && height == src.GetHeight())
		return src;

	ProfileRange r1("ImageSource::Render.Scale");
	//return src.Scale(width, height, wxIMAGE_QUALITY_HIGH);
	return src.Scale(width, height, wxIMAGE_QUALITY_NORMAL);
}

void anet::ImageSource::SavePng(const std::string& filename, int width, int height) const {
	std::filesystem::path path(filename);
	if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
	wxImage img = Render(width, height);
	if (!img.IsOk()) { LOG::error() << "Render() returned invalid wxImage"; return; }
	img.SaveFile(filename, wxBITMAP_TYPE_PNG);
}


// ============================================================
// ImageProviderFactory
// ============================================================

static anet::HeatMapFlags ToFlag(const std::string& flag_str)
{
	//	HM_AutoNormX = 1 << 2,  // X軸の自動スケーリング
	//	HM_AutoNormY = 1 << 3,  // Y軸の自動スケーリング
	//	HM_MeanMode = 1 << 4,  // 平均モード
	//	HM_SumMode = 1 << 5,  // 合計モード
	//	HM_AutoScaleAxis = 1 << 6,  // 値軸(データ軸)のレンジ自動追従（NaN指定側のみEMA更新）
	//	HM_LogScaleAxis = 1 << 7,  // 値軸を対数スケール表示（log1p(|v|)の符号付）
	//	HM_ShowZeroLine = 1 << 8,  // 値0の位置に水平ラインを描画
	//	HM_FlipY = 1 << 9,  // Y軸方向を反転して描画（上が大値）

	if (flag_str == "LogScaleValue") return anet::HeatMapFlags::HM_LogScaleValue;
	if (flag_str == "AutoNormValue") return anet::HeatMapFlags::HM_AutoNormValue;
	if (flag_str == "AutoNormX") return anet::HeatMapFlags::HM_AutoNormX;
	if (flag_str == "AutoNormY") return anet::HeatMapFlags::HM_AutoNormY;
	if (flag_str == "MeanMode") return anet::HeatMapFlags::HM_MeanMode;
	if (flag_str == "SumMode") return anet::HeatMapFlags::HM_SumMode;
	if (flag_str == "AutoScaleAxis") return anet::HeatMapFlags::HM_AutoScaleAxis;
	if (flag_str == "LogScaleAxis") return anet::HeatMapFlags::HM_LogScaleAxis;
	if (flag_str == "ShowZeroLine") return anet::HeatMapFlags::HM_ShowZeroLine;
	if (flag_str == "FlipY") return anet::HeatMapFlags::HM_FlipY;
	return anet::HeatMapFlags::HM_None;
}

static uint32_t GetFlags(const ImageProviderConfig& config)
{
	uint64_t flags = 0;
	for (const auto& flag_str : config.flags) {
		flags |= ToFlag(flag_str);
	}
	return flags;
}

static std::shared_ptr<VectorProbe> MakeExperienceProbe(const ProbeConfig& probe_config, const EnvSpec& env_spec)
{
	//static constexpr const char* STATE_OBS = "experience.state.obs";
	//static constexpr const char* STATE_DONE = "experience.state.done";
	//static constexpr const char* STATE_TRUNCATED = "experience.state.truncated";
	//static constexpr const char* STATE_EPISODE_START = "experience.state.episode_start";
	//static constexpr const char* ACTION_ACTION = "experience.action.action";
	//static constexpr const char* REWARD = "experience.reward";
	//static constexpr const char* NEXT_STATE_OBS = "experience.next_state.obs";
	//static constexpr const char* NEXT_STATE_DONE = "experience.next_state.done";
	//static constexpr const char* NEXT_STATE_TRUNCATED = "experience.next_state.truncated";
	//static constexpr const char* NEXT_STATE_EPISODE_START = "experience.next_state.episode_start";

	auto key = probe_config.key;

	// spec準備
	const anet::rl::StateSpec* state_spec = nullptr;
	const anet::rl::ActionSpec* action_spec = nullptr;
	if (key == BatchExperience::STATE_OBS || key == BatchExperience::NEXT_STATE_OBS)
		state_spec = &env_spec.state_spec;
	if (key == BatchExperience::ACTION_ACTION)
		action_spec = &env_spec.action_spec;

	// BatchExperienceVectorProbe生成
	auto probe = std::make_shared<BatchExperienceVectorProbe>(
		key, probe_config.index, state_spec, action_spec);

	return probe;
}

static std::shared_ptr<VectorProbe> MakeAgentProbe(const ProbeConfig& probe_config, const EnvSpec& env_spec)
{
	//const std::string& key,
	//	int index,
	//	const anet::rl::StateSpec* state_spec = nullptr,
	//	const anet::rl::ActionSpec* action_spec = nullptr,
	//	AutoScaleMode auto_scale_mode = AutoScaleMode::DISABLE,
	//	std::optional<float> min_override = std::nullopt,
	//	std::optional<float> max_override = std::nullopt,

	//static constexpr const char* STATE_OBS = "experience.state.obs";
	//static constexpr const char* STATE_DONE = "experience.state.done";
	//static constexpr const char* STATE_TRUNCATED = "experience.state.truncated";
	//static constexpr const char* STATE_EPISODE_START = "experience.state.episode_start";
	//static constexpr const char* ACTION_ACTION = "experience.action.action";
	//static constexpr const char* REWARD = "experience.reward";
	//static constexpr const char* NEXT_STATE_OBS = "experience.next_state.obs";
	//static constexpr const char* NEXT_STATE_DONE = "experience.next_state.done";
	//static constexpr const char* NEXT_STATE_TRUNCATED = "experience.next_state.truncated";
	//static constexpr const char* NEXT_STATE_EPISODE_START = "experience.next_state.episode_start";

	auto key = probe_config.key;

	// spec準備
	const anet::rl::StateSpec* state_spec = nullptr;
	const anet::rl::ActionSpec* action_spec = nullptr;
	if (key == ReplayBuffer::STATE_OBS || key == ReplayBuffer::NEXT_STATE_OBS)
		state_spec = &env_spec.state_spec;
	if (key == ReplayBuffer::ACTION)
		action_spec = &env_spec.action_spec;

	//// AutoScaleMode
	//AgentTensorVectorProbe::AutoScaleMode auto_scale_mode = AgentTensorVectorProbe::AutoScaleMode::DISABLE;
	//if (probe_config.auto_scale_mode == kAutoScaleMode_PerStep)
	//	auto_scale_mode = AgentTensorVectorProbe::AutoScaleMode::PER_STEP;
	//else if (probe_config.auto_scale_mode == kAutoScaleMode_Global)
	//	auto_scale_mode = AgentTensorVectorProbe::AutoScaleMode::GLOBAL;

	// AgentTensorVectorProbe生成
	auto probe = std::make_shared<AgentTensorVectorProbe>(
		key, probe_config.index, state_spec, action_spec);

	return probe;
}

static std::shared_ptr<VectorProbe> MakeProbe(const ProbeConfig& config, const EnvSpec& env_spec)
{
	if (config.source == kProbeType_Experience)
		return MakeExperienceProbe(config, env_spec);
	if (config.source == kProbeType_Agent)
		return MakeAgentProbe(config, env_spec);
	return nullptr;
}

static std::shared_ptr<HeatMapVectorObserver> MakeHeatMapVectorObserver(
	const std::string& tag, const ImageProviderConfig& config, const EnvSpec& env_spec, std::shared_ptr<Agent> agent)
{
	// flags
	uint64_t flags = GetFlags(config);

	// observer_config
	HeatMapObserverConfig observer_config {
		config.heatmap.width,
		config.heatmap.height,
		config.interval,
		config.heatmap.max_points
	};

	// probe
	std::shared_ptr<VectorProbe> x_probe = MakeProbe(config.heatmap.probe.x, env_spec);
	std::shared_ptr<VectorProbe> y_probe = MakeProbe(config.heatmap.probe.y, env_spec);
	std::shared_ptr<VectorProbe> v_probe = MakeProbe(config.heatmap.probe.value, env_spec);
	ANET_CHECK(x_probe != nullptr);
	ANET_CHECK(y_probe != nullptr);
	ANET_CHECK(v_probe != nullptr);

	// HeatMapVectorObserver
	auto obs = std::make_shared<HeatMapVectorObserver>(tag, observer_config, x_probe, y_probe, v_probe);
	return obs;
}

static std::shared_ptr<SweepedHeatMapObserver> MakeSweepedHeatMapObserver(
	const std::string& tag, const ImageProviderConfig& config, const EnvSpec& env_spec, std::shared_ptr<Agent> agent)
{
	// flags
	uint32_t flags = GetFlags(config);

	// config
	SweepedHeatMapObserverConfig observer_config {
		config.interval,			// log_interval
		flags,						// flags
		config.sweep_obs.width,		// grid_width
		config.sweep_obs.height,	// gird_height
		config.image_width,
		config.image_height
	};

	// extractor
	anet::rl::ValueExtractFunction extractor;
	const auto& sweep_obs = config.sweep_obs;
	if (sweep_obs.extractor_name == "max") {
		extractor = [sweep_obs](
			const torch::Tensor& t, const std::unordered_set<std::string>& req)
			{
				auto ret = anet::rl::extractor::MaxIdxExtractor(t, req, sweep_obs.extractor_index);
				return ret;
			};
	} else if (sweep_obs.extractor_name == "std") {
		extractor = [sweep_obs](
			const torch::Tensor& t, const std::unordered_set<std::string>& req)
			{
				auto ret = anet::rl::extractor::StdIdxExtractor(t, req, sweep_obs.extractor_index);
				return ret;
			};
	} else if (sweep_obs.extractor_name == "min") {
		extractor = [sweep_obs](
			const torch::Tensor& t, const std::unordered_set<std::string>& req)
			{
				auto ret = anet::rl::extractor::MinIdxExtractor(t, req, sweep_obs.extractor_index);
				return ret;
			};
	} else if (sweep_obs.extractor_name == "argmax") {
		extractor = [sweep_obs](
			const torch::Tensor& t, const std::unordered_set<std::string>& req)
			{
				auto ret = anet::rl::extractor::ArgmaxIdxExtractor(t, req, sweep_obs.extractor_index);
				return ret;
			};
	} else {
		extractor = [sweep_obs](
			const torch::Tensor& t, const std::unordered_set<std::string>& req)
			{
				auto ret = anet::rl::extractor::MeanIdxExtractor(t, req, sweep_obs.extractor_index);
				return ret;
			};
	}

	// StateSweepProcessor
	auto processor = std::make_shared<anet::rl::StateSweepProcessor>(
		env_spec.state_spec,
		config.sweep_obs.x_index,  // x_index = x
		config.sweep_obs.y_index,  // y_index = y
		extractor
	);

	// TensorFunction
	std::optional<anet::TensorFunction> forward_func = agent->GetTensorFunction(config.sweep_obs.network_key);
	if (!forward_func.has_value()) {
		ANET_SYSTEM_ERROR("Failed to get TensorFunction: " << config.sweep_obs.network_key);
		return nullptr;
	}

	// SweepedHeatMapObserver
	auto obs = std::make_shared<SweepedHeatMapObserver>(tag, observer_config, processor, *forward_func, processor);
	return obs;
}

static std::shared_ptr<TimeHistogramObserver> MakeTimeHistogramObserver(const std::string& tag, const ImageProviderConfig& config, const EnvSpec& env_spec)
{
	// flags
	uint32_t flags = GetFlags(config);

	//enum class TimeFrameMode {
	//	Scale = 0,
	//	Overwrite,
	//	Scroll
	//}

	// config
	anet::TimeFrameMode frame_mode = anet::TimeFrameMode::Scale;
	if (config.histgram.frame_mode == kImageFrameMode_Scale) {
		frame_mode = anet::TimeFrameMode::Scale;
	} else if (config.histgram.frame_mode == kImageFrameMode_Overwrite) {
		frame_mode = anet::TimeFrameMode::Overwrite;
	} else if (config.histgram.frame_mode == kImageFrameMode_Scroll) {
		frame_mode = anet::TimeFrameMode::Scroll;
	}

	// observer_config
	anet::rl::TimeHistogramObserverConfig observer_config {
		config.histgram.bins,	// bins
		config.histgram.max_frames,
		config.image_height,
		config.image_width,
		frame_mode,			// mode
		flags,				// flags
		config.interval,	// log_interval
		config.histgram.frame_interval,	// frame_interval
		std::numeric_limits<float>::quiet_NaN(),		// base_min
		std::numeric_limits<float>::quiet_NaN(),		// base_max
		config.histgram.alpha
	};

	// probe
	std::shared_ptr<VectorProbe> value_probe = MakeProbe(config.histgram.probe.value, env_spec);
	ANET_CHECK(value_probe != nullptr);

	// SweepedHeatMapObserver
	auto obs = std::make_shared<TimeHistogramObserver>(tag, observer_config, value_probe);
	return obs;

	//struct TimeHistogramObserverConfig {
	//	int bins = 128;
	//	int max_frames = 1000;
	//	int image_height = -1;
	//	int image_width = -1;
	//	TimeFrameMode mode = TimeFrameMode::Scale;
	//	uint32_t flags = anet::HeatMapFlags::HM_Default;
	//	int log_interval = 100;
	//	int frame_interval = 10;
	//	float base_min = std::numeric_limits<float>::quiet_NaN();
	//	float base_max = std::numeric_limits<float>::quiet_NaN();
	//	float alpha = 1.0f;
	//};
}

template<typename T>
static ImageProviderBundle MakeBundle(std::shared_ptr<T> observer)
{
	auto observer_wp = std::weak_ptr<T>(observer);
	return {
		observer,
		[observer](auto n) { n->Attach(observer); },		// attach
		[observer_wp](auto n) { if (auto p = observer_wp.lock()) n->Detach(p); }	// detach
	};
}

ImageProviderBundle ImageProviderFactory::CreateImageProvider(const std::string& tag, ImageProviderConfig config, const EnvSpec& env_spec, std::shared_ptr<Agent>agent)
{
	if (config.type == kImageType_StateSweepedHeatMap) {
		auto observer = MakeSweepedHeatMapObserver(tag, config, env_spec, agent);
		return MakeBundle<SweepedHeatMapObserver>(observer);
	}
	if (config.type == kImageType_HeatMap) {
		auto observer = MakeHeatMapVectorObserver(tag, config, env_spec, agent);
		return MakeBundle<HeatMapVectorObserver>(observer);
	}
	if (config.type == kImageType_TimeHistgram) {
		auto observer = MakeTimeHistogramObserver(tag, config, env_spec);
		return MakeBundle<TimeHistogramObserver>(observer);
	}

	ANET_SYSTEM_ERROR("Unknown ImageProvider type: " << config.type << " config=" << config.ToString());
	return {};
}


// ============================================================
// ImageProviderManager
// ============================================================

ImageProviderManager::ImageProviderManager(const EnvSpec& env_spec, std::shared_ptr<Notifier> notifier)
	: env_spec_(env_spec), notifier_(notifier)
{
	;
}

void ImageProviderManager::CreateAndRegister(const std::string& tag, const ImageProviderConfig& config, std::shared_ptr<Agent> agent)
{
	// 重複チェック
	if (registry_.find(tag) != registry_.end()) {
		LOG::warn() << "ImageProvider tag '" << tag << "' already exists.";
		// 上書き
	}

	//  Factoryを使って生成 (Bundleを受け取る)
	auto bundle = factory_.CreateImageProvider(tag, config, env_spec_, agent);
	ANET_CHECK(bundle.image_provider);

	//  Notifierへの登録 (Attach)
	if (notifier_ && bundle.attach)
		bundle.attach(notifier_);

	// 内部管理エントリの作成
	ManageEntry entry { { tag, bundle.image_provider, config }, bundle.detach };

	// レジストリに保存
	registry_[tag] = std::move(entry);

	LOG::info() << "Registered ImageProvider: " << tag;
	MetricsLogger::Instance()->Log(tag, config);
}

static constexpr const char* CONFIG_KEY_METRICS_IMAGE_DEFAULT_PREFIX = "metrics.image.";
static constexpr const char* CONFIG_KEY_METRICS_IMAGE_PREFIX = "metrics.image.[";
static constexpr const char* CONFIG_KEY_METRICS_IMAGE_SUFFIX = "]";

void ImageProviderManager::CreateAndRegister(const anet::ConfigData& config_data, std::shared_ptr<Agent> agent, const std::string& key_prefix)
{
	// 一つのtagが複数行にわたるので一旦tag単位でまとめる
	std::unordered_map<std::string, std::unordered_map<std::string, std::string>> tag_map;
	std::unordered_map<std::string, std::string> default_map;

	auto config_map = config_data.Map();
	for (const auto& kv : config_map) {
		const std::string& config_key = kv.first;
		const std::string& config_value = kv.second;

		// 設定Keyからtagを抽出
		auto config_tag = anet::ExtractBetween(config_key, CONFIG_KEY_METRICS_IMAGE_PREFIX, CONFIG_KEY_METRICS_IMAGE_SUFFIX);
		if (config_tag.empty()) {
			// tagがなかったらデフォルト設定判定
			if (config_key.find(CONFIG_KEY_METRICS_IMAGE_DEFAULT_PREFIX) == 0) {
				auto default_key = anet::RemovePrefix(config_key, CONFIG_KEY_METRICS_IMAGE_DEFAULT_PREFIX);
				default_map[default_key] = config_value;
			}
			continue;
		}

		// サブキーを抽出
		auto pos = config_key.find(CONFIG_KEY_METRICS_IMAGE_SUFFIX);
		auto size = config_key.size();
		if ((pos + 2) >= config_key.size()) continue;
		auto config_sub_key = config_key.substr(pos + 2);
		if (config_sub_key.empty()) continue;

		// 集約Mapに保存
		tag_map[config_tag][config_sub_key] = config_value;
	}

	// デフォルトConfigData
	ConfigData default_config_data;
	for (const auto& kv : default_map) {
		const auto& config_sub_key = kv.first;
		const auto& config_value = kv.second;
		default_config_data.Set(config_sub_key, config_value);
	}

	// 取り出したConfig一式でtag単位で回す
	for (const auto& kv : tag_map) {
		const auto& tag = kv.first;
		const auto& config_map = kv.second;

		// デフォルトConfigDataをベースとして、対象tag分のConfigDataを作る
		ConfigData sub_config_data = default_config_data;
		for (const auto& kv2 : config_map) {
			const auto& config_sub_key = kv2.first;
			const auto& config_value = kv2.second;
			sub_config_data.Set(config_sub_key, config_value);
		}

		// tagとConfigDataを使って生成＆登録
		auto config_key_prefix = CONFIG_KEY_METRICS_IMAGE_PREFIX + tag + CONFIG_KEY_METRICS_IMAGE_SUFFIX;
		ImageProviderConfig provider_config(sub_config_data, config_key_prefix);
		this->CreateAndRegister(tag, provider_config, agent);
	}
}

void ImageProviderManager::Remove(const std::string& tag, std::shared_ptr<Notifier> notifier)
{
	// 対象が見つからないイレギュラー
	auto itr = registry_.find(tag);
	if (itr == registry_.end()) {
		ANET_ASSERT_MSG(false, "Remove target not found. tag=" << tag);		// 普通じゃないのでASSERTで落とす
		return;
	}

	// 登録時にFactoryが作ったDetach関数を実行
	if (notifier && itr->second.detach) {
		itr->second.detach(notifier);
	}

	// レジストリから削除
	registry_.erase(itr);

}

std::vector<ImageProviderManager::Entry> ImageProviderManager::List() const
{
	std::vector<ImageProviderManager::Entry> list;
	list.reserve(registry_.size());
	for (const auto& pair : registry_) {
		list.push_back(pair.second.public_data);
	}
	return list;
}

