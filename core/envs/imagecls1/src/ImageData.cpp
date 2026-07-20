#include "ImageData.hpp"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>

#include <wx/image.h>
#include <wx/init.h>

#include "anet/log.hpp"
#include "anet/env.hpp"
#include "anet/profile.hpp"

using namespace anet::img;
namespace LOG = anet::log;

namespace {

std::filesystem::path NormalizePath(const std::filesystem::path& path)
{
    if (path.empty()) return {};
    return std::filesystem::absolute(path).lexically_normal();
}

std::string ComparablePath(const std::filesystem::path& path)
{
    auto value = path.generic_string();
#ifdef _WIN32
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
#endif
    return value;
}

std::string CacheModeName(ImageCacheMode mode)
{
    switch (mode) {
    case ImageCacheMode::Auto: return "auto";
    case ImageCacheMode::None: return "none";
    case ImageCacheMode::FullRam: return "full_ram";
    }
    ANET_SYSTEM_ERROR("Unknown ImageCacheMode value.");
    return "unknown";
}

ImageCacheMode ParseCacheMode(const std::string& value)
{
    if (value == "auto") return ImageCacheMode::Auto;
    if (value == "none") return ImageCacheMode::None;
    if (value == "full_ram") return ImageCacheMode::FullRam;
    ANET_SYSTEM_ERROR("Invalid ImageDataset.cache.mode='" << value
        << "'. Expected auto, none, or full_ram.");
    return ImageCacheMode::None;
}

template <typename T>
void ReadDatasetField(
    const anet::ConfigData& config_data,
    const DatasetKey& key,
    const std::string& field,
    T& value)
{
    config_data.Read("ImageDataset." + field, value, value);
    config_data.Read("ImageDataset.[" + key + "]." + field, value, value);
}

void EnsureWxImageSupport()
{
    // wxImageのglobal初期化はprocess内で一度だけ行い、全decode workerで共有する。
    static std::once_flag flag;
    static bool initialized = false;
    std::call_once(flag, [] {
        initialized = wxInitialize();
        if (initialized && wxImage::FindHandler(wxBITMAP_TYPE_PNG) == nullptr) {
            wxInitAllImageHandlers();
        }
    });
    if (!initialized) {
        ANET_SYSTEM_ERROR("Failed to initialize wxImage support.");
    }
}

ImageManifest LoadManifest(const DatasetKey& key, const ImageDatasetConfig& config)
{
    ANET_PROFILE_SCOPE(ImageDataset_LoadManifest);

    // classes.txtを宣言順に読み、manifest全体で共有するclass indexを確定する。
    std::ifstream classes_file(config.classes_txt_path);
    if (!classes_file.is_open()) {
        ANET_SYSTEM_ERROR("Failed to open ImageDataset classes file. dataset_key='" << key
            << "' path='" << config.classes_txt_path.string() << "'");
    }

    ImageManifest manifest;
    std::map<std::string, int64_t> class_to_index;
    std::string line;
    int64_t line_number = 0;
    while (std::getline(classes_file, line)) {
        ++line_number;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        if (class_to_index.contains(line)) {
            ANET_SYSTEM_ERROR("Duplicate ImageDataset class. dataset_key='" << key
                << "' line=" << line_number << " class='" << line << "'");
        }
        class_to_index.emplace(line, static_cast<int64_t>(manifest.class_names.size()));
        manifest.class_names.push_back(line);
    }
    if (manifest.class_names.empty()) {
        ANET_SYSTEM_ERROR("ImageDataset classes file is empty. dataset_key='" << key
            << "' path='" << config.classes_txt_path.string() << "'");
    }

    // sample listをclass indexへ解決し、壊れた参照をdataset公開前に検出する。
    std::ifstream list_file(config.list_txt_path);
    if (!list_file.is_open()) {
        ANET_SYSTEM_ERROR("Failed to open ImageDataset list file. dataset_key='" << key
            << "' path='" << config.list_txt_path.string() << "'");
    }

    line_number = 0;
    while (std::getline(list_file, line)) {
        ++line_number;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        const auto separator = line.find_first_of("/\\");
        if (separator == std::string::npos || separator == 0 || separator + 1 >= line.size()) {
            ANET_SYSTEM_ERROR("Malformed ImageDataset list entry. dataset_key='" << key
                << "' line=" << line_number << " value='" << line << "'");
        }
        const auto class_name = line.substr(0, separator);
        const auto class_it = class_to_index.find(class_name);
        if (class_it == class_to_index.end()) {
            ANET_SYSTEM_ERROR("Unknown ImageDataset class. dataset_key='" << key
                << "' line=" << line_number << " class='" << class_name << "'");
        }
        manifest.paths.push_back((config.root_dir / (line + config.suffix)).lexically_normal());
        manifest.targets.push_back(class_it->second);
    }
    if (manifest.paths.empty()) {
        ANET_SYSTEM_ERROR("ImageDataset sample list is empty. dataset_key='" << key
            << "' path='" << config.list_txt_path.string() << "'");
    }
    return manifest;
}

uint64_t MixSeed(uint64_t seed, uint64_t a, uint64_t b)
{
    auto mix = [](uint64_t value) {
        value += 0x9e3779b97f4a7c15ULL;
        value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
        return value ^ (value >> 31);
    };
    return mix(seed ^ mix(a) ^ mix(b));
}

const std::set<std::string>& DatasetFields()
{
    static const std::set<std::string> fields = {
        "root_dir", "list_txt_path", "classes_txt_path", "suffix",
        "image_width", "image_height", "cache.mode", "cache.max_bytes",
    };
    return fields;
}

const std::set<std::string>& ImageClsEnvFields()
{
    static const std::set<std::string> fields = {
        "max_steps",
        "train.dataset_key",
        "train.augment.enabled",
        "train.augment.hflip_p",
        "train.augment.rrc_scale_min",
        "train.augment.rrc_scale_max",
        "train.augment.rrc_ratio_min",
        "train.augment.rrc_ratio_max",
        "eval.dataset_key",
        "eval.eval_window.mode",
        "eval.eval_window.rotating.size",
    };
    return fields;
}

void AuditImageClsEnvKeys(const anet::ConfigData& config_data, const std::string& config_prefix)
{
    // default tagとEnv固有tagの両方を走査し、ImageClsEnvが所有する設定だけを監査する。
    const std::string default_prefix = "ImageClsEnv.";
    const std::string override_prefix = config_prefix.empty() ? "" : config_prefix + ".";
    for (const auto& [key, value] : config_data.Map()) {
        (void)value;
        std::string field;
        bool active = false;
        if (key.starts_with(default_prefix)) {
            field = key.substr(default_prefix.size());
            active = true;
        } else if (!override_prefix.empty() && key.starts_with(override_prefix)) {
            field = key.substr(override_prefix.size());
            active = true;
        }
        if (!active) continue;

        // 現行schemaのfieldだけを許可し、旧schemaと各role配下のtypoをfail-fastする。
        if (ImageClsEnvFields().contains(field)) continue;
        ANET_SYSTEM_ERROR("Unknown or obsolete ImageClsEnv config key='" << key
            << "'. Use ImageDataset.* for shared dataset fields and ImageClsEnv.train.* or"
            << " ImageClsEnv.eval.* for role-specific fields.");
    }
}

} // namespace

ImageDatasetConfig ImageDatasetConfig::Resolve(const anet::ConfigData& config_data, const DatasetKey& key)
{
    if (key.empty()) {
        ANET_SYSTEM_ERROR("ImageDataset DatasetKey must not be empty.");
    }

    ImageDatasetConfig config;
    std::string root_dir;
    std::string list_txt_path;
    std::string classes_txt_path;
    std::string cache_mode = "auto";
    // catalog共通値の上へdataset-key固有値を重ね、1 dataset分の設定を確定する。
    ReadDatasetField(config_data, key, "root_dir", root_dir);
    ReadDatasetField(config_data, key, "list_txt_path", list_txt_path);
    ReadDatasetField(config_data, key, "classes_txt_path", classes_txt_path);
    ReadDatasetField(config_data, key, "suffix", config.suffix);
    ReadDatasetField(config_data, key, "image_width", config.image_width);
    ReadDatasetField(config_data, key, "image_height", config.image_height);
    ReadDatasetField(config_data, key, "cache.mode", cache_mode);
    ReadDatasetField(config_data, key, "cache.max_bytes", config.cache_max_bytes);

    // I/O開始前にpath、shape、cache上限の構築時検証を完了する。
    if (root_dir.empty() || list_txt_path.empty() || classes_txt_path.empty()) {
        ANET_SYSTEM_ERROR("Missing required ImageDataset path. dataset_key='" << key
            << "' root_dir='" << root_dir << "' list_txt_path='" << list_txt_path
            << "' classes_txt_path='" << classes_txt_path << "'");
    }
    if (config.image_width <= 0 || config.image_height <= 0) {
        ANET_SYSTEM_ERROR("Invalid ImageDataset image shape. dataset_key='" << key
            << "' width=" << config.image_width << " height=" << config.image_height
            << ". Expected positive values.");
    }
    if (config.cache_max_bytes == 0) {
        ANET_SYSTEM_ERROR("Invalid ImageDataset.cache.max_bytes=0. dataset_key='" << key
            << "'. Expected a positive uint64 value.");
    }

    // 登録済み設定との比較が実行directoryに依存しないよう絶対pathへ正規化する。
    config.root_dir = NormalizePath(root_dir);
    config.list_txt_path = NormalizePath(list_txt_path);
    config.classes_txt_path = NormalizePath(classes_txt_path);
    config.cache_mode = ParseCacheMode(cache_mode);
    return config;
}

bool ImageDatasetConfig::Equals(const ImageDatasetConfig& other, std::string* first_different_field) const
{
    const auto different = [&](const char* field) {
        if (first_different_field != nullptr) *first_different_field = field;
        return false;
    };
    if (ComparablePath(root_dir) != ComparablePath(other.root_dir)) return different("root_dir");
    if (ComparablePath(list_txt_path) != ComparablePath(other.list_txt_path)) return different("list_txt_path");
    if (ComparablePath(classes_txt_path) != ComparablePath(other.classes_txt_path)) return different("classes_txt_path");
    if (suffix != other.suffix) return different("suffix");
    if (image_width != other.image_width) return different("image_width");
    if (image_height != other.image_height) return different("image_height");
    if (cache_mode != other.cache_mode) return different("cache.mode");
    if (cache_max_bytes != other.cache_max_bytes) return different("cache.max_bytes");
    return true;
}

anet::ConfigData ImageDatasetConfig::ToConfigData(const DatasetKey& key) const
{
    // catalog共通値を含めて解決済みの1 datasetを、dataset-key固有スコープへ展開する。
    const auto prefix = "ImageDataset.[" + key + "].";
    anet::ConfigData config_data;
    config_data.Set(prefix + "root_dir", root_dir.string());
    config_data.Set(prefix + "list_txt_path", list_txt_path.string());
    config_data.Set(prefix + "classes_txt_path", classes_txt_path.string());
    config_data.Set(prefix + "suffix", suffix);
    config_data.Set(prefix + "image_width", image_width);
    config_data.Set(prefix + "image_height", image_height);
    config_data.Set(prefix + "cache.mode", CacheModeName(cache_mode));
    config_data.Set(prefix + "cache.max_bytes", cache_max_bytes);
    return config_data;
}

std::string ImageDatasetConfig::ToString() const
{
    std::ostringstream oss;
    oss << "{root_dir='" << root_dir.string() << "', list_txt_path='" << list_txt_path.string()
        << "', classes_txt_path='" << classes_txt_path.string() << "', suffix='" << suffix
        << "', image_width=" << image_width << ", image_height=" << image_height
        << ", cache.mode=" << CacheModeName(cache_mode)
        << ", cache.max_bytes=" << cache_max_bytes << "}";
    return oss.str();
}

ResolvedImageDatasetCatalog anet::img::ResolveImageDatasetCatalog(const anet::ConfigData& config_data)
{
    // catalogを一巡してdataset keyを収集し、未知fieldや壊れたkey構文を先に排除する。
    std::set<DatasetKey> keys;
    const std::string prefix = "ImageDataset.[";
    for (const auto& [config_key, value] : config_data.Map()) {
        (void)value;
        if (!config_key.starts_with("ImageDataset.")) continue;
        if (!config_key.starts_with(prefix)) {
            const auto field = config_key.substr(std::string("ImageDataset.").size());
            if (!DatasetFields().contains(field)) {
                ANET_SYSTEM_ERROR("Unknown ImageDataset config key='" << config_key << "'.");
            }
            continue;
        }
        const auto close = config_key.find(']', prefix.size());
        if (close == std::string::npos || close == prefix.size()
            || close + 1 >= config_key.size() || config_key[close + 1] != '.') {
            ANET_SYSTEM_ERROR("Malformed ImageDataset catalog key='" << config_key << "'");
        }
        const auto field = config_key.substr(close + 2);
        if (!DatasetFields().contains(field)) {
            ANET_SYSTEM_ERROR("Unknown ImageDataset catalog field. key='" << config_key
                << "' field='" << field << "'");
        }
        keys.insert(config_key.substr(prefix.size(), close - prefix.size()));
    }
    if (keys.empty()) {
        ANET_SYSTEM_ERROR("ImageDataset catalog has no declared DatasetKey. Expected ImageDataset.[key].*.");
    }

    // 全keyの解決が成功した場合だけ、呼出し側へ完成済みcatalogを返す。
    ResolvedImageDatasetCatalog catalog;
    for (const auto& key : keys) catalog.emplace(key, ImageDatasetConfig::Resolve(config_data, key));
    return catalog;
}

ImageDataset::ImageDataset(DatasetKey key, ImageDatasetConfig config)
    : key_(std::move(key)), config_(std::move(config)), manifest_(LoadManifest(key_, config_))
{
    // manifest確定後にentry単位のcache状態を同じ要素数で初期化する。
    entry_states_.resize(manifest_.paths.size(), EntryState::Empty);
    entry_failures_.resize(manifest_.paths.size());
}

uint64_t ImageDataset::RequiredPayloadBytes() const
{
    // FullRamが保持するNCHW uint8 payloadの必要量を、乗算ごとにoverflow確認して求める。
    uint64_t value = static_cast<uint64_t>(manifest_.paths.size());
    for (const auto factor : { uint64_t{3}, static_cast<uint64_t>(config_.image_height), static_cast<uint64_t>(config_.image_width) }) {
        if (factor != 0 && value > std::numeric_limits<uint64_t>::max() / factor) {
            ANET_SYSTEM_ERROR("ImageDataset payload size overflow. dataset_key='" << key_ << "'");
        }
        value *= factor;
    }
    return value;
}

void ImageDataset::PrepareCache()
{
    // 最初の呼出しだけがcache方式を決定し、並行呼出しは決定完了まで待機する。
    std::unique_lock lock(mutex_);
    while (cache_prepare_state_ == CachePrepareState::Preparing) condition_.wait(lock);
    if (cache_prepare_state_ == CachePrepareState::Ready) return;
    if (cache_prepare_state_ == CachePrepareState::Failed) std::rethrow_exception(cache_prepare_failure_);
    cache_prepare_state_ = CachePrepareState::Preparing;
    lock.unlock();

    try {
        // autoは上限内ならFullRamを選び、容量超過時は一度だけNoneへ固定する。
        const auto required_bytes = RequiredPayloadBytes();
        auto selected_mode = config_.cache_mode;
        if (selected_mode == ImageCacheMode::Auto) {
            selected_mode = required_bytes <= config_.cache_max_bytes
                ? ImageCacheMode::FullRam : ImageCacheMode::None;
            if (selected_mode == ImageCacheMode::None) {
                LOG::warn() << "ImageDataset auto cache disabled because payload exceeds cap. dataset_key='"
                    << key_ << "' required_bytes=" << required_bytes
                    << " max_bytes=" << config_.cache_max_bytes;
            }
        }

        torch::Tensor payload;
        if (selected_mode == ImageCacheMode::FullRam) {
            // 明示FullRamは確保不能をエラーにし、autoの確保失敗だけNoneへfallbackする。
            if (required_bytes > config_.cache_max_bytes) {
                ANET_SYSTEM_ERROR("ImageDataset full_ram cache exceeds cap. dataset_key='" << key_
                    << "' required_bytes=" << required_bytes << " max_bytes=" << config_.cache_max_bytes);
            }
            try {
                payload = torch::empty(
                    { static_cast<int64_t>(manifest_.paths.size()), 3, config_.image_height, config_.image_width },
                    torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
            } catch (const std::exception& e) {
                if (config_.cache_mode != ImageCacheMode::Auto) {
                    ANET_SYSTEM_ERROR("ImageDataset full_ram allocation failed. dataset_key='"
                        << key_ << "' required_bytes=" << required_bytes
                        << " max_bytes=" << config_.cache_max_bytes << " reason='" << e.what() << "'");
                }
                LOG::warn() << "ImageDataset auto cache allocation failed; using none. dataset_key='"
                    << key_ << "' required_bytes=" << required_bytes
                    << " max_bytes=" << config_.cache_max_bytes;
                selected_mode = ImageCacheMode::None;
            } catch (...) {
                if (config_.cache_mode != ImageCacheMode::Auto) {
                    ANET_SYSTEM_ERROR("ImageDataset full_ram allocation failed. dataset_key='"
                        << key_ << "' required_bytes=" << required_bytes
                        << " max_bytes=" << config_.cache_max_bytes << " reason='<unknown>'");
                }
                LOG::warn() << "ImageDataset auto cache allocation failed; using none. dataset_key='"
                    << key_ << "' required_bytes=" << required_bytes
                    << " max_bytes=" << config_.cache_max_bytes;
                selected_mode = ImageCacheMode::None;
            }
        }

        // 完成した選択とpayloadを一括公開し、待機中のreaderから部分状態を隠す。
        lock.lock();
        effective_cache_mode_ = selected_mode;
        cache_payload_ = std::move(payload);
        cache_prepare_state_ = CachePrepareState::Ready;
        lock.unlock();
        condition_.notify_all();
        LOG::info() << "ImageDataset cache ready. dataset_key='" << key_
            << "' requested_mode=" << CacheModeName(config_.cache_mode)
            << " effective_mode=" << CacheModeName(selected_mode)
            << " required_bytes=" << required_bytes
            << " retained_payload_bytes="
            << (selected_mode == ImageCacheMode::FullRam ? required_bytes : 0);
    } catch (...) {
        // 初回失敗を保持し、後続呼出しにも同じ例外を再送出する。
        lock.lock();
        cache_prepare_failure_ = std::current_exception();
        cache_prepare_state_ = CachePrepareState::Failed;
        lock.unlock();
        condition_.notify_all();
        std::rethrow_exception(cache_prepare_failure_);
    }
}

torch::Tensor ImageDataset::Decode(size_t index) const
{
    ANET_PROFILE_SCOPE(ImageDataset_Get_decode);
    // process共有のdecoderを初期化してから、manifestで組み立て済みのpathを読み込む。
    EnsureWxImageSupport();
    wxImage image;
    const auto path = manifest_.paths[index].string();
    if (!image.LoadFile(path)) {
        ANET_SYSTEM_ERROR("Failed to decode ImageDataset image. dataset_key='" << key_
            << "' index=" << index << " path='" << path << "'");
    }
    // dataset契約の固定shapeへ揃え、batch collate可能な画像だけを返す。
    if (image.GetWidth() != config_.image_width || image.GetHeight() != config_.image_height) {
        image = image.Scale(config_.image_width, config_.image_height, wxIMAGE_QUALITY_BILINEAR);
    }
    // wxImageの一時bufferをcloneして所有権を切り離し、CHW contiguousへ変換する。
    auto tensor = torch::from_blob(
        image.GetData(), { config_.image_height, config_.image_width, 3 }, torch::kUInt8).clone();
    return tensor.permute({ 2, 0, 1 }).contiguous();
}

torch::data::Example<> ImageDataset::Get(size_t index)
{
    ANET_PROFILE_SCOPE(cache_lookup);
    // cache状態へ触れる前にindexを検証し、初回ならcache方式を確定する。
    if (index >= manifest_.paths.size()) {
        ANET_SYSTEM_ERROR("ImageDataset index out of range. dataset_key='" << key_
            << "' index=" << index << " size=" << manifest_.paths.size());
    }
    PrepareCache();

    std::unique_lock lock(mutex_);
    if (effective_cache_mode_ == ImageCacheMode::None) {
        // NoCacheは共有状態を保持せず、呼出しごとにfresh tensorをdecodeする。
        lock.unlock();
        return { Decode(index), torch::tensor(manifest_.targets[index], torch::kInt64) };
    }
    // 同じentryのdecodeは1 workerに限定し、完了済みpayloadまたはsticky failureを共有する。
    while (entry_states_[index] == EntryState::Loading) condition_.wait(lock);
    if (entry_states_[index] == EntryState::Ready) {
        return { cache_payload_[static_cast<int64_t>(index)], torch::tensor(manifest_.targets[index], torch::kInt64) };
    }
    if (entry_states_[index] == EntryState::Failed) std::rethrow_exception(entry_failures_[index]);
    entry_states_[index] = EntryState::Loading;
    lock.unlock();

    try {
        // mutex外でdecodeし、成功後だけ予約済みcache slotへ結果を公開する。
        ANET_PROFILE_SCOPE_NEXT(cache_fill);
        auto decoded = Decode(index);
        lock.lock();
        cache_payload_[static_cast<int64_t>(index)].copy_(decoded);
        entry_states_[index] = EntryState::Ready;
        lock.unlock();
        condition_.notify_all();
        return { cache_payload_[static_cast<int64_t>(index)], torch::tensor(manifest_.targets[index], torch::kInt64) };
    } catch (...) {
        // entry単位の失敗を保持して待機workerを起こし、同じ失敗を再送出する。
        lock.lock();
        entry_failures_[index] = std::current_exception();
        entry_states_[index] = EntryState::Failed;
        lock.unlock();
        condition_.notify_all();
        std::rethrow_exception(entry_failures_[index]);
    }
}

ImageDatasetManager& ImageDatasetManager::Instance()
{
    static ImageDatasetManager instance;
    return instance;
}

void ImageDatasetManager::RegisterCatalog(const ResolvedImageDatasetCatalog& catalog)
{
    std::lock_guard lock(mutex_);
    size_t added = 0;
    // 既存keyとの競合をcatalog全件で先に検証し、失敗時は登録状態を一切変更しない。
    for (const auto& [key, config] : catalog) {
        const auto it = entries_.find(key);
        if (it == entries_.end()) continue;
        std::string field;
        if (!it->second->config.Equals(config, &field)) {
            ANET_SYSTEM_ERROR("ImageDataset catalog conflict. dataset_key='" << key
                << "' field='" << field << "' registered=" << it->second->config.ToString()
                << " requested=" << config.ToString());
        }
    }
    // 検証完了後に未登録entryだけをまとめて公開する。
    for (const auto& [key, config] : catalog) {
        if (entries_.contains(key)) continue;
        auto entry = std::make_shared<Entry>();
        entry->config = config;
        entries_.emplace(key, std::move(entry));
        ++added;
    }
    if (added > 0) {
        LOG::info() << "ImageDataset catalog registered. added=" << added
            << " registered_total=" << entries_.size();
    }
}

std::shared_ptr<ImageDataset> ImageDatasetManager::Acquire(const DatasetKey& key)
{
    std::shared_ptr<Entry> entry;
    {
        // keyごとに構築担当を1 threadへ限定し、他threadは完成または失敗を待つ。
        std::unique_lock lock(mutex_);
        const auto it = entries_.find(key);
        if (it == entries_.end()) {
            ANET_SYSTEM_ERROR("ImageDataset DatasetKey is not registered. dataset_key='" << key << "'");
        }
        entry = it->second;
        while (entry->state == State::Loading) entry->condition.wait(lock);
        if (entry->state == State::Ready) return entry->dataset;
        if (entry->state == State::Failed) std::rethrow_exception(entry->failure);
        entry->state = State::Loading;
    }

    try {
        // manifest I/Oをmanager mutex外で行い、成功したdatasetだけをentryへ公開する。
        auto dataset = std::make_shared<ImageDataset>(key, entry->config);
        std::lock_guard lock(mutex_);
        entry->dataset = std::move(dataset);
        entry->state = State::Ready;
        entry->condition.notify_all();
        LOG::info() << "ImageDataset ready. dataset_key='" << key
            << "' sample_count=" << entry->dataset->Size();
        return entry->dataset;
    } catch (...) {
        // 構築失敗をentryへ固定し、以後のAcquireにも同じ例外を返す。
        std::lock_guard lock(mutex_);
        entry->failure = std::current_exception();
        entry->state = State::Failed;
        entry->condition.notify_all();
        std::rethrow_exception(entry->failure);
    }
}

TrainImageDataSourceConfig::TrainImageDataSourceConfig(
    const anet::ConfigData& config_data, const std::string& config_prefix)
    : anet::Config(
        config_data,
        "ImageClsEnv.train",
        config_prefix.empty() ? "" : config_prefix + ".train")
{
    // train roleが所有するdataset参照とaugmentation設定だけを読み込む。
    AuditImageClsEnvKeys(config_data, config_prefix);
    ANET_READ_CONFIG(config_data, dataset_key);
    ANET_READ_CONFIG(config_data, augment.enabled);
    ANET_READ_CONFIG(config_data, augment.hflip_p);
    ANET_READ_CONFIG(config_data, augment.rrc_scale_min);
    ANET_READ_CONFIG(config_data, augment.rrc_scale_max);
    ANET_READ_CONFIG(config_data, augment.rrc_ratio_min);
    ANET_READ_CONFIG(config_data, augment.rrc_ratio_max);

    // dataset参照とaugmentation範囲をSource構築前に検証する。
    if (dataset_key.empty()) {
        ANET_SYSTEM_ERROR("ImageClsEnv.train.dataset_key must not be empty.");
    }
    if (augment.hflip_p < 0.0 || augment.hflip_p > 1.0) {
        ANET_SYSTEM_ERROR("Invalid ImageClsEnv.train.augment.hflip_p=" << augment.hflip_p
            << ". Expected [0, 1].");
    }
    if (augment.rrc_scale_min <= 0.0 || augment.rrc_scale_min > augment.rrc_scale_max
        || augment.rrc_scale_max > 1.0) {
        ANET_SYSTEM_ERROR("Invalid ImageClsEnv.train.augment.rrc_scale range. min="
            << augment.rrc_scale_min << " max=" << augment.rrc_scale_max);
    }
    if (augment.rrc_ratio_min <= 0.0 || augment.rrc_ratio_min > augment.rrc_ratio_max) {
        ANET_SYSTEM_ERROR("Invalid ImageClsEnv.train.augment.rrc_ratio range. min="
            << augment.rrc_ratio_min << " max=" << augment.rrc_ratio_max);
    }
}

EvalImageDataSourceConfig::EvalImageDataSourceConfig(
    const anet::ConfigData& config_data, const std::string& config_prefix)
    : anet::Config(
        config_data,
        "ImageClsEnv.eval",
        config_prefix.empty() ? "" : config_prefix + ".eval")
{
    // eval roleが所有するdataset参照とwindow設定だけを読み込む。
    AuditImageClsEnvKeys(config_data, config_prefix);
    ANET_READ_CONFIG(config_data, dataset_key);
    ANET_READ_CONFIG(config_data, eval_window.mode);
    ANET_READ_CONFIG(config_data, eval_window.rotating.size);

    // fullでもrotating用sizeを正値として保持し、mode切替時の未設定概念を持ち込まない。
    if (dataset_key.empty()) {
        ANET_SYSTEM_ERROR("ImageClsEnv.eval.dataset_key must not be empty.");
    }
    if (eval_window.mode != "full" && eval_window.mode != "rotating") {
        ANET_SYSTEM_ERROR("Invalid ImageClsEnv.eval.eval_window.mode='" << eval_window.mode
            << "'. Expected full or rotating.");
    }
    if (eval_window.rotating.size <= 0) {
        ANET_SYSTEM_ERROR("ImageClsEnv.eval.eval_window.rotating.size must be positive. actual="
            << eval_window.rotating.size);
    }
}

ImageDataSource::ImageDataSource(
    TrainImageDataSourceConfig config, int batch_size, anet::seed_t seed,
    int worker_type, int worker_threads)
    : ImageDataSource(
        Role::Train,
        std::move(config.dataset_key),
        config.augment.enabled,
        config.augment.hflip_p,
        config.augment.rrc_scale_min,
        config.augment.rrc_scale_max,
        config.augment.rrc_ratio_min,
        config.augment.rrc_ratio_max,
        "full",
        1,
        batch_size,
        seed,
        worker_type,
        worker_threads)
{
}

ImageDataSource::ImageDataSource(
    EvalImageDataSourceConfig config, int batch_size, anet::seed_t seed,
    int worker_type, int worker_threads)
    : ImageDataSource(
        Role::Eval,
        std::move(config.dataset_key),
        false,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
        std::move(config.eval_window.mode),
        config.eval_window.rotating.size,
        batch_size,
        seed,
        worker_type,
        worker_threads)
{
}

ImageDataSource::ImageDataSource(
    Role role,
    std::string dataset_key,
    bool augment_enabled,
    double augment_hflip_p,
    double augment_rrc_scale_min,
    double augment_rrc_scale_max,
    double augment_rrc_ratio_min,
    double augment_rrc_ratio_max,
    std::string eval_window_mode,
    int64_t eval_window_rotating_size,
    int batch_size,
    anet::seed_t seed,
    int worker_type,
    int worker_threads)
    : role_(role)
    , dataset_key_(std::move(dataset_key))
    , augment_enabled_(augment_enabled)
    , augment_hflip_p_(augment_hflip_p)
    , augment_rrc_scale_min_(augment_rrc_scale_min)
    , augment_rrc_scale_max_(augment_rrc_scale_max)
    , augment_rrc_ratio_min_(augment_rrc_ratio_min)
    , augment_rrc_ratio_max_(augment_rrc_ratio_max)
    , eval_window_mode_(std::move(eval_window_mode))
    , eval_window_rotating_size_(eval_window_rotating_size)
    , batch_size_(batch_size)
    , augment_seed_(MixSeed(seed, 0x6175676d656e74ULL, 0))
    , dataset_(ImageDatasetManager::Instance().Acquire(dataset_key_))
    , sampler_random_(MixSeed(seed, 0x73616d706c6572ULL, 0))
    , worker_type_(worker_type)
    , worker_threads_(worker_threads)
{
    // batchとworker指定を検証し、Source専有poolの実効worker数を一度だけ決定する。
    if (batch_size_ <= 0) {
        ANET_SYSTEM_ERROR("ImageDataSource batch_size must be positive. dataset_key='"
            << dataset_key_ << "' batch_size=" << batch_size_);
    }
    if (worker_type_ != anet::rl::WorkerType::AUTO
        && worker_type_ != anet::rl::WorkerType::SINGLE_THREAD
        && worker_type_ != anet::rl::WorkerType::THREAD_POOL) {
        ANET_SYSTEM_ERROR("Invalid env.worker_type=" << worker_type_
            << ". Expected AUTO(0), SINGLE_THREAD(1), or THREAD_POOL(2).");
    }
    const bool use_pool = worker_type_ == anet::rl::WorkerType::THREAD_POOL
        || (worker_type_ == anet::rl::WorkerType::AUTO && batch_size_ > 1);
    worker_count_ = use_pool
        ? anet::rl::WorkerThreadResolver::Resolve(worker_threads_, batch_size_) : 1;
    // rotating windowがdataset外を要求しないことをdataset取得後に確認する。
    if (role_ == Role::Eval && eval_window_mode_ == "rotating"
        && eval_window_rotating_size_ > static_cast<int64_t>(dataset_->Size())) {
        ANET_SYSTEM_ERROR("ImageClsEnv.eval.eval_window.rotating.size exceeds dataset size. dataset_key='"
            << dataset_key_ << "' size=" << eval_window_rotating_size_
            << " dataset_size=" << dataset_->Size());
    }
    // trainとrotating evalだけ順序をshuffleし、full evalはmanifest順を維持する。
    permutation_.resize(dataset_->Size());
    std::iota(permutation_.begin(), permutation_.end(), size_t{0});
    if (role_ == Role::Train || eval_window_mode_ == "rotating") {
        std::shuffle(permutation_.begin(), permutation_.end(), sampler_random_);
    }
}

ImageDataSource::~ImageDataSource()
{
    Shutdown();
}

ImageDataSource::SampleSelection ImageDataSource::SelectTrainBatch()
{
    // 全laneへ実sampleを割り当て、dataset末尾を跨いでもbatch sizeを厳密に満たす。
    SampleSelection selection;
    selection.indices.reserve(batch_size_);
    selection.epoch_tags.reserve(batch_size_);
    for (int i = 0; i < batch_size_; ++i) {
        selection.indices.push_back(permutation_[cursor_]);
        selection.epoch_tags.push_back(cycle_);
        ++cursor_;
        if (cursor_ == permutation_.size()) {
            // dataset一周をepochとして数え、次epoch用の順序を直ちに作り直す。
            cursor_ = 0;
            ++cycle_;
            ++selection.completed_cycles;
            std::shuffle(permutation_.begin(), permutation_.end(), sampler_random_);
        }
    }
    selection.valid_count = batch_size_;
    return selection;
}

ImageDataSource::SampleSelection ImageDataSource::SelectEvalBatch()
{
    // fullはdataset全体、rotatingは指定sizeを1 windowとして今回の有効件数を求める。
    const int64_t window_size = eval_window_mode_ == "full"
        ? static_cast<int64_t>(dataset_->Size()) : eval_window_rotating_size_;
    const int64_t valid_count = std::min<int64_t>(batch_size_, window_size - eval_window_progress_);

    SampleSelection selection;
    selection.valid_count = valid_count;
    selection.indices.reserve(batch_size_);
    selection.epoch_tags.reserve(batch_size_);
    // 有効sampleだけcursorを進め、dataset周回とwindow終端を別々に記録する。
    for (int64_t i = 0; i < valid_count; ++i) {
        const size_t index = eval_window_mode_ == "full" ? cursor_ : permutation_[cursor_];
        selection.indices.push_back(index);
        selection.epoch_tags.push_back(cycle_);
        ++cursor_;
        if (cursor_ == dataset_->Size()) {
            cursor_ = 0;
            ++cycle_;
            ++selection.completed_cycles;
            if (eval_window_mode_ == "rotating") {
                std::shuffle(permutation_.begin(), permutation_.end(), sampler_random_);
            }
        }
    }
    eval_window_progress_ += valid_count;
    selection.window_end = eval_window_progress_ == window_size;
    if (selection.window_end) eval_window_progress_ = 0;

    // window末尾の不足laneは最後の有効sampleで埋め、valid_countで集計対象から除外する。
    const auto padding_index = selection.indices.back();
    const auto padding_epoch = selection.epoch_tags.back();
    while (static_cast<int>(selection.indices.size()) < batch_size_) {
        selection.indices.push_back(padding_index);
        selection.epoch_tags.push_back(padding_epoch);
    }
    return selection;
}

torch::Tensor ImageDataSource::ApplyRandomResizedCrop(const torch::Tensor& image, std::mt19937_64& random)
{
    const int64_t in_h = image.size(1);
    const int64_t in_w = image.size(2);
    const int64_t out_h = dataset_->GetConfig().image_height;
    const int64_t out_w = dataset_->GetConfig().image_width;
    const double area = static_cast<double>(in_h * in_w);
    std::uniform_real_distribution<double> scale(augment_rrc_scale_min_, augment_rrc_scale_max_);
    std::uniform_real_distribution<double> ratio(
        std::log(augment_rrc_ratio_min_), std::log(augment_rrc_ratio_max_));

    // torchvision相当の候補を最大10回試し、成立しなければ元画像全体を採用する。
    int64_t crop_h = in_h;
    int64_t crop_w = in_w;
    int64_t top = 0;
    int64_t left = 0;
    for (int attempt = 0; attempt < 10; ++attempt) {
        const double target_area = area * scale(random);
        const double aspect_ratio = std::exp(ratio(random));
        const auto candidate_w = static_cast<int64_t>(std::round(std::sqrt(target_area * aspect_ratio)));
        const auto candidate_h = static_cast<int64_t>(std::round(std::sqrt(target_area / aspect_ratio)));
        if (candidate_w <= 0 || candidate_w > in_w || candidate_h <= 0 || candidate_h > in_h) continue;
        crop_w = candidate_w;
        crop_h = candidate_h;
        std::uniform_int_distribution<int64_t> top_dist(0, in_h - crop_h);
        std::uniform_int_distribution<int64_t> left_dist(0, in_w - crop_w);
        top = top_dist(random);
        left = left_dist(random);
        break;
    }
    // 選択領域を切り出し、必要な場合だけ契約shapeへbilinear resizeする。
    auto cropped = image.index({
        torch::indexing::Slice(), torch::indexing::Slice(top, top + crop_h),
        torch::indexing::Slice(left, left + crop_w) });
    if (crop_h == out_h && crop_w == out_w) return cropped.contiguous();
    auto resized = torch::nn::functional::interpolate(
        cropped.unsqueeze(0).to(torch::kFloat32),
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{ out_h, out_w }).mode(torch::kBilinear).align_corners(false));
    return resized.squeeze(0).round().clamp(0, 255).to(torch::kUInt8).contiguous();
}

torch::Tensor ImageDataSource::ApplyAugment(
    const torch::Tensor& image, uint64_t epoch_tag, size_t dataset_index)
{
    if (!augment_enabled_ || role_ != Role::Train) return image;
    // epochとdataset indexから乱数列を再構成し、worker実行順に依存しないaugmentにする。
    std::mt19937_64 random(MixSeed(augment_seed_, epoch_tag, dataset_index));
    auto result = ApplyRandomResizedCrop(image, random);
    std::bernoulli_distribution horizontal_flip(augment_hflip_p_);
    if (horizontal_flip(random)) result = result.flip({ 2 });
    return result.contiguous();
}

ImageBatch ImageDataSource::NextBatch()
{
    // Source専有samplerから、このbatchが採点するindexとcycle metadataを確定する。
    ANET_PROFILE_SCOPE(sample);
    dataset_->PrepareCache();
    auto selection = role_ == Role::Train ? SelectTrainBatch() : SelectEvalBatch();

    std::vector<size_t> unique_indices;
    std::unordered_map<size_t, size_t> unique_positions;
    // eval末尾paddingを含む重複indexは一度だけdecodeし、slot組立時に再利用する。
    for (const auto index : selection.indices) {
        if (unique_positions.contains(index)) continue;
        unique_positions.emplace(index, unique_indices.size());
        unique_indices.push_back(index);
    }

    // padding等の重複indexをまとめ、同期またはSource専有poolでdecode/cache lookupする。
    ANET_PROFILE_SCOPE_NEXT(decode_cache);
    std::vector<std::optional<torch::data::Example<>>> decoded(unique_indices.size());
    const bool use_pool = worker_type_ == anet::rl::WorkerType::THREAD_POOL
        || (worker_type_ == anet::rl::WorkerType::AUTO && batch_size_ > 1);
    if (use_pool && decode_pool_ == nullptr) {
        decode_pool_ = std::make_shared<anet::PinnedThreadPool>(worker_count_);
    }
    if (decode_pool_ == nullptr) {
        for (size_t i = 0; i < unique_indices.size(); ++i) {
            decoded[i] = dataset_->Get(unique_indices[i]);
        }
    } else {
        // worker例外は最初の1件を呼出しthreadへ戻し、部分batchを公開しない。
        std::mutex failure_mutex;
        std::exception_ptr failure;
        for (size_t i = 0; i < unique_indices.size(); ++i) {
            decode_pool_->Enqueue(static_cast<int>(i % static_cast<size_t>(worker_count_)),
                [&, i] {
                    ANET_PROFILE_SCOPE_FULL(decode, "ImageDataSource::NextBatch.decode");
                    try {
                        decoded[i] = dataset_->Get(unique_indices[i]);
                    } catch (...) {
                        std::lock_guard lock(failure_mutex);
                        if (failure == nullptr) failure = std::current_exception();
                    }
                });
        }
        decode_pool_->WaitAll();
        if (failure != nullptr) std::rethrow_exception(failure);
    }

    // slotごとのepoch/indexから再現可能なaugmentを適用し、targetと順序を揃える。
    ANET_PROFILE_SCOPE_NEXT(augment);
    std::vector<torch::Tensor> images;
    std::vector<int64_t> targets;
    images.reserve(batch_size_);
    targets.reserve(batch_size_);
    for (int i = 0; i < batch_size_; ++i) {
        const auto index = selection.indices[static_cast<size_t>(i)];
        const auto& example = decoded[unique_positions.at(index)].value();
        images.push_back(ApplyAugment(example.data, selection.epoch_tags[static_cast<size_t>(i)], index));
        targets.push_back(example.target.item<int64_t>());
    }

    // 過去batchとstorageを共有しないfresh Tensorへcollateして所有権を渡す。
    ANET_PROFILE_SCOPE_NEXT(collate);
    return ImageBatch{
        .grid = torch::stack(images).contiguous(),
        .targets = torch::tensor(targets, torch::TensorOptions().dtype(torch::kInt64)),
        .epoch_tags = std::move(selection.epoch_tags),
        .valid_count = selection.valid_count,
        .window_end = selection.window_end,
        .completed_cycles = selection.completed_cycles,
    };
}

void ImageDataSource::Shutdown()
{
    // Source専有workerを停止してから参照を解放し、破棄後のcallback実行を防ぐ。
    if (decode_pool_ != nullptr) {
        decode_pool_->Stop();
        decode_pool_.reset();
    }
}
