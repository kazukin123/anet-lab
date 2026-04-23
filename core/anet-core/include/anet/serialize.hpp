// anet/serialize.hpp

#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <type_traits>
#include <string_view>
#include <torch/torch.h>
#include "anet/common.hpp"
#include "anet/log.hpp"

namespace anet {

    class OutputArchive;
    class InputArchive;

    // =========================================================
    // インタフェース
    // =========================================================

    class Saveable {
    public:
        virtual int64_t Save(OutputArchive& archive) const = 0;
        virtual ~Saveable() = default;
    };

    class Loadable {
    public:
        virtual int64_t Load(InputArchive& archive) = 0;
        virtual ~Loadable() = default;
    };

	class Serializable : public Saveable, public Loadable {
    public:
        virtual ~Serializable() = default;
    };

    // =========================================================
    // OutputArchive (Serializer)
    // =========================================================

    class OutputArchive {
    public:
        // ファイルストリームでもstringstreamでも受け取れるように参照で保持
        explicit OutputArchive(std::ostream& stream, const std::string& name = "") : stream_(stream), name_(name) {}

        void CheckStream()
        {
            if (!stream_) {
                ANET_SYSTEM_ERROR("OutputArchive: Failed to write data.");
            }
        }

		std::string GetName() const { return name_; }

        // Primitive Types (int, float, double, bool, etc.)
        template <typename T> requires std::is_arithmetic_v<T>
        int64_t Write(const T& value)
        {
            stream_.write(reinterpret_cast<const char*>(&value), sizeof(T));
            CheckStream();
            return sizeof(T);
        }

        // std::string
        int64_t Write(const std::string& value)
        {
            // 長さを先に書き込む (int64_t固定)
            int64_t size = static_cast<int64_t>(value.size());
            int64_t written_bytes = Write(size);
            if (size > 0) {
                stream_.write(value.data(), size);
                CheckStream();
                written_bytes += size; // データ本体のバイト数を加算
            }
            return written_bytes;
        }

        // torch::Tensor
        int64_t Write(const torch::Tensor& tensor)
        {
            return WriteTorchObject(tensor);
        }

        // Any Saveable Object (Recursive)
        int64_t Write(const Saveable& obj)
        {
            return obj.Save(*this);
        }

        // Generic Torch Object (Module, Optimizer, Tensor)
        // torch::saveで保存可能なものは全てここで処理
        template <typename T>
        int64_t WriteTorchObject(const T& torch_obj)
        {
            // バッファリング
            std::ostringstream ss(std::ios::binary);

            // torch::save は ostream を要求するので ss を渡す
            try {
                torch::save(torch_obj, ss);
            } catch (const std::exception& e) {
                ANET_SYSTEM_ERROR("OutputArchive: torch::save failed: " << e.what());
            }

            // C++20: ゼロコピー参照を取得
            std::string_view view = ss.view();

            // サイズヘッダ + バイナリ本体 を書き込む
            int64_t size = view.size();
            ANET_LOG_DEBUG("size=" << size);
            int64_t written_bytes = Write(size); // 8バイトを書き込み
            stream_.write(view.data(), size);
            CheckStream();
            written_bytes += size; // データ本体のバイト数を加算

			return written_bytes;
        }

    private:
        std::ostream& stream_;
        const std::string name_;
    };

    // =========================================================
    // InputArchive (Deserializer)
    // =========================================================

    class InputArchive {
    public:
        explicit InputArchive(std::istream& stream, const std::string& name = "") : stream_(stream), name_(name) {}

        std::string GetName() const { return name_; }

        int64_t ReadBytes(char* dest, size_t size)
        {
            stream_.read(dest, size);

            if (stream_.eof()) {
                ANET_SYSTEM_ERROR("InputArchive: Unexpected End Of File (EOF). The file might be corrupted or truncated.");
            }
            if (stream_.fail() || stream_.bad()) {
                ANET_SYSTEM_ERROR("InputArchive: Stream read error.");
            }
            if (static_cast<size_t>(stream_.gcount()) != size) {
                ANET_SYSTEM_ERROR("InputArchive: Incomplete read. Expected "
                    << size << " bytes, but got " << stream_.gcount());
            }
            return size;
        }

        // Primitive Types
        template <typename T> requires std::is_arithmetic_v<T>
        int64_t Read(T& out_value)
        {
            ReadBytes(reinterpret_cast<char*>(&out_value), sizeof(T));
            return sizeof(T);
        }

        // std::string
        int64_t Read(std::string& out_value)
        {
            // サイズを読む
            int64_t size = 0;
            Read(size);
            if (size < 0) {
                ANET_SYSTEM_ERROR("InputArchive: Invalid string size (negative).");
            }
            if (size > 1024 * 1024 * 1024) {
                ANET_SYSTEM_ERROR("InputArchive: String size too large (>1GB). Potential corruption.");
            }

            // 本体をバイト列として読む
            out_value.resize(size);
            if (size > 0) {
                ReadBytes(&out_value[0], size);
            }
            return size;
        }

        // torch::Tensor
        int64_t Read(torch::Tensor& out_tensor)
        {
            return ReadTorchObject(out_tensor);
        }

        // Any Loadable Object (Recursive)
        int64_t Read(Loadable& obj)
        {
            return obj.Load(*this);
        }

        // Generic Torch Object
        template <typename T>
        int64_t ReadTorchObject(T& out_torch_obj)
        {
            // サイズを読み取る
            int64_t size = 0;
            Read(size);
            ANET_LOG_DEBUG("size=" << size);
            if (size < 0) {
                ANET_SYSTEM_ERROR("InputArchive: Invalid Torch object size (negative).");
            }

            // サイズ分だけバッファに読み込む
            std::string buffer;
            buffer.resize(size);
            if (size > 0) {
                ReadBytes(&buffer[0], size);
            }

            // torch::load は例外(c10::Error)を投げる可能性があるのでキャッチする
            try {
                std::istringstream ss(buffer, std::ios::binary);    /// @todo コピーを回避?
                torch::load(out_torch_obj, ss);
            } catch (const std::exception& e) {
                ANET_SYSTEM_ERROR("InputArchive: torch::load failed: " << e.what());
            }
            return size;
        }

    private:
        std::istream& stream_;
        const std::string name_;
    };


    // =========================================================
    // Archive File Header (Magic Number & Versioning)
    // =========================================================

    class ArchiveHeader : public Serializable {
    public:
        inline static const std::string kMagicWord = "ANET_ARCHIVE"; // typo修正: ARCIVE -> ARCHIVE
        inline static const int kFormatVersion = 1;
    public:
		explicit ArchiveHeader(const std::string& info = "") : info(info) {}

        std::string GetInfo() const { return info; }

        int64_t Save(OutputArchive& ar) const override
        {
			int64_t size = 0;
            size += ar.Write(kMagicWord);
            size += ar.Write(kFormatVersion);
            size += ar.Write(info);
            return size;
        }
        
        int64_t Load(InputArchive& ar) override
        {
            int64_t size = 0;

            // magic word
            std::string magic;
            size += ar.Read(magic);
            if (magic != kMagicWord) {
                ANET_SYSTEM_ERROR("ArchiveHeader: Magic word mismatch. (Expected: " << kMagicWord << ", Actual: " << magic << ")");
            }

            // format version
            int version;
            size += ar.Read(version);
            if (version != kFormatVersion) {
                ANET_SYSTEM_ERROR("ArchiveHeader: Format version mismatch. (Expected: " << kFormatVersion << ", Actual: " << version << ")");
            }

            // info
            size += ar.Read(info);
 
            // ret
            return size;
        }

    public:
        std::string info = "";
    };


}	// namespace anet
