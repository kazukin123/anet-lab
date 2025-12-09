#pragma once
#include <memory>
#include <optional>
#include <type_traits>

#include <string>
#include <sstream>
#include <stdexcept>
#include <nlohmann/json.hpp>
#include <torch/torch.h>
#include "anet/util.hpp"

#if ANET_ENABLE_DEBUGINFO
#ifndef ANET_ENABLE_ASSERT
#define ANET_ENABLE_ASSERT 1
#endif
#endif

#if ANET_ENABLE_ASSERT
#define ANET_ASSERT(cond)                                                      \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::stringstream ss;                                              \
            ss << "ANET_ASSERT failed: "                                       \
               << " | Condition: " << #cond                                    \
               << " | File: " << __FILE__ << ":" << __LINE__;                  \
            throw std::runtime_error(ss.str());                                \
        }                                                                      \
    } while (0)
#define ANET_ASSERT_MSG(cond, msg)                                             \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::stringstream ss;                                              \
            ss << "ANET_ASSERT failed: " << (msg)                              \
               << " | Condition: " << #cond                                    \
               << " | File: " << __FILE__ << ":" << __LINE__;                  \
            throw std::runtime_error(ss.str());                                \
        }                                                                      \
    } while (0)
#else
#define ANET_ASSERT(cond) do {} while (0)
#define ANET_ASSERT_MSG(cond, msg) do {} while (0)
#endif

namespace anet {

    using TensorFunction = std::function<torch::Tensor(const torch::Tensor&)>;

    //class Module : public std::enable_shared_from_this<Module> {
    //public:
    //    virtual std::string GetClassName() = 0;
    //    virtual std::string GetInstanceName() = 0;
    //    //virtual OrderedMap<std::string, Module>  GetChildlen() = 0;
    //public:
    //    virtual anet::OrderedMap<std::string, std::string> GetParameters() = 0;
    //    virtual anet::OrderedMap<std::string, std::vector<torch::Tensor>> GetTensorVector() = 0;

    //    virtual std::optional<std::string> GetParam(const std::string& key) = 0;
    //    virtual std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string key) = 0;
    //    virtual TensorFunction GetTensorFunction(const std::string& key) const = 0;
    //public:
    //    virtual void Load(std::istream& stream) const = 0;
    //    virtual void Save(std::ostream& stream) const = 0;
    //    virtual void Print(std::ostream& stream) const = 0;
    //    virtual nlohmann::json ToJson(bool recursive = true) = 0;
    //public:
    //    virtual ~Module() = default;
    //};

    class DataExporter {
    public:
        virtual std::optional<float> GetScalar(const std::string& key) const = 0;
        virtual std::optional<torch::Tensor> GetTensor(const std::string& key) const = 0;
        virtual std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string& key) const = 0;
        virtual ~DataExporter() = default;
    };

    class DataExporterBase {
    public:
        virtual std::optional<float> GetScalar(const std::string& key) const { return std::nullopt; }
        virtual std::optional<torch::Tensor> GetTensor(const std::string& key) const { return std::nullopt; }
        virtual std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string& key) const { return std::nullopt; }
        virtual ~DataExporterBase() = default;
    };

    // ToString() を持つかどうか判定するメタ関数
    template<typename T>
    using has_ToString = decltype(std::declval<const T&>().ToString());

    // ToString()があれば何でもOKな operator<<
    template<typename T,typename = has_ToString<T>>
    std::ostream& operator<<(std::ostream& os, const T& v)
    {
        return (os << v.ToString());
    }

}   // namespace anet
