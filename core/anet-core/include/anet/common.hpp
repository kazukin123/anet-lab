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
#include "anet/tensor_util.hpp"

#if ANET_ENABLE_DEBUGINFO

#ifndef ANET_ENABLE_ASSERT
#define ANET_ENABLE_ASSERT 1
#endif

#else

#ifndef ANET_ENABLE_ASSERT
#define ANET_ENABLE_ASSERT 0
#endif

#endif


#if ANET_ENABLE_ASSERT

#define ANET_ASSERT(cond)                                                           \
    do {                                                                            \
        if (!(cond)) {                                                              \
            anet::ThrowError(__FILE__, __LINE__, "ANET_ASSERT failed: ", #cond);     \
        }                                                                           \
    } while (0)

#define ANET_ASSERT_MSG(cond, stream_args)                                          \
    do {                                                                            \
        if (!(cond)) {                                                              \
            std::ostringstream ss;                                                  \
            ss << stream_args;                                                      \
            anet::ThrowError(__FILE__, __LINE__, "ANET_ASSERT failed: ", #cond, ss.str());  \
        }                                                                           \
    } while (0)

#else
#define ANET_ASSERT(cond) do {} while (0)
#define ANET_ASSERT_MSG(cond, msg) do {} while (0)
#endif


#define ANET_CHECK(cond)                                                            \
    do {                                                                            \
        if (!(cond)) {                                                              \
            anet::ThrowError(__FILE__, __LINE__, "ANET_CHECK failed: ", #cond);     \
        }                                                                           \
    } while (0)

#define ANET_CHECK_MSG(cond, stream_args)                                           \
    do {                                                                            \
        if (!(cond)) {                                                              \
            std::ostringstream ss;                                                  \
            ss << stream_args;                                                      \
            anet::ThrowError(__FILE__, __LINE__, "ANET_CHECK failed: ", #cond, ss.str());  \
        }                                                                           \
    } while (0)

#define ANET_SYSTEM_ERROR(stream_args)                                              \
    do {                                                                            \
        std::ostringstream ss;                                                      \
        ss << stream_args;                                                          \
        anet::ThrowError(__FILE__, __LINE__, "System error: ", nullptr, ss.str());  \
    } while (0)


namespace anet {

    using json = nlohmann::json;

    void ThrowError(const char* file, int line, const char* prefix, const char* cond, const std::string& msg = "");

    json round_numbers(const json& j, int precision = 6);

    using TensorFunction = std::function<torch::Tensor(const torch::Tensor&)>;
    using TensorDictFunction = std::function<anet::TensorDict(const torch::Tensor&)>;

    class TensorFunctionProvider {
    public:
        virtual std::optional<TensorFunction> GetTensorFunction(const std::string& key) = 0;
        virtual ~TensorFunctionProvider() = default;
	};

    class TensorDictFunctionProvider {
    public:
        virtual std::optional<TensorDictFunction> GetTensorDictFunction(const std::string& key) { return std::nullopt; }
        virtual ~TensorDictFunctionProvider() = default;
    };

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
    //    virtual anet::json ToJson(bool recursive = true) = 0;
    //public:
    //    virtual ~Module() = default;
    //};

    class Module {
    public:
        virtual std::optional<float> GetScalar(const std::string& key, int64_t index = -1) const = 0;
        virtual std::optional<torch::Tensor> GetTensor(const std::string& key, int64_t index = -1) const = 0;
        virtual std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string& key, int64_t index = -1) const = 0;
        virtual ~Module() = default;
    };

    class ModuleBase : virtual public Module {
    public:
        virtual std::optional<float> GetScalar(const std::string& key, int64_t = -1) const override
            { return std::nullopt; }
        virtual std::optional<torch::Tensor> GetTensor(const std::string& key, int64_t = -1) const override
            { return std::nullopt; }
        virtual std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string& key, int64_t = -1) const override
            { return std::nullopt; }
        virtual ~ModuleBase() = default;
    };

    // ToString() を持つかどうか判定するメタ関数
    //template<typename T>
    //using has_ToString = decltype(std::declval<const T&>().ToString());

    // ToString()があれば何でもOKな operator<<
    //template<typename T,typename = has_ToString<T>>
    //std::ostream& operator<<(std::ostream& os, const T& v)
    //{
    //    return (os << v.ToString());
    //}

}   // namespace anet
