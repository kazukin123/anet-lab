#pragma once
#include <torch/torch.h>
#include <string>
#include <sstream>
#include "anet/common.hpp"

#define ANET_ENABLE_TENSOR_CHECK

namespace anet {

    // Device と dtype を一元管理する軽量コンテキスト
    struct TensorContext {
        torch::Device device;
        torch::Dtype float_dtype = torch::kFloat;
        torch::Dtype long_dtype = torch::kLong;
        torch::Dtype bool_dtype = torch::kBool;

        explicit TensorContext(torch::Device dev = torch::kCPU) : device(dev) {}

        inline torch::TensorOptions FloatOpt() const {
            return torch::TensorOptions().dtype(float_dtype).device(device);
        }
        inline torch::TensorOptions LongOpt() const {
            return torch::TensorOptions().dtype(long_dtype).device(device);
        }
        inline torch::TensorOptions BoolOpt() const {
            return torch::TensorOptions().dtype(bool_dtype).device(device);
        }
    };

    // =========================
    // 転送／基本ユーティリティ
    // =========================

    // 必要なときだけ .to(device) する安全版
    //inline torch::Tensor ToDevice(const torch::Tensor& t, const torch::Device& dev) {
    //    return (t.device() == dev) ? t : t.to(dev);
    //}

    // detach + to(device)
    inline torch::Tensor ToDeviceDetached(const torch::Tensor& t, const torch::Device& dev) {
        return t.detach().to(dev);
    }

    // スカラ取得（GPU→CPU同期を含む）※今回は現状維持する方針
    template<typename T = float>
    inline T ToScalar(const torch::Tensor& t) {
        return t.cpu().item<T>();
    }

    // =========================
    // テンソル生成ヘルパ
    // =========================

    inline torch::Tensor FullLike(const torch::Tensor& ref, float val, const TensorContext& ctx) {
        return torch::full(ref.sizes(), val, ctx.FloatOpt());
    }
    inline torch::Tensor OnesLike(const torch::Tensor& ref, const TensorContext& ctx) {
        return torch::ones(ref.sizes(), ctx.FloatOpt());
    }
    inline torch::Tensor ZerosLike(const torch::Tensor& ref, const TensorContext& ctx) {
        return torch::zeros(ref.sizes(), ctx.FloatOpt());
    }
    inline torch::Tensor BoolFullLike(const torch::Tensor& ref, bool val, const TensorContext& ctx) {
        return torch::full(ref.sizes(), val, ctx.BoolOpt());
    }

    // デバッグ用：テンソルを簡易文字列化
    inline std::string ToString(const torch::Tensor& t, int precision = 4) {
        std::ostringstream oss;
        oss.precision(precision);
        oss << t;
        return oss.str();
    }

    inline float itemf(const at::Tensor& t) {
        auto s = t.detach().to(torch::kCPU);
        TORCH_CHECK(s.numel() == 1, "itemf expects scalar, got numel=", s.numel(), " shape=", s.sizes());
        return s.item<float>();
    }

    inline void CheckDevice(const torch::Tensor& t, const torch::Device& expected, const char* name = "tensor")
    {
        if (t.device() != expected) {
            std::stringstream ss;
            ss << "Device mismatch in " << name << ": tensor=" << t.device()
                << " expected=" << expected;
            wxASSERT_MSG(false, ss.str());
        }
    }
    inline void CheckDeviceCPU(const torch::Tensor& t, const char* name = "tensor")
    {
        CheckDevice(t, torch::kCPU, name);
    }
    inline void CheckDeviceCUDA(const torch::Tensor& t, const char* name = "tensor")
    {
        CheckDevice(t, torch::kCUDA, name);
    }

#ifdef ANET_ENABLE_TENSOR_CHECK
#define ANET_CHECK_TENSOR_DEVICE(t, e, msg) anet::CheckDevice(t, e, msg)
#define ANET_CHECK_TENSOR_DEVICE_CPU(t, msg) anet::CheckDeviceCPU(t, msg)
#define ANET_CHECK_TENSOR_DEVICE_CUDA(t, msg) anet::CheckDeviceCUDA(t, msg)
#else   // ifdef ANET_ENABLE_TENSOR_CHECK
#define ANET_CHECK_TENSOR_DEVICE(t, e, msg) ((void)0)
#define ANET_CHECK_TENSOR_DEVICE_CPU(t, msg) ((void)0)
#define ANET_CHECK_TENSOR_DEVICE_CPU(t, msg) ((void)0)
#endif

} // namespace anet::util
