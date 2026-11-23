#include "anet/common.hpp"
using namespace anet;


// =============================================================
// TensorSpec
// =============================================================

static std::string SpaceTypeToString(SpaceType type)
{
    switch (type) {
    case SpaceType::Vector: return "Vector";
    case SpaceType::Grid: return "Grid";
    case SpaceType::Sequence: return "Sequence";
    default: return "Unknown";
    }
}

anet::json TensorSpec::ToJson() const
{
    static constexpr char kScaleField[] = "scale";
    static constexpr char kStackAxisField[] = "stack_axis";
    anet::json j;
    j["type"] = SpaceTypeToString(type);
    j["shape"] = shape;
    j["dtype"] = std::string(c10::toString(dtype)); /// @todo LibTorchの型名変換を利用
    j["num_classes"] = num_classes;
    j["labels"] = labels;
    j["min_values"] = min_values;
    j["max_values"] = max_values;
    j[kScaleField] = scale == ValueScale::Scaled ? "Scaled" : "Unscaled";
    j[kStackAxisField] = has_stack_axis;
    return j;
}

std::string TensorSpec::ToString() const
{
    return ToJson().dump(2); // 2-space indent for pretty print
}
