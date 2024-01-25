#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

namespace onnx_cse {

TORCH_API bool EliminateONNXCommonSubexpression(
    const std::shared_ptr<Graph>& graph);
}
}
} // namespace torch

