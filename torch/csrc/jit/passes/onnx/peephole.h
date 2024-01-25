#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

bool DropONNXUselessCasts(const std::shared_ptr<Graph>& graph);

void PeepholeOptimizeONNX(
    std::shared_ptr<Graph>& graph,
    int opset_version,
    bool fixed_batch_size);

} // namespace jit
} // namespace torch
