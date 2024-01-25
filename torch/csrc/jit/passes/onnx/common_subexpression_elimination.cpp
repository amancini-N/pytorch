#include <torch/csrc/jit/passes/onnx/common_subexpression_elimination.h>

#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/node_hashing.h>
#include <torch/csrc/jit/jit_log.h>

#include <unordered_map>

namespace torch {
namespace jit {
namespace onnx_cse {
namespace {

namespace onnx {
using namespace ::c10::onnx;
}

struct CommonSubexpressionEliminator {
  CommonSubexpressionEliminator(std::shared_ptr<Graph> graph)
      : graph_(std::move(graph)) {}

  bool run(std::function<Node*(Node*)> parent_lookup_fn) {
    return run(graph_->block(), std::move(parent_lookup_fn));
  }

  bool run(Block* block, std::function<Node*(Node*)> parent_lookup_fn) {
    std::unordered_set<Node*, HashNode, EqualNode> subexprs;
    bool changed = false;
    for (auto it = block->nodes().begin(); it != block->nodes().end(); ++it) {
      auto node = *it;

      if (node->isNondeterministic()) {
        GRAPH_DEBUG("Node was skipped due to its non determinism:\n", *node);
        continue;
      }

      // We need to skip if a node is used as a return value of an if block
      bool have_to_skip = false;
      for (auto* output : node->outputs()) {
        auto output_uses = output->uses();
        for (auto use : output_uses) {
            have_to_skip |= use.user->kind() == prim::Return &&
                            use.user->owningBlock()->owningNode() &&
                            use.user->owningBlock()->owningNode()->kind() == onnx::If;
        }
      }
      if (have_to_skip) {
        GRAPH_DEBUG("Node was skipped as its output is used by an onnx::If subgraph output: \n", *node);
        continue;
      }

      if (!node->blocks().empty()) {
        // Traverse sub-blocks.
        for (auto block : node->blocks()) {
          changed |= run(block, [&](Node* n) {
            auto existing = subexprs.find(n);
            if (existing != subexprs.end()) {
              return *existing;
            }

            return parent_lookup_fn(n);
          });
        }

        continue;
      }

      // Check for CSE opportunities in the parent block.
      auto parent_lookup = parent_lookup_fn(node);
      auto g_out = node->owningGraph()->outputs();
      if (parent_lookup != nullptr) {
        GRAPH_UPDATE("Replacing\n", *node, "with\n", *parent_lookup);
        changed = true;
        node->replaceAllUsesWith(parent_lookup);
        it.destroyCurrent();
        continue;
      }

      // Check whether the same subexpression already exists.
      auto subit = subexprs.insert(node);
      if (!subit.second) {
        // Subexpression exists, replace the uses of node, and destroy it.
        auto existing = *subit.first;

        GRAPH_UPDATE("Replacing\n", *node, "with\n", *existing);
        changed = true;
        node->replaceAllUsesWith(existing);
        // Destroy the node.
        it.destroyCurrent();
      }
    }

    return changed;
  }

  private:
    std::shared_ptr<Graph> graph_;
};

}

bool EliminateONNXCommonSubexpression(const std::shared_ptr<Graph>& graph) {
  GRAPH_DUMP("Before CSE", graph);
  CommonSubexpressionEliminator cse(graph);
  return cse.run([](Node*) { return nullptr; });
}

}
}
}