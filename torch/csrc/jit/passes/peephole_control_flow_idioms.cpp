#include <torch/csrc/jit/passes/peephole.h>
#include <torch/csrc/jit/passes/peephole_control_flow_idioms.h>

#include <ATen/core/jit_type.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/ir/ir_views.h>
#include <torch/csrc/jit/jit_log.h>

namespace torch {
namespace jit {

namespace {


bool removeInvariantTensorsFromLoop(Node* loop_node) {
  // In this pass, we check for unmutated outputs. If the input does not have mutation
  // inside the loop body, we can remove the input/output pair on the loop.
  bool changed = false;
  Block* loop_body = loop_node->blocks().at(0);
  for (size_t i_input = 2; i_input < loop_node->inputs().size();) {
    Value* i_param = loop_body->inputs().at(i_input - 1);
    Value* i_return = loop_body->outputs().at(i_input - 1);
    if (i_param == i_return) {
      loop_node->outputs().at(i_input - 2)->replaceAllUsesWith(
          loop_node->inputs().at(i_input));
      loop_body->eraseInput(i_input - 1);
      loop_body->eraseOutput(i_input - 1);
      loop_node->removeInput(i_input);
      loop_node->eraseOutput(i_input - 2);
      changed = true;
    } else {
      ++i_input;
    }

  }
  return changed;
}

// bool testAndRemoveConditionalInvariantsFromLoop(Node* loop_node) {
//   // Here's a tricky one: we want to check whether a loop input is altered inside an if statement.
//   // We'll restrict this to is/isnot None inputs for now. If the input is only used inside the if condition,
//   // it means the None input won't be altered in any iteration of the loop and it can easily removed from the if block.
//   bool changed = false;
//   for (Value* block_input : loop_node->blocks().at(0)->inputs()) {
//     if (block_input->type()->cast<NoneType>()) {
//       std::unordered_set<Node*> if_nodes_using_input;
//       for (auto input_use: block_input->uses()) {
//         bool invariant_input = true;
//         Node* user_node = input_use.user;
//         bool is_isnot_node = (user_node->kind() == aten::__is__ || user_node->kind() == aten::__isnot__) && 
//           ((user_node->inputs().at(0)->node() == prim::Constant && user_node->inputs().at(0)->type()->cast<NoneType>() != nullptr) ||
//            (user_node->inputs().at(1)->node() == prim::Constant && user_node->inputs().at(1)->type()->cast<NoneType>() != nullptr));
//         if (is_isnot_node) {
//           for (auto is_isnot_use: user_node->output()->uses()) {
//             Node* if_node = is_isnot_use.user;
//             if (if_node->kind() == prim::If) {
//               if_nodes_using_input.insert(if_node);
//             } else {
//               invariant_input = false;
//             }
//           }
//         } else {
//           Node* owning_node = user_node->owningBlock()->owningNode();
//           auto found_if = if_nodes_using_input.find(owning_node);
//           invariant_input &= found_if != if_nodes_using_input.end();
//         }
//       }

//     }
//   }

//   return changed;
// }

}

struct PeepholeOptimizeForLoopIdiomsImpl {
  // NOLINTNEXTLINE(modernize-pass-by-value)
  PeepholeOptimizeForLoopIdiomsImpl(const std::shared_ptr<Graph>& graph)
      : graph_(graph) {}

  bool run() {
    return optimizeBlock(graph_->block());
  }

  bool optimizeBlock(Block* block) {
    bool changed = false;
    for (auto it = block->nodes().begin(); it != block->nodes().end(); ++it) {
      Node* node = *it;

      for (Block* sub_block : node->blocks()) {
        changed |= optimizeBlock(sub_block);
      }

      if (node->kind() == prim::Loop) {
        changed |= removeInvariantTensorsFromLoop(node);
      }
      
      // In this pass, we check for None inputs. If the input has mutations, we have to change
      // the input type from NoneTYpe to Optional, and copy all type metadata from mutation.
      for (size_t i_input = 0; i_input < node->inputs().size(); ++i_input) {
        auto input = node->inputs().at(i_input);

        if (input->type()->cast<NoneType>()) {

        }
      }
    }
    return changed;
  }

 private:
  std::shared_ptr<Graph> graph_;
};

bool PeepholeOptimizeControlFlowIdioms(const std::shared_ptr<Graph>& graph) {
  PeepholeOptimizeForLoopIdiomsImpl peephole(graph);
  bool changed = peephole.run();
  GRAPH_DUMP("After PeepholeOptimize: ", graph);
  return changed;
}

} // namespace jit
} // namespace torch
