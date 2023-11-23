# Owner(s): ["module: onnx"]
import onnxruntime
import pytorch_test_common

import torch
from pytorch_test_common import skipIfNoCuda
from torch.onnx import verification
from torch.onnx._globals import GLOBALS
from torch.testing._internal import common_utils


def _jit_graph_to_onnx_model(graph, operator_export_type, opset_version):
    r"""
    This function exports torch::jit::Graph object
    to serialized ONNX ModelProto.
    This function is for testing purpose.
    It only keeps the essential parts for IR graph conversions.
    It also does not interact with actual PyTorch modules nor
    PyTorch tensor inputs.
    """

    GLOBALS.export_onnx_opset_version = opset_version
    graph = torch.onnx.utils._optimize_graph(
        graph, operator_export_type, params_dict={}
    )
    proto, _, _, _ = graph._export_onnx(
        {},
        opset_version,
        {},
        False,
        operator_export_type,
        False,
        False,
        {},
        True,
        "",
        {},
    )
    return proto


class _TestJITIRToONNX:
    """Abstract base class for test cases.

    Intentionally not a sub-class of unittest.TestCase so that unittest / pytest
    don't run it directly. unitest.TestCase is mixed in as another base class when
    creating concrete sub-types. See MakeTestCase().
    """

    opset_version = -1  # Sub-classes must override
    ort_providers = ["CPUExecutionProvider"]
    check_shape = True
    check_dtype = True
    ignore_none = True  # True for tracing, and Flase for scripting

    def run_test(self, graph_ir, example_inputs):
        graph = torch._C.parse_ir(graph_ir)
        jit_outs = torch._C._jit_interpret_graph(graph, example_inputs)

        onnx_proto = _jit_graph_to_onnx_model(
            graph, torch.onnx.OperatorExportTypes.ONNX, self.opset_version
        )
        ort_sess = onnxruntime.InferenceSession(
            onnx_proto, providers=self.ort_providers
        )
        ort_outs = verification._run_onnx(ort_sess, example_inputs)

        options = verification.VerificationOptions(
            rtol=1e-3,
            atol=1e-7,
            check_shape=self.check_shape,
            check_dtype=self.check_dtype,
            ignore_none=self.ignore_none,
            acceptable_error_percentage=None,
        )
        verification._compare_onnx_pytorch_outputs(
            ort_outs,
            jit_outs,
            options,
        )

    def test_example_ir(self):
        graph_ir = """
        graph(%1 : Float(2, 3),
              %2 : Float(2, 3)):
          %3 : int = prim::Constant[value=1]()
          %4 : Float(2, 3) = aten::add(%1, %2, %3)
          return (%4)
        """
        a = torch.randn(2, 3)
        b = torch.randn(2, 3)
        self.run_test(graph_ir, (a, b))

    def test_add_sub_with_graph_inputs(self):
        for op in ["add", "sub", "rsub"]:
            graph_ir = f"""
            graph(%1 : Float(2, 3),
                  %2 : Float(2, 3),
                  %3 : int):
              %4 : Float(2, 3) = aten::{op}(%1, %2, %3)
              return (%4)
            """
            a = torch.randn(2, 3)
            b = torch.randn(2, 3)
            self.run_test(graph_ir, (a, b, 2))

    def test_native_layer_norm(self):
        graph_ir = """
        graph(%x : Float(2, 3, 2),
              %w : Float(3, 2),
              %b : Float(3, 2)):
          %5 : int = prim::Constant[value=3]()
          %6 : int = prim::Constant[value=2]()
          %7 : int[] = prim::ListConstruct(%5, %6)
          %10 : float = prim::Constant[value=1.0000000000000001e-05]()
          %11 : Float(2, 3, 2), %12 : Float(2, 1, 1), %13 : Float(2, 1, 1) = aten::native_layer_norm(%x, %7, %w, %b, %10)
          return (%11, %12, %13)
        """
        x = torch.randn(2, 3, 2)
        w = torch.randn(3, 2)
        b = torch.randn(3, 2)
        self.run_test(graph_ir, (x, w, b))

    def test_convolution(self):
        graph_ir = """
        graph(%1 : Tensor,
              %2 : Tensor):
          %3 : NoneType = prim::Constant()
          %4 : int[] = prim::Constant[value=[1, 1]]()
          %5 : int[] = prim::Constant[value=[0, 0]]()
          %6 : bool = prim::Constant[value=0]()
          %7 : int = prim::Constant[value=1]()
          %8 : Tensor = aten::convolution(%1, %2, %3, %4, %5, %4, %6, %5, %7)
          return (%8)
        """
        x = torch.randn(8, 1, 5, 5)
        w = torch.randn(4, 1, 3, 3)
        self.run_test(graph_ir, (x, w))

    def test_log_softmax(self):
        graph_ir = """
        graph(%x: Tensor):
          %half_to_float: bool = prim::Constant[value=0]()
          %dim: int = prim::Constant[value=1]()
          %y = aten::_log_softmax(%x, %dim, %half_to_float)
          return (%y)
        """
        x = torch.randn(5, 2)
        self.run_test(graph_ir, (x,))

    @skipIfNoCuda
    def test_log_softmax_half_to_float(self):
        graph_ir = """
        graph(%x: Tensor):
          %half_to_float: bool = prim::Constant[value=1]()
          %dim: int = prim::Constant[value=1]()
          %y = aten::_log_softmax(%x, %dim, %half_to_float)
          return (%y)
        """
        x = torch.randn(5, 2).half().to("cuda")
        self.run_test(graph_ir, (x,))

    def test_native_dropout(self):
        graph_ir = """
        graph(%1 : Float(2, 3)):
          %2 : float = prim::Constant[value=0.0]()
          %training : bool = prim::Constant[value=1]()
          %3 : Tensor, %4 : Tensor = aten::native_dropout(%1, %2, %training)
          return (%3, %4)
        """
        a = torch.randn(2, 3)
        self.run_test(graph_ir, (a,))

    def test_dict_with_contains_use_in_inner_module(self):
        graph_ir = """
        graph(%d.1 : Tensor):
          %19 : int = prim::Constant[value=1]()
          %3 : str = prim::Constant[value="a"]()
          %2 : str = prim::Constant[value="res"]()
          %5 : Dict(str, Tensor) = prim::DictConstruct(%3, %d.1)
          %v.1 : Dict(str, Dict(str, Tensor)) = prim::DictConstruct(%2, %5)
          %9 : Dict(str, Tensor) = aten::__getitem__(%v.1, %2)
          %x.1 : Tensor = aten::__getitem__(%9, %3)
          %15 : bool = aten::__contains__(%v.1, %2)
          %y : Tensor = prim::If(%15)
            block0():
              %y.1 : Tensor = aten::add(%x.1, %x.1, %19)
              -> (%y.1)
            block1():
              -> (%x.1)
          return (%y)
        """
        a = torch.randn(2, 3)
        self.run_test(graph_ir, (a,))

    def test_dict_with_format_use_in_inner_module(self):
        graph_ir = """
        graph(%d.1 : Tensor):
          %7 : str = prim::Constant[value="a"]()
          %6 : str = prim::Constant[value="res.1"]()
          %4 : str = prim::Constant[value="1"]()
          %3 : str = prim::Constant[value="res"]()
          %2 : str = prim::Constant[value="{}.{}"]()
          %key.1 : str = aten::format(%2, %3, %4)
          %9 : Dict(str, Tensor) = prim::DictConstruct(%7, %d.1)
          %v.1 : Dict(str, Dict(str, Tensor)) = prim::DictConstruct(%6, %9)
          %13 : Dict(str, Tensor) = aten::__getitem__(%v.1, %key.1)
          %x.1 : Tensor = aten::__getitem__(%13, %7)
          return (%x.1)
        """
        a = torch.randn(2, 3)
        self.run_test(graph_ir, (a,))

    def test_dict_with_format_number_use_in_inner_module(self):
        graph_ir = """
        graph(%d.1 : Tensor):
          %7 : str = prim::Constant[value="a"]()
          %6 : str = prim::Constant[value="res,1"]()
          %4 : int = prim::Constant[value=1]()
          %3 : str = prim::Constant[value="res"]()
          %2 : str = prim::Constant[value="{},{}"]()
          %key.1 : str = aten::format(%2, %3, %4)
          %9 : Dict(str, Tensor) = prim::DictConstruct(%7, %d.1)
          %v.1 : Dict(str, Dict(str, Tensor)) = prim::DictConstruct(%6, %9)
          %13 : Dict(str, Tensor) = aten::__getitem__(%v.1, %key.1)
          %x.1 : Tensor = aten::__getitem__(%13, %7)
          return (%x.1)
        """
        a = torch.randn(2, 3)
        self.run_test(graph_ir, (a,))

    def test_dict_with_setitem_use_in_inner_module(self):
        graph_ir = """
        graph(%d.1 : Tensor):
          %7 : str = prim::Constant[value="res"]()
          %3 : str = prim::Constant[value="a"]()
          %v.1 : Dict(str, Dict(str, Tensor)) = prim::DictConstruct()
          %5 : Dict(str, Tensor) = prim::DictConstruct(%3, %d.1)
          aten::_set_item(%v.1, %7, %5)
          %11 : Dict(str, Tensor) = aten::__getitem__(%v.1, %7)
          %x.1 : Tensor = aten::__getitem__(%11, %3)
          return (%x.1)
        """
        a = torch.randn(2, 3)
        self.run_test(graph_ir, (a,))

    def test_list_accumulating_dicts_in_loop(self):
        from typing import Dict, List

        class InnerModelFormat(torch.nn.Module):

            def forward(self, d: torch.Tensor) -> torch.Tensor:
                v: List[Dict[str, torch.Tensor]] = []
                for i in range(5):
                    v.append({"test": d*i})
                y = torch.zeros_like(d)
                for elem in v:
                    y = y + elem["test"]
                return y

        graph_ir = """
        graph(%d.1 : Tensor):
          %31 : int = prim::Constant[value=1]()
          %16 : NoneType = prim::Constant()
          %9 : str = prim::Constant[value="test"]() # /workspaces/pytorch/test/onnx/test_pytorch_jit_onnx.py:237:30
          %6 : bool = prim::Constant[value=1]() # /workspaces/pytorch/test/onnx/test_pytorch_jit_onnx.py:236:16
          %3 : int = prim::Constant[value=5]() # /workspaces/pytorch/test/onnx/test_pytorch_jit_onnx.py:236:31
          %v.1 : Dict(str, Tensor)[] = prim::ListConstruct()
          prim::Loop(%3, %6) # /workspaces/pytorch/test/onnx/test_pytorch_jit_onnx.py:236:16
            block0(%i.1 : int):
              %12 : Tensor = aten::mul(%d.1, %i.1) # /workspaces/pytorch/test/onnx/test_pytorch_jit_onnx.py:237:38
              %13 : Dict(str, Tensor) = prim::DictConstruct(%9, %12)
              %14 : Dict(str, Tensor)[] = aten::append(%v.1, %13) # /workspaces/pytorch/test/onnx/test_pytorch_jit_onnx.py:237:20
              -> (%6)
          %y.1 : Tensor = aten::zeros_like(%d.1, %16, %16, %16, %16, %16) # /workspaces/pytorch/test/onnx/test_pytorch_jit_onnx.py:238:20
          %23 : int = aten::len(%v.1) # /workspaces/pytorch/test/onnx/test_pytorch_jit_onnx.py:239:16
          %y : Tensor = prim::Loop(%23, %6, %y.1) # /workspaces/pytorch/test/onnx/test_pytorch_jit_onnx.py:239:16
            block0(%25 : int, %y.11 : Tensor):
              %elem.1 : Dict(str, Tensor) = aten::__getitem__(%v.1, %25) # /workspaces/pytorch/test/onnx/test_pytorch_jit_onnx.py:239:16
              %30 : Tensor = aten::__getitem__(%elem.1, %9) # /workspaces/pytorch/test/onnx/test_pytorch_jit_onnx.py:240:28
              %y.5 : Tensor = aten::add(%y.11, %30, %31) # /workspaces/pytorch/test/onnx/test_pytorch_jit_onnx.py:240:24
              -> (%6, %y.5)
          return (%y)
        """
        # graph_ir = str(torch.jit.script(InnerModelFormat()).graph)
        # print(graph_ir)
        a = torch.randn(2, 3)
        self.run_test(graph_ir, (a,))

    def test_lists_accumulated_on_outer_dict_in_loop(self):
        from typing import Dict, List

        class InnerModelFormat(torch.nn.Module):

            def forward(self, d: torch.Tensor) -> torch.Tensor:
                v: Dict[str, List[torch.Tensor]] = {"test": []}
                for i in range(5):
                    v["test"].append(d*i)
                y = torch.zeros_like(d)
                for elem in v["test"]:
                    y = y + elem
                return y

        graph_ir = """
        graph(%d.1 : Tensor):
          %33 : int = prim::Constant[value=1]()
          %18 : NoneType = prim::Constant()
          %8 : bool = prim::Constant[value=1]() # /workspaces/pytorch/test/onnx/test_pytorch_jit_onnx.py:279:16
          %2 : str = prim::Constant[value="test"]() # /workspaces/pytorch/test/onnx/test_pytorch_jit_onnx.py:278:52
          %5 : int = prim::Constant[value=5]() # /workspaces/pytorch/test/onnx/test_pytorch_jit_onnx.py:279:31
          %50 : Tensor[] = prim::ListConstruct()
          %v.1 : Dict(str, Tensor[]) = prim::DictConstruct(%2, %50)
          prim::Loop(%5, %8) # /workspaces/pytorch/test/onnx/test_pytorch_jit_onnx.py:279:16
            block0(%i.1 : int):
              %12 : Tensor[] = aten::__getitem__(%v.1, %2) # /workspaces/pytorch/test/onnx/test_pytorch_jit_onnx.py:280:20
              %15 : Tensor = aten::mul(%d.1, %i.1) # /workspaces/pytorch/test/onnx/test_pytorch_jit_onnx.py:280:37
              %16 : Tensor[] = aten::append(%12, %15) # /workspaces/pytorch/test/onnx/test_pytorch_jit_onnx.py:280:20
              -> (%8)
          %y.1 : Tensor = aten::zeros_like(%d.1, %18, %18, %18, %18, %18) # /workspaces/pytorch/test/onnx/test_pytorch_jit_onnx.py:281:20
          %26 : Tensor[] = aten::__getitem__(%v.1, %2) # /workspaces/pytorch/test/onnx/test_pytorch_jit_onnx.py:282:28
          %27 : int = aten::len(%26) # /workspaces/pytorch/test/onnx/test_pytorch_jit_onnx.py:282:16
          %y : Tensor = prim::Loop(%27, %8, %y.1) # /workspaces/pytorch/test/onnx/test_pytorch_jit_onnx.py:282:16
            block0(%29 : int, %y.11 : Tensor):
              %elem.1 : Tensor = aten::__getitem__(%26, %29) # /workspaces/pytorch/test/onnx/test_pytorch_jit_onnx.py:282:16
              %y.5 : Tensor = aten::add(%y.11, %elem.1, %33) # /workspaces/pytorch/test/onnx/test_pytorch_jit_onnx.py:283:24
              -> (%8, %y.5)
          return (%y)
        """
        # graph_ir = str(torch.jit.script(InnerModelFormat()).graph)
        # print(graph_ir)
        a = torch.randn(2, 3)
        self.run_test(graph_ir, (a,))

    def test_lists_accumulated_on_outer_dict_using_zip_in_loop(self):
        from typing import Dict, List
        from collections import namedtuple

        # class X:
        #     def __init__(self, test: torch.Tensor):
        #         self.test = test

        X = namedtuple("X", "test")

        class InnerModelFormat(torch.nn.Module):

            def forward(self, d: torch.Tensor) -> torch.Tensor:
                v: List[X] = []
                for i in range(5):
                    v.append(X(test=d*i))
                y = torch.zeros_like(d)
                for elem in v:
                    y = y + elem.test
                return y

        graph_ir = """
        graph(%d.1 : Tensor):
          %30 : int = prim::Constant[value=1]()
          %28 : int = prim::Constant[value=0]()
          %15 : NoneType = prim::Constant()
          %6 : bool = prim::Constant[value=1]() # /workspaces/pytorch/test/onnx/test_pytorch_jit_onnx.py:330:16
          %3 : int = prim::Constant[value=5]() # /workspaces/pytorch/test/onnx/test_pytorch_jit_onnx.py:330:31
          %v.1 : NamedTuple(test : Tensor)[] = prim::ListConstruct()
          prim::Loop(%3, %6) # /workspaces/pytorch/test/onnx/test_pytorch_jit_onnx.py:330:16
            block0(%i.1 : int):
              %11 : Tensor = aten::mul(%d.1, %i.1) # /workspaces/pytorch/test/onnx/test_pytorch_jit_onnx.py:331:36
              %12 : NamedTuple(test : Tensor) = prim::TupleConstruct(%11) # /workspaces/pytorch/test/onnx/test_pytorch_jit_onnx.py:331:29
              %13 : NamedTuple(test : Tensor)[] = aten::append(%v.1, %12) # /workspaces/pytorch/test/onnx/test_pytorch_jit_onnx.py:331:20
              -> (%6)
          %y.1 : Tensor = aten::zeros_like(%d.1, %15, %15, %15, %15, %15) # /workspaces/pytorch/test/onnx/test_pytorch_jit_onnx.py:332:20
          %22 : int = aten::len(%v.1) # /workspaces/pytorch/test/onnx/test_pytorch_jit_onnx.py:333:16
          %y : Tensor = prim::Loop(%22, %6, %y.1) # /workspaces/pytorch/test/onnx/test_pytorch_jit_onnx.py:333:16
            block0(%24 : int, %y.11 : Tensor):
              %elem.1 : NamedTuple(test : Tensor) = aten::__getitem__(%v.1, %24) # /workspaces/pytorch/test/onnx/test_pytorch_jit_onnx.py:333:16
              %29 : Tensor = prim::TupleIndex(%elem.1, %28)
              %y.5 : Tensor = aten::add(%y.11, %29, %30) # /workspaces/pytorch/test/onnx/test_pytorch_jit_onnx.py:334:24
              -> (%6, %y.5)
          return (%y)
        """
        # graph_ir = str(torch.jit.script(InnerModelFormat()).graph)
        # print(graph_ir)
        a = torch.randn(2, 3)
        self.run_test(graph_ir, (a,))

    def test_dict_accumulated_in_loop(self):
        from typing import Dict, List

        class InnerModelFormat(torch.nn.Module):

            def forward(self, d: torch.Tensor) -> torch.Tensor:
                v: Dict[str, torch.Tensor] = {}
                keys = ["x", "y"]
                ones = torch.ones_like(d)
                values = [ones, ones*2]
                for i in range(2):
                    v[keys[i]] = values[i]
                y = torch.zeros_like(d)
                for elem in v:
                    y = y + v[elem]
                return y

        graph_ir = """
        graph(%d.1 : Tensor):
          %47 : int = prim::Constant[value=1]()
          %20 : bool = prim::Constant[value=1]() # /workspaces/pytorch/test/onnx/test_pytorch_jit_onnx.py:282:16
          %7 : NoneType = prim::Constant()
          %4 : str = prim::Constant[value="y"]() # /workspaces/pytorch/test/onnx/test_pytorch_jit_onnx.py:279:29
          %3 : str = prim::Constant[value="x"]() # /workspaces/pytorch/test/onnx/test_pytorch_jit_onnx.py:279:24
          %15 : int = prim::Constant[value=2]() # /workspaces/pytorch/test/onnx/test_pytorch_jit_onnx.py:281:37
          %v.1 : Dict(str, Tensor) = prim::DictConstruct()
          %keys.1 : str[] = prim::ListConstruct(%3, %4)
          %ones.1 : Tensor = aten::ones_like(%d.1, %7, %7, %7, %7, %7) # /workspaces/pytorch/test/onnx/test_pytorch_jit_onnx.py:280:23
          %16 : Tensor = aten::mul(%ones.1, %15) # /workspaces/pytorch/test/onnx/test_pytorch_jit_onnx.py:281:32
          %values.1 : Tensor[] = prim::ListConstruct(%ones.1, %16)
          prim::Loop(%15, %20) # /workspaces/pytorch/test/onnx/test_pytorch_jit_onnx.py:282:16
            block0(%i.1 : int):
              %24 : Tensor = aten::__getitem__(%values.1, %i.1) # /workspaces/pytorch/test/onnx/test_pytorch_jit_onnx.py:283:33
              %28 : str = aten::__getitem__(%keys.1, %i.1) # /workspaces/pytorch/test/onnx/test_pytorch_jit_onnx.py:283:22
              aten::_set_item(%v.1, %28, %24) # /workspaces/pytorch/test/onnx/test_pytorch_jit_onnx.py:283:20
              -> (%20)
          %y.1 : Tensor = aten::zeros_like(%d.1, %7, %7, %7, %7, %7) # /workspaces/pytorch/test/onnx/test_pytorch_jit_onnx.py:284:20
          %38 : str[] = aten::keys(%v.1) # /workspaces/pytorch/test/onnx/test_pytorch_jit_onnx.py:285:16
          %39 : int = aten::len(%38) # /workspaces/pytorch/test/onnx/test_pytorch_jit_onnx.py:285:16
          %y : Tensor = prim::Loop(%39, %20, %y.1) # /workspaces/pytorch/test/onnx/test_pytorch_jit_onnx.py:285:16
            block0(%41 : int, %y.11 : Tensor):
              %elem.1 : str = aten::__getitem__(%38, %41) # /workspaces/pytorch/test/onnx/test_pytorch_jit_onnx.py:285:16
              %46 : Tensor = aten::__getitem__(%v.1, %elem.1) # /workspaces/pytorch/test/onnx/test_pytorch_jit_onnx.py:286:28
              %y.5 : Tensor = aten::add(%y.11, %46, %47) # /workspaces/pytorch/test/onnx/test_pytorch_jit_onnx.py:286:24
              -> (%20, %y.5)
          return (%y)
        """
        # graph_ir = str(torch.jit.script(InnerModelFormat()).graph)
        # print(graph_ir)
        a = torch.randn(2, 3)
        self.run_test(graph_ir, (a,))


def MakeTestCase(opset_version: int) -> type:
    name = f"TestJITIRToONNX_opset{opset_version}"
    return type(
        str(name),
        (pytorch_test_common.ExportTestCase,),
        dict(_TestJITIRToONNX.__dict__, opset_version=opset_version),
    )


TestJITIRToONNX_opset14 = MakeTestCase(14)

if __name__ == "__main__":
    common_utils.run_tests()
