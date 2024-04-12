import numpy as np
import torch
import torch.nn as nn
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

from .harness import DispatchTestCase

empty_ops = [
    (
        "empty_one_dimension",
        [1],
        None,
        None,
    ),
    (
        "empty_two_dimension",
        [1, 2],
        None,
        None,
    ),
    (
        "empty_three_dimension",
        [2, 3, 4],
        None,
        None,
    ),
    (
        "empty_one_dimension_dtype",
        [1],
        torch.float32,
        None,
    ),
    (
        "empty_two_dimension_dtype",
        [2, 3],
        torch.float32,
        None,
    ),
    (
        "empty_one_dimension_dtype_device",
        [1],
        torch.float32,
        "cuda",
    ),
    (
        "empty_two_dimension_dtype_device",
        [2, 3],
        torch.float32,
        "cuda",
    ),
]


class TestRandConverter(DispatchTestCase):
    @parameterized.expand(
        [(empty_op[0], empty_op[1], empty_op[2], empty_op[3]) for empty_op in empty_ops]
    )
    def test_empty(self, name, shape_or_input, data_type, device):
        class TestModule(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                # print("The size is====", shape_or_input, "function name is", name)
                shape_or_input[0] = x.shape[0]
                return torch.empty(shape_or_input)

        empty_model = TestModule()
        # cannot use self.run_test() since it expects input in form of tensor

        inputs = [torch.randint(1, 3, shape_or_input, dtype=torch.int32)]
        comparator_shape_dtype_device = (
            lambda x, y, check_dtype: x.shape == y.shape
            and (x.dtype == y.dtype if check_dtype else True)
            and (x.get_device() == y.get_device)
        )
        expected_ops = []
        if "device" in name:
            self.run_test_comparator(
                empty_model,
                inputs,
                expected_ops,
                [(comparator_shape_dtype_device, [True, True])],
                use_dynamo_tracer=True,
            )
        elif "dtype" in name:
            self.run_test_comparator(
                empty_model,
                inputs,
                expected_ops,
                [(comparator_shape_dtype_device, [True, False])],
                use_dynamo_tracer=True,
            )
        else:
            self.run_test_comparator(
                empty_model,
                inputs,
                expected_ops,
                [(comparator_shape_dtype_device, [False, False])],
                use_dynamo_tracer=True,
            )


if __name__ == "__main__":
    run_tests()
