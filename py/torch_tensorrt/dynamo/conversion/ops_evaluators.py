import logging
import operator
from typing import Dict, Sequence, Tuple, Union

import numpy as np
import torch
from torch.fx.node import Argument, Node, Target
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
    ConverterRegistry,
    dynamo_tensorrt_converter,
)
from torch_tensorrt.fx.types import TRTTensor
from torch_tensorrt.fx.utils import Frameworks, unified_dtype_converter

_LOGGER: logging.Logger = logging.getLogger(__name__)


def getitem_validator(getitem_node: Node) -> bool:
    from torch_tensorrt.dynamo.conversion._ConverterRegistry import DYNAMO_CONVERTERS

    # Getitem nodes can only be converted if their parent node also can
    return getitem_node.args[0] in DYNAMO_CONVERTERS


# TODO: Subsequent evaluators should be registered here with their own validators
@dynamo_tensorrt_converter(operator.getitem, capability_validator=getitem_validator)
@dynamo_tensorrt_converter(torch.ops.aten.detach.default)
def generic_evaluator(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    _LOGGER.debug(
        f"Evaluating {ConverterRegistry.qualified_name_or_str(target)} on object with name: {name}"
    )
    return target(*args)


@dynamo_tensorrt_converter(torch.ops.aten.arange.start_step)
def aten_ops_arange_start_step(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return np.arange(*args)


def empty_validator(empty_node: Node) -> bool:
    layout = empty_node.kwargs.get("layout", None)
    pin_memory = empty_node.kwargs.get("pin_memory", None)
    memory_format = empty_node.kwargs.get("memory_format", None)
    if layout is not None:
        _LOGGER.debug(f"Currently we don't support specifying layout, got {layout}.")
        return False
    if pin_memory is not None:
        _LOGGER.debug(
            f"Currently we don't support specifying pin_memory, got {pin_memory}."
        )
        return False
    if memory_format is not None:
        _LOGGER.debug(
            f"Currently we don't support specifying layout, got {memory_format}."
        )
        return False
    return True


@dynamo_tensorrt_converter(
    torch.ops.aten.empty.memory_format, capability_validator=empty_validator
)
def aten_ops_empty(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    if kwargs.get("device") is not None:
        return np.empty(*args[0], dtype=kwargs.get("dtype")).to(
            device=kwargs.get("device")
        )
    return np.empty(
        *args[0], dtype=unified_dtype_converter(kwargs.get("dtype"), Frameworks.NUMPY)
    )
