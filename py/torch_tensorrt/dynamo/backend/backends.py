import logging
from typing import Sequence
import torch
from functools import partial
import torch._dynamo as td

from torch_tensorrt.dynamo import CompilationSettings
from torch_tensorrt.dynamo.lowering._decompositions import (
    get_decompositions,
)
from torch_tensorrt.dynamo.lowering._pre_aot_lowering import (
    pre_aot_substitutions,
)
from torch_tensorrt.dynamo.lowering._partition import (
    partition,
    get_submod_inputs,
)
from torch_tensorrt._Device import Device
from torch_tensorrt.dynamo.utils import parse_dynamo_kwargs
from torch_tensorrt.dynamo.conversion import (
    convert_module,
    repair_long_or_double_inputs,
)
from torch_tensorrt.dynamo.runtime import TorchTensorRTModule

from torch._functorch.aot_autograd import aot_module_simplified, make_boxed_compiler


logger = logging.getLogger(__name__)


@td.register_backend(name="torch_tensorrt")
def torch_tensorrt_backend(
    gm: torch.fx.GraphModule, sample_inputs: Sequence[torch.Tensor], **kwargs
):
    DEFAULT_BACKEND = aot_torch_tensorrt_aten_backend

    return DEFAULT_BACKEND(gm, sample_inputs, **kwargs)


@td.register_backend(name="aot_torch_tensorrt_aten")
def aot_torch_tensorrt_aten_backend(
    gm: torch.fx.GraphModule, sample_inputs: Sequence[torch.Tensor], **kwargs
):
    settings = parse_dynamo_kwargs(kwargs)

    custom_backend = partial(
        _pretraced_backend,
        settings=settings,
    )

    # Perform Pre-AOT Lowering for Module-Level Replacement
    gm = pre_aot_substitutions(gm)

    # Invoke AOTAutograd to translate operators to aten
    return aot_module_simplified(
        gm,
        sample_inputs,
        fw_compiler=make_boxed_compiler(custom_backend),
        decompositions=get_decompositions(),
    )


def _pretraced_backend(
    gm: torch.fx.GraphModule,
    sample_inputs: Sequence[torch.Tensor],
    settings: CompilationSettings = CompilationSettings(),
):
    """Helper function to manage translation of traced FX module to TRT engines

    Args:
        module: FX GraphModule to convert
        inputs: Inputs to the module
        settings: Compilation settings
    Returns:
        Compiled FX GraphModule
    """
    try:
        logger.debug("Post-AOT Autograd graph:\n" + str(gm.graph))

        trt_compiled = _compile_module(
            gm,
            sample_inputs,
            settings=settings,
        )
        return trt_compiled
    except:
        if not settings.pass_through_build_failures:
            logger.warning(
                "TRT conversion failed on the subgraph. See trace above. "
                + "Returning GraphModule forward instead.",
                exc_info=True,
            )
            return gm.forward
        else:
            logger.critical(
                "Halting compilation on build failure since "
                + "pass_through_build_failures was specified as True. "
                + "To return the default Torch implementation and avoid "
                + "halting compilation on engine build failures, "
                + "specify pass_through_build_failures=False."
            )
            raise


def _compile_module(
    gm: torch.fx.GraphModule,
    sample_inputs: Sequence[torch.Tensor],
    settings: CompilationSettings = CompilationSettings(),
) -> torch.fx.GraphModule:
    """Compile a traced FX module

    Includes: Partitioning + Conversion Phases

    Args:
        module: FX GraphModule to convert
        inputs: Inputs to the module
        settings: Compilation settings
    Returns:
        Compiled FX GraphModule
    """
    # Partition module into components that can be TRT-accelerated
    partitioned_module = partition(
        gm,
        verbose=settings.debug,
        min_block_size=settings.min_block_size,
        torch_executed_ops=settings.torch_executed_ops,
    )

    # Store TRT replicas of Torch subgraphs
    trt_modules = {}

    # Iterate over all components that can be accelerated
    # Generate the corresponding TRT Module for those
    for name, _ in partitioned_module.named_children():
        submodule = getattr(partitioned_module, name)

        # Get submodule inputs
        submodule_inputs = get_submod_inputs(
            partitioned_module, submodule, sample_inputs
        )

        # Handle long/double inputs if requested by the user
        if settings.truncate_long_and_double:
            submodule_inputs = repair_long_or_double_inputs(
                partitioned_module, submodule, submodule_inputs, name
            )
        
        # Store the input nodes for this TRT subgraph
        submodule_input_nodes = []
        submodule_node = None
        for node in partitioned_module.graph.nodes:
            if node.name == name:
                submodule_node = node
                if len(node.args) > 0:
                    for i in range(len(node.args)):
                        if node.args[i].op == "placeholder":
                            submodule_input_nodes.append(node.args[i])

        # Create TRT Module from submodule
        trt_mod = convert_module(
            submodule,
            submodule_inputs,
            settings=settings,
            name=name,
        )
        # Add the engine as input to the execute engine op node.
        import io
        engine_str = None
        with io.BytesIO() as engine_bytes:
            engine_bytes.write(trt_mod.engine.serialize())
            engine_str = engine_bytes.getvalue()
        engine_with_metadata = torch.classes.tensorrt.Engine(
                [
                    torch.ops.tensorrt.ABI_VERSION(),
                    name + "_engine" if name != "" else "tensorrt_engine",
                    Device._current_device()._to_serialized_rt_device(),
                    engine_str,
                    TorchTensorRTModule._pack_binding_names(trt_mod.input_names),
                    TorchTensorRTModule._pack_binding_names(trt_mod.output_names),
                ]
            )
        
        
        engine_torch = torch.frombuffer(engine_str, dtype=torch.uint8)
        partitioned_module.register_buffer("engine", engine_torch)
        engine_node = partitioned_module.graph.get_attr("engine")
        submodule_input_nodes.append(engine_node)
        new_node = partitioned_module.graph.create_node("call_function", torch.ops.tensorrt.execute_engine, tuple(submodule_input_nodes))
        submodule_node.replace_all_uses_with(new_node)
        # import pdb; pdb.set_trace()
        partitioned_module.graph.erase_node(submodule_node)
        # trt_modules[name] = trt_mod
        # partitioned_module.recompile()
        # import pdb; pdb.set_trace()
        # print("done")
 
    
    # Replace all FX Modules with TRT Modules
    # for name, trt_mod in trt_modules.items():
    #     setattr(partitioned_module, name, trt_mod)

    return partitioned_module
