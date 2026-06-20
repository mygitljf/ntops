import torch

import ntops
from ntops.torch import _vendor_triton
from ntops.torch.utils import _cached_make, _flatten_kernel_tensors, _prepare_out

# Saved reference to the real torch module.
# InfiniCore.__init__ rebinds the 'torch' name in operator __globals__ to the
# infinicore module, so torch-only utility functions (broadcast_shapes, etc.)
# would otherwise become unavailable.  _rt always points to the real module.
_rt = __import__("torch")


def _to_nt(torch_dtype):
    import ninetoothed

    mapping = {
        _rt.float16: ninetoothed.float16,
        _rt.bfloat16: ninetoothed.bfloat16,
        _rt.float32: ninetoothed.float32,
        _rt.float64: ninetoothed.float64,
        _rt.int8: ninetoothed.int8,
        _rt.int16: ninetoothed.int16,
        _rt.int32: ninetoothed.int32,
        _rt.int64: ninetoothed.int64,
        _rt.uint8: ninetoothed.uint8,
    }
    result = mapping.get(torch_dtype)
    if result is not None:
        return result
    name = str(torch_dtype).rpartition(".")[2]
    name_map = {
        "float16": ninetoothed.float16,
        "bfloat16": ninetoothed.bfloat16,
        "float32": ninetoothed.float32,
        "float64": ninetoothed.float64,
        "int8": ninetoothed.int8,
        "int16": ninetoothed.int16,
        "int32": ninetoothed.int32,
        "int64": ninetoothed.int64,
        "uint8": ninetoothed.uint8,
    }
    return name_map.get(name)


def heaviside(input, values, *, out=None):
    # Hot path for the dominant same-dtype/same-shape/contiguous call: skip
    # result_type/broadcast_shapes/.to()/_prepare_out host dispatch, which is a
    # measurable fraction of these sub-0.05ms bandwidth-bound launches on C500.
    if (
        out is None
        and input.dtype == values.dtype
        and input.shape == values.shape
        and input.is_contiguous()
        and values.is_contiguous()
        and (
            input.dtype in _vendor_triton._HEAVISIDE_FLOAT
            or input.dtype in _vendor_triton._HEAVISIDE_INT
        )
    ):
        fast_out = torch.empty_like(input)
        if _vendor_triton.heaviside_fast_path(input, values, fast_out, input.dtype):
            return fast_out

    result_dtype = (
        torch.result_type(input, values) if hasattr(torch, "result_type") else input.dtype
    )

    nt_dtype = _to_nt(result_dtype)
    if nt_dtype is None:
        raise NotImplementedError(f"heaviside kernel does not support {result_dtype}")

    out_shape = _rt.broadcast_shapes(input.shape, values.shape)

    input_c = input.to(result_dtype) if input.dtype != result_dtype else input
    values_c = values.to(result_dtype) if values.dtype != result_dtype else values

    if (
        out is None
        and not input_c.is_contiguous()
        and tuple(input_c.shape) == tuple(out_shape)
    ):
        # Preserve input's (transpose-style) strided layout so the linear physical
        # kernel pairs out[p] with in[p] at the same logical index without a copy.
        out = torch.empty_like(input_c)
    else:
        out = _prepare_out(out, out_shape, result_dtype, input.device)

    # Vendor fast path handles the contiguous, scalar-broadcast (single stride-0 read,
    # no materialization) and transpose-style dense-strided operands in-kernel, which
    # skips the broadcast + .contiguous() materialization of the general path below.
    if _vendor_triton.heaviside_fast_path(input_c, values_c, out, result_dtype):
        return out

    # General path (bf16 variant, shape-changing broadcast, non-dense strides):
    # materialize broadcast operands, run the NineToothed elementwise kernel.
    try:
        input_b, values_b = torch.broadcast_tensors(input_c, values_c)
    except AttributeError:
        # infinicore context: broadcast_tensors not available;
        # same-shape inputs (the common case) are already broadcast.
        if tuple(input_c.shape) == tuple(values_c.shape) == tuple(out_shape):
            input_b, values_b = input_c, values_c
        else:
            raise NotImplementedError(
                "broadcast_tensors not available for shape-mismatched inputs"
            )
    if not input_b.is_contiguous():
        input_b = input_b.contiguous()
    if not values_b.is_contiguous():
        values_b = values_b.contiguous()

    kernel_out = out
    if not out.is_contiguous():
        kernel_out = torch.empty(out_shape, dtype=result_dtype, device=out.device)

    in_view, values_view, out_view = _flatten_kernel_tensors(input_b, values_b, kernel_out)

    block_size = ntops.kernels.heaviside.BLOCK_SIZE
    if result_dtype == torch.float64:
        block_size = 512

    kernel = _cached_make(
        ntops.kernels.heaviside.premake,
        in_view.ndim,
        dtype=nt_dtype,
        block_size=block_size,
        num_warps=4,
        max_num_configs=1,
    )

    kernel(in_view, values_view, out_view)

    if kernel_out is not out:
        out.copy_(kernel_out)
    return out
