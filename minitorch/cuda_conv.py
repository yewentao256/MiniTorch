# type: ignore
# Currently pyright doesn't support numba.cuda
from typing import Any, TypeVar
import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    to_index,
)
from .tensor_ops import TensorOps

FakeCUDAKernel = Any

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs: Any) -> Fn:
    """Decorator to create a cuda jit function."""
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn: Fn, **kwargs: Any) -> FakeCUDAKernel:
    """Decorator to create a cuda jit function."""
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


@cuda.jit(device=True)
def _conv1d_single_out_element(
    in_storage: numba.cuda.cudadrv.devicearray.DeviceNDArray,
    in_shape: numba.types.UniTuple(numba.int32, 3),
    in_strides: numba.types.UniTuple(numba.int32, 3),
    weight_storage: numba.cuda.cudadrv.devicearray.DeviceNDArray,
    weight_shape: numba.types.UniTuple(numba.int32, 3),
    weight_strides: numba.types.UniTuple(numba.int32, 3),
    out_b: int,
    out_c: int,
    out_w: int,
    reverse: bool,
) -> float:
    """Compute a single output element for conv1d.
    Given input shape: (B, C_in, W_in)
    weight shape: (C_out, C_in, K_w)
    output shape: (B, C_out, W_in).
    """
    B, C_in, W_in = in_shape[0], in_shape[1], in_shape[2]
    C_out, _, K_w = weight_shape[0], weight_shape[1], weight_shape[2]

    if out_b >= B or out_c >= C_out or out_w >= W_in:
        return 0.0

    val = 0.0
    for c_in in range(C_in):
        for kw in range(K_w):
            if reverse:
                # reverse=True, anchored right
                in_w = out_w + kw - (K_w - 1)
            else:
                # reverse=False, anchored left
                in_w = out_w + kw

            # Check bounds for in_w
            if 0 <= in_w < W_in:
                in_pos = (
                    out_b * in_strides[0] + c_in * in_strides[1] + in_w * in_strides[2]
                )
                w_pos = (
                    out_c * weight_strides[0]
                    + c_in * weight_strides[1]
                    + kw * weight_strides[2]
                )
                val += in_storage[in_pos] * weight_storage[w_pos]

    return val


@cuda.jit
def tensor_conv1d(
    out: numba.cuda.cudadrv.devicearray.DeviceNDArray,
    out_shape: numba.types.UniTuple(numba.int32, 3),
    out_strides: numba.types.UniTuple(numba.int32, 3),
    out_size: int,
    in_storage: numba.cuda.cudadrv.devicearray.DeviceNDArray,
    in_shape: numba.types.UniTuple(numba.int32, 3),
    in_strides: numba.types.UniTuple(numba.int32, 3),
    weight_storage: numba.cuda.cudadrv.devicearray.DeviceNDArray,
    weight_shape: numba.types.UniTuple(numba.int32, 3),
    weight_strides: numba.types.UniTuple(numba.int32, 3),
    reverse: bool,
) -> None:
    """CUDA kernel for 1D convolution with same output width.

    Output shape: (B, C_out, W_in)
    """
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if i >= out_size:
        return

    # Convert i (linear index) to multidimensional index out[b, c_out, w_out]
    out_index = cuda.local.array(MAX_DIMS, numba.int32)
    to_index(i, out_shape, out_index)
    b, c, w = out_index[0], out_index[1], out_index[2]

    val = _conv1d_single_out_element(
        in_storage,
        in_shape,
        in_strides,
        weight_storage,
        weight_shape,
        weight_strides,
        b,
        c,
        w,
        reverse,
    )

    out_pos = index_to_position(out_index, out_strides)
    out[out_pos] = val


@cuda.jit(device=True)
def _conv2d_single_out_element(
    in_storage: numba.cuda.cudadrv.devicearray.DeviceNDArray,
    in_shape: numba.types.UniTuple(numba.int32, 4),
    in_strides: numba.types.UniTuple(numba.int32, 4),
    weight_storage: numba.cuda.cudadrv.devicearray.DeviceNDArray,
    weight_shape: numba.types.UniTuple(numba.int32, 4),
    weight_strides: numba.types.UniTuple(numba.int32, 4),
    out_b: int,
    out_c: int,
    out_h: int,
    out_w: int,
) -> float:
    """Compute a single output element for conv2d.
    Similar logic can be applied as 1D to not reduce dimension,
    i.e. output shape: (B, C_out, H_in, W_in).
    """
    B, C_in, H_in, W_in = in_shape[0], in_shape[1], in_shape[2], in_shape[3]
    C_out, _, K_h, K_w = (
        weight_shape[0],
        weight_shape[1],
        weight_shape[2],
        weight_shape[3],
    )

    if out_b >= B or out_c >= C_out or out_h >= H_in or out_w >= W_in:
        return 0.0

    val = 0.0
    for c_in in range(C_in):
        for kh in range(K_h):
            for kw in range(K_w):
                in_h = out_h + kh
                in_w = out_w + kw
                if 0 <= in_h < H_in and 0 <= in_w < W_in:
                    in_pos = (
                        out_b * in_strides[0]
                        + c_in * in_strides[1]
                        + in_h * in_strides[2]
                        + in_w * in_strides[3]
                    )

                    w_pos = (
                        out_c * weight_strides[0]
                        + c_in * weight_strides[1]
                        + kh * weight_strides[2]
                        + kw * weight_strides[3]
                    )
                    val += in_storage[in_pos] * weight_storage[w_pos]

    return val


@cuda.jit
def tensor_conv2d(
    out: numba.cuda.cudadrv.devicearray.DeviceNDArray,
    out_shape: numba.types.UniTuple(numba.int32, 4),
    out_strides: numba.types.UniTuple(numba.int32, 4),
    out_size: int,
    in_storage: numba.cuda.cudadrv.devicearray.DeviceNDArray,
    in_shape: numba.types.UniTuple(numba.int32, 4),
    in_strides: numba.types.UniTuple(numba.int32, 4),
    weight_storage: numba.cuda.cudadrv.devicearray.DeviceNDArray,
    weight_shape: numba.types.UniTuple(numba.int32, 4),
    weight_strides: numba.types.UniTuple(numba.int32, 4),
) -> None:
    """CUDA kernel for 2D convolution with same output height and width.
    Output shape: (B, C_out, H_in, W_in)
    """
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if i >= out_size:
        return

    out_index = cuda.local.array(MAX_DIMS, numba.int32)
    to_index(i, out_shape, out_index)
    b, c, h, w = out_index[0], out_index[1], out_index[2], out_index[3]

    val = _conv2d_single_out_element(
        in_storage,
        in_shape,
        in_strides,
        weight_storage,
        weight_shape,
        weight_strides,
        b,
        c,
        h,
        w,
    )

    out_pos = index_to_position(out_index, out_strides)
    out[out_pos] = val


class CudaConvOps(TensorOps):
    cuda = True

    @staticmethod
    def conv1d(input: Tensor, weight: Tensor, reverse: bool = False) -> Tensor:
        """Perform a 1D convolution with the given input and weight on CUDA.
        Output shape: (B, C_out, W_in), no dimension reduction.

        reverse: bool - if True, kernel anchored to the right,
                        else anchored to the left.
        """
        B, C_in, W_in = input.shape
        C_out, C_in_w, K_w = weight.shape
        assert C_in == C_in_w, "Input and weight channel sizes must match."

        out = input.zeros((B, C_out, W_in))

        threadsperblock = THREADS_PER_BLOCK
        blockspergrid = (out.size + threadsperblock - 1) // threadsperblock

        tensor_conv1d[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *input.tuple(), *weight.tuple(), reverse
        )
        return out

    @staticmethod
    def conv2d(input: Tensor, weight: Tensor) -> Tensor:
        """Perform a 2D convolution with the given input and weight on CUDA.
        Output shape: (B, C_out, H_in, W_in), no dimension reduction.
        """
        B, C_in, H_in, W_in = input.shape
        C_out, C_in_w, K_h, K_w = weight.shape
        assert C_in == C_in_w, "Input and weight channel sizes must match."

        # Output has same height and width as input
        out = input.zeros((B, C_out, H_in, W_in))

        threadsperblock = THREADS_PER_BLOCK
        blockspergrid = (out.size + threadsperblock - 1) // threadsperblock

        tensor_conv2d[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *input.tuple(), *weight.tuple()
        )
        return out
