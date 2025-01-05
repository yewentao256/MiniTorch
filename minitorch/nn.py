from typing import Optional, Tuple
from minitorch.tensor_data import TensorData

from .autodiff import Context
from .tensor import Tensor
from .tensor_functions import Function, rand


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0

    new_height = height // kh
    new_width = width // kw

    # reshape the input tensor to group the pooling regions
    input = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)
    # permute the dimensions to bring kernel dimensions together
    input = input.permute(0, 1, 2, 4, 3, 5)
    # flatten the kernel dimensions
    input = input.contiguous().view(batch, channel, new_height, new_width, kh * kw)

    return input, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled average pooling 2D"""
    batch, channel, _, _ = input.shape
    input_tiled, new_height, new_width = tile(input, kernel)
    # Compute the average over the last dimension
    result = input_tiled.mean(dim=len(input_tiled.shape) - 1)
    return result.view(batch, channel, new_height, new_width)


class Max(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, dim: Tensor) -> Tensor:
        """Compute the maximum values along a specified dimension."""
        max_val = t1.f.max_reduce(t1, int(dim.item()))
        eq_mask = t1 == max_val
        ctx.save_for_backward(eq_mask, dim)
        return max_val

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Calculate the gradient for Max."""
        eq_mask, dim = ctx.saved_values

        true_shape = TensorData.shape_broadcast(grad_output.shape, eq_mask.shape)
        grad_broadcast = grad_output.zeros(true_shape)
        grad_output.f.id_map(grad_output, grad_broadcast)
        grad_input = grad_broadcast * eq_mask / eq_mask.sum(dim=int(dim.item()))
        return grad_input, 0.0


def max(input: Tensor, dim: Optional[int] = None) -> Tensor:
    """Compute the maximum values along a specified dimension."""
    if dim is None:
        return Max.apply(input.contiguous().view(input.size), input._ensure_tensor(0))
    else:
        return Max.apply(input, input._ensure_tensor(dim))


def argmax(input: Tensor, dim: Optional[int] = None) -> Tensor:
    """Compute the argmax as a 1-hot tensor along a specified dimension."""
    if dim is None:
        flat = input.contiguous().view(input.size)
        max_val = max(flat)
        argm = flat == max_val
        return argm.view(*input.shape)
    else:
        max_val = max(input, dim=dim)
        shape = list(input.shape)
        shape[dim] = 1
        max_val_broadcast = max_val.view(*shape)
        argm = input == max_val_broadcast
        return argm


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled max pooling 2D"""
    batch, channel, _, _ = input.shape
    input_tiled, new_height, new_width = tile(input, kernel)
    # Compute the max over the last dimension (the tiled region)
    result = max(input_tiled, dim=len(input_tiled.shape) - 1)
    return result.view(batch, channel, new_height, new_width)


def softmax(input: Tensor, dim: Optional[int] = None) -> Tensor:
    """Compute the softmax along a specified dimension.

    softmax(x) = exp(x) / sum(exp(x), dim)
    """
    if dim is None:
        # If no dim is given, assume dim=0 for a flat tensor
        input = input.contiguous().view(input.size)
        dim = 0

    exp_inp = input.exp()
    sum_exp = exp_inp.sum(dim=dim)
    return exp_inp / sum_exp


def logsoftmax(input: Tensor, dim: Optional[int] = None) -> Tensor:
    """Compute the log of the softmax along a specified dimension using the log-sum-exp trick.

    logsoftmax(x) = x - log(sum(exp(x), dim))
    To avoid numerical instability, we use:
    logsumexp = m + log(sum(exp(x - m), dim)) where m = max(x, dim)
    """
    if dim is None:
        input = input.contiguous().view(input.size)
        dim = 0

    max_val = max(input, dim=dim)
    shape = list(input.shape)
    shape[dim] = 1
    m = max_val.view(*shape)

    # logsumexp calculation
    logsumexp = (input - m).exp().sum(dim=dim).log() + m
    return input - logsumexp


def dropout(input: Tensor, p: float = 0.5, ignore: bool = False) -> Tensor:
    """Apply dropout to the input."""
    if ignore:
        return input
    if p >= 1.0:
        return input.zeros()
    mask = rand(input.shape) > p

    # Scale the output by 1/(1-p) to keep expected value constant
    return input * mask / (1 - p)
