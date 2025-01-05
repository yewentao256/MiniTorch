import random
import minitorch
from minitorch.cuda_conv import CudaConvOps

Fast_Backend = minitorch.TensorBackend(minitorch.FastOps)
Cuda_Backend = minitorch.TensorBackend(minitorch.CudaOps)


def test_conv1d_cuda_simple() -> None:
    # A known simple test input
    # CPU version
    t_cpu = minitorch.tensor([0, 1, 2, 3], backend=Fast_Backend).view(1, 1, 4)
    w_cpu = minitorch.tensor([1, 2, 3], backend=Fast_Backend).view(1, 1, 3)
    out_cpu = minitorch.Conv1dFun.apply(t_cpu, w_cpu)

    # GPU version
    t_gpu = minitorch.tensor([0, 1, 2, 3], backend=Cuda_Backend).view(1, 1, 4)
    w_gpu = minitorch.tensor([1, 2, 3], backend=Cuda_Backend).view(1, 1, 3)
    out_gpu = CudaConvOps.conv1d(t_gpu, w_gpu)

    # Ensure outputs match
    for i in range(4):
        assert out_cpu[0, 0, i] == out_gpu[0, 0, i]


def test_conv1d_cuda_random() -> None:
    # random input list of size 100
    input_size = 100
    weight_size = 5
    input_list = random.choices(range(10086), k=input_size)
    weight_list = random.choices(range(10086), k=weight_size)

    # CPU version
    t_cpu = minitorch.tensor(input_list, backend=Fast_Backend).view(1, 1, input_size)
    w_cpu = minitorch.tensor(weight_list, backend=Fast_Backend).view(1, 1, weight_size)
    out_cpu = minitorch.Conv1dFun.apply(t_cpu, w_cpu)

    # GPU version
    t_gpu = minitorch.tensor(input_list, backend=Cuda_Backend).view(1, 1, input_size)
    w_gpu = minitorch.tensor(weight_list, backend=Cuda_Backend).view(1, 1, weight_size)
    out_gpu = CudaConvOps.conv1d(t_gpu, w_gpu)

    # Verify outputs
    for i in range(input_size):
        assert out_cpu[0, 0, i] == out_gpu[0, 0, i]


def test_conv2d_cuda_simple() -> None:
    # A simple known-pattern test
    # CPU version
    t_cpu = minitorch.tensor([[0, 1], [2, 3]], backend=Fast_Backend).view(1, 1, 2, 2)
    w_cpu = minitorch.tensor([[1, 1], [1, 1]], backend=Fast_Backend).view(1, 1, 2, 2)
    out_cpu = minitorch.Conv2dFun.apply(t_cpu, w_cpu)

    # GPU version
    t_gpu = minitorch.tensor([[0, 1], [2, 3]], backend=Cuda_Backend).view(1, 1, 2, 2)
    w_gpu = minitorch.tensor([[1, 1], [1, 1]], backend=Cuda_Backend).view(1, 1, 2, 2)
    out_gpu = CudaConvOps.conv2d(t_gpu, w_gpu)

    for i in range(2):
        for j in range(2):
            assert out_cpu[0, 0, i, j] == out_gpu[0, 0, i, j]


def test_conv2d_cuda_random() -> None:
    # Random input tensor of size 4x4 and weight tensor of size 2x2
    input_size = (100, 100)
    weight_size = (5, 5)
    input_list = random.choices(range(10086), k=input_size[0] * input_size[1])
    weight_list = random.choices(range(10086), k=weight_size[0] * weight_size[1])

    # CPU version
    t_cpu = minitorch.tensor(input_list, backend=Fast_Backend).view(1, 1, *input_size)
    w_cpu = minitorch.tensor(weight_list, backend=Fast_Backend).view(1, 1, *weight_size)
    out_cpu = minitorch.Conv2dFun.apply(t_cpu, w_cpu)

    # GPU version
    t_gpu = minitorch.tensor(input_list, backend=Cuda_Backend).view(1, 1, *input_size)
    w_gpu = minitorch.tensor(weight_list, backend=Cuda_Backend).view(1, 1, *weight_size)
    out_gpu = CudaConvOps.conv2d(t_gpu, w_gpu)

    # Verify outputs
    for i in range(input_size[0]):
        for j in range(input_size[1]):
            assert out_cpu[0, 0, i, j] == out_gpu[0, 0, i, j]
