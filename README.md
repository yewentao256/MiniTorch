# MiniTorch

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

Solution for Cornell MiniTorch: [https://github.com/minitorch/minitorch]

Docs: [https://minitorch.github.io/]

## Task 4.4 - Bonus Convolution

Code: [cuda_conv.py](minitorch/cuda_conv.py)

Test case: [test_cuda_conv.py](tests/test_cuda_conv.py)

Result:

```bash
PS C:\Users\Peter\Desktop\mod4-yewentao256> pytest .\tests\test_cuda_conv.py
======================================================== test session starts ========================================================
platform win32 -- Python 3.11.5, pytest-8.3.2, pluggy-1.5.0
rootdir: C:\Users\Peter\Desktop\mod4-yewentao256
configfile: pyproject.toml
plugins: hypothesis-6.54.0, env-1.1.3
collected 4 items

tests\test_cuda_conv.py ....                                                                                                   [100%]

========================================================= warnings summary ==========================================================
tests/test_cuda_conv.py::test_conv1d_cuda_simple
tests/test_cuda_conv.py::test_conv2d_cuda_simple
  C:\Users\Peter\AppData\Local\Programs\Python\Python311\Lib\site-packages\numba\cuda\dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_cuda_conv.py::test_conv1d_cuda_simple
tests/test_cuda_conv.py::test_conv1d_cuda_random
tests/test_cuda_conv.py::test_conv2d_cuda_simple
tests/test_cuda_conv.py::test_conv2d_cuda_random
  C:\Users\Peter\AppData\Local\Programs\Python\Python311\Lib\site-packages\numba\cuda\cudadrv\devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
    warn(NumbaPerformanceWarning(msg))

tests/test_cuda_conv.py::test_conv1d_cuda_random
  C:\Users\Peter\AppData\Local\Programs\Python\Python311\Lib\site-packages\numba\cuda\dispatcher.py:536: NumbaPerformanceWarning: Grid size 4 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=================================================== 4 passed, 7 warnings in 5.56s ===================================================
```

## Task 4.5

### Mnist

[Logs](mnist.txt)

### Sentiment

[Logs](sentiment.txt)
