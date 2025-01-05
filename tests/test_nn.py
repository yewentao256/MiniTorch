import pytest
from hypothesis import given

import minitorch
from minitorch import Tensor
from minitorch.tensor_data import TensorData

from .strategies import assert_close
from .tensor_strategies import tensors


@pytest.mark.task4_3
@given(tensors(shape=(1, 1, 4, 4)))
def test_avg(t: Tensor) -> None:
    out = minitorch.avgpool2d(t, (2, 2))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(2) for j in range(2)]) / 4.0
    )

    out = minitorch.avgpool2d(t, (2, 1))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(2) for j in range(1)]) / 2.0
    )

    out = minitorch.avgpool2d(t, (1, 2))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(1) for j in range(2)]) / 2.0
    )
    minitorch.grad_check(lambda t: minitorch.avgpool2d(t, (2, 2)), t)


@pytest.mark.task4_4
@given(tensors(shape=(2, 3, 4)))
def test_max_random(t: Tensor) -> None:
    expected_max = max(
        t[i, j, k]
        for i in range(t.shape[0])
        for j in range(t.shape[1])
        for k in range(t.shape[2])
    )
    computed_max = minitorch.max(t, dim=None)
    assert_close(computed_max.item(), expected_max)


@pytest.mark.task4_4
def test_max_1d() -> None:
    backend = minitorch.TensorBackend(minitorch.FastOps)
    t = Tensor(TensorData([1.0, 3.0, 2.0], (3,)), backend=backend)
    t.requires_grad_(True)
    expected_max = 3.0
    computed_max = minitorch.max(t, dim=None)
    assert_close(computed_max.item(), expected_max)

    computed_max.backward()
    expected_grad = [0.0, 1.0, 0.0]
    assert t.grad is not None
    for i, grad in enumerate(expected_grad):
        assert_close(t.grad[i], grad)


@pytest.mark.task4_4
def test_max_2d() -> None:
    backend = minitorch.TensorBackend(minitorch.FastOps)
    # Define a 2D tensor
    t = Tensor(
        TensorData(
            [1.0, 5.0, 3.0, 4.0, 2.0, 6.0, 7.0, 0.0, 8.0],
            (3, 3),
        ),
        backend=backend,
    )
    t.requires_grad_(True)
    expected_max = 8.0
    computed_max = minitorch.max(t, dim=None)
    assert_close(computed_max.item(), expected_max)

    computed_max.backward()
    # Gradient should be 1 for the maximum element, 0 elsewhere
    expected_grad = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    assert t.grad is not None
    for i in range(3):
        for j in range(3):
            assert_close(t.grad[i, j], expected_grad[i][j])


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_max_pool(t: Tensor) -> None:
    out = minitorch.maxpool2d(t, (2, 2))
    print(out)
    print(t)
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(2) for j in range(2)])
    )

    out = minitorch.maxpool2d(t, (2, 1))
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(2) for j in range(1)])
    )

    out = minitorch.maxpool2d(t, (1, 2))
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(1) for j in range(2)])
    )


@pytest.mark.task4_4
def test_max_3d() -> None:
    backend = minitorch.TensorBackend(minitorch.FastOps)
    # Define a 3D tensor
    t = Tensor(
        TensorData(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            (2, 2, 2),
        ),
        backend=backend,
    )
    t.requires_grad_(True)
    expected_max = 8.0
    computed_max = minitorch.max(t, dim=None)
    assert_close(computed_max.item(), expected_max)

    computed_max.backward()
    # Gradient should be 1 for the maximum element, 0 elsewhere
    expected_grad = [
        [
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        [
            [0.0, 0.0],
            [0.0, 1.0],
        ],
    ]
    assert t.grad is not None
    for i in range(2):
        for j in range(2):
            for k in range(2):
                assert_close(t.grad[i, j, k], expected_grad[i][j][k])


@pytest.mark.task4_4
def test_max_1d_multiple_max() -> None:
    backend = minitorch.TensorBackend(minitorch.FastOps)
    # Define a 1D tensor with multiple maximum values
    t = Tensor(
        TensorData([1.0, 3.0, 2.0, 3.0], (4,)),
        backend=backend,
    )
    t.requires_grad_(True)
    expected_max = 3.0
    computed_max = minitorch.max(t, dim=None)
    assert_close(computed_max.item(), expected_max)

    computed_max.backward()
    # Assuming gradient is equally distributed among all maxima
    expected_grad = [0.0, 0.5, 0.0, 0.5]
    assert t.grad is not None
    for i, grad in enumerate(expected_grad):
        assert_close(t.grad[i], grad)


@pytest.mark.task4_4
def test_max_2d_multiple_max() -> None:
    backend = minitorch.TensorBackend(minitorch.FastOps)
    # Define a 2D tensor with multiple maximum values
    t = Tensor(
        TensorData(
            [1.0, 5.0, 5.0, 4.0, 2.0, 6.0, 7.0, 0.0, 6.0],
            (3, 3),
        ),
        backend=backend,
    )
    t.requires_grad_(True)
    expected_max = 7.0
    computed_max = minitorch.max(t, dim=None)
    assert_close(computed_max.item(), expected_max)

    computed_max.backward()
    # Gradient should be 1 for each occurrence of the maximum value (7.0)
    expected_grad = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ]
    assert t.grad is not None
    for i in range(3):
        for j in range(3):
            assert_close(t.grad[i, j], expected_grad[i][j])


@pytest.mark.task4_4
def test_max_3d_multiple_max() -> None:
    backend = minitorch.TensorBackend(minitorch.FastOps)
    # Define a 3D tensor with multiple maximum values
    t = Tensor(
        TensorData(
            [1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 7.0, 8.0],
            (2, 2, 2),
        ),
        backend=backend,
    )
    t.requires_grad_(True)
    expected_max = 8.0
    computed_max = minitorch.max(t, dim=None)
    assert_close(computed_max.item(), expected_max)

    computed_max.backward()
    # Gradient should be equally distributed among all maxima (two occurrences of 8.0)
    expected_grad = [
        [
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        [
            [0.0, 0.5],
            [0.0, 0.5],
        ],
    ]
    assert t.grad is not None
    for i in range(2):
        for j in range(2):
            for k in range(2):
                assert_close(t.grad[i, j, k], expected_grad[i][j][k])


@pytest.mark.task4_4
@given(tensors())
def test_drop(t: Tensor) -> None:
    q = minitorch.dropout(t, 0.0)
    idx = q._tensor.sample()
    assert q[idx] == t[idx]
    q = minitorch.dropout(t, 1.0)
    assert q[q._tensor.sample()] == 0.0
    q = minitorch.dropout(t, 1.0, ignore=True)
    idx = q._tensor.sample()
    assert q[idx] == t[idx]


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_softmax(t: Tensor) -> None:
    q = minitorch.softmax(t, 3)
    x = q.sum(dim=3)
    assert_close(x[0, 0, 0, 0], 1.0)

    q = minitorch.softmax(t, 1)
    x = q.sum(dim=1)
    assert_close(x[0, 0, 0, 0], 1.0)

    minitorch.grad_check(lambda a: minitorch.softmax(a, dim=2), t)


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_log_softmax(t: Tensor) -> None:
    q = minitorch.softmax(t, 3)
    q2 = minitorch.logsoftmax(t, 3).exp()
    for i in q._tensor.indices():
        assert_close(q[i], q2[i])

    minitorch.grad_check(lambda a: minitorch.logsoftmax(a, dim=2), t)
