"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable, List

#
# Implementation of a prelude of elementary functions.


def mul(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


def id(a: float) -> float:
    """Identity function."""
    return a


def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def neg(a: float) -> float:
    """Negate a number."""
    return -a


def lt(a: float, b: float) -> float:
    """Check if a is less than b."""
    return 1.0 if a < b else 0.0


def eq(a: float, b: float) -> float:
    """Check if a is equal to b."""
    return 1.0 if a == b else 0.0


def max(a: float, b: float) -> float:
    """Return the maximum of two numbers."""
    return a if a > b else b


def is_close(a: float, b: float) -> bool:
    """Check if two numbers are close."""
    return (a - b < 1e-2) and (b - a < 1e-2)


def sigmoid(a: float) -> float:
    """Compute the sigmoid function."""
    if a >= 0:
        return 1.0 / (1.0 + math.exp(-a))
    else:
        # why is this exp(a) instead of exp(-a)?
        # because if a is negative, then exp(-a) will be very large, and exp(a) will be very small
        # so we need to use exp(a) to make the result more accurate
        return math.exp(a) / (1.0 + math.exp(a))


def relu(a: float) -> float:
    """Compute the ReLU function."""
    return a if a > 0 else 0.0


EPS = 1e-6


def log(a: float) -> float:
    """Compute the natural logarithm."""
    return math.log(a + EPS)


def exp(a: float) -> float:
    """Compute the exponential function."""
    return math.exp(a)


def log_back(a: float, d: float) -> float:
    """Compute the derivative of the logarithm function."""
    # d/dx log(x) = 1/x
    return d / (a + EPS)


def inv(a: float) -> float:
    """Compute the inverse of a number."""
    return 1.0 / a


def inv_back(a: float, d: float) -> float:
    """Compute the derivative of the inverse function."""
    # d/dx 1/x = -1/x^2
    return -d / (a**2)


def relu_back(a: float, d: float) -> float:
    """Compute the derivative of the ReLU function."""
    # d/dx relu(x) = 1 if x > 0 else 0
    return d if a > 0 else 0.0


def map(f: Callable[[float], float], iter: Iterable[float]) -> List[float]:
    """Apply a function to each element of an iterable."""
    return [f(x) for x in iter]


def zipWith(
    f: Callable[[float, float], float], iter1: Iterable[float], iter2: Iterable[float]
) -> List[float]:
    """Apply a function to corresponding elements of two iterables."""
    return [f(x, y) for x, y in zip(iter1, iter2)]


def reduce(
    f: Callable[[float, float], float], iter: Iterable[float], init: float
) -> float:
    """Reduce an iterable to a single value by applying a function cumulatively."""
    result = init
    for x in iter:
        result = f(result, x)
    return result


def negList(iter: Iterable[float]) -> List[float]:
    """Negate a list."""
    return map(neg, iter)


def addLists(iter1: Iterable[float], iter2: Iterable[float]) -> List[float]:
    """Add two lists together."""
    return zipWith(add, iter1, iter2)


def sum(iter: Iterable[float]) -> float:
    """Sum a list."""
    return reduce(add, iter, 0)


def prod(iter: Iterable[float]) -> float:
    """Multiply a list."""
    return reduce(mul, iter, 1)
