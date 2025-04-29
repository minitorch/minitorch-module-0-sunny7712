"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, Iterable

# ## Task 0.1

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.


def mul(x: float, y: float) -> float:
    """Multiply two numbers."""
    return x * y


def id(x: float) -> float:
    """Identity function."""
    return x


def add(x: float, y: float) -> float:
    """Add two numbers."""
    return x + y


def neg(x: float) -> float:
    """Negate a number."""
    return -x


def lt(x: float, y: float) -> bool:
    """Less than comparison."""
    return x < y


def eq(x: float, y: float) -> bool:
    """Equality comparison."""
    return x == y


def max(x: float, y: float) -> float:
    """Maximum of two numbers."""
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """Check if two numbers are close."""
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Sigmoid function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Relu function."""
    return max(0, x)


def log(x: float) -> float:
    """Natural logarithm."""
    return math.log(x)


def exp(x: float) -> float:
    """Exponential function."""
    return math.exp(x)


def log_back(x: float, dout: float) -> float:
    """Derivative of the natural logarithm."""
    return dout / x


def inv(x: float) -> float:
    """Inverse function."""
    return 1.0 / x


def inv_back(x: float, dout: float) -> float:
    """Derivative of the inverse function."""
    return -dout / (x * x)


def relu_back(x: float, dout: float) -> float:
    """Derivative of the relu function."""
    return dout if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Map a function over a list."""
    def apply(ls : Iterable[float]) -> Iterable[float]:
        return [fn(x) for x in ls]
    return apply

def zipWith(fn: Callable[[float, float], float]) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Zip two lists and apply a function."""
    def apply(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        return [fn(x, y) for x, y in zip(ls1, ls2)]
    return apply

def reduce(fn: Callable[[float, float], float]) -> Callable[[Iterable[float]], float]:
    """Reduce a list using a function."""
    def apply(ls: Iterable[float]) -> float:
        ls = list(ls)
        result  = 0
        if len(ls) > 0:
            result = ls[0]
        if len(ls) > 1:
            for x in ls[1:]:
                result = fn(result, x)
        return result
    return apply

def negList(ls: Iterable[float]) -> Iterable[float]:
    """Negate a list."""
    return map(neg)(ls)

def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add two lists together."""
    return zipWith(add)(ls1, ls2)

def sum(ls: Iterable[float]) -> float:
    """Sum a list."""
    return reduce(add)(ls)

def prod(ls: Iterable[float]) -> float:
    """Take the product of a list."""
    return reduce(mul)(ls)


