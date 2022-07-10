from pulp import (
    LpProblem,
    LpVariable,
    LpConstraintVar, 
    LpConstraint,
    LpAffineExpression
)
import numpy as np
from typing import Iterable

class LpProblemIterable(LpProblem):
    """Wrapper for handling iterable added values."""

    def __iadd__(self, other):
        if isinstance(other, tuple):
            other, name = other
        else:
            name = None
        is_native = (
            isinstance(other, LpConstraintVar) or
            isinstance(other, LpConstraint) or
            isinstance(other, LpAffineExpression)
        )
        if isinstance(other, Iterable) and not is_native:
            if isinstance(other, CompArray) and other.ndim == 0:
                super().__iadd__((other.item(), name))
            else:
                for i, element in enumerate(other):
                    if name:
                        self += element, f'{name}_{i}'
                    else:
                        self += element
        else:
            return super().__iadd__((other, name))
        
        return self


class CompArray(np.ndarray):
    """Override eq, le, ge, so that expression creation works properly."""

    def __new__(cls, a: np.ndarray):
        return a.view(cls)
    
    def __eq__(self, other):
        return np.vectorize(lambda l, r: l == r)(self, other)
    
    def __le__(self, other):
        return np.vectorize(lambda l, r: l <= r)(self, other)
    
    def __ge__(self, other):
        return np.vectorize(lambda l, r: l >= r)(self, other)


def lpvar(name: str, *dims: list[Iterable | int], **kwargs) -> CompArray:
    """
    Create n-dimensional array of variables.
    
    A dimension can be either iterable or integer. Integers are
    converted to ranges before assinging variable names.
    """
    def build(name: str, iters: list) -> list:
        iters = iters.copy()
        dim = iters.pop()
        if iters:
            return [
                build(name + f'_{d}', iters) for d in dim
            ]
        else:
            return [
                LpVariable(name + f'_{d}', **kwargs) for d in dim
            ]

    iters = []
    for dim in dims:
        if type(dim) is int:
            iters.append(range(dim))
        else:
            iters.append(dim)
    iters.reverse()
    return CompArray(np.array(build(name, iters)))
