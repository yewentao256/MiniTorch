from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    # Convert vals (tuple) to lists only when modifying the arg-th value
    vals_forward = tuple(
        val + (epsilon if i == arg else 0) for i, val in enumerate(vals)
    )
    vals_backward = tuple(
        val - (epsilon if i == arg else 0) for i, val in enumerate(vals)
    )

    f_forward = f(*vals_forward)
    f_backward = f(*vals_backward)

    return (f_forward - f_backward) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulates the derivative of the output with respect to this variable."""
        ...

    @property
    def unique_id(self) -> int:
        """Returns a unique identifier for this variable."""
        ...

    def is_leaf(self) -> bool:
        """True if this variable is a leaf node in the computation graph."""
        ...

    def is_constant(self) -> bool:
        """True if this variable is a constant (no derivative)"""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Returns the parents of this variable."""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Returns the derivatives of the parents of this variable."""
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    visited = set()
    topo_order = []

    def dfs(v: Variable) -> None:
        """Recursive depth-first search to visit nodes."""
        if v.unique_id in visited or v.is_constant():
            return
        visited.add(v.unique_id)
        for parent in v.parents:
            dfs(parent)
        if not v.is_constant():
            topo_order.append(v)

    # Start DFS from the output variable
    dfs(variable)
    return reversed(topo_order)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
    ----
    variable: The right-most variable
    deriv : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through
    `accumulate_derivative`.

    """
    topo_order = topological_sort(variable)
    derivatives = {variable.unique_id: deriv}

    for var in topo_order:
        deriv = derivatives[var.unique_id]
        if var.is_leaf():
            var.accumulate_derivative(deriv)
        else:
            for v, d in var.chain_rule(deriv):
                if v.is_constant():
                    continue
                derivatives.setdefault(v.unique_id, 0.0)
                derivatives[v.unique_id] = derivatives[v.unique_id] + d


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during back-propagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Returns the saved values for back-propagation."""
        return self.saved_values
