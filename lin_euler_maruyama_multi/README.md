# Linear Euler-Maruyama for Stochastic Differential Equations with Multiple Noise Sources

This module provides a solver for systems of linear stochastic differential equations (SDEs) using the Euler-Maruyama method. It is specifically designed for equations with multiple independent Wiener processes (noise sources).

## Solved Equation

The solver handles SDEs of the form:

`dX(t) = (A*X(t) + a) dt + Σ_{j=1}^{m} (B_j*X(t) + b_j) dW_j(t)`

Where:
*   `X(t)` is the state vector.
*   `A` and `B_j` are matrices.
*   `a` and `b_j` are vectors.
*   `dW_j(t)` represent independent Wiener processes.
