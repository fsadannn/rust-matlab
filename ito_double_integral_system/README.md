
# Ito Double Integral System

This module computes the Ito double integral for `m` Wiener processes.

The implementation uses the stochastic differential equation representation of the associated Ito double integral. This representation was presented by Milstein.

The method requires a sub-sampling of the Wiener process between $t_n$ and $t_{n+1}$. The size of the sub-sampling is of order $h^{2 \cdot \text{order}}$, which means that we need at least $1/h^{2 \cdot \text{order} - 1}$ Wiener samples of the subinterval.

## Input

The input to the system is a 2D array representing the Wiener processes of the subinterval. Each column of this array contains `m` Wiener samples at a specific time point, $t_n + i \cdot h^{2 \cdot \text{order}}$, where $i$ is the index of the subinterval sample.
