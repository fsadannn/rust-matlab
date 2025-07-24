# Rust-based Padé Implementation for Triangular Matrices for MATLAB

This project provides a Rust implementation of the diagonal Padé approximation for the matrix exponential, specifically optimized for **upper triangular matrices**. It is designed to be called from MATLAB and offers a significant performance improvement over native MATLAB code by exploiting the matrix structure and using optimized BLAS and LAPACK operations.

## Features

- **High-performance:** Core computations are implemented in Rust, compiled to a native MEX library for maximum speed.
- **Optimized for Triangular Matrices:** Uses `dtrmm` and `dtrsm` BLAS routines for efficient matrix multiplication and solving linear systems with triangular matrices.
- **Diagonal Padé approximation:** Implements the Padé approximation for the matrix exponential.
- **MATLAB wrapper:** Provides a simple interface to call the Rust function from MATLAB.

## Prerequisites

- MATLAB
- A C compiler compatible with MATLAB's MEX setup (e.g., MinGW-w64 on Windows, GCC on Linux, Clang on macOS).
- The Rust toolchain (including `cargo`).

## Building the MEX file

1.  **Install Rust:** If you don't have it already, install Rust from [rustup.rs](https://rustup.rs/).
2.  **Build the project:**
    ```bash
    cargo build --release
    ```
3.  **Locate the compiled library:** The compiled MEX file will be in the `target/release` directory, with a name like `rpade_triangular.dll` (Windows), `librpade_triangular.so` (Linux).
4.  **Copy to MATLAB path:** Copy the compiled library to a directory on your MATLAB path.
5.  **Rename the library:** Rename the compiled library to `rpade_triangular.mexa64` (Linux) or `rpade_triangular.mexw64` (Windows).

## Usage

The MEX function has the following signature in MATLAB:

```matlab
% A is an upper triangular matrix
% scaling calculation
normA = norm(A,'inf');
[~,e] = log2(normA);
s = max(0,e+1);
% polinomial degree 2-7
p = 3;
E = rpade_triangular(A, p, s);
```

- `A`: The **upper triangular** input matrix.
- `p`: The degree of the Padé approximation (between 2 and 7).
- `s`: A scaling parameter.

The function returns `E`, the Padé approximation of the matrix exponential of `A`.