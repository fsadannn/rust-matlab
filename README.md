# Rust-based Matrix Exponential for MATLAB

This repository provides Rust implementations of the Padé approximation for the matrix exponential, designed to be called from MATLAB. It offers a significant performance improvement over native MATLAB code by leveraging Rust's speed and memory safety, along with optimized BLAS and LAPACK operations.

## Crates

This repository is structured as a Cargo workspace and contains the following crates:

- **`rpade`**: A Rust implementation of the diagonal Padé approximation for general matrices. This version requires the norm and scaling factor to be provided as parameters.
- **`rpade_no_norm`**: A version of `rpade` that computes the norm and scaling factor internally.
- **`rpade_triangular`**: An optimized version of the Padé approximation for upper triangular matrices. This version requires the norm and scaling factor to be provided as parameters.
- **`rpade_triangular_no_norm`**: A version of `rpade_triangular` that computes the norm and scaling factor internally.
- **`rpade_shared`**: Internal helper functions shared between the `rpade` crates.
- **`gemv3d`**: A specialized matrix-vector multiplication routine.
- **`lin_euler_maruyama_multi`**: An implementation of the Euler-Maruyama method for systems of linear stochastic differential equations.
- **`lin_taylor_2_1`**: An implementation of a 2,1 Taylor method for systems of linear ordinary differential equations with commutative noises.
- **`math_helpers`**: A collection of mathematical helper functions used by the other crates.
- **`matlab_base_wrapper`**: A wrapper for the MATLAB C MEX API.
- **`matlab_blas_wrapper`**: A wrapper for the BLAS library.
- **`matlab_lapack_wrapper`**: A wrapper for the LAPACK library.

## Features

- **High-performance:** Core computations are implemented in Rust, compiled to a native MEX library for maximum speed.
- **BLAS/LAPACK integration:** Uses `dgemm` and `dgesv` for efficient matrix multiplication and solving linear systems.
- **Optimized for Triangular Matrices:** The `rpade_triangular` crate is specifically optimized for upper triangular matrices, using `dtrmm` and `dtrsm` for improved performance.
- **MATLAB wrapper:** Provides a simple interface to call the Rust functions from MATLAB.

## Prerequisites

- MATLAB
- A C compiler compatible with MATLAB's MEX setup (e.g., MinGW-w64 on Windows, GCC on Linux, Clang on macOS).
- The Rust toolchain (including `cargo`).

## Building the MEX files

1.  **Install Rust:** If you don't have it already, install Rust from [rustup.rs](https://rustup.rs/).
2.  **Build the project:**
    ```bash
    cargo build --release
    ```
3.  **Locate the compiled libraries:** The compiled MEX files will be in the `target/release` directory, with names like `rpade.dll` and `rpade_triangular.dll` (Windows), or `librpade.so` and `librpade_triangular.so` (Linux).
4.  **Copy to MATLAB path:** Copy the compiled libraries to a directory on your MATLAB path.
5.  **Rename the libraries:** Rename the compiled libraries to `rpade.mexa64` and `rpade_triangular.mexa64` (Linux) or `rpade.mexw64` and `rpade_triangular.mexw64` (Windows).

## Usage

See the `README.md` files in the individual crate directories for usage instructions for each specific function.
