# Rust-MATLAB: High-Performance Mathematical Tools

A collection of high-performance mathematical tools and stochastic differential equation (SDE) solvers implemented in Rust, designed for seamless integration with MATLAB via MEX extensions.

This repository leverages Rust's safety and speed, along with hardware-level optimizations (SIMD/FMA) and industry-standard libraries (BLAS/LAPACK), to provide significant performance improvements over native MATLAB implementations.

## Project Structure

This repository is a Cargo workspace containing several specialized crates:

### Matrix Exponential (Padé Approximation)
- **[`rpade`](file:///c:/Users/SadaNN/Desktop/rust-m/rust-matlab/rpade)**: Diagonal Padé approximation for general matrices.
- **[`rpade_no_norm`](file:///c:/Users/SadaNN/Desktop/rust-m/rust-matlab/rpade_no_norm)**: Version of `rpade` that computes the norm and scaling factor internally.
- **[`rpade_no_norm_2x2`](file:///c:/Users/SadaNN/Desktop/rust-m/rust-matlab/rpade_no_norm_2x2)**: Highly optimized 2x2 matrix exponential.
- **[`rpade_triangular`](file:///c:/Users/SadaNN/Desktop/rust-m/rust-matlab/rpade_triangular)**: Padé approximation optimized for upper triangular matrices.
- **[`rpade_triangular_no_norm`](file:///c:/Users/SadaNN/Desktop/rust-m/rust-matlab/rpade_triangular_no_norm)**: Version of `rpade_triangular` that computes internal scaling.

### Stochastic Differential Equations (SDE)
- **[`lin_euler_maruyama_multi`](file:///c:/Users/SadaNN/Desktop/rust-m/rust-matlab/lin_euler_maruyama_multi)**: Euler-Maruyama method for systems of linear SDEs.
- **[`lin_taylor_2_1`](file:///c:/Users/SadaNN/Desktop/rust-m/rust-matlab/lin_taylor_2_1)**: 2,1 Taylor method for linear ODEs with commutative noise.
- **[`ito_double_integral_system`](file:///c:/Users/SadaNN/Desktop/rust-m/rust-matlab/ito_double_integral_system)**: Implementation of Ito double integrals for systems.

### Specialized Models (SIMD Optimized)
These crates implement specific models with manual SIMD (SSE4.2, FMA) optimizations for maximum throughput:
- **[`vander_pol_2_15_mul`](file:///c:/Users/SadaNN/Desktop/rust-m/rust-matlab/vander_pol_2_15_mul)** / **[`vander_pol_2_15_mix`](file:///c:/Users/SadaNN/Desktop/rust-m/rust-matlab/vander_pol_2_15_mix)**: Van der Pol oscillator models.
- **[`lambert_2_15_additive`](file:///c:/Users/SadaNN/Desktop/rust-m/rust-matlab/lambert_2_15_additive)** / **[`lambert_2_15_mul`](file:///c:/Users/SadaNN/Desktop/rust-m/rust-matlab/lambert_2_15_mul)**: Lambert models with additive or multiplicative noise.
- **[`landau_2_15`](file:///c:/Users/SadaNN/Desktop/rust-m/rust-matlab/landau_2_15)**: Landau model implementation.

### Utilities & Wrappers
- **[`gem3d`](file:///c:/Users/SadaNN/Desktop/rust-m/rust-matlab/gem3d)**: Specialized 3D matrix-matrix/vector multiplication.
- **[`matlab_base_wrapper`](file:///c:/Users/SadaNN/Desktop/rust-m/rust-matlab/matlab_base_wrapper)**: Low-level wrapper for the MATLAB C MEX API.
- **[`matlab_blas_wrapper`](file:///c:/Users/SadaNN/Desktop/rust-m/rust-matlab/matlab_blas_wrapper)** / **[`matlab_lapack_wrapper`](file:///c:/Users/SadaNN/Desktop/rust-m/rust-matlab/matlab_lapack_wrapper)**: Rust interfaces for MATLAB's built-in BLAS and LAPACK libraries.
- **[`math_helpers`](file:///c:/Users/SadaNN/Desktop/rust-m/rust-matlab/math_helpers)**: Common mathematical functions and SIMD abstractions.

## Features

- **Blazing Fast:** Core logic written in Rust with manual SIMD (SSE4.2, FMA) optimizations.
- **Native Integration:** Direct use of MATLAB's internal BLAS/LAPACK for linear algebra.
- **Automated Builds:** Simple one-command build process for all extensions using `cargo xtask`.
- **Cross-Platform:** Supports Windows (`.mexw64`) and Linux (`.mexa64`).
- **CI/CD:** Automated releases via GitHub Actions.

## Prerequisites

- **MATLAB** (R2023a recommended for compatibility).
- **Rust Toolchain** (Install via [rustup.rs](https://rustup.rs/)).
- **C Compiler:**
  - **Windows:** MinGW-w64 or MSVC.
  - **Linux:** GCC or Clang.

## Installation & Building

### 1. Download Pre-compiled Binaries
You can find pre-compiled MEX files for Windows and Linux in the [GitHub Releases](https://github.com/fsnaranjo/rust_matlab/releases).

### 2. Building from Source
To build all MEX extensions at once:

```bash
cargo xtask dist
```

This will:
1. Compile all crates in release mode.
2. Rename the resulting libraries to the appropriate MEX extension (`.mexw64` or `.mexa64`).
3. Collect all binaries in the `dist/` directory.

Simply add the `dist/` folder to your MATLAB path to start using the functions.

## License

This project is licensed under the [Apache License 2.0](LICENSE).
