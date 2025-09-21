/// Performs the DAXPY operation: Y = alpha * X + Y, using x86_64 AVX SIMD intrinsics.
///
/// # Safety
/// This function is `unsafe` because it operates on raw pointers and uses SIMD intrinsics.
/// The caller must ensure that:
/// - `source_x` and `dest_y` are valid, non-null pointers.
/// - The memory regions pointed to by `source_x` and `dest_y` are valid for at least `size`
///   `f64` elements.
/// - The `source_x` and `dest_y` memory regions do not overlap unless `source_x` is also `dest_y`
///   (which would be an in-place operation on Y, not a standard DAXPY).
/// - The target CPU supports the necessary x86_64 AVX SIMD instructions.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
pub fn daxpy_avx(alpha: f64, source_x: *const f64, dest_y: *mut f64, size: usize) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{
        __m256d, _mm256_add_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_set1_pd, _mm256_storeu_pd,
    };
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{
        __m256d, _mm256_add_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_set1_pd, _mm256_storeu_pd,
    };

    const UNROLL_FACTOR: usize = 4; // Process 4 f64 elements per SIMD operation
    let remainder = size % UNROLL_FACTOR;
    let unrolled_limit = size - remainder;

    // Broadcast the alpha factor into a 256-bit SIMD register.
    let alpha_vec: __m256d = _mm256_set1_pd(alpha);

    // Main unrolled loop: processes UNROLL_FACTOR elements at a time using SIMD.
    for i in (0..unrolled_limit).step_by(UNROLL_FACTOR) {
        // Load 4 f64 values from vector X
        let x_vec: __m256d = unsafe { _mm256_loadu_pd(source_x.add(i)) };
        // Load 4 f64 values from vector Y
        let y_vec: __m256d = unsafe { _mm256_loadu_pd(dest_y.add(i)) };

        // Perform alpha * X
        let alpha_x_vec: __m256d = _mm256_mul_pd(alpha_vec, x_vec);

        // Perform (alpha * X) + Y
        let result_vec: __m256d = _mm256_add_pd(alpha_x_vec, y_vec);

        // Store the result back into Y
        unsafe { _mm256_storeu_pd(dest_y.add(i), result_vec) };
    }

    match remainder {
        1 => unsafe {
            *dest_y.add(unrolled_limit) =
                alpha * (*source_x.add(unrolled_limit)) + (*dest_y.add(unrolled_limit));
        },
        2 => unsafe {
            *dest_y.add(unrolled_limit) =
                alpha * (*source_x.add(unrolled_limit)) + (*dest_y.add(unrolled_limit));
            *dest_y.add(unrolled_limit + 1) =
                alpha * (*source_x.add(unrolled_limit + 1)) + (*dest_y.add(unrolled_limit + 1));
        },
        3 => unsafe {
            *dest_y.add(unrolled_limit) =
                alpha * (*source_x.add(unrolled_limit)) + (*dest_y.add(unrolled_limit));
            *dest_y.add(unrolled_limit + 1) =
                alpha * (*source_x.add(unrolled_limit + 1)) + (*dest_y.add(unrolled_limit + 1));
            *dest_y.add(unrolled_limit + 2) =
                alpha * (*source_x.add(unrolled_limit + 2)) + (*dest_y.add(unrolled_limit + 2));
        },
        _ => (),
    }
}
