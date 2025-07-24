#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
fn _dtri_maxmy_simd(alpha: f64, source_x: *const f64, dest_y: *mut f64, n: usize) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{
        __m128d, _mm_add_pd, _mm_loadu_pd, _mm_mul_pd, _mm_set1_pd, _mm_storeu_pd,
    };
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{
        __m128d, _mm_add_pd, _mm_loadu_pd, _mm_mul_pd, _mm_set1_pd, _mm_storeu_pd,
    };

    unsafe {
        *dest_y.add(0) = alpha * (*source_x.add(0)) + (*dest_y.add(0));
    }

    const UNROLL_FACTOR: usize = 2; // Process 2 f64 elements per SIMD operation

    // Broadcast the alpha factor into a 128-bit SIMD register.
    let alpha_vec: __m128d = _mm_set1_pd(alpha);

    for j in 1..n {
        let gap = j * n;
        let remainder = (j + 1) % UNROLL_FACTOR;
        let unrolled_limit = (j + 1) - remainder;
        for i in (0..(unrolled_limit)).step_by(UNROLL_FACTOR) {
            unsafe {
                // Load 2 f64 values from vector X
                let x_vec: __m128d = _mm_loadu_pd(source_x.add(gap + i));
                // Load 2 f64 values from vector Y
                let y_vec: __m128d = _mm_loadu_pd(dest_y.add(gap + i));

                // Perform alpha * X
                let alpha_x_vec: __m128d = _mm_mul_pd(alpha_vec, x_vec);

                // Perform (alpha * X) + Y
                let result_vec: __m128d = _mm_add_pd(alpha_x_vec, y_vec);

                // Store the result back into Y
                _mm_storeu_pd(dest_y.add(gap + i), result_vec);
            }
        }
        if remainder != 0 {
            unsafe {
                *dest_y.add(gap + unrolled_limit) = alpha * (*source_x.add(gap + unrolled_limit))
                    + (*dest_y.add(gap + unrolled_limit));
            }
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
fn _dtri_maxmy_simd256(alpha: f64, source_x: *const f64, dest_y: *mut f64, n: usize) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{
        __m256d, _mm256_add_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_set1_pd, _mm256_storeu_pd,
    };
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{
        __m256d, _mm256_add_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_set1_pd, _mm256_storeu_pd,
    };

    unsafe {
        *dest_y.add(0) = alpha * (*source_x.add(0)) + (*dest_y.add(0));
    }

    if n >= 2 {
        unsafe {
            *dest_y.add(n) = alpha * (*source_x.add(n)) + (*dest_y.add(n));
            *dest_y.add(n + 1) = alpha * (*source_x.add(n + 1)) + (*dest_y.add(n + 1));
        }
    }

    if n >= 3 {
        unsafe {
            let gap = 2 * n;
            *dest_y.add(gap) = alpha * (*source_x.add(gap)) + (*dest_y.add(gap));
            *dest_y.add(gap + 1) = alpha * (*source_x.add(gap + 1)) + (*dest_y.add(gap + 1));
            *dest_y.add(gap + 2) = alpha * (*source_x.add(gap + 2)) + (*dest_y.add(gap + 2));
        }
    }

    if n == 3 {
        return;
    }

    const UNROLL_FACTOR: usize = 4; // Process 4 f64 elements per SIMD operation
    // Broadcast the alpha factor into a 256-bit SIMD register.
    let alpha_vec: __m256d = _mm256_set1_pd(alpha);

    for j in 3..n {
        let gap = j * n;
        let remainder = (j + 1) % UNROLL_FACTOR;
        let unrolled_limit = (j + 1) - remainder;

        for i in (0..unrolled_limit).step_by(UNROLL_FACTOR) {
            unsafe {
                // Load 4 f64 values from vector X
                let x_vec: __m256d = _mm256_loadu_pd(source_x.add(gap + i));
                // Load 4 f64 values from vector Y
                let y_vec: __m256d = _mm256_loadu_pd(dest_y.add(gap + i));

                // Perform alpha * X
                let alpha_x_vec: __m256d = _mm256_mul_pd(alpha_vec, x_vec);

                // Perform (alpha * X) + Y
                let result_vec: __m256d = _mm256_add_pd(alpha_x_vec, y_vec);

                // Store the result back into Y
                _mm256_storeu_pd(dest_y.add(gap + i), result_vec);
            }
        }

        match remainder {
            0 => (),
            1 => unsafe {
                *dest_y.add(gap + unrolled_limit) = alpha * (*source_x.add(gap + unrolled_limit))
                    + (*dest_y.add(gap + unrolled_limit));
            },
            2 => unsafe {
                *dest_y.add(gap + unrolled_limit) = alpha * (*source_x.add(gap + unrolled_limit))
                    + (*dest_y.add(gap + unrolled_limit));
                *dest_y.add(gap + unrolled_limit + 1) = alpha
                    * (*source_x.add(gap + unrolled_limit + 1))
                    + (*dest_y.add(gap + unrolled_limit + 1));
            },
            3 => unsafe {
                *dest_y.add(gap + unrolled_limit) = alpha * (*source_x.add(gap + unrolled_limit))
                    + (*dest_y.add(gap + unrolled_limit));
                *dest_y.add(gap + unrolled_limit + 1) = alpha
                    * (*source_x.add(gap + unrolled_limit + 1))
                    + (*dest_y.add(gap + unrolled_limit + 1));
                *dest_y.add(gap + unrolled_limit + 2) = alpha
                    * (*source_x.add(gap + unrolled_limit + 2))
                    + (*dest_y.add(gap + unrolled_limit + 2));
            },
            _ => (),
        }
    }
}

// Fallback for non-x86_64 architectures for daxpy_simd
pub fn _dtri_maxmy_fallback(alpha: f64, source_x: *const f64, dest_y: *mut f64, n: usize) {
    // Fallback to scalar operations if not on x86_64
    unsafe {
        *dest_y.add(0) = alpha * (*source_x.add(0)) + (*dest_y.add(0));
    }

    for j in 1..n {
        let gap = j * n;
        for i in 0..(j + 1) {
            unsafe {
                *dest_y.add(gap + i) = alpha * (*source_x.add(gap + i)) + (*dest_y.add(gap + i));
            }
        }
    }
}

/// Performs the operation for upper triangular column-major stored matrices:
/// Y = alpha * X + Y, using x86_64 SIMD intrinsics.
///
/// # Arguments
/// * `alpha` - The scalar factor (f64).
/// * `source_x` - A raw pointer to the source upper triangular matrix X (f64 values).
/// * `dest_y` - A mutable raw pointer to the destination matrix Y (f64 values),
///              which will be updated in-place.
/// * `size` - The number of columns in the matrices.
///
/// # Safety
/// This function is `unsafe` because it operates on raw pointers and uses SIMD intrinsics.
/// The caller must ensure that:
/// - `source_x` and `dest_y` are valid, non-null pointers.
/// - The memory regions pointed to by `source_x` and `dest_y` are valid for a triangular
///   matrix with `size` columns (i.e., size * (size + 1) / 2 `f64` elements).
/// - The `source_x` and `dest_y` memory regions do not overlap unless `source_x` is also `dest_y`
///   (which would be an in-place operation on Y).
/// - The target CPU supports the necessary x86_64 SIMD instructions (e.g., SSE2 for _mm_pd operations).
///   This typically means compiling with `target_feature="+sse2"` or running on a modern x86_64 CPU.
pub unsafe fn dtri_maxmy_simd(alpha: f64, source_x: *const f64, dest_y: *mut f64, size: usize) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx") {
            return unsafe { _dtri_maxmy_simd256(alpha, source_x, dest_y, size) };
        }
        if is_x86_feature_detected!("sse2") {
            return unsafe { _dtri_maxmy_simd(alpha, source_x, dest_y, size) };
        }
    }

    _dtri_maxmy_fallback(alpha, source_x, dest_y, size);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtri_maxmy_simd() {
        let n = 3;
        let mut x = vec![0.0; n * n];
        let mut y = vec![0.0; n * n];
        let alpha = 2.0;

        x[0] = 1.0;
        x[3] = 2.0;
        x[4] = 3.0;
        y[0] = 4.0;
        y[6] = 5.0;
        y[7] = 6.0;

        unsafe {
            dtri_maxmy_simd(alpha, x.as_mut_ptr(), y.as_mut_ptr(), n);
        }

        let mut ans = vec![0.0; n * n];
        ans[0] = 6.0;
        ans[3] = 4.0;
        ans[4] = 6.0;
        ans[6] = 5.0;
        ans[7] = 6.0;

        for i in 0..(n * n) {
            assert!((y[i] - ans[i]).abs() < 1e-12);
        }
    }
}
