/// # Safety
/// This function is `unsafe` because it uses raw pointers and relies on the caller
/// to ensure that:
/// - `source_x` and `dest_y` are valid, non-null pointers.
/// - The memory regions pointed to by `source_x` and `dest_y` are valid for at least `n`
///   `f64` elements.
/// - The `source_x` and `dest_y` memory regions do not overlap unless `source_x` is also `dest_y`
///   (which would be an in-place operation on Y, not a standard DAXPY).
/// - The target CPU supports the necessary x86_64 AVX SIMD instructions.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
pub unsafe fn dtri_maxmy_avx(alpha: f64, source_x: *const f64, dest_y: *mut f64, n: usize) {
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
            // Load 4 f64 values from vector X
            let x_vec: __m256d = unsafe { _mm256_loadu_pd(source_x.add(gap + i)) };
            // Load 4 f64 values from vector Y
            let y_vec: __m256d = unsafe { _mm256_loadu_pd(dest_y.add(gap + i)) };

            // Perform alpha * X
            let alpha_x_vec: __m256d = _mm256_mul_pd(alpha_vec, x_vec);

            // Perform (alpha * X) + Y
            let result_vec: __m256d = _mm256_add_pd(alpha_x_vec, y_vec);

            // Store the result back into Y
            unsafe { _mm256_storeu_pd(dest_y.add(gap + i), result_vec) };
        }

        match remainder {
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
