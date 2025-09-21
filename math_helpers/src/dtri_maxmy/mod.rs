pub mod avx;
pub mod fallback;
pub mod simd;

pub use self::avx::dtri_maxmy_avx;
pub use self::fallback::dtri_maxmy_fallback;
pub use self::simd::dtri_maxmy_simd;

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
pub unsafe fn dtri_maxmy(alpha: f64, source_x: *const f64, dest_y: *mut f64, size: usize) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx") && size >= 4 {
            return unsafe { dtri_maxmy_avx(alpha, source_x, dest_y, size) };
        }
        if is_x86_feature_detected!("sse2") && size >= 2 {
            return unsafe { dtri_maxmy_simd(alpha, source_x, dest_y, size) };
        }
    }

    unsafe { dtri_maxmy_fallback(alpha, source_x, dest_y, size) }
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
