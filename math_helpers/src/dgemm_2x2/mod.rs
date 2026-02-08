use crate::dgemm_2x2::{avx2::dgemm_2x2_avx2, fallback::dgemm_2x2_fallback, sse2::dgemm_2x2_sse2};

mod avx2;
mod fallback;
mod sse2;

pub type FnDGEM22 = unsafe fn(*const f64, *const f64, *mut f64) -> ();

pub fn dgemm_2x2(a: *const f64, b: *const f64, out: *mut f64) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dgemm_2x2_avx2(a, b, out) };
        } else if is_x86_feature_detected!("sse2") {
            return unsafe { dgemm_2x2_sse2(a, b, out) };
        }
    }

    unsafe { dgemm_2x2_fallback(a, b, out) }
}
