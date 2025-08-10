use std::sync::atomic::{AtomicBool, Ordering};

use ctor::ctor;

pub static HAS_AVX: AtomicBool = AtomicBool::new(false);
pub static HAS_SSE2: AtomicBool = AtomicBool::new(false);

#[ctor]
unsafe fn set_avx() {
    HAS_AVX.store(is_x86_feature_detected!("avx"), Ordering::Relaxed);
    HAS_SSE2.store(is_x86_feature_detected!("sse2"), Ordering::Relaxed);
}
