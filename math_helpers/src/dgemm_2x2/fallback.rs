pub unsafe fn dgemm_2x2_fallback(a: *const f64, b: *const f64, out: *mut f64) {
    unsafe { *out.add(0) = (*a.add(0)) * (*b.add(0)) + (*a.add(2)) * (*b.add(1)) };
    unsafe { *out.add(1) = (*a.add(1)) * (*b.add(0)) + (*a.add(3)) * (*b.add(1)) };
    unsafe { *out.add(2) = (*a.add(0)) * (*b.add(2)) + (*a.add(2)) * (*b.add(3)) };
    unsafe { *out.add(3) = (*a.add(1)) * (*b.add(2)) + (*a.add(3)) * (*b.add(3)) };
}
