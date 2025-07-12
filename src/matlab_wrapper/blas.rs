// #[link(name = "libmwblas")]
unsafe extern "C" {
    pub fn dgemm(
        transa: *const u8,
        transb: *const u8,
        m: *const usize,
        n: *const usize,
        k: *const usize,
        alpha: *const f64,
        a: *const f64,
        lda: *const usize,
        b: *const f64,
        ldb: *const usize,
        beta: *const f64,
        c: *mut f64,
        ldc: *const usize,
    );
    pub fn dgesv(
        n: *const usize,
        nrhs: *const usize,
        a: *mut f64,
        lda: *const usize,
        ipiv: *mut isize,
        b: *mut f64,
        ldb: *const usize,
        info: *mut isize,
    );
}
