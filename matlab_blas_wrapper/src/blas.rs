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
    pub fn dtrmm(
        side: *const u8,
        uplo: *const u8,
        transa: *const u8,
        diag: *const u8,
        m: *const usize,
        n: *const usize,
        alpha: *const f64,
        a: *const f64,
        lda: *const usize,
        b: *const f64,
        ldb: *const usize,
    );
    pub fn dtrsm(
        side: *const u8,
        uplo: *const u8,
        transa: *const u8,
        diag: *const u8,
        m: *const usize,
        n: *const usize,
        alpha: *const f64,
        a: *const f64,
        lda: *const usize,
        b: *const f64,
        ldb: *const usize,
    );
    pub fn dgemv(
        trans: *const u8,
        m: *const usize,
        n: *const usize,
        alpha: *const f64,
        a: *const f64,
        lda: *const usize,
        x: *const f64,
        incx: *const usize,
        beta: *const f64,
        c: *mut f64,
        incy: *const usize,
    );
}
