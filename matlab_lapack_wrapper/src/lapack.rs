// #[link(name = "libmwlapack")]
unsafe extern "C" {
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
    pub fn dlange(
        norm: *const u8,
        m: *const usize,
        n: *const usize,
        a: *const f64,
        lda: *const usize,
        work: *mut f64,
    ) -> f64;
    pub fn dlantr(
        norm: *const u8,
        uplo: *const u8,
        diag: *const u8,
        m: *const usize,
        n: *const usize,
        a: *const f64,
        lda: *const usize,
        work: *mut f64,
    ) -> f64;
}
