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
}
