use crate::lapack;

/// Computes the infinity norm of a square matrix.
///
/// The infinity norm of a matrix is defined as the maximum absolute row sum of the matrix.
/// This function uses the LAPACK `dlange` function to compute the infinity norm.
///
/// # Arguments
///
/// * `n` - The order of the matrix (number of rows and columns).
/// * `a` - A pointer to the first element of the matrix stored in column-major order.
/// * `lda` - The leading dimension of the matrix `a` (the number of elements between successive
///   rows in memory, typically equal to or greater than `n`).
///
/// # Returns
///
/// The infinity norm of the matrix as a `f64` value.
///
/// # Safety
///
/// This function is `unsafe` because it calls an external C function and operates on raw pointers.
/// The caller must ensure that `a` is a valid pointer to a matrix with dimensions at least `n` x `n`,
/// and that the memory is properly aligned for `f64` values.
pub unsafe fn norm_inf(n: usize, a: *const f64, lda: usize) -> f64 {
    let mut work = vec![0.0; n];
    unsafe { lapack::dlange(b"I".as_ptr(), &n, &n, a, &lda, work.as_mut_ptr()) }
}

/// Computes the infinity norm of a square matrix.
///
/// The infinity norm of a matrix is defined as the maximum absolute row sum of the matrix.
/// This function uses the LAPACK `dlange` function to compute the infinity norm.
///
/// # Safety
///
/// This function is `unsafe` because it calls an external C function and operates on raw pointers.
/// The caller must ensure that `a` is a valid pointer to a matrix with dimensions at least `n` x `n`,
/// and that the memory is properly aligned for `f64` values.
pub unsafe fn norm_inf_tri_upper(n: usize, a: *const f64, lda: usize) -> f64 {
    let mut work = vec![0.0; n];
    unsafe {
        lapack::dlantr(
            b"I".as_ptr(),
            b"U".as_ptr(),
            b"N".as_ptr(),
            &n,
            &n,
            a,
            &lda,
            work.as_mut_ptr(),
        )
    }
}
