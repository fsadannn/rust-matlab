use math_helpers::{dxpy, scale_unrolled};
use matlab_blas_wrapper::blas::dgemv;

pub fn exp_matrix_action(
    A: *const f64,    // d*d matrix
    x: *const f64,    // d-element input vector
    result: *mut f64, // d-element output vector
    term_a: *mut f64, // First d-element temporary vector
    term_b: *mut f64, // Second d-element temporary vector
    d: usize,         // Dimension of the matrix and vectors
) {
    // Initialize `result` = x (the k=0 term).
    unsafe { std::ptr::copy_nonoverlapping(x, result, d) };

    // Initialize the first term vector `term_a` = x. This will hold A^k*x / k!
    unsafe { std::ptr::copy_nonoverlapping(x, term_a, d) };

    // Loop to compute and add the remaining terms.
    for k in 1..=13 {
        // Step a: Multiply the previous term by A.
        // DGEMV: output_term := 1.0 * A * input_term + 0.0 * output_term
        unsafe {
            dgemv(
                "N\0".as_ptr(),
                &d,
                &d,
                &(1f64),
                A,
                &d,
                term_a,
                &(1usize),
                &(0f64),
                term_b,
                &(1usize),
            )
        };

        // Step b: Scale the `output_term` by 1/k. It now holds the complete new term.
        let scale = 1.0 / (k as f64);
        unsafe { scale_unrolled(term_b, term_a, d, scale) };

        // Step c: Add the new term (now in `output_term`) to the total sum in `result`.
        unsafe { dxpy(term_a, result, d) };
    }
}
