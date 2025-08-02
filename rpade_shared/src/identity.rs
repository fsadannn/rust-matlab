/// Sets the diagonal elements of two matrices P and Q to 1.0.
/// This version uses loop unrolling for potential performance improvement.
///
/// # Arguments
/// * `P` - A mutable raw pointer to the first matrix (f64).
/// * `Q` - A mutable raw pointer to the second matrix (f64).
/// * `nrows` - The number of rows in the matrices.
/// * `ncols` - The number of columns in the matrices.
#[allow(non_snake_case)]
pub fn set_identity2_unrolled(P: *mut f64, Q: *mut f64, nrows: usize, ncols: usize) {
    // Calculate the number of iterations that can be processed in chunks of 4
    let unrolled_limit = nrows - (nrows % 4);

    // Unrolled loop: processes 4 iterations at a time
    for i in (0..unrolled_limit).step_by(4) {
        unsafe {
            // Process i
            *P.wrapping_add(i * ncols + i) = 1.0;
            *Q.wrapping_add(i * ncols + i) = 1.0;

            // Process i + 1
            *P.wrapping_add((i + 1) * ncols + (i + 1)) = 1.0;
            *Q.wrapping_add((i + 1) * ncols + (i + 1)) = 1.0;

            // Process i + 2
            *P.wrapping_add((i + 2) * ncols + (i + 2)) = 1.0;
            *Q.wrapping_add((i + 2) * ncols + (i + 2)) = 1.0;

            // Process i + 3
            *P.wrapping_add((i + 3) * ncols + (i + 3)) = 1.0;
            *Q.wrapping_add((i + 3) * ncols + (i + 3)) = 1.0;
        }
    }

    // Cleanup loop: processes any remaining iterations (0 to 3 iterations)
    for i in unrolled_limit..nrows {
        unsafe {
            *P.wrapping_add(i * ncols + i) = 1.0;
            *Q.wrapping_add(i * ncols + i) = 1.0;
        }
    }
}

// Example usage (for demonstration, you can put this in a `main` function or test)
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_identity_unrolled() {
        let nrows = 5;
        let ncols = 5;
        let mut p_data = vec![0.0; nrows * ncols];
        let mut q_data = vec![0.0; nrows * ncols];

        // Get raw pointers
        let p_ptr = p_data.as_mut_ptr();
        let q_ptr = q_data.as_mut_ptr();

        set_identity2_unrolled(p_ptr, q_ptr, nrows, ncols);

        // Verify the results
        for i in 0..nrows {
            for j in 0..ncols {
                let index = i * ncols + j;
                if i == j {
                    assert_eq!(p_data[index], 1.0);
                    assert_eq!(q_data[index], 1.0);
                } else {
                    assert_eq!(p_data[index], 0.0);
                    assert_eq!(q_data[index], 0.0);
                }
            }
        }

        // Test with nrows that is a multiple of 4
        let nrows_mult_4 = 8;
        let ncols_mult_4 = 8;
        let mut p_data_mult_4 = vec![0.0; nrows_mult_4 * ncols_mult_4];
        let mut q_data_mult_4 = vec![0.0; nrows_mult_4 * ncols_mult_4];
        let p_ptr_mult_4 = p_data_mult_4.as_mut_ptr();
        let q_ptr_mult_4 = q_data_mult_4.as_mut_ptr();

        set_identity2_unrolled(p_ptr_mult_4, q_ptr_mult_4, nrows_mult_4, ncols_mult_4);

        for i in 0..nrows_mult_4 {
            for j in 0..ncols_mult_4 {
                let index = i * ncols_mult_4 + j;
                if i == j {
                    assert_eq!(p_data_mult_4[index], 1.0);
                    assert_eq!(q_data_mult_4[index], 1.0);
                } else {
                    assert_eq!(p_data_mult_4[index], 0.0);
                    assert_eq!(q_data_mult_4[index], 0.0);
                }
            }
        }
    }
}
