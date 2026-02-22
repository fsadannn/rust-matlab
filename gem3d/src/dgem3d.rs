use math_helpers::daxpy_simd;
use matlab_blas_wrapper::blas::{dgemm, dgemv};

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_feature = "sse2")))]
#[deny(warnings)]
compile_error!("This module only supports x86 and x86_64 architectures with sse3");

pub fn dgem3d(
    A: *const f64,
    a_dims: &[usize; 3],
    B: *const f64,
    b_dims: &[usize; 3],
    y: *const f64,
    y_dims: &[usize; 3],
    out: *mut f64,
) -> Result<(), String> {
    /* The options for A and B be compatible since A: A1xA2xA3 and B: B1xB2xB3
     * - A2 == B1 meaning we can multiply A and B across pages
     *  + A3 == B3 have the same pages
     *      ++ OUT: A1xB2xA3
     *  + A3 == B2 and B3 == 1, B is a vector across pages
     *      ++ OUT: A1x1xA3 -> simplified to a matrix of A1XA3
     * - A2 != B1 and B1 == 1 and B2 == A3 and B3 == 1, meaning B is a numeric constant across pages
     *      and have the same pages
     *  ++ OUT: A1xA2xA3
     */
    if (a_dims[1] == b_dims[0]
        && a_dims[2] != b_dims[2]
        && (a_dims[2] != b_dims[1] || b_dims[2] != 1))
        || (a_dims[1] != b_dims[0] && (b_dims[0] != 1 || b_dims[1] != a_dims[2] || b_dims[2] != 1))
    {
        return Err("Dimension mismatch for A and B".into());
    }

    let cross_page_dims = [
        a_dims[0],
        if a_dims[1] == b_dims[0] {
            if b_dims[2] == 1 { 1 } else { b_dims[1] }
        } else {
            a_dims[1]
        },
        a_dims[2],
    ];

    /* The options for C=A*B and Y be compatible since C: C1xC2xC3 and B: Y1xY2xY3
     * - C1 == Y1 and C2 = Y2 meaning we can add C and Y across pages
     *  + C3 == Y3 have the same pages
     *  + C3 != Y3 broadcast Y across pages
     *      ++ OUT: A1x1xA3 -> simplified to a matrix of A1XA3
     * - Y1==Y2==Y3==1 Y=0 meaning ignore the additive part
     */
    if (y_dims[0] != cross_page_dims[0]
        || y_dims[1] != cross_page_dims[1]
        || (y_dims[2] != cross_page_dims[2] && y_dims[2] != 1))
        && (y_dims.iter().sum::<usize>() != 3 || unsafe { *y.add(0) != 0.0 })
    {
        return Err("Dimension mismatch for A*B and C".into());
    }

    // if has sum term then we copy over output
    if y_dims.iter().sum::<usize>() == 3 && unsafe { *y.add(0) == 0.0 } {
        if y_dims[2] == cross_page_dims[2] {
            unsafe { std::ptr::copy_nonoverlapping(y, out, cross_page_dims.iter().product()) };
        } else {
            let n_elements = y_dims[0] * y_dims[1];
            for i in 0..cross_page_dims[2] {
                unsafe { std::ptr::copy_nonoverlapping(y, out.add(n_elements * i), n_elements) };
            }
        }
    }

    // B is constant cross pages
    if b_dims[0] == 1 {
        let n_elements = cross_page_dims[0] * cross_page_dims[1];
        for i in 0..cross_page_dims[2] {
            let factor = unsafe { *B.add(i) };
            unsafe {
                daxpy_simd(
                    factor,
                    A.add(i * n_elements),
                    out.add(i * n_elements),
                    n_elements,
                );
            }
        }

        return Ok(());
    }

    const CHN: *const u8 = "N\0".as_ptr();
    const ONE: *const f64 = &(1f64);

    // B is a vector
    if b_dims[0] == 1 && cross_page_dims[1] == 1 {
        const ONEI: *const usize = &(1usize);
        let n_elements_a = a_dims[1] * a_dims[0];
        let n_elements_b = b_dims[0];
        let rows = a_dims[0] as *const usize;
        let cols = a_dims[1] as *const usize;
        for i in 0..cross_page_dims[2] {
            unsafe {
                dgemv(
                    CHN,
                    rows,
                    cols,
                    ONE,
                    A.add(n_elements_a * i),
                    rows,
                    B.add(n_elements_b * i),
                    ONEI,
                    ONE,
                    out.add(n_elements_b * i),
                    ONEI,
                )
            };
        }

        return Ok(());
    }

    // B is a matrix
    let n_elements_a = a_dims[1] * a_dims[0];
    let n_elements_b = b_dims[0] * b_dims[1];
    let n_elements_c = cross_page_dims[0] * cross_page_dims[1];
    let rows = a_dims[0] as *const usize;
    let cols = a_dims[1] as *const usize;
    let colres = cross_page_dims[1] as *const usize;
    for i in 0..cross_page_dims[2] {
        unsafe {
            dgemm(
                CHN,
                CHN,
                rows,
                colres,
                cols,
                ONE,
                A.add(n_elements_a * i),
                rows,
                B.add(n_elements_b * i),
                cols,
                ONE,
                out.add(n_elements_c * i),
                rows,
            )
        };
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn test_dimensions() {
        let mut out = vec![0.0; 12];
        // A2 == B1 and A3 != B3 and A3 != B2
        let mut res = dgem3d(
            vec![0.0; 12].as_ptr(),
            &[2, 2, 3],
            vec![0.0; 12].as_ptr(),
            &[2, 1, 2],
            vec![0.0; 12].as_ptr(),
            &[1, 1, 1],
            out.as_mut_ptr(),
        );
        assert!(res.is_err() && res.unwrap_err() == "Dimension mismatch for A and B");
        // A2 == B1 and A3 != B3 and A3 == B2 and B3 != 1
        res = dgem3d(
            vec![0.0; 12].as_ptr(),
            &[2, 2, 3],
            vec![0.0; 12].as_ptr(),
            &[2, 3, 2],
            vec![0.0; 12].as_ptr(),
            &[1, 1, 1],
            out.as_mut_ptr(),
        );
        assert!(res.is_err() && res.unwrap_err() == "Dimension mismatch for A and B");
        // A2 != B1 and B1 != 1
        res = dgem3d(
            vec![0.0; 12].as_ptr(),
            &[2, 2, 3],
            vec![0.0; 12].as_ptr(),
            &[3, 1, 1],
            vec![0.0; 12].as_ptr(),
            &[1, 1, 1],
            out.as_mut_ptr(),
        );
        assert!(res.is_err() && res.unwrap_err() == "Dimension mismatch for A and B");
        // A2 != B1 and B1 == 1 and B2 != A3
        res = dgem3d(
            vec![0.0; 12].as_ptr(),
            &[2, 2, 3],
            vec![0.0; 12].as_ptr(),
            &[1, 2, 1],
            vec![0.0; 12].as_ptr(),
            &[1, 1, 1],
            out.as_mut_ptr(),
        );
        assert!(res.is_err() && res.unwrap_err() == "Dimension mismatch for A and B");
        // A2 != B1 and B1 == 1 and B2 == A3 and B3 != 1
        res = dgem3d(
            vec![0.0; 12].as_ptr(),
            &[2, 2, 3],
            vec![0.0; 12].as_ptr(),
            &[1, 3, 2],
            vec![0.0; 12].as_ptr(),
            &[1, 1, 1],
            out.as_mut_ptr(),
        );
        assert!(res.is_err() && res.unwrap_err() == "Dimension mismatch for A and B");
    }
}
