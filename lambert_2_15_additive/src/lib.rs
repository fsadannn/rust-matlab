#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
use std::{mem::swap, os::raw::c_int};

use math_helpers::{FnDGEM22, M128dAsF64s, dgemm_2x2_sse2, matrix_exp_22::matrix_exp_2x2};
use matlab_base_wrapper::{
    mex::mexErrMsgTxt,
    mx::mxCreateDoubleMatrix,
    raw::{Rhs, mxArray, mxComplexity_mxREAL},
};

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_feature = "sse3")))]
#[deny(warnings)]
compile_error!("This module only supports x86 and x86_64 architectures with sse3");

#[allow(unused_variables, clippy::not_unsafe_ptr_arg_deref)]
#[unsafe(no_mangle)]
pub extern "C" fn mexFunction(
    nlhs: c_int,
    plhs: *mut *mut mxArray,
    nrhs: c_int,
    prhs: *mut *mut mxArray,
) {
    let rhslice: Rhs =
        unsafe { ::std::slice::from_raw_parts(prhs as *const &mxArray, nrhs as usize) };
    let lhslice = unsafe {
        ::std::slice::from_raw_parts_mut(plhs as *mut Option<&mut mxArray>, nlhs as usize)
    };

    if !is_x86_feature_detected!("sse2") {
        unsafe {
            mexErrMsgTxt(
                "lambert_2_15: SSE2 instruction set is not supported on this platform.\n\0"
                    .as_ptr(),
            );
        }
        return;
    }

    if nrhs != 5 {
        // Letting the standard library do the work of making Rusts strings C-compatible
        unsafe {
            mexErrMsgTxt("lambert_2_15: 5 input arguments required.\n\0".as_ptr());
        }
    }

    if nlhs > 1 {
        // Letting the standard library do the work of making Rusts strings C-compatible
        unsafe {
            mexErrMsgTxt("lambert_2_15: too many output arguments.\n\0".as_ptr());
        }
    }

    if nlhs < 1 {
        // Letting the standard library do the work of making Rusts strings C-compatible
        unsafe {
            mexErrMsgTxt("lambert_2_15: too few output arguments.\n\0".as_ptr());
        }
    }

    let sigma_1 = match rhslice.first() {
        Some(a) => {
            if !a.is_scalar() {
                unsafe {
                    mexErrMsgTxt(b"lambert_2_15: first argument must be a number.\n\0".as_ptr());
                }
            }
            a.get_scalar()
        }
        None => {
            unsafe {
                mexErrMsgTxt(b"lambert_2_15: first argument must be a number.\n\0".as_ptr());
            }
            return;
        }
    };
    let x0 = match rhslice.get(1) {
        Some(a) => {
            let max_dim = *a.dimensions().iter().max().unwrap_or(&0);
            if max_dim != 2 {
                unsafe {
                    mexErrMsgTxt(
                        b"lambert_2_15: second argument must be a vector of 2 components.\n\0"
                            .as_ptr(),
                    );
                }
            }
            a.get_slice()
        }
        None => {
            unsafe {
                mexErrMsgTxt(b"lambert_2_15: second argument must be a vector.\n\0".as_ptr());
            }
            return;
        }
    };

    let (t, n) = match rhslice.get(2) {
        Some(a) => {
            if a.dimensions().len() > 2 {
                unsafe {
                    mexErrMsgTxt(b"lambert_2_15: 3rd argument must be a vector.\n\0".as_ptr());
                }
            }
            let n: usize = *a.dimensions().iter().max().unwrap();
            (a.get_slice(), n)
        }
        None => {
            unsafe {
                mexErrMsgTxt(b"lambert_2_15: 3rd argument must be a vector.\n\0".as_ptr());
            }
            return;
        }
    };

    let dW = match rhslice.get(3) {
        Some(a) => {
            if a.dimensions().len() > 2 {
                unsafe {
                    mexErrMsgTxt(b"lambert_2_15: 4rd argument must be a mxArray.\n\0".as_ptr());
                }
            }
            let nn: usize = *a.dimensions().iter().max().unwrap();
            if nn != n {
                unsafe {
                    mexErrMsgTxt(
                        b"lambert_2_15: 3rd and 4rd argument must be the same size.\n\0".as_ptr(),
                    );
                }
            }
            a.get_slice()
        }
        None => {
            unsafe {
                mexErrMsgTxt(b"lambert_2_15: 4th argument must be mxArray.\n\0".as_ptr());
            }
            return;
        }
    };

    let dZ = match rhslice.get(4) {
        Some(a) => {
            if a.dimensions().len() > 2 {
                unsafe {
                    mexErrMsgTxt(b"lambert_2_15: 5rd argument must be a mxArray.\n\0".as_ptr());
                }
            }
            let nn: usize = *a.dimensions().iter().max().unwrap();
            if nn != n {
                unsafe {
                    mexErrMsgTxt(
                        b"lambert_2_15: 3rd and 5rd argument must be the same size.\n\0".as_ptr(),
                    );
                }
            }
            a.get_slice()
        }
        None => {
            unsafe {
                mexErrMsgTxt(b"lambert_2_15: 5th argument must be mxArray.\n\0".as_ptr());
            }
            return;
        }
    };

    let mut dgemm: FnDGEM22 = dgemm_2x2_sse2;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            use math_helpers::dgemm_2x2_avx2;

            dgemm = dgemm_2x2_avx2
        }
    }

    let ans_matrix: *mut mxArray = unsafe { mxCreateDoubleMatrix(2, n, mxComplexity_mxREAL) };
    let res = unsafe { ans_matrix.as_mut().unwrap().get_ptr() };
    unsafe { *plhs.add(0) = ans_matrix };
    unsafe {
        *res.add(0) = x0[0];
    };
    unsafe {
        *res.add(1) = x0[1];
    };
    let mut yn = M128dAsF64s {
        scalars: [x0[0], x0[1]],
    };

    let mut a = M128dAsF64s { scalars: [0.0; 2] };
    let mut b = M128dAsF64s { scalars: [0.0; 2] };

    // assume the evenly spaced partition
    let h = unsafe { *t.get_unchecked(1) - *t.get_unchecked(0) };
    let h2_2 = h.powf(2.0) / 2.0;
    const A: [f64; 4] = [-2.0, 998.0, 1.0, -999.0];
    let Ah: [f64; 4] = [A[0] * h, A[1] * h, A[2] * h, A[3] * h];
    let expAh = matrix_exp_2x2(&Ah, 6);
    let mut expAt = [1.0, 0.0, 0.0, 1.0];
    let mut aux = [0.0; 4];

    let A_row_1 = M128dAsF64s {
        scalars: [A[0], A[2]],
    };
    let A_row_2 = M128dAsF64s {
        scalars: [A[1], A[3]],
    };

    if is_x86_feature_detected!("fma") {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::{
            _mm_add_pd, _mm_fmadd_pd, _mm_hadd_pd, _mm_load1_pd, _mm_mul_pd, _mm_set_pd,
            _mm_set1_pd, _mm_storeu_pd,
        };
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::{
            _mm_add_pd, _mm_fmadd_pd, _mm_hadd_pd, _mm_load1_pd, _mm_mul_pd, _mm_set_pd,
            _mm_set1_pd, _mm_storeu_pd,
        };

        let h_vec = unsafe { _mm_set1_pd(h) };
        let h2_2_vec = unsafe { _mm_set1_pd(h2_2) };
        let sigma_vec = unsafe { _mm_set1_pd(sigma_1) };

        for i in 1..n {
            let t_n = unsafe { *t.get_unchecked(i - 1) };
            let (t_sin, t_cos) = t_n.sin_cos();

            unsafe {
                // 1. Load scalars directly into SIMD (Broadcast)
                let v_I1 = _mm_load1_pd(dW.as_ptr().add(i));
                let v_I10 = _mm_load1_pd(dZ.as_ptr().add(i));

                // 2. Compute 'a' vector (A * yn + trig)
                let mul_a1 = _mm_mul_pd(A_row_1.simd, yn.simd);
                let mul_a2 = _mm_mul_pd(A_row_2.simd, yn.simd);
                let dot_a = _mm_hadd_pd(mul_a1, mul_a2);
                let trig_vec = _mm_set_pd(998.0 * (t_cos - t_sin), 2.0 * t_sin);
                a.simd = _mm_add_pd(dot_a, trig_vec);

                // 3. Compute 'b' vector
                let exp_cols = _mm_set_pd(expAt[1] + expAt[3], expAt[0] + expAt[2]);
                b.simd = _mm_mul_pd(sigma_vec, exp_cols);

                // 4. Matrix updates (Unchanged, usually handled by dgemm)
                dgemm(expAt.as_ptr(), expAh.as_ptr(), aux.as_mut_ptr());
                swap(&mut expAt, &mut aux);

                // 5. Final yn update using FMA: yn = yn + (a * h) + (b * I1) + (Aa * h2_2) + (Ab * I10)
                // Compute Aa and Ab dot products
                let Aa = _mm_hadd_pd(
                    _mm_mul_pd(A_row_1.simd, a.simd),
                    _mm_mul_pd(A_row_2.simd, a.simd),
                );
                let Ab = _mm_hadd_pd(
                    _mm_mul_pd(A_row_1.simd, b.simd),
                    _mm_mul_pd(A_row_2.simd, b.simd),
                );

                // Use FMA for accumulation: acc = (a * h) + yn
                let mut acc = _mm_fmadd_pd(a.simd, h_vec, yn.simd);
                // acc = (b * I1) + acc
                acc = _mm_fmadd_pd(b.simd, v_I1, acc);
                // acc = (Aa * h2_2) + acc
                acc = _mm_fmadd_pd(Aa, h2_2_vec, acc);
                // acc = (Ab * I10) + acc
                yn.simd = _mm_fmadd_pd(Ab, v_I10, acc);

                // 6. Store result
                _mm_storeu_pd(res.add(2 * i), yn.simd);
            }
        }
    } else {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::{
            _mm_add_pd, _mm_hadd_pd, _mm_load1_pd, _mm_mul_pd, _mm_set_pd, _mm_set1_pd,
            _mm_storeu_pd,
        };
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::{
            _mm_add_pd, _mm_hadd_pd, _mm_load1_pd, _mm_mul_pd, _mm_set_pd, _mm_set1_pd,
            _mm_storeu_pd,
        };
        // Pre-calculate constants for the loop
        let h_vec = unsafe { _mm_set1_pd(h) };
        let h2_2_vec = unsafe { _mm_set1_pd(h2_2) };
        let sigma_vec = unsafe { _mm_set1_pd(sigma_1) };

        for i in 1..n {
            let t_n = unsafe { *t.get_unchecked(i - 1) };

            // 1. Efficiently load and broadcast I_1 and I_10
            let v_I1 = unsafe { _mm_load1_pd(dW.as_ptr().add(i)) };
            let v_I10 = unsafe { _mm_load1_pd(dZ.as_ptr().add(i)) };

            let (t_sin, t_cos) = t_n.sin_cos();
            // a[0] = A[0] * yn[0] + A[2] * yn[1] + 2.0 * t_sin;
            // a[1] = A[1] * yn[0] + A[3] * yn[1] + 998.0 * (t_n.cos() - t_sin);
            // 2. Compute a = A * yn + trig_constants
            // We use the "HADD" pattern or manual shuffle to sum the dot products
            unsafe {
                let mul1 = _mm_mul_pd(A_row_1.simd, yn.simd);
                let mul2 = _mm_mul_pd(A_row_2.simd, yn.simd);

                // Horizontal add trick: [m1.0+m1.1, m2.0+m2.1]
                let dot_products = _mm_hadd_pd(mul1, mul2);

                let trig_vec = _mm_set_pd(998.0 * (t_cos - t_sin), 2.0 * t_sin);
                a.simd = _mm_add_pd(dot_products, trig_vec);
            }

            // 3. Compute b = sigma_1 * [expAt[0]+expAt[2], expAt[1]+expAt[3]]
            unsafe {
                let exp_cols = _mm_set_pd(expAt[1] + expAt[3], expAt[0] + expAt[2]);
                b.simd = _mm_mul_pd(sigma_vec, exp_cols);
            }

            // 4. Matrix updates
            unsafe {
                dgemm(expAt.as_ptr(), expAh.as_ptr(), aux.as_mut_ptr());
            }
            swap(&mut expAt, &mut aux);

            // 5. Final yn update (Vectorized version of your commented math)
            // Logic: yn += (a * h) + (b * I_1) + (A*a * h2_2) + (A*b * I_10)
            unsafe {
                let term1 = _mm_mul_pd(a.simd, h_vec);
                let term2 = _mm_mul_pd(b.simd, v_I1);

                // Dot products for A*a and A*b
                let Aa = _mm_hadd_pd(
                    _mm_mul_pd(A_row_1.simd, a.simd),
                    _mm_mul_pd(A_row_2.simd, a.simd),
                );
                let Ab = _mm_hadd_pd(
                    _mm_mul_pd(A_row_1.simd, b.simd),
                    _mm_mul_pd(A_row_2.simd, b.simd),
                );

                let term3 = _mm_mul_pd(Aa, h2_2_vec);
                let term4 = _mm_mul_pd(Ab, v_I10);

                // Sum everything into yn
                yn.simd = _mm_add_pd(yn.simd, _mm_add_pd(term1, term2));
                yn.simd = _mm_add_pd(yn.simd, _mm_add_pd(term3, term4));

                // 6. Store back to result matrix
                _mm_storeu_pd(res.add(2 * i), yn.simd);
            }
        }
    }
}
