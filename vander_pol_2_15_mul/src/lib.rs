#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
use std::os::raw::c_int;

use math_helpers::M128dAsF64s;
use matlab_base_wrapper::{
    mex::mexErrMsgTxt,
    mx::mxCreateDoubleMatrix,
    raw::{Rhs, mxArray, mxComplexity_mxREAL},
};

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_feature = "sse3")))]
#[deny(warnings)]
compile_error!("This module only supports x86 and x86_64 architectures with sse3");

// fn rust_to_cstring(rust_string: String) -> CString {
//     // CString::new takes anything that can be turned into a Vec<u8>,
//     // including String and &str.
//     match CString::new(rust_string) {
//         Ok(c_string) => c_string,
//         Err(e) => {
//             // Handle the error (e.g., log it, panic, or return a default/error CString)
//             // A common scenario for FFI might involve returning an error code to C.
//             panic!("Failed to convert Rust string to CString: {}", e);
//         }
//     }
// }

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
                "vander_pol_2_15_aditive: SSE2 instruction set is not supported on this platform.\n\0"
                    .as_ptr(),
            );
        }
        return;
    }

    if nrhs != 7 {
        // Letting the standard library do the work of making Rusts strings C-compatible
        unsafe {
            mexErrMsgTxt("vander_pol_2_15_aditive: 7 input arguments required.\n\0".as_ptr());
        }
    }

    if nlhs > 1 {
        // Letting the standard library do the work of making Rusts strings C-compatible
        unsafe {
            mexErrMsgTxt("vander_pol_2_15_aditive: too many output arguments.\n\0".as_ptr());
        }
    }

    if nlhs < 1 {
        // Letting the standard library do the work of making Rusts strings C-compatible
        unsafe {
            mexErrMsgTxt("vander_pol_2_15_aditive: too few output arguments.\n\0".as_ptr());
        }
    }

    let alpha = match rhslice.first() {
        Some(a) => {
            if !a.is_scalar() {
                unsafe {
                    mexErrMsgTxt(
                        b"vander_pol_2_15_aditive: first argument must be a number.\n\0".as_ptr(),
                    );
                }
            }
            a.get_scalar()
        }
        None => {
            unsafe {
                mexErrMsgTxt(
                    b"vander_pol_2_15_aditive: first argument must be a number.\n\0".as_ptr(),
                );
            }
            return;
        }
    };
    let omega = match rhslice.get(1) {
        Some(a) => {
            if !a.is_scalar() {
                unsafe {
                    mexErrMsgTxt(
                        b"vander_pol_2_15_aditive: first argument must be a number.\n\0".as_ptr(),
                    );
                }
            }
            a.get_scalar()
        }
        None => {
            unsafe {
                mexErrMsgTxt(
                    b"vander_pol_2_15_aditive: first argument must be a number.\n\0".as_ptr(),
                );
            }
            return;
        }
    };
    let sigma_1 = match rhslice.get(2) {
        Some(a) => {
            if !a.is_scalar() {
                unsafe {
                    mexErrMsgTxt(
                        b"vander_pol_2_15_aditive: first argument must be a number.\n\0".as_ptr(),
                    );
                }
            }
            a.get_scalar()
        }
        None => {
            unsafe {
                mexErrMsgTxt(
                    b"vander_pol_2_15_aditive: first argument must be a number.\n\0".as_ptr(),
                );
            }
            return;
        }
    };
    let x0 = match rhslice.get(3) {
        Some(a) => {
            let max_dim = *a.dimensions().iter().max().unwrap_or(&0);
            if max_dim != 2 {
                unsafe {
                    mexErrMsgTxt(
                        b"vander_pol_2_15_aditive: second argument must be a vector of 2 components.\n\0"
                            .as_ptr(),
                    );
                }
            }
            a.get_slice()
        }
        None => {
            unsafe {
                mexErrMsgTxt(
                    b"vander_pol_2_15_aditive: second argument must be a vector.\n\0".as_ptr(),
                );
            }
            return;
        }
    };

    let (t, n) = match rhslice.get(4) {
        Some(a) => {
            if a.dimensions().len() > 2 {
                unsafe {
                    mexErrMsgTxt(
                        b"vander_pol_2_15_aditive: 3rd argument must be a vector.\n\0".as_ptr(),
                    );
                }
            }
            let n: usize = *a.dimensions().iter().max().unwrap();
            (a.get_slice(), n)
        }
        None => {
            unsafe {
                mexErrMsgTxt(
                    b"vander_pol_2_15_aditive: 3rd argument must be a vector.\n\0".as_ptr(),
                );
            }
            return;
        }
    };

    let dW = match rhslice.get(5) {
        Some(a) => {
            if a.dimensions().len() > 2 {
                unsafe {
                    mexErrMsgTxt(
                        b"vander_pol_2_15_aditive: 4rd argument must be a mxArray.\n\0".as_ptr(),
                    );
                }
            }
            let nn: usize = *a.dimensions().iter().max().unwrap();
            if nn != n {
                unsafe {
                    mexErrMsgTxt(
                        b"vander_pol_2_15_aditive: 3rd and 4rd argument must be the same size.\n\0"
                            .as_ptr(),
                    );
                }
            }
            a.get_slice()
        }
        None => {
            unsafe {
                mexErrMsgTxt(
                    b"vander_pol_2_15_aditive: 4th argument must be mxArray.\n\0".as_ptr(),
                );
            }
            return;
        }
    };

    let dZ = match rhslice.get(6) {
        Some(a) => {
            if a.dimensions().len() > 2 {
                unsafe {
                    mexErrMsgTxt(
                        b"vander_pol_2_15_aditive: 4rd argument must be a mxArray.\n\0".as_ptr(),
                    );
                }
            }
            let nn: usize = *a.dimensions().iter().max().unwrap();
            if nn != n {
                unsafe {
                    mexErrMsgTxt(
                        b"vander_pol_2_15_aditive: 3rd and 4rd argument must be the same size.\n\0"
                            .as_ptr(),
                    );
                }
            }
            a.get_slice()
        }
        None => {
            unsafe {
                mexErrMsgTxt(
                    b"vander_pol_2_15_aditive: 4th argument must be mxArray.\n\0".as_ptr(),
                );
            }
            return;
        }
    };

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
    // let mut b = M128dAsF64s { scalars: [0.0; 2] };

    // assume the evenly spaced partition
    let h = unsafe { *t.get_unchecked(1) - *t.get_unchecked(0) };
    let h2_2 = h.powf(2.0) / 2.0;

    if is_x86_feature_detected!("fma") {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::{_mm_fmadd_pd, _mm_load1_pd, _mm_set1_pd, _mm_storeu_pd};
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::{_mm_fmadd_pd, _mm_load1_pd, _mm_set1_pd, _mm_storeu_pd};

        let h_vec = unsafe { _mm_set1_pd(h) };
        let h2_2_vec = unsafe { _mm_set1_pd(h2_2) };

        for i in 1..n {
            unsafe {
                // 1. Load scalars directly into SIMD (Broadcast)
                let v_I10 = _mm_load1_pd(dZ.as_ptr().add(i));

                // 2. Compute 'a' vector
                let x = yn.scalars[0];
                let y = yn.scalars[1];
                let alpha_1_x2 = alpha * (1.0 - x * x);

                a.scalars[0] = y;
                a.scalars[1] = alpha_1_x2 * y - omega * x;

                // 3. Compute 'b' vector
                let b = sigma_1 * x;

                // 4. Final yn update using FMA: yn = yn + (a * h) + (b * I1) + (Aa * h2_2) + (Ab * I10)
                // Compute Aa and Ab dot products
                let Aa = M128dAsF64s {
                    scalars: [
                        a.scalars[1],
                        -(2.0 * alpha * x * y + omega) * a.scalars[0] + alpha_1_x2 * a.scalars[1],
                    ],
                };
                let Ab = M128dAsF64s {
                    scalars: [b, b * alpha_1_x2],
                };

                // acc = (b * I1) + acc
                yn.scalars[1] += b * (*dW.as_ptr().add(i));
                // Use FMA for accumulation: acc = (a * h) + yn
                yn.simd = _mm_fmadd_pd(a.simd, h_vec, yn.simd);
                // acc = (Aa * h2_2) + acc
                yn.simd = _mm_fmadd_pd(Aa.simd, h2_2_vec, yn.simd);
                // acc = (Ab * I_10) + acc
                yn.simd = _mm_fmadd_pd(Ab.simd, v_I10, yn.simd);
                // acc = (Ba * I_01) + acc
                yn.scalars[1] += y * (h * (*dW.as_ptr().add(i)) - (*dZ.as_ptr().add(i)));

                // 6. Store result
                _mm_storeu_pd(res.add(2 * i), yn.simd);
            }
        }
    } else {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::{_mm_add_pd, _mm_load1_pd, _mm_mul_pd, _mm_set1_pd, _mm_storeu_pd};
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::{_mm_add_pd, _mm_load1_pd, _mm_mul_pd, _mm_set1_pd, _mm_storeu_pd};
        // Pre-calculate constants for the loop
        let h_vec = unsafe { _mm_set1_pd(h) };
        let h2_2_vec = unsafe { _mm_set1_pd(h2_2) };

        for i in 1..n {
            unsafe {
                // 1. Efficiently load and broadcast I_1 and I_10
                let v_I1 = _mm_load1_pd(dW.as_ptr().add(i));
                let v_I10 = _mm_load1_pd(dZ.as_ptr().add(i));

                let x = yn.scalars[0];
                let y = yn.scalars[1];
                let alpha_1_x2 = alpha * (1.0 - x * x);

                a.scalars[0] = yn.scalars[1];
                a.scalars[1] = alpha_1_x2 * y - omega * x;

                // 3. Compute 'b' vector
                let b = sigma_1 * x;

                // 5. Final yn update (Vectorized version of your commented math)
                // Logic: yn += (a * h) + (b * I_1) + (A*a * h2_2) + (A*b * I_10)

                let mut term1 = M128dAsF64s {
                    simd: _mm_mul_pd(a.simd, h_vec),
                };
                term1.scalars[1] += b * (*dW.as_ptr().add(i));

                // Dot products for A*a and A*b
                let Aa = M128dAsF64s {
                    scalars: [
                        a.scalars[1],
                        -(2.0 * alpha * x * y + omega) * a.scalars[0] + alpha_1_x2 * a.scalars[1],
                    ],
                };
                let Ab = M128dAsF64s {
                    scalars: [b, b * alpha_1_x2],
                };

                let term3 = _mm_mul_pd(Aa.simd, h2_2_vec);
                let term4 = _mm_mul_pd(Ab.simd, v_I10);

                // Sum everything into yn
                yn.simd = _mm_add_pd(yn.simd, term1.simd);
                yn.simd = _mm_add_pd(yn.simd, _mm_add_pd(term3, term4));
                yn.scalars[1] += y * (h * (*dW.as_ptr().add(i)) - (*dZ.as_ptr().add(i)));

                // 6. Store back to result matrix
                _mm_storeu_pd(res.add(2 * i), yn.simd);
            }
        }
    }
}
