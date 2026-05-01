#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
use std::os::raw::c_int;

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
                "landau: SSE2 instruction set is not supported on this platform.\n\0".as_ptr(),
            );
        }
        return;
    }

    if nrhs != 6 {
        // Letting the standard library do the work of making Rusts strings C-compatible
        unsafe {
            mexErrMsgTxt("landau: 7 input arguments required.\n\0".as_ptr());
        }
    }

    if nlhs > 1 {
        // Letting the standard library do the work of making Rusts strings C-compatible
        unsafe {
            mexErrMsgTxt("landau: too many output arguments.\n\0".as_ptr());
        }
    }

    if nlhs < 1 {
        // Letting the standard library do the work of making Rusts strings C-compatible
        unsafe {
            mexErrMsgTxt("landau: too few output arguments.\n\0".as_ptr());
        }
    }

    let alpha = match rhslice.first() {
        Some(a) => {
            if !a.is_scalar() {
                unsafe {
                    mexErrMsgTxt(b"landau: first argument must be a number.\n\0".as_ptr());
                }
            }
            a.get_scalar()
        }
        None => {
            unsafe {
                mexErrMsgTxt(b"landau: first argument must be a number.\n\0".as_ptr());
            }
            return;
        }
    };
    let omega = match rhslice.get(1) {
        Some(a) => {
            if !a.is_scalar() {
                unsafe {
                    mexErrMsgTxt(b"landau: 2nd argument must be a number.\n\0".as_ptr());
                }
            }
            a.get_scalar()
        }
        None => {
            unsafe {
                mexErrMsgTxt(b"landau: 2nd argument must be a number.\n\0".as_ptr());
            }
            return;
        }
    };
    let x0 = match rhslice.get(2) {
        Some(a) => {
            if !a.is_scalar() {
                unsafe {
                    mexErrMsgTxt(b"landau: 3rd argument must be a number.\n\0".as_ptr());
                }
            }
            a.get_scalar()
        }
        None => {
            unsafe {
                mexErrMsgTxt(b"landau: 3rd argument must be a number.\n\0".as_ptr());
            }
            return;
        }
    };
    let (t, n) = match rhslice.get(3) {
        Some(a) => {
            if a.dimensions().len() > 2 {
                unsafe {
                    mexErrMsgTxt(b"landau: 4th argument must be a vector.\n\0".as_ptr());
                }
            }
            let n: usize = *a.dimensions().iter().max().unwrap();
            (a.get_slice(), n)
        }
        None => {
            unsafe {
                mexErrMsgTxt(b"landau: 4th argument must be a vector.\n\0".as_ptr());
            }
            return;
        }
    };
    let dW = match rhslice.get(4) {
        Some(a) => {
            if a.dimensions().len() > 2 {
                unsafe {
                    mexErrMsgTxt(b"landau: 5th argument must be a mxArray.\n\0".as_ptr());
                }
            }
            let nn: usize = *a.dimensions().iter().max().unwrap();
            if nn != n {
                unsafe {
                    mexErrMsgTxt(
                        b"landau: 5th and 4th argument must be the same size.\n\0".as_ptr(),
                    );
                }
            }
            a.get_slice()
        }
        None => {
            unsafe {
                mexErrMsgTxt(b"landau: 5th argument must be mxArray.\n\0".as_ptr());
            }
            return;
        }
    };
    let dZ = match rhslice.get(5) {
        Some(a) => {
            if a.dimensions().len() > 2 {
                unsafe {
                    mexErrMsgTxt(b"landau: 6th argument must be a mxArray.\n\0".as_ptr());
                }
            }
            let nn: usize = *a.dimensions().iter().max().unwrap();
            if nn != n {
                unsafe {
                    mexErrMsgTxt(
                        b"landau: 6th and 4th argument must be the same size.\n\0".as_ptr(),
                    );
                }
            }
            a.get_slice()
        }
        None => {
            unsafe {
                mexErrMsgTxt(b"landau: 6th argument must be mxArray.\n\0".as_ptr());
            }
            return;
        }
    };

    let ans_matrix: *mut mxArray = unsafe { mxCreateDoubleMatrix(1, n, mxComplexity_mxREAL) };
    let res = unsafe { ans_matrix.as_mut().unwrap().get_ptr() };
    unsafe { *plhs.add(0) = ans_matrix };
    unsafe {
        *res.add(0) = x0;
    };
    let omega2 = omega * omega;
    let coef = alpha + omega2 * 0.5;

    let h = unsafe { *t.get_unchecked(1) - *t.get_unchecked(0) };
    let h2_2 = h.powf(2.0) * 0.5;
    let mut y_n = x0;

    for (i, (&dw, &dz)) in dW.iter().zip(dZ.iter()).enumerate() {
        let a = coef * y_n - y_n.powi(3);
        let ax = coef - 3.0 * (y_n * y_n);
        let b = y_n * omega;

        y_n += a * h
            + b * dw
            + (omega * b) * (0.5 * (dw * dw - h))
            + (ax * a + 0.5 * (b * b * (-6.0 * y_n))) * h2_2
            + (ax * b) * dz
            + (omega * a) * (dw * h - dz)
            + (omega2 * b) * (0.5 * (((dw * dw) / 3.0 - h) * dw));
        unsafe {
            *res.add(i) = y_n;
        };
    }
}
