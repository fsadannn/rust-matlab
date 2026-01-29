#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
use std::os::raw::c_int;

use matlab_base_wrapper::{
    mex::mexErrMsgTxt,
    mx::mxCreateDoubleMatrix,
    raw::{Rhs, mxArray, mxComplexity_mxREAL},
};

fn expm(a: f64, b: f64) -> (f64, f64) {
    let exp_a = a.exp();
    let mut phi = b;
    if a.abs() > 1e-16 {
        phi = ((exp_a - 1.0) / a) * b;
    }
    (exp_a, phi)
}

#[allow(unused_variables)]
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

    if nrhs != 6 {
        // Letting the standard library do the work of making Rusts strings C-compatible
        unsafe {
            mexErrMsgTxt("lin_euler_maruyama_multi: 6 input arguments required.\n\0".as_ptr());
        }
    }

    if nlhs > 1 {
        // Letting the standard library do the work of making Rusts strings C-compatible
        unsafe {
            mexErrMsgTxt("lin_euler_maruyama_multi: too many output arguments.\n\0".as_ptr());
        }
    }

    if nlhs < 1 {
        // Letting the standard library do the work of making Rusts strings C-compatible
        unsafe {
            mexErrMsgTxt("lin_euler_maruyama_multi: too few output arguments.\n\0".as_ptr());
        }
    }

    let amx = match rhslice.first() {
        Some(a) => a,
        None => {
            unsafe {
                mexErrMsgTxt(
                    b"lin_euler_maruyama_multi: first argument must be a mxArray.\n\0".as_ptr(),
                );
            }
            return;
        }
    };
    let bmx = match rhslice.get(1) {
        Some(a) => a,
        None => {
            unsafe {
                mexErrMsgTxt(
                    b"lin_euler_maruyama_multi: second argument must be a mxArray.\n\0".as_ptr(),
                );
            }
            return;
        }
    };
    let sigmamx = match rhslice.get(2) {
        Some(a) => a,
        None => {
            unsafe {
                mexErrMsgTxt(
                    b"lin_euler_maruyama_multi: second argument must be a mxArray.\n\0".as_ptr(),
                );
            }
            return;
        }
    };
    let x0mx = match rhslice.get(3) {
        Some(a) => a,
        None => {
            unsafe {
                mexErrMsgTxt(
                    b"lin_euler_maruyama_multi: second argument must be a mxArray.\n\0".as_ptr(),
                );
            }
            return;
        }
    };
    let tmx = match rhslice.get(4) {
        Some(a) => a,
        None => {
            unsafe {
                mexErrMsgTxt(
                    b"lin_euler_maruyama_multi: second argument must be a mxArray.\n\0".as_ptr(),
                );
            }
            return;
        }
    };
    let dWmx = match rhslice.get(5) {
        Some(a) => a,
        None => {
            unsafe {
                mexErrMsgTxt(
                    b"lin_euler_maruyama_multi: second argument must be a mxArray.\n\0".as_ptr(),
                );
            }
            return;
        }
    };

    let n: usize = *tmx.dimensions().iter().max().unwrap();

    let a: f64 = amx.get_scalar();
    let b: f64 = bmx.get_scalar();
    let sigma: f64 = sigmamx.get_scalar();
    let t: *mut f64 = tmx.get_ptr();
    let dW: *mut f64 = dWmx.get_ptr();

    let ans_matrix: *mut mxArray = unsafe { mxCreateDoubleMatrix(1, n, mxComplexity_mxREAL) };
    let res = unsafe { ans_matrix.as_mut().unwrap().get_ptr() };
    unsafe { *plhs.add(0) = ans_matrix };

    unsafe {
        *res.add(0) = x0mx.get_scalar();
    };

    let mut yn: f64 = unsafe { *res.add(0) };

    for i in 1..n {
        let h: f64 = unsafe { *t.add(i) } - unsafe { *t.add(i - 1) };
        let fx = a - 3.0 * b * (yn * yn);
        let gx = sigma;
        let c_1 = h * (fx - 0.5 * (gx * gx)) + unsafe { *dW.add(i) } * gx;
        let c_2 = h * (a * yn - b * yn.powi(3) - fx * yn);
        let (e1, e2) = expm(c_1, c_2);

        let yn1 = e1 * yn + e2;
        unsafe { *res.add(i) = yn1 };
        yn = yn1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expm_1() {
        let (a, b) = expm(1.0, 0.0);
        assert!((a - 1f64.exp()).abs() < 1e-16);
        assert!((b - 0.0).abs() < 1e-16);
    }

    #[test]
    fn test_expm_2() {
        let (a, b) = expm(1.0, 1.0);
        assert!((a - 1f64.exp()).abs() < 1e-15);
        assert!((b - 1.718281828459045).abs() < 1e-15);
    }

    #[test]
    fn test_expm_3() {
        let (a, b) = expm(0.010874482458667, 0.018512488484005);
        assert!((a - 0.010874482458667f64.exp()).abs() < 1e-15);
        assert!((b - 0.018613511207509f64).abs() < 1e-15);
    }
}
