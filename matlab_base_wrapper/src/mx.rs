#![allow(non_camel_case_types)]
use super::raw::{mwSize, mxArray};

type size_t = usize;
pub type mxComplexity = ::std::os::raw::c_uint;

// #[link(name = "libmx")]
unsafe extern "C" {
    pub fn mxGetDimensions_800(pa: *const mxArray) -> *const mwSize;
    pub fn mxGetNumberOfDimensions_800(pa: *const mxArray) -> mwSize;
    pub fn mxGetNumberOfElements_800(pa: *const mxArray) -> size_t;
    pub fn mxIsComplex_800(pa: *const mxArray) -> bool;
    pub fn mxIsSparse_800(pa: *const mxArray) -> bool;
    pub fn mxIsDouble_800(pa: *const mxArray) -> bool;
    pub fn mxGetPr_800(pa: *const mxArray) -> *mut f64;
    pub fn mxCreateDoubleMatrix_800(m: mwSize, n: mwSize, flag: mxComplexity) -> *mut mxArray;
    pub fn mxGetScalar_800(pa: *const mxArray) -> f64;
}

pub use self::{
    mxCreateDoubleMatrix_800 as mxCreateDoubleMatrix, mxGetDimensions_800 as mxGetDimensions,
    mxGetNumberOfDimensions_800 as mxGetNumberOfDimensions,
    mxGetNumberOfElements_800 as mxGetNumberOfElements, mxGetPr_800 as mxGetPr,
    mxGetScalar_800 as mxGetScalar, mxIsComplex_800 as mxIsComplex, mxIsDouble_800 as mxIsDouble,
    mxIsSparse_800 as mxIsSparse,
};
