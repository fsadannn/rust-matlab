// #[link(name = "libmex")]
#[allow(unused)]
unsafe extern "C" {
    pub fn mexPrintf(fmt: *const u8, ...);
    pub fn mexErrMsgTxt(fmt: *const u8);
}
