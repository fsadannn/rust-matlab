#[inline(always)]
pub fn ito_double_integral(n: usize, m: usize, dW: *const f64, h: f64, res: *mut f64) {
    if m == 1 {
        let mut sum = 0f64;
        for i in 0..n {
            sum += unsafe { *dW.add(i) };
        }
        unsafe { *res.add(0) = 0.5f64 * (sum * sum - h) };
        return;
    }

    let mut acc: Vec<f64> = vec![0f64; m];
    let mut sum: Vec<f64> = vec![0f64; m];
    assert!(acc.len() == m);
    assert!(sum.len() == m);

    for i in 0..(m - 1) {
        for j in (i + 1)..m {
            acc[j] = 0f64;
            sum[j] = 0f64;
        }
        for k in 0..n {
            let idx = k * m;
            let val = unsafe { *dW.add(idx + i) };
            acc[i] += val;
            for j in (i + 1)..m {
                sum[j] += val * acc[j];
                acc[j] += unsafe { *dW.add(idx + j) };
            }
        }
        unsafe { *res.add(i * m + i) = 0.5f64 * (acc[i] * acc[i] - h) };
        for j in (i + 1)..m {
            unsafe { *res.add(j * m + i) = sum[j] };
            unsafe { *res.add(i * m + j) = acc[i] * acc[j] - sum[j] };
        }
    }
    unsafe { *res.add(m * m - 1) = 0.5f64 * (acc[m - 1] * acc[m - 1] - h) };
}
