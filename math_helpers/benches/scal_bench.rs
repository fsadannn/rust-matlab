use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use math_helpers::{
    scale_unrolled, scale_unrolled_avx, scale_unrolled_fallback, scale_unrolled_simd,
};
use rand::{Rng, rng};
use std::hint::black_box;

#[derive(Clone)]
struct ScaleBenchCase {
    size: usize,
}

fn bench_scal(c: &mut Criterion) {
    let cases = vec![
        ScaleBenchCase { size: 4 },
        ScaleBenchCase { size: 25 },
        ScaleBenchCase { size: 64 },
        ScaleBenchCase { size: 81 },
        ScaleBenchCase { size: 100 },
        ScaleBenchCase { size: 256 },
    ];

    for case in cases {
        let case = case.clone();
        let mut name = format!("scale_unrolled{}", case.size);
        c.bench_function(&name, |b| {
            b.iter_batched(
                || {
                    let mut v = vec![0.0_f64; case.size];
                    let res = vec![0.0_f64; case.size];
                    let mut ran = rng();
                    ran.fill(v.as_mut_slice());
                    let alpha = ran.random::<f64>();
                    (v, res, alpha)
                },
                |(v, mut res, alpha)| {
                    unsafe {
                        scale_unrolled(
                            black_box(v.as_ptr()),
                            black_box(res.as_mut_ptr()),
                            black_box(v.len()),
                            black_box(alpha),
                        )
                    };
                },
                BatchSize::SmallInput,
            )
        });

        name = format!("scale_unrolled_avx{}", case.size);
        c.bench_function(&name, |b| {
            b.iter_batched(
                || {
                    let mut v = vec![0.0_f64; case.size];
                    let res = vec![0.0_f64; case.size];
                    let mut ran = rng();
                    ran.fill(v.as_mut_slice());
                    let alpha = ran.random::<f64>();
                    (v, res, alpha)
                },
                |(v, mut res, alpha)| {
                    unsafe {
                        scale_unrolled_avx(
                            black_box(v.as_ptr()),
                            black_box(res.as_mut_ptr()),
                            black_box(v.len()),
                            black_box(alpha),
                        )
                    };
                },
                BatchSize::SmallInput,
            )
        });
        name = format!("scale_unrolled_simd_{}", case.size);
        c.bench_function(&name, |b| {
            b.iter_batched(
                || {
                    let mut v = vec![0.0_f64; case.size];
                    let res = vec![0.0_f64; case.size];
                    let mut ran = rng();
                    ran.fill(v.as_mut_slice());
                    let alpha = ran.random::<f64>();
                    (v, res, alpha)
                },
                |(v, mut res, alpha)| {
                    unsafe {
                        scale_unrolled_simd(
                            black_box(v.as_ptr()),
                            black_box(res.as_mut_ptr()),
                            black_box(v.len()),
                            black_box(alpha),
                        )
                    };
                },
                BatchSize::SmallInput,
            )
        });
        name = format!("scale_unrolled_fallback{}", case.size);
        c.bench_function(&name, |b| {
            b.iter_batched(
                || {
                    let mut v = vec![0.0_f64; case.size];
                    let res = vec![0.0_f64; case.size];
                    let mut ran = rng();
                    ran.fill(v.as_mut_slice());
                    let alpha = ran.random::<f64>();
                    (v, res, alpha)
                },
                |(v, mut res, alpha)| {
                    unsafe {
                        scale_unrolled_fallback(
                            black_box(v.as_ptr()),
                            black_box(res.as_mut_ptr()),
                            black_box(v.len()),
                            black_box(alpha),
                        )
                    };
                },
                BatchSize::SmallInput,
            )
        });
    }
}

criterion_group!(benches, bench_scal);
criterion_main!(benches);
