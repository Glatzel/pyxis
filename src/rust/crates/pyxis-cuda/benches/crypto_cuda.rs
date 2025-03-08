use std::sync::LazyLock;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use pyxis::crypto::*;
static COORDS: LazyLock<(Vec<f64>, Vec<f64>)> = LazyLock::new(|| {
    let num_points = 100;
    let x_min = 72.004;
    let x_max = 137.8347;
    let y_min = 0.8293;
    let y_max = 55.8271;

    let x_step = (x_max - x_min) / (num_points as f64).sqrt(); // sqrt(num_points) for equal distribution
    let y_step = (y_max - y_min) / (num_points as f64).sqrt();

    let mut vx = Vec::new();
    let mut vy = Vec::new();

    for i in 0..(num_points as f64).sqrt() as usize {
        for j in 0..(num_points as f64).sqrt() as usize {
            let x = x_min + i as f64 * x_step;
            let y = y_min + j as f64 * y_step;
            vx.push(x);
            vy.push(y);
        }
    }
    (vx, vy)
});
fn bench_crypto_exact_lonlat(c: &mut Criterion) {
    let mut group = c.benchmark_group("crypto_exact_cuda");
    let ctx = pyxis_cuda::PyxisCudaContext::default();
    for i in [4, 7, 10, 13].iter() {
        let threshold = 10.0f64.powi(-i);
        group.bench_with_input(BenchmarkId::new("bd2gcj", i), i, |b, _| {
            let lon = COORDS.0.clone();
            let lat = COORDS.1.clone();
            let mut dlon = ctx.from_slice(&lon);
            let mut dlat = ctx.from_slice(&lat);
            b.iter(|| {
                ctx.crypto_exact_cuda(
                    &mut dlon,
                    &mut dlat,
                    CryptoSpace::BD09,
                    CryptoSpace::GCJ02,
                    threshold,
                    CryptoThresholdMode::LonLat,
                    100,
                );
            })
        });
        group.bench_with_input(BenchmarkId::new("bd2wgs", i), i, |b, _| {
            let lon = COORDS.0.clone();
            let lat = COORDS.1.clone();
            let mut dlon = ctx.from_slice(&lon);
            let mut dlat = ctx.from_slice(&lat);
            b.iter(|| {
                ctx.crypto_exact_cuda(
                    &mut dlon,
                    &mut dlat,
                    CryptoSpace::BD09,
                    CryptoSpace::WGS84,
                    threshold,
                    CryptoThresholdMode::LonLat,
                    100,
                );
            })
        });
        group.bench_with_input(BenchmarkId::new("gcj2wgs", i), i, |b, _| {
            let lon = COORDS.0.clone();
            let lat = COORDS.1.clone();
            let mut dlon = ctx.from_slice(&lon);
            let mut dlat = ctx.from_slice(&lat);
            b.iter(|| {
                ctx.crypto_exact_cuda(
                    &mut dlon,
                    &mut dlat,
                    CryptoSpace::GCJ02,
                    CryptoSpace::WGS84,
                    threshold,
                    CryptoThresholdMode::LonLat,
                    100,
                );
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_crypto_exact_lonlat);
criterion_main!(benches);
