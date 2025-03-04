use std::sync::LazyLock;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use pyxis::crypto::*;
static COORDS: LazyLock<Vec<(f64, f64)>> = LazyLock::new(|| {
    let num_points = 100;
    let x_min = 72.004;
    let x_max = 137.8347;
    let y_min = 0.8293;
    let y_max = 55.8271;

    let x_step = (x_max - x_min) / (num_points as f64).sqrt(); // sqrt(num_points) for equal distribution
    let y_step = (y_max - y_min) / (num_points as f64).sqrt();

    let mut coordinates = Vec::new();

    for i in 0..(num_points as f64).sqrt() as usize {
        for j in 0..(num_points as f64).sqrt() as usize {
            let x = x_min + i as f64 * x_step;
            let y = y_min + j as f64 * y_step;
            coordinates.push((x, y));
        }
    }

    coordinates
});
fn bench_crypto(c: &mut Criterion) {
    let mut group = c.benchmark_group("crypto");

    group.bench_function("bd2gcj", |b| {
        b.iter(|| bd09_to_gcj02(black_box(121.0), black_box(30.0)))
    });
    group.bench_function("bd2wgs", |b| {
        b.iter(|| bd09_to_wgs84(black_box(121.0), black_box(30.0)))
    });
    group.bench_function("gcj2wgs", |b| {
        b.iter(|| gcj02_to_wgs84(black_box(121.0), black_box(30.0)))
    });
    group.bench_function("gcj2bd", |b| {
        b.iter(|| gcj02_to_bd09(black_box(121.0), black_box(30.0)))
    });
    group.bench_function("wgs2bd", |b| {
        b.iter(|| wgs84_to_bd09(black_box(121.0), black_box(30.0)))
    });
    group.bench_function("wgs2gcj", |b| {
        b.iter(|| wgs84_to_gcj02(black_box(121.0), black_box(30.0)))
    });
    group.finish();
}
fn bench_crypto_exact_lonlat(c: &mut Criterion) {
    let mut group = c.benchmark_group("crypto_exact");

    for i in [4, 7, 10, 13].iter() {
        let threshold = 10.0f64.powi(-i);
        group.bench_with_input(BenchmarkId::new("bd2gcj", i), i, |b, _| {
            b.iter(|| {
                for p in COORDS.iter() {
                    crypto_exact(
                        black_box(p.0),
                        black_box(p.1),
                        &bd09_to_gcj02,
                        &gcj02_to_bd09,
                        threshold,
                        CryptoThresholdMode::LonLat,
                        1000,
                    );
                }
            })
        });
        group.bench_with_input(BenchmarkId::new("bd2wgs", i), i, |b, _| {
            b.iter(|| {
                for p in COORDS.iter() {
                    crypto_exact(
                        black_box(p.0),
                        black_box(p.1),
                        &bd09_to_wgs84,
                        &wgs84_to_bd09,
                        threshold,
                        CryptoThresholdMode::LonLat,
                        1000,
                    );
                }
            })
        });
        group.bench_with_input(BenchmarkId::new("gcj2wgs", i), i, |b, _| {
            b.iter(|| {
                for p in COORDS.iter() {
                    crypto_exact(
                        black_box(p.0),
                        black_box(p.1),
                        &gcj02_to_wgs84,
                        &wgs84_to_gcj02,
                        threshold,
                        CryptoThresholdMode::LonLat,
                        1000,
                    );
                }
            })
        });
    }
    group.finish();
}
fn bench_crypto_exact_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("crypto-exact-distance");
    for i in [3, 6].iter() {
        // [1m, 1cm, 1mm]
        let threshold = 10.0f64.powi(-i);
        group.bench_with_input(BenchmarkId::new("bd2gcj", i), i, |b, _| {
            b.iter(|| {
                for p in COORDS.iter() {
                    crypto_exact(
                        black_box(p.0),
                        black_box(p.1),
                        &bd09_to_gcj02,
                        &gcj02_to_bd09,
                        threshold,
                        CryptoThresholdMode::Distance,
                        1000,
                    );
                }
            })
        });
        group.bench_with_input(BenchmarkId::new("bd2wgs", i), i, |b, _| {
            b.iter(|| {
                for p in COORDS.iter() {
                    crypto_exact(
                        black_box(p.0),
                        black_box(p.1),
                        &bd09_to_wgs84,
                        &wgs84_to_bd09,
                        threshold,
                        CryptoThresholdMode::Distance,
                        1000,
                    );
                }
            })
        });
        group.bench_with_input(BenchmarkId::new("gcj2wgs", i), i, |b, _| {
            b.iter(|| {
                for p in COORDS.iter() {
                    crypto_exact(
                        black_box(p.0),
                        black_box(p.1),
                        &gcj02_to_wgs84,
                        &wgs84_to_gcj02,
                        threshold,
                        CryptoThresholdMode::Distance,
                        1000,
                    );
                }
            })
        });
    }
    group.finish();
}
criterion_group!(
    benches,
    bench_crypto,
    bench_crypto_exact_lonlat,
    bench_crypto_exact_distance
);
criterion_main!(benches);
