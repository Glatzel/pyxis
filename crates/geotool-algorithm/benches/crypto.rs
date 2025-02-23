use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use geotool_algorithm::*;
use rand::Rng;
fn bench_crypto(c: &mut Criterion) {
    let mut group = c.benchmark_group("crypto");

    group.bench_function("bd2gcj", |b| {
        b.iter(|| geotool_algorithm::bd09_to_gcj02(black_box(121.0), black_box(30.0)))
    });
    group.bench_function("bd2wgs", |b| {
        b.iter(|| geotool_algorithm::bd09_to_wgs84(black_box(121.0), black_box(30.0)))
    });
    group.bench_function("gcj2wgs", |b| {
        b.iter(|| geotool_algorithm::gcj02_to_wgs84(black_box(121.0), black_box(30.0)))
    });
    group.bench_function("gcj2bd", |b| {
        b.iter(|| geotool_algorithm::gcj02_to_bd09(black_box(121.0), black_box(30.0)))
    });
    group.bench_function("wgs2bd", |b| {
        b.iter(|| geotool_algorithm::wgs84_to_bd09(black_box(121.0), black_box(30.0)))
    });
    group.bench_function("wgs2gcj", |b| {
        b.iter(|| geotool_algorithm::wgs84_to_gcj02(black_box(121.0), black_box(30.0)))
    });
    group.finish();
}
fn bench_crypto_exact(c: &mut Criterion) {
    let mut group = c.benchmark_group("crypto_exact");
    let mut rng = rand::rng();
    for i in [4, 7, 10, 13].iter() {
        let threshold = 10.0f64.powi(-i);
        group.bench_with_input(BenchmarkId::new("lonlat", i), i, |b, _| {
            b.iter(|| {
                for _ in 0..1000 {
                    let wgs = (
                        rng.random_range(72.004..137.8347),
                        rng.random_range(0.8293..55.8271),
                    );
                    let gcj = wgs84_to_gcj02(wgs.0, wgs.1);
                    let bd = wgs84_to_bd09(wgs.0, wgs.1);

                    crypto_exact(
                        bd.0,
                        bd.1,
                        bd09_to_gcj02,
                        gcj02_to_bd09,
                        threshold,
                        CryptoThresholdMode::LonLat,
                        1000,
                    );

                    crypto_exact(
                        bd.0,
                        bd.1,
                        bd09_to_wgs84,
                        wgs84_to_bd09,
                        threshold,
                        CryptoThresholdMode::LonLat,
                        1000,
                    );

                    crypto_exact(
                        gcj.0,
                        gcj.1,
                        gcj02_to_wgs84,
                        wgs84_to_gcj02,
                        threshold,
                        CryptoThresholdMode::LonLat,
                        1000,
                    );
                }
            })
        });
    }
    for i in [2, 3].iter() {
        // [1m, 1cm, 1mm]
        let threshold = 10.0f64.powi(-i);

        group.bench_with_input(BenchmarkId::new("distance", i), i, |b, _| {
            b.iter(|| {
                for _ in 0..1000 {
                    let wgs = (
                        rng.random_range(72.004..137.8347),
                        rng.random_range(0.8293..55.8271),
                    );
                    let gcj = wgs84_to_gcj02(wgs.0, wgs.1);
                    let bd = wgs84_to_bd09(wgs.0, wgs.1);

                    crypto_exact(
                        bd.0,
                        bd.1,
                        bd09_to_gcj02,
                        gcj02_to_bd09,
                        threshold,
                        CryptoThresholdMode::Distance,
                        1000,
                    );

                    crypto_exact(
                        bd.0,
                        bd.1,
                        bd09_to_wgs84,
                        wgs84_to_bd09,
                        threshold,
                        CryptoThresholdMode::Distance,
                        1000,
                    );

                    crypto_exact(
                        gcj.0,
                        gcj.1,
                        gcj02_to_wgs84,
                        wgs84_to_gcj02,
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
criterion_group!(benches, bench_crypto, bench_crypto_exact);
criterion_main!(benches);
