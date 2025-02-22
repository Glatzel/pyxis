use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
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
    for i in [5, 8, 11, 14].iter() {
        let threshold = 10.0f64.powi(-i);
        group.bench_with_input(BenchmarkId::new("bd2wgs-exact", i), i, |b, i| {
            b.iter(|| {
                geotool_algorithm::bd09_to_wgs84_exact(
                    black_box(121.10271732371203),
                    black_box(30.61484572185035),
                    threshold,
                    1000,
                )
            })
        });
        group.bench_with_input(BenchmarkId::new("bd2gcj-exact", i), i, |b, i| {
            b.iter(|| {
                geotool_algorithm::bd09_to_gcj02_exact(
                    black_box(121.10271732371203),
                    black_box(30.61484572185035),
                    threshold,
                    1000,
                )
            })
        });
        group.bench_with_input(BenchmarkId::new("gcj2wgs-exact", i), i, |b, i| {
            b.iter(|| {
                geotool_algorithm::gcj02_to_wgs84_exact(
                    black_box(121.09626935575027),
                    black_box(30.608604331756705),
                    threshold,
                    1000,
                )
            })
        });
    }
    group.finish();
}
criterion_group!(benches, bench_crypto, bench_crypto_exact);
criterion_main!(benches);
