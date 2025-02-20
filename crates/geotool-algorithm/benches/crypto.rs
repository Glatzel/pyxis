use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
fn bench_gcj02_to_wgs84(c: &mut Criterion) {
    let mut group = c.benchmark_group("GCJ02 to WGS84");

    group.bench_function("fast", |b| {
        b.iter(|| {
            geotool_algorithm::gcj02_to_wgs84(
                black_box(121.09626935575027),
                black_box(30.608604331756705),
            )
        })
    });
    for i in [1e-5, 1e-10].iter() {
        group.bench_with_input(BenchmarkId::new("exact", format!("{i:.2e}")), i, |b, i| {
            b.iter(|| {
                geotool_algorithm::gcj02_to_wgs84_exact(
                    black_box(121.09626935575027),
                    black_box(30.608604331756705),
                    *i,
                    1000,
                )
            })
        });
    }
    group.finish();
}
criterion_group!(benches, bench_gcj02_to_wgs84);
criterion_main!(benches);
