use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn bench_datum_compense(c: &mut Criterion) {
    c.bench_function("datum_compense", |b| {
        b.iter(|| {
            geotool_algorithm::datum_compense(
                black_box(469704.6693f64),
                black_box(2821940.796f64),
                black_box(400.0f64),
                black_box(6_378_137.0f64),
                black_box(500_000.0f64),
                black_box(0.0f64),
            )
        })
    });
}

criterion_group!(benches, bench_datum_compense);
criterion_main!(benches);
