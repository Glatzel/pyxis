use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_datum_compense(c: &mut Criterion) {
    c.bench_function("datum_compense", |b| {
        b.iter(|| {
            geotool_algorithm::datum_compense(
                black_box(469704.6693),
                black_box(2821940.796),
                black_box(400.0),
                black_box(6378_137.0),
                black_box(500_000.0),
                black_box(0.0),
            )
        })
    });
}

criterion_group!(benches, bench_datum_compense);
criterion_main!(benches);
