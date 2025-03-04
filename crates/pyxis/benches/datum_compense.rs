use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn bench_datum_compense(c: &mut Criterion) {
    let parms = pyxis::DatumCompenseParms::new(400.0, 6_378_137.0, 500_000.0, 0.0);
    c.bench_function("datum_compense", |b| {
        b.iter(|| {
            pyxis::datum_compense(
                black_box(469704.6693f64),
                black_box(2821940.796f64),
                &parms,
            )
        })
    });
}

criterion_group!(benches, bench_datum_compense);
criterion_main!(benches);
