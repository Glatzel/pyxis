use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn bench_datum_compensate(c: &mut Criterion) {
    let parms = pyxis::DatumCompensateParms::new(400.0, 6_378_137.0, 500_000.0, 0.0);
    c.bench_function("datum_compensate", |b| {
        b.iter(|| {
            pyxis::datum_compensate(black_box(469704.6693f64), black_box(2821940.796f64), &parms)
        })
    });
}

criterion_group!(benches, bench_datum_compensate);
criterion_main!(benches);
