use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::Rng;
fn bench_datum_compense_cuda(c: &mut Criterion) {
    let mut group = c.benchmark_group("datum_compense_cuda");
    let mut rng = rand::rng();
    let ctx = &pyxis_cuda::PyxisCudaContext::new();
    for i in [0, 2, 4, 6, 8].iter() {
        let parms = pyxis::DatumCompenseParms::new(400.0, 6_378_137.0, 500_000.0, 0.0);
        let count = 10i32.pow(*i);
        let xc: Vec<f64> = (0..count)
            .map(|_| 469704.6693 + rng.random::<f64>())
            .collect();
        let yc: Vec<f64> = (0..count)
            .map(|_| 2821940.796 + rng.random::<f64>())
            .collect();

        let mut dxc = ctx.from_slice(&xc);
        let mut dyc = ctx.from_slice(&yc);
        group.bench_with_input(BenchmarkId::new("length", i), i, |b, _| {
            b.iter(|| ctx.datum_compense_cuda(&mut dxc, &mut dyc, &parms))
        });
    }
}

criterion_group!(bench, bench_datum_compense_cuda);
criterion_main!(bench);
