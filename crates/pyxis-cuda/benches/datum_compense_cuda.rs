use criterion::{Criterion, criterion_group, criterion_main};
use rand::Rng;
fn bench_datum_compense_cuda(c: &mut Criterion) {
    let _ctx = cust::quick_init().unwrap();

    let mut rng = rand::rng();
    let parms = pyxis::DatumCompenseParms::new(400.0, 6_378_137.0, 500_000.0, 0.0);
    let xc: Vec<f64> = (0..1000)
        .map(|_| 469704.6693 + rng.random::<f64>())
        .collect();
    let yc: Vec<f64> = (0..1000)
        .map(|_| 2821940.796 + rng.random::<f64>())
        .collect();
    let ctx = pyxis_cuda::PyxisCudaContext::new();
    let mut dxc = ctx.from_slice(&xc);
    let mut dyc = ctx.from_slice(&yc);
    c.bench_function("datum_compense", |b| {
        b.iter(|| ctx.datum_compense_cuda(&mut dxc, &mut dyc, &parms))
    });
}

criterion_group!(bench, bench_datum_compense_cuda);
criterion_main!(bench);
