import cupy as cp
import pytest
from pyxis_cuda import DatumCompensateCuda


@pytest.mark.cuda
def test_datum_compensate_cuda():
    x = cp.array([469704.6693] * 2, dtype=cp.float64)
    y = cp.array([2821940.796] * 2, dtype=cp.float64)
    module = DatumCompensateCuda()
    module.datum_compensate_cuda("double", x, y, 400.0, 6378137.0, 500000.0, 0.0)
    x = cp.asnumpy(x)
    y = cp.asnumpy(y)
    assert pytest.approx(x[1]) == 469706.56912942487
    assert pytest.approx(y[1]) == 2821763.831232311
