import os

import cupy as cp  # type: ignore
import numpy as np
import pytest
import pyxis_cuda

rng = np.random.default_rng(1337)
npr = rng.random
group = "datum compensate vector"


def data_figures():
    if os.getenv("CI"):
        return [5]
    else:  # pragma: nocover
        return [5]


@pytest.fixture(params=data_figures(), scope="module")
def sample_coords(request):
    # We pass in the complete file contents, because we don't want file IO
    # to skew results.
    return 10**request.param, 2821940.796, 469704.6693, 400.0


@pytest.mark.cuda
def test_pyxis_cuda(benchmark, sample_coords):
    def vector(x, y, h):
        module = pyxis_cuda.DatumCompensateCuda()
        module.datum_compensate_cuda("double", x, y, h, radius=6378_137, x0=500_000, y0=0)

    benchmark.group = group + str(sample_coords[0])
    benchmark(
        vector,
        cp.asarray(npr(sample_coords[0]) + 2821940.796),
        cp.asarray(npr(sample_coords[0]) + 2821940.796 + 469704.6693),
        400,
    )
