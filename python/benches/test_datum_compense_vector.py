import os

import numpy as np
import pytest

rng = np.random.default_rng(1337)
npr = rng.random
group = "datum compense vector"


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


def test_numba(benchmark, sample_coords):
    numba = pytest.importorskip("numba")

    @numba.njit()
    def vector(x, y, h, r=6378_137.0, x0=0.0, y0=500000.0):  # pragma: nocover
        q = h / r
        factor = q / (1 + q)
        for i in range(x.shape[0]):
            x[i] = x[i] - factor * (x[i] - x0)
        for i in range(y.shape[0]):
            y[i] = y[i] - factor * (y[i] - y0)
        return x, y

    benchmark.group = group + str(sample_coords[0])
    benchmark(vector, npr(sample_coords[0]) + 2821940.796, npr(sample_coords[0]) + 2821940.796 + 469704.6693, 400)


def test_numba_numpy(benchmark, sample_coords):
    numba = pytest.importorskip("numba")

    @numba.njit()
    def vector(x, y, h, r=6378_137.0, x0=0.0, y0=500000.0):  # pragma: nocover
        q = h / r
        factor = q / (1 + q)
        x = x - factor * (x - x0)
        y = y - factor * (y - y0)
        return x, y

    benchmark.group = group + str(sample_coords[0])
    benchmark(vector, npr(sample_coords[0]) + 2821940.796, npr(sample_coords[0]) + 2821940.796 + 469704.6693, 400)


def test_numpy(benchmark, sample_coords):
    def vector(x, y, h, r=6378_137.0, x0=0.0, y0=500000.0):
        q = h / r
        factor = q / (1 + q)
        x = x - factor * (x - x0)
        y = y - factor * (y - y0)
        return x, y

    benchmark.group = group + str(sample_coords[0])
    benchmark(vector, npr(sample_coords[0]) + 2821940.796, npr(sample_coords[0]) + 2821940.796 + 469704.6693, 400)


def test_numexpr(benchmark, sample_coords):
    ne = pytest.importorskip("numexpr")

    def vector(x, y, h, r=6378_137.0, x0=0.0, y0=500000.0):
        q = h / r
        factor = q / (1 + q)  # noqa: F841
        ne.evaluate("x - factor * (x - 500000)", out=x)
        ne.evaluate("y - factor * (y - 0)", out=y)
        return x, y

    benchmark.group = group + str(sample_coords[0])
    benchmark(vector, npr(sample_coords[0]) + 2821940.796, npr(sample_coords[0]) + 2821940.796 + 469704.6693, 400)


def test_pyxis_inplace(benchmark, sample_coords):
    import pyxis

    def vector(x, y, h):
        pyxis.datum_compense(x, y, h, radius=6378_137, x0=500_000, y0=0, clone=False)

    benchmark.group = group + str(sample_coords[0])
    benchmark(vector, npr(sample_coords[0]) + 2821940.796, npr(sample_coords[0]) + 2821940.796 + 469704.6693, 400)


def test_pyxis_copy(benchmark, sample_coords):
    import pyxis

    def vector(x, y, h):
        pyxis.datum_compense(x, y, h, radius=6378_137, x0=500_000, y0=0)

    benchmark.group = group + str(sample_coords[0])
    benchmark(vector, npr(sample_coords[0]) + 2821940.796, npr(sample_coords[0]) + 2821940.796 + 469704.6693, 400)
