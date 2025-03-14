import cupy as cp

from pyxis.pyxis_cuda._utils import PTX_PATH, TDTYPE, TCoordArrayCuda, TCoordScalarCuda, array_util, scalar_util


class DatumCompenseCuda:
    def __init__(self) -> None:
        self.module = cp.RawModule(path=str(PTX_PATH / "datum_compense_cuda.ptx"))

    def datum_compense_cuda(
        self,
        dtype: TDTYPE,
        xc: TCoordArrayCuda,
        yc: TCoordArrayCuda,
        hb: TCoordScalarCuda,
        radius: TCoordScalarCuda,
        x0: TCoordScalarCuda,
        y0: TCoordScalarCuda,
    ) -> tuple[cp.ndarray, cp.ndarray]:
        xc = array_util(xc, dtype)
        yc = array_util(yc, dtype)
        x0 = scalar_util(x0, dtype)
        y0 = scalar_util(y0, dtype)
        fn = self.module.get_function("datum_compense_cuda_double")
        ratio = hb / radius / (1.0 + hb / radius)
        fn((100,), (100,), (xc, yc, ratio, x0, y0, xc, yc))
        return xc, yc
