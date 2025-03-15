import cupy as cp


from pyxis.pyxis_cuda._utils import PTX_PATH, TDTYPE


class DatumCompenseCuda:
    def __init__(self) -> None:
        self.module = cp.RawModule(path=str(PTX_PATH / "datum_compense_cuda.ptx"))

    def datum_compense_cuda(
        self,
        dtype: TDTYPE,
        xc: cp.ndarray,
        yc: cp.ndarray,
        hb: float,
        radius: float,
        x0: float,
        y0: float,
    ) -> tuple[cp.ndarray, cp.ndarray]:
        fn = self.module.get_function(f"datum_compense_cuda_{dtype}")
        ratio = hb / radius / (1.0 + hb / radius)
        fn((100,), (100,), (xc, yc, ratio, x0, y0, xc, yc))
        return xc, yc
