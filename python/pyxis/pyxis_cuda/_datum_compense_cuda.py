import cupy as cp

from pyxis.pyxis_cuda import get_grid_block
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
    ):
        fn = self.module.get_function(f"datum_compense_cuda_{dtype}")
        ratio = hb / radius / (1.0 + hb / radius)
        grid_size, block_size = get_grid_block(xc.size)
        fn((grid_size,), (block_size,), (xc, yc, ratio, x0, y0, xc, yc))
        cp.cuda.Stream.null.synchronize()
