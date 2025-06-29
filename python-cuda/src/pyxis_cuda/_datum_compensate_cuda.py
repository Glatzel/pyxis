import cupy as cp  # type: ignore

from ._context import get_grid_block
from ._utils import PTX_PATH, TDTYPE


class DatumCompensateCuda:
    def __init__(self) -> None:
        self.module = cp.RawModule(path=str(PTX_PATH / "datum_compensate_cuda.ptx"))

    def datum_compensate_cuda(
        self,
        dtype: TDTYPE,
        xc: cp.ndarray,
        yc: cp.ndarray,
        hb: float,
        radius: float,
        x0: float,
        y0: float,
    ):
        fn = self.module.get_function(f"datum_compensate_cuda_{dtype}")
        ratio = hb / radius / (1.0 + hb / radius)
        grid_size, block_size = get_grid_block(xc.size)
        print(grid_size, block_size)
        fn((grid_size,), (block_size,), (xc.size, xc, yc, ratio, x0, y0, xc, yc))
        cp.cuda.Stream.null.synchronize()
