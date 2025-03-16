import cupy as cp

from pyxis import COORD_CRYPTO_SPACE
from pyxis.pyxis_cuda import get_grid_block
from pyxis.pyxis_cuda._utils import PTX_PATH, TDTYPE


class CryptoCuda:
    def __init__(self) -> None:
        self.module = cp.RawModule(path=str(PTX_PATH / "crypto_cuda.ptx"))

    def crypto(
        self,
        dtype: TDTYPE,
        lon: cp.ndarray,
        lat: cp.ndarray,
        from_space: COORD_CRYPTO_SPACE,
        to_space: COORD_CRYPTO_SPACE,
    ):
        fn = self.module.get_function(f"{from_space.lower()}_to_{to_space.lower()}_cuda_{dtype}")
        grid_size, block_size = get_grid_block(lon.size)
        fn((grid_size,), (block_size,), (lon, lat))
        cp.cuda.Stream.null.synchronize()

    def crypto_exact(
        self,
        dtype: TDTYPE,
        lon: cp.ndarray,
        lat: cp.ndarray,
        from_space: COORD_CRYPTO_SPACE,
        to_space: COORD_CRYPTO_SPACE,
        threshold: float,
        max_iter: int,
    ):
        fn = self.module.get_function(f"{from_space.lower()}_to_{to_space.lower()}_exact_cuda_{dtype}")
        grid_size, block_size = get_grid_block(lon.size)
        fn((grid_size,), (block_size,), (lon, lat, threshold, False, max_iter))
        cp.cuda.Stream.null.synchronize()
