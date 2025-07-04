from enum import Enum

import cupy as cp  # type: ignore

from pyxis_cuda import get_grid_block
from pyxis_cuda._utils import PTX_PATH, TDTYPE


class COORD_CRYPTO_SPACE(str, Enum):
    """COORD_CRYPTO_SPACE."""

    WGS84 = "WGS84"
    GCJ02 = "GCJ02"
    BD09 = "BD09"

    @classmethod
    def list(cls):
        return list(map(str, cls))


class CryptoCuda:
    def __init__(self) -> None:
        self.module = cp.RawModule(path=str(PTX_PATH / "crypto_cuda.ptx"))

    def crypto(
        self,
        dtype: TDTYPE,
        lon: cp.ndarray,
        lat: cp.ndarray,
        crypto_from: COORD_CRYPTO_SPACE,
        crypto_to: COORD_CRYPTO_SPACE,
    ):
        if (
            (str(crypto_from).upper() not in COORD_CRYPTO_SPACE.list())
            or (str(crypto_to).upper() not in COORD_CRYPTO_SPACE.list())
            or str(crypto_to).upper() == str(crypto_from).upper()
        ):
            msg = f"Unsupported: from `{crypto_from}` to `{crypto_to}`."
            raise TypeError(msg)

        fn = self.module.get_function(f"{crypto_from.lower()}_to_{crypto_to.lower()}_cuda_{dtype}")
        grid_size, block_size = get_grid_block(lon.size)
        fn((grid_size,), (block_size,), (lon.size, lon, lat, lon, lat))
        cp.cuda.Stream.null.synchronize()

    def crypto_exact(
        self,
        dtype: TDTYPE,
        lon: cp.ndarray,
        lat: cp.ndarray,
        crypto_from: COORD_CRYPTO_SPACE,
        crypto_to: COORD_CRYPTO_SPACE,
        threshold: float,
        max_iter: int,
    ):
        match crypto_from.upper(), crypto_to.upper():
            case COORD_CRYPTO_SPACE.GCJ02, COORD_CRYPTO_SPACE.WGS84:
                pass
            case COORD_CRYPTO_SPACE.BD09, COORD_CRYPTO_SPACE.GCJ02:
                pass
            case COORD_CRYPTO_SPACE.BD09, COORD_CRYPTO_SPACE.WGS84:
                pass
            case _:
                msg = f"Unsupported: from `{crypto_from}` to `{crypto_to}`."
                raise TypeError(msg)
        fn = self.module.get_function(f"{crypto_from.lower()}_to_{crypto_to.lower()}_exact_cuda_{dtype}")
        grid_size, block_size = get_grid_block(lon.size)
        fn((grid_size,), (block_size,), (lon.size, lon, lat, threshold, False, max_iter, lon, lat))
        cp.cuda.Stream.null.synchronize()
