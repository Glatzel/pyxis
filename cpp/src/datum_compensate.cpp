#include "pyxis.h"
template <typename T>
CUDA_HOST_DEVICE void datum_compensate(
    T xc,
    T yc,
    T factor,
    T x0,
    T y0,
    T &out_xc,
    T &out_yc)
{
    out_xc = xc - factor * (xc - x0);
    out_yc = yc - factor * (yc - y0);
}
