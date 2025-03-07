#include "pyxis.h"

CUDA_HOST_DEVICE void datum_compense(double xc,
                                double yc,
                                double factor,
                                double x0,
                                double y0,
                                double &out_xc,
                                double &out_yc)
{
    out_xc = xc - factor * (xc - x0);
    out_yc = yc - factor * (yc - y0);
}
