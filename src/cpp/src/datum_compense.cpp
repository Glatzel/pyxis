#include "datum_compense.h"

template <typename T>
CUDA_HOST_DEVICE void datum_compense(
    T xc,
    T yc,
    T factor,
    T x0,
    T y0,
    T *out_xc,
    T *out_yc)
{
    *out_xc = xc - factor * (xc - x0);
    *out_yc = yc - factor * (yc - y0);
}
extern "C" {
    CUDA_HOST_DEVICE void datum_compense_float(
        float xc,
        float yc,
        float factor,
        float x0,
        float y0,
        float *out_xc,
        float *out_yc)
    {
        datum_compense(xc,yc,factor,x0,y0,out_xc,out_yc);
    };
    CUDA_HOST_DEVICE void datum_compense_double(
        double xc,
        double yc,
        double factor,
        double x0,
        double y0,
        double *out_xc,
        double *out_yc)
    {
        datum_compense(xc,yc,factor,x0,y0,out_xc,out_yc);
    };
}