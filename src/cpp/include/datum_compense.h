#pragma once
#include "cuda_macro.h"

#pragma region cpp
#ifdef __cplusplus
template <typename T>
CUDA_HOST_DEVICE void datum_compense(T xc,
                                     T yc,
                                     T factor,
                                     T x0,
                                     T y0,
                                     T *out_xc,
                                     T *out_yc);
#endif
#pragma endregion
// #############################################################################
#pragma region c
#ifdef __cplusplus
extern "C"
{
#endif
    CUDA_HOST_DEVICE void datum_compense_float(
        float xc,
        float yc,
        float factor,
        float x0,
        float y0,
        float *out_xc,
        float *out_yc);
    CUDA_HOST_DEVICE void datum_compense_double(
        double xc,
        double yc,
        double factor,
        double x0,
        double y0,
        double *out_xc,
        double *out_yc);

#ifdef __cplusplus
}
#endif
#pragma endregion
