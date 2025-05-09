#pragma once
#include "cuda_macro.h"

template <typename T>
CUDA_HOST_DEVICE void datum_compensate(T xc,
                                     T yc,
                                     T factor,
                                     T x0,
                                     T y0,
                                     T *out_xc,
                                     T *out_yc);
