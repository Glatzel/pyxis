#pragma once
#include "cuda_macro.h"
CUDA_HOST_DEVICE void datum_compense(double xc,
                                double yc,
                                double factor,
                                double x0,
                                double y0,
                                double &out_xc,
                                double &out_yc);
