#pragma once
#ifdef __CUDACC__ // If compiled with nvcc
#define CUDA_DEVICE __device__
#else
#define CUDA_DEVICE // Empty for normal C++ compilation
#endif
CUDA_DEVICE void datum_compense(double xc,
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
