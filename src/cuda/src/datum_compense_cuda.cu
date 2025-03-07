#include "datum_compense.h"
#include <cuda_runtime.h>
__global__ void datum_compense_cuda(double *xc,
                                               double *yc,
                                               double factor,
                                               double x0,
                                               double y0)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    datum_compense(xc[i], yc[i], factor, x0, y0, xc[i], yc[i]);
}
