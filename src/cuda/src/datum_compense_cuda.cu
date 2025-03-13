#include "datum_compense.cpp"
// template <typename T>
// __global__ void datum_compense_cuda(
//     T *xc,
//     T *yc,
//     T factor,
//     T x0,
//     T y0)
// {
//     int i = threadIdx.x + blockIdx.x * blockDim.x;
//     datum_compense(xc[i], yc[i], factor, x0, y0, xc[i], yc[i]);
// }
extern "C"
{
    __global__ int datum_compense_cuda_c_float(float *xc, float *yc, float factor, float x0, float y0)
    {
        return datum_compense<float>(*xc, *yc, factor, x0, y0);
    }
}