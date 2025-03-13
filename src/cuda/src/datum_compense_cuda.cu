#include "datum_compense.cpp"
template <typename T>
__global__ void datum_compense_cuda(
    T *xc,
    T *yc,
    T factor,
    T x0,
    T y0,
    T *out_xc,
    T *out_yc)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    datum_compense(xc[i], yc[i], factor, x0, y0, &out_xc[i], &out_yc[i]);
};
extern "C"
{
    __global__ void datum_compense_cuda_float(
        float *xc,
        float *yc,
        float factor,
        float x0,
        float y0,
        float *out_xc,
        float *out_yc)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        datum_compense(xc[i], yc[i], factor, x0, y0, xc[i], yc[i]);
    };
}
extern "C"
{
    __global__ void datum_compense_cuda_double(
        double *xc,
        double *yc,
        double factor,
        double x0,
        double y0,
        double *out_xc,
        double *out_yc)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        datum_compense(xc[i], yc[i], factor, x0, y0, xc[i], yc[i]);
    };
}