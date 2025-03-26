#include "datum_compense.cpp"
#pragma region generics
template <typename T>
__global__ void datum_compense_cuda(
    const int N,
    const T *xc,
    const T *yc,
    const T factor,
    const T x0,
    const T y0,
    T *out_xc,
    T *out_yc)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= N)
        return;
    datum_compense(xc[i], yc[i], factor, x0, y0, out_xc[i], out_yc[i]);
};
#pragma endregion
#pragma region float
extern "C"
{
    __global__ void datum_compense_cuda_float(
        const int N,
        const float *xc,
        const float *yc,
        const float factor,
        const float x0,
        const float y0,
        float *out_xc,
        float *out_yc)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= N)
            return;
        datum_compense(xc[i], yc[i], factor, x0, y0, out_xc[i], out_yc[i]);
    };
}
#pragma endregion
#pragma region double
extern "C"
{
    __global__ void datum_compense_cuda_double(
        const int N,
        const double *xc,
        const double *yc,
        const double factor,
        const double x0,
        const double y0,
        double *out_xc,
        double *out_yc)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= N)
            return;
        datum_compense(xc[i], yc[i], factor, x0, y0, out_xc[i], out_yc[i]);
    };
}
#pragma endregion
