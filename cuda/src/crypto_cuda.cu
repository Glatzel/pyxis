#include "crypto.cpp"

template <typename T>
__global__ void bd09_to_gcj02_cuda(int N,
                                   const T *lon,
                                   const T *lat,
                                   T *out_lon,
                                   T *out_lat)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= N)
        return;
    bd09_to_gcj02(lon[i], lat[i], out_lon[i], out_lat[i]);
};

template <typename T>
__global__ void gcj02_to_bd09_cuda(int N,
                                   const T *lon,
                                   const T *lat, T *out_lon,
                                   T *out_lat)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= N)
        return;
    gcj02_to_bd09(lon[i], lat[i], out_lon[i], out_lat[i]);
};
template <typename T>
__global__ void gcj02_to_wgs84_cuda(int N,
                                    const T *lon,
                                    const T *lat,
                                    T *out_lon,
                                    T *out_lat)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= N)
        return;
    gcj02_to_wgs84(lon[i], lat[i], out_lon[i], out_lat[i]);
};
template <typename T>
__global__ void wgs84_to_gcj02_cuda(int N,
                                    const T *lon,
                                    const T *lat,
                                    T *out_lon,
                                    T *out_lat)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= N)
        return;
    wgs84_to_gcj02(lon[i], lat[i], out_lon[i], out_lat[i]);
};
template <typename T>
__global__ void wgs84_to_bd09_cuda(int N,
                                   const T *lon,
                                   const T *lat,
                                   T *out_lon,
                                   T *out_lat)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= N)
        return;
    wgs84_to_bd09(lon[i], lat[i], out_lon[i], out_lat[i]);
};
template <typename T>
__global__ void bd09_to_wgs84_cuda(int N,
                                   const T *lon,
                                   const T *lat,
                                   T *out_lon,
                                   T *out_lat)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= N)
        return;
    bd09_to_wgs84(lon[i], lat[i], out_lon[i], out_lat[i]);
};

template <typename T>
__global__ void gcj02_to_wgs84_exact_cuda(int N,
                                          const T *lon,
                                          const T *lat,
                                          const T threshold,
                                          const bool distance_mode,
                                          const int max_iter,
                                          T *out_lon,
                                          T *out_lat)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= N)
        return;
    crypto_exact(lon[i], lat[i], gcj02_to_wgs84, wgs84_to_gcj02, threshold, distance_mode, max_iter, out_lon[i], out_lat[i]);
};
template <typename T>
__global__ void bd09_to_wgs84_exact_cuda(int N,
                                         const T *lon,
                                         const T *lat,
                                         const T threshold,
                                         const bool distance_mode,
                                         const int max_iter,
                                         T *out_lon,
                                         T *out_lat)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= N)
        return;
    crypto_exact(lon[i], lat[i], bd09_to_wgs84, wgs84_to_bd09, threshold, distance_mode, max_iter, out_lon[i], out_lat[i]);
};
template <typename T>
__global__ void bd09_to_gcj02_exact_cuda(int N,
                                         const T *lon,
                                         const T *lat,
                                         const T threshold,
                                         const bool distance_mode,
                                         const int max_iter,
                                         T *out_lon,
                                         T *out_lat)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= N)
        return;
    crypto_exact(lon[i], lat[i], bd09_to_gcj02, gcj02_to_bd09, threshold, distance_mode, max_iter, out_lon[i], out_lat[i]);
};
// float
extern "C"
{
    __global__ void bd09_to_gcj02_cuda_float(int N,
                                             const float *lon,
                                             const float *lat,
                                             float *out_lon,
                                             float *out_lat)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= N)
            return;
        bd09_to_gcj02<float>(lon[i], lat[i], out_lon[i], out_lat[i]);
    };

    __global__ void gcj02_to_bd09_cuda_float(int N,
                                             const float *lon,
                                             const float *lat,
                                             float *out_lon,
                                             float *out_lat)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= N)
            return;
        gcj02_to_bd09<float>(lon[i], lat[i], out_lon[i], out_lat[i]);
    };

    __global__ void gcj02_to_wgs84_cuda_float(int N,
                                              const float *lon,
                                              const float *lat,
                                              float *out_lon,
                                              float *out_lat)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= N)
            return;
        gcj02_to_wgs84<float>(lon[i], lat[i], out_lon[i], out_lat[i]);
    };

    __global__ void wgs84_to_gcj02_cuda_float(int N,
                                              const float *lon,
                                              const float *lat,
                                              float *out_lon,
                                              float *out_lat)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= N)
            return;
        wgs84_to_gcj02<float>(lon[i], lat[i], out_lon[i], out_lat[i]);
    };

    __global__ void wgs84_to_bd09_cuda_float(int N,
                                             const float *lon,
                                             const float *lat,
                                             float *out_lon,
                                             float *out_lat)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= N)
            return;
        wgs84_to_bd09<float>(lon[i], lat[i], out_lon[i], out_lat[i]);
    };

    __global__ void bd09_to_wgs84_cuda_float(int N,
                                             const float *lon,
                                             const float *lat,
                                             float *out_lon,
                                             float *out_lat)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= N)
            return;
        bd09_to_wgs84<float>(lon[i], lat[i], out_lon[i], out_lat[i]);
    };

    __global__ void gcj02_to_wgs84_exact_cuda_float(int N,
                                                    const float *lon,
                                                    const float *lat,
                                                    const float threshold,
                                                    const bool distance_mode,
                                                    const int max_iter,
                                                    float *out_lon,
                                                    float *out_lat)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= N)
            return;
        crypto_exact<float>(lon[i], lat[i], gcj02_to_wgs84, wgs84_to_gcj02, threshold, distance_mode, max_iter, out_lon[i], out_lat[i]);
    };

    __global__ void bd09_to_wgs84_exact_cuda_float(int N,
                                                   const float *lon,
                                                   const float *lat,
                                                   const float threshold,
                                                   const bool distance_mode,
                                                   const int max_iter,
                                                   float *out_lon,
                                                   float *out_lat)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= N)
            return;
        crypto_exact<float>(lon[i], lat[i], bd09_to_wgs84, wgs84_to_bd09, threshold, distance_mode, max_iter, out_lon[i], out_lat[i]);
    };

    __global__ void bd09_to_gcj02_exact_cuda_float(int N,
                                                   const float *lon,
                                                   const float *lat,
                                                   const float threshold,
                                                   const bool distance_mode,
                                                   const int max_iter,
                                                   float *out_lon,
                                                   float *out_lat)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= N)
            return;
        crypto_exact<float>(lon[i], lat[i], bd09_to_gcj02, gcj02_to_bd09, threshold, distance_mode, max_iter, out_lon[i], out_lat[i]);
    };
}
// double
extern "C"
{
    __global__ void bd09_to_gcj02_cuda_double(int N,
                                              const double *lon,
                                              const double *lat,
                                              double *out_lon,
                                              double *out_lat)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= N)
            return;
        bd09_to_gcj02<double>(lon[i], lat[i], out_lon[i], out_lat[i]);
    };

    __global__ void gcj02_to_bd09_cuda_double(int N,
                                              const double *lon,
                                              const double *lat,
                                              double *out_lon,
                                              double *out_lat)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= N)
            return;
        gcj02_to_bd09<double>(lon[i], lat[i], out_lon[i], out_lat[i]);
    };

    __global__ void gcj02_to_wgs84_cuda_double(int N,
                                               const double *lon,
                                               const double *lat,
                                               double *out_lon,
                                               double *out_lat)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= N)
            return;
        gcj02_to_wgs84<double>(lon[i], lat[i], out_lon[i], out_lat[i]);
    };

    __global__ void wgs84_to_gcj02_cuda_double(int N,
                                               const double *lon,
                                               const double *lat,
                                               double *out_lon,
                                               double *out_lat)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= N)
            return;
        wgs84_to_gcj02<double>(lon[i], lat[i], out_lon[i], out_lat[i]);
    };

    __global__ void wgs84_to_bd09_cuda_double(int N,
                                              const double *lon,
                                              const double *lat,
                                              double *out_lon,
                                              double *out_lat)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= N)
            return;
        wgs84_to_bd09<double>(lon[i], lat[i], out_lon[i], out_lat[i]);
    };

    __global__ void bd09_to_wgs84_cuda_double(int N,
                                              const double *lon,
                                              const double *lat,
                                              double *out_lon,
                                              double *out_lat)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= N)
            return;
        bd09_to_wgs84<double>(lon[i], lat[i], out_lon[i], out_lat[i]);
    };

    __global__ void gcj02_to_wgs84_exact_cuda_double(int N,
                                                     const double *lon,
                                                     const double *lat,
                                                     const double threshold,
                                                     const bool distance_mode,
                                                     const int max_iter,
                                                     double *out_lon,
                                                     double *out_lat)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= N)
            return;
        crypto_exact<double>(lon[i], lat[i], gcj02_to_wgs84, wgs84_to_gcj02, threshold, distance_mode, max_iter, out_lon[i], out_lat[i]);
    };

    __global__ void bd09_to_wgs84_exact_cuda_double(int N,
                                                    const double *lon,
                                                    const double *lat,
                                                    const double threshold,
                                                    const bool distance_mode,
                                                    const int max_iter,
                                                    double *out_lon,
                                                    double *out_lat)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= N)
            return;
        crypto_exact<double>(lon[i], lat[i], bd09_to_wgs84, wgs84_to_bd09, threshold, distance_mode, max_iter, out_lon[i], out_lat[i]);
    };

    __global__ void bd09_to_gcj02_exact_cuda_double(int N,
                                                    const double *lon,
                                                    const double *lat,
                                                    const double threshold,
                                                    const bool distance_mode,
                                                    const int max_iter,
                                                    double *out_lon,
                                                    double *out_lat)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= N)
            return;
        crypto_exact<double>(lon[i], lat[i], bd09_to_gcj02, gcj02_to_bd09, threshold, distance_mode, max_iter, out_lon[i], out_lat[i]);
    };
}
