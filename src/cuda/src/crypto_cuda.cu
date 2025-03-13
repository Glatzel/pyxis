#include "crypto.cpp"

template <typename T>
__global__ void bd09_to_gcj02_cuda(T *lon,
                                   T *lat)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    bd09_to_gcj02(lon[i], lat[i], lon[i], lat[i]);
};

template <typename T>
__global__ void gcj02_to_bd09_cuda(T *lon,
                                   T *lat)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    gcj02_to_bd09(lon[i], lat[i], lon[i], lat[i]);
};
template <typename T>
__global__ void gcj02_to_wgs84_cuda(T *lon,
                                    T *lat)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    gcj02_to_wgs84(lon[i], lat[i], lon[i], lat[i]);
};
template <typename T>
__global__ void wgs84_to_gcj02_cuda(T *lon,
                                    T *lat)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    wgs84_to_gcj02(lon[i], lat[i], lon[i], lat[i]);
};
template <typename T>
__global__ void wgs84_to_bd09_cuda(T *lon,
                                   T *lat)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    wgs84_to_bd09(lon[i], lat[i], lon[i], lat[i]);
};
template <typename T>
__global__ void bd09_to_wgs84_cuda(T *lon,
                                   T *lat)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    bd09_to_wgs84(lon[i], lat[i], lon[i], lat[i]);
};

template <typename T>
__global__ void gcj02_to_wgs84_exact_cuda(T *lon,
                                          T *lat,
                                          const T threshold,
                                          const bool distance_mode,
                                          const int max_iter)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    crypto_exact(lon[i], lat[i], gcj02_to_wgs84, wgs84_to_gcj02, threshold, distance_mode, max_iter, lon[i], lat[i]);
};
template <typename T>
__global__ void bd09_to_wgs84_exact_cuda(T *lon,
                                         T *lat,
                                         const T threshold,
                                         const bool distance_mode,
                                         const int max_iter)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    crypto_exact(lon[i], lat[i], bd09_to_wgs84, wgs84_to_bd09, threshold, distance_mode, max_iter, lon[i], lat[i]);
};
template <typename T>
__global__ void bd09_to_gcj02_exact_cuda(T *lon,
                                         T *lat,
                                         const T threshold,
                                         const bool distance_mode,
                                         const int max_iter)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    crypto_exact(lon[i], lat[i], bd09_to_gcj02, gcj02_to_bd09, threshold, distance_mode, max_iter, lon[i], lat[i]);
};
