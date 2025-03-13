#pragma once

#ifdef __CUDACC__ // If compiled with nvcc
#define CUDA_DEVICE __device__
#define CUDA_HOST __host__
#define CUDA_HOST_DEVICE __host__ __device__
#else
#define CUDA_DEVICE
#define CUDA_HOST
#define CUDA_HOST_DEVICE
#endif

// datum_compense.cpp
template <typename T>
CUDA_HOST_DEVICE void datum_compense(T xc,
                                     T yc,
                                     T factor,
                                     T x0,
                                     T y0,
                                     T &out_xc,
                                     T &out_yc);

// crypto.cpp
template <typename T>
CUDA_HOST_DEVICE void bd09_to_gcj02(
    const T bd09_lon, const T bd09_lat,
    T &gcj02_lon, T &gcj02_lat);
template <typename T>
CUDA_HOST_DEVICE void gcj02_to_bd09(
    const T gcj02_lon, const T gcj02_lat,
    T &bd09_lon, T &bd09_lat);
template <typename T>
CUDA_HOST_DEVICE void gcj02_to_wgs84(
    const T gcj02_lon, const T gcj02_lat,
    T &wgs84_lon, T &wgs84_lat);
template <typename T>
CUDA_HOST_DEVICE void wgs84_to_gcj02(
    const T wgs84_lon, const T wgs84_lat,
    T &gcj02_lon, T &gcj02_lat);
template <typename T>
CUDA_HOST_DEVICE void wgs84_to_bd09(
    const T wgs84_lon, const T wgs84_lat,
    T &bd09_lon, T &bd09_lat);
template <typename T>
CUDA_HOST_DEVICE void bd09_to_wgs84(
    const T bd09_lon, const T bd09_lat,
    T &wgs84_lon, T &wgs84_lat);
template <typename T>
CUDA_HOST_DEVICE T to_radians(const T degrees);
template <typename T>
CUDA_HOST_DEVICE T haversine_distance(const T lon_a, const T lat_a,
                                           const T lon_b, const T lat_b);
template <typename T>
CUDA_HOST_DEVICE void crypto_exact(
    const T src_lon,
    const T src_lat,
    void (*crypto_fn)(const T, const T, T &, T &),
    void (*inv_crypto_fn)(const T, const T, T &, T &),
    const T threshold,
    const bool distance_mode,
    const int max_iter,
    T &out_lon,
    T &out_lat);
