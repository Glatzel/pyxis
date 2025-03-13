#pragma once
#include "cuda_macro.h"

// include bool in c
#ifndef __cplusplus
#include <stdbool.h>
#endif

#pragma region cpp
#ifdef __cplusplus
template <typename T>
CUDA_HOST_DEVICE void bd09_to_gcj02(
    const T bd09_lon, const T bd09_lat,
    T *gcj02_lon, T *gcj02_lat);
template <typename T>
CUDA_HOST_DEVICE void gcj02_to_bd09(
    const T gcj02_lon, const T gcj02_lat,
    T *bd09_lon, T *bd09_lat);
template <typename T>
CUDA_HOST_DEVICE void gcj02_to_wgs84(
    const T gcj02_lon, const T gcj02_lat,
    T *wgs84_lon, T *wgs84_lat);
template <typename T>
CUDA_HOST_DEVICE void wgs84_to_gcj02(
    const T wgs84_lon, const T wgs84_lat,
    T *gcj02_lon, T *gcj02_lat);
template <typename T>
CUDA_HOST_DEVICE void wgs84_to_bd09(
    const T wgs84_lon, const T wgs84_lat,
    T *bd09_lon, T *bd09_lat);
template <typename T>
CUDA_HOST_DEVICE void bd09_to_wgs84(
    const T bd09_lon, const T bd09_lat,
    T *wgs84_lon, T *wgs84_lat);
template <typename T>
CUDA_HOST_DEVICE T to_radians(const T degrees);
template <typename T>
CUDA_HOST_DEVICE T haversine_distance(const T lon_a, const T lat_a,
                                      const T lon_b, const T lat_b);
template <typename T>
CUDA_HOST_DEVICE void crypto_exact(
    const T src_lon,
    const T src_lat,
    void (*crypto_fn)(const T, const T, T *, T *),
    void (*inv_crypto_fn)(const T, const T, T *, T *),
    const T threshold,
    const bool distance_mode,
    const int max_iter,
    T *out_lon,
    T *out_lat);
#endif
#pragma endregion
// #############################################################################
#pragma region c float
#ifdef __cplusplus
extern "C"
{
#endif

    CUDA_HOST_DEVICE void bd09_to_gcj02_float(
        const float bd09_lon, const float bd09_lat,
        float *gcj02_lon, float *gcj02_lat);

    CUDA_HOST_DEVICE void gcj02_to_bd09_float(
        const float gcj02_lon, const float gcj02_lat,
        float *bd09_lon, float *bd09_lat);

    CUDA_HOST_DEVICE void gcj02_to_wgs84_float(
        const float gcj02_lon, const float gcj02_lat,
        float *wgs84_lon, float *wgs84_lat);

    CUDA_HOST_DEVICE void wgs84_to_gcj02_float(
        const float wgs84_lon, const float wgs84_lat,
        float *gcj02_lon, float *gcj02_lat);

    CUDA_HOST_DEVICE void wgs84_to_bd09_float(
        const float wgs84_lon, const float wgs84_lat,
        float *bd09_lon, float *bd09_lat);

    CUDA_HOST_DEVICE void bd09_to_wgs84_float(
        const float bd09_lon, const float bd09_lat,
        float *wgs84_lon, float *wgs84_lat);

    CUDA_HOST_DEVICE float to_radians_float(const float degrees);

    CUDA_HOST_DEVICE float haversine_distance_float(const float lon_a, const float lat_a,
                                              const float lon_b, const float lat_b);

    CUDA_HOST_DEVICE void crypto_exact_float(
        const float src_lon,
        const float src_lat,
        void (*crypto_fn)(const float, const float, float *, float *),
        void (*inv_crypto_fn)(const float, const float, float *, float *),
        const float threshold,
        const bool distance_mode,
        const int max_iter,
        float *out_lon,
        float *out_lat);

#ifdef __cplusplus
}
#endif
#pragma endregion
// #############################################################################
#pragma region c double
#ifdef __cplusplus
extern "C"
{
#endif

    CUDA_HOST_DEVICE void bd09_to_gcj02_double(
        const double bd09_lon, const double bd09_lat,
        double *gcj02_lon, double *gcj02_lat);

    CUDA_HOST_DEVICE void gcj02_to_bd09_double(
        const double gcj02_lon, const double gcj02_lat,
        double *bd09_lon, double *bd09_lat);

    CUDA_HOST_DEVICE void gcj02_to_wgs84_double(
        const double gcj02_lon, const double gcj02_lat,
        double *wgs84_lon, double *wgs84_lat);

    CUDA_HOST_DEVICE void wgs84_to_gcj02_double(
        const double wgs84_lon, const double wgs84_lat,
        double *gcj02_lon, double *gcj02_lat);

    CUDA_HOST_DEVICE void wgs84_to_bd09_double(
        const double wgs84_lon, const double wgs84_lat,
        double *bd09_lon, double *bd09_lat);

    CUDA_HOST_DEVICE void bd09_to_wgs84_double(
        const double bd09_lon, const double bd09_lat,
        double *wgs84_lon, double *wgs84_lat);

    CUDA_HOST_DEVICE double to_radians_double(const double degrees);

    CUDA_HOST_DEVICE double haversine_distance_double(const double lon_a, const double lat_a,
                                               const double lon_b, const double lat_b);

    CUDA_HOST_DEVICE void crypto_exact_double(
        const double src_lon,
        const double src_lat,
        void (*crypto_fn)(const double, const double, double *, double *),
        void (*inv_crypto_fn)(const double, const double, double *, double *),
        const double threshold,
        const bool distance_mode,
        const int max_iter,
        double *out_lon,
        double *out_lat);
#ifdef __cplusplus
}
#endif
