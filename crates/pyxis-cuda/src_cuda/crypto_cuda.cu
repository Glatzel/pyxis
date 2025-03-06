extern "C"
{
#include "crypto.cpp"
}
extern "C" __global__ void bd09_to_gcj02_cuda(double *lon,
                                              double *lat)
{
    int i = threadIdx.x;
    bd09_to_gcj02(lon[i], lat[i], lon[i], lat[i]);
}
extern "C" __global__ void gcj02_to_bd09_cuda(double *lon,
                                              double *lat)
{
    int i = threadIdx.x;
    gcj02_to_bd09(lon[i], lat[i], lon[i], lat[i]);
}
extern "C" __global__ void gcj02_to_wgs84_cuda(double *lon,
                                               double *lat)
{
    int i = threadIdx.x;
    gcj02_to_wgs84(lon[i], lat[i], lon[i], lat[i]);
}
extern "C" __global__ void wgs84_to_gcj02_cuda(double *lon,
                                               double *lat)
{
    int i = threadIdx.x;
    wgs84_to_gcj02(lon[i], lat[i], lon[i], lat[i]);
}
extern "C" __global__ void wgs84_to_bd09_cuda(double *lon,
                                              double *lat)
{
    int i = threadIdx.x;
    wgs84_to_bd09(lon[i], lat[i], lon[i], lat[i]);
}
extern "C" __global__ void bd09_to_wgs84_cuda(double *lon,
                                              double *lat)
{
    int i = threadIdx.x;
    bd09_to_wgs84(lon[i], lat[i], lon[i], lat[i]);
}