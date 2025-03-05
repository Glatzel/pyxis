#pragma once
#ifdef __CUDACC__ // If compiled with nvcc
#define CUDA_DEVICE __device__
#else
#define CUDA_DEVICE // Empty for normal C++ compilation
#endif
#include <cmath>
#define M_PI 3.14159265358979323846
CUDA_DEVICE void bd09_to_gcj02(double bd09_lon, double bd09_lat,
                               double &gcj02_lon, double &gcj02_lat)
{
    double x_pi=M_PI * 3000.0 / 180.0;
    double x = bd09_lon - 0.0065;
    double y = bd09_lat - 0.006;
    double z = sqrt(x * x + y * y) - 0.00002 * sin(y * x_pi);
    double theta = atan2(y, x) - 0.000003 * cos(x * x_pi);
    gcj02_lon = z * cos(theta);
    gcj02_lat = z * sin(theta);
}
CUDA_DEVICE void gcj02_to_bd09(double gcj02_lon, double gcj02_lat,
                               double &bd09_lon, double &bd09_lat)

{
    double x_pi=M_PI * 3000.0 / 180.0;
    double z = sqrt(gcj02_lon * gcj02_lon + gcj02_lat * gcj02_lat) + 0.00002 * sin(gcj02_lat * x_pi);
    double theta = atan2(gcj02_lat, gcj02_lon) + 0.000003 * cos(gcj02_lon * x_pi);
    bd09_lon = z * cos(theta) + 0.0065;
    bd09_lat = z * sin(theta) + 0.006;
}