#pragma once

#ifdef __CUDACC__ // If compiled with nvcc
#define CUDA_DEVICE __device__
#define CUDA_HOST __host__
#define CUDA_HOST_DEVICE __host__ __device__
#else
#define CUDA_DEVICE // Empty for normal C++ compilation
#define CUDA_HOST // Empty for normal C++ compilation
#define CUDA_HOST_DEVICE // Empty for normal C++ compilation
#endif