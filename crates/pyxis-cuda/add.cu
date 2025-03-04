extern "C" __global__ void vector_add(const float* a, const float* b, float* result, int N) {
    __shared__ float s_a[256]; // Shared memory for block
    __shared__ float s_b[256];

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        s_a[threadIdx.x] = a[i];  // Load data into shared memory
        s_b[threadIdx.x] = b[i];
        __syncthreads();  // Ensure all threads load data before computing

        result[i] = s_a[threadIdx.x] + s_b[threadIdx.x]; // Compute using shared memory
    }
}