extern "C" __global__ void datum_compense(double *xc,
                                          double *yc,
                                          double factor,
                                          double x0,
                                          double y0)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
        xc[i] = xc[i] - factor * (xc[i] - x0);
        yc[i] = yc[i] - factor * (yc[i] - y0);
}
