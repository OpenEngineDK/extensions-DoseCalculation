#include "DoseCalc.h"
#include <Meta/CUDA.h>

typedef unsigned char uchar;

void SetupDoseCalc(int pbo, int w, int h, int d) {
    
}

__global__ void doseCalc(uint *d_output) {
}

void RunDoseCalc(int pbo, int w, int h, int d, Beam beam, int beamlet_x, int beamlet_y, float dx, float dy, float dz) {
    cudaGLRegisterBufferObject(pbo);
    CHECK_FOR_CUDA_ERROR();

    uint* p;
    cudaGLMapBufferObject((void**)&p,pbo);
    CHECK_FOR_CUDA_ERROR();

    cudaMemset(p, 0, 4 * w * h * d);
    CHECK_FOR_CUDA_ERROR();
    
    const dim3 blockSize(16, 16, 1);
    const dim3 gridSize(blockSize.x, blockSize.y);

    doseCalc<<<gridSize, blockSize>>>(p);
    CHECK_FOR_CUDA_ERROR();

    cudaGLUnmapBufferObject(pbo);
    CHECK_FOR_CUDA_ERROR();
    
    cudaGLUnregisterBufferObject(pbo);
}
