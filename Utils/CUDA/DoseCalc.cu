#include <Meta/CUDA.h>
#include "DoseCalc.h"

#include <stdlib.h>


typedef unsigned char uchar;

void SetupDoseCalc(int pbo, int w, int h, int d) {
    
}

__global__ void radioDepth(cudaArray* output, float dx, float dy, float dz) {

}

__global__ void doseCalc(uint *d_output) {
}

void RunDoseCalc(int pbo, int w, int h, int d, Beam beam, int beamlet_x, int beamlet_y, float dx, float dy, float dz) {
    cudaArray *d_arr = NULL;
    cudaExtent ext;
    ext.width  = w;
    ext.height = h;
    ext.depth  = d;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaMalloc3DArray(&d_arr, &channelDesc, ext);
    CHECK_FOR_CUDA_ERROR();

	dim3 blockDim(512,1,1);
	dim3 gridDim((unsigned int)(ceil((double)(w*h*d)/blockDim.x)), 1, 1);

	// Invoke kernel
    radioDepth<<<gridDim, blockDim>>>(d_arr, dx, dy, dz);
    CHECK_FOR_CUDA_ERROR();

    cudaFreeArray(d_arr);
    CHECK_FOR_CUDA_ERROR();
    /* cudaGLRegisterBufferObject(pbo); */
    /* CHECK_FOR_CUDA_ERROR(); */

    /* uint* p; */
    /* cudaGLMapBufferObject((void**)&p,pbo); */
    /* CHECK_FOR_CUDA_ERROR(); */

    /* cudaMemset(p, 0, 4 * w * h * d); */
    /* CHECK_FOR_CUDA_ERROR(); */
    
    // const dim3 blockSize(16, 16, 1);
    // const dim3 gridSize(blockSize.x, blockSize.y);

    // doseCalc<<<gridSize, blockSize>>>(p);
    // CHECK_FOR_CUDA_ERROR();

    // cudaGLUnmapBufferObject(pbo);
    // CHECK_FOR_CUDA_ERROR();
    
    // cudaGLUnregisterBufferObject(pbo);
}
