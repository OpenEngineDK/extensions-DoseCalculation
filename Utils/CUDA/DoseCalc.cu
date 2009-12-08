#include <Utils/CUDA/uint_util.hcu>
#include <Meta/CUDA.h>

#include "DoseCalc.h"

#include <Utils/CUDA/DozeCuda.h>

#include <stdlib.h>

typedef unsigned char uchar;

texture<float, 3, cudaReadModeElementType> tex;
uint3 dimensions; // should be placd in constant memory

void SetupDoseCalc(GLuint pbo, int w, int h, int d) {
    cudaGLRegisterBufferObject(pbo);
    CHECK_FOR_CUDA_ERROR();

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();    
    tex.normalized = true;
    tex.filterMode = cudaFilterModeLinear;
    tex.addressMode[0] = cudaAddressModeClamp;
    tex.addressMode[1] = cudaAddressModeClamp;
    tex.addressMode[2] = cudaAddressModeClamp;

    cudaBindTextureToArray(tex, GetVolumeArray(), channelDesc);
    CHECK_FOR_CUDA_ERROR();

    dimensions = make_uint3(w, h, d);
}

__global__ void radioDepth(float* input, float* output, uint3 dims) {
    const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    // lookup via tex3D(...);

    uint3 coordinate = idx_to_co(idx, dims);
    
}

__global__ void doseCalc(uint *d_output) {

}

void RunDoseCalc(GLuint pbo, int w, int h, int d, Beam beam, int beamlet_x, int beamlet_y, float dx, float dy, float dz) {
    // Map the buffer object that we want to write the radiological depth to.
    float* radiologicalDepth;
    cudaGLMapBufferObject( (void**)&radiologicalDepth, pbo);

    dim3 blockDim(512,1,1);
	dim3 gridDim((unsigned int)(ceil((double)(w*h*d)/blockDim.x)), 1, 1);

    radioDepth<<< gridDim, blockDim >>>((float*) GetVolumeArray(), 
                                        radiologicalDepth, 
                                        dimensions);

    cudaGLUnmapBufferObject(pbo);
    
    /*
	dim3 blockDim(512,1,1);
	dim3 gridDim((unsigned int)(ceil((double)(w*h*d)/blockDim.x)), 1, 1);

	// Invoke kernel
    radioDepth<<<gridDim, blockDim>>>((float*)d_arr, dx, dy, dz);
    CHECK_FOR_CUDA_ERROR();

    cudaFreeArray(d_arr);
    CHECK_FOR_CUDA_ERROR();
    */

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
