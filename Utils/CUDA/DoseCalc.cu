#include <Meta/CUDA.h>

#include <Utils/CUDA/DoseCalc.h>
#include <Utils/CUDA/uint_util.hcu>
#include <Utils/CUDA/DozeCuda.h>

#include <stdlib.h>

typedef unsigned char uchar;
typedef unsigned int  uint;

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

    /* CHECK_FOR_CUDA_ERROR(); */
    dimensions = make_uint3(w, h, d);

}

__global__ void radioDepth(float* output, uint3 dims) {
    const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    // lookup via tex3D(...);

    uint3 coordinate = idx_to_co(idx, dims);

    output[idx] = (coordinate.x / dims.x + coordinate.y / dims.y + coordinate.z / dims.z) * 0.25f;
}

__global__ void doseCalc(uint *d_output) {

}

void RunDoseCalc(GLuint pbo, Beam beam, int beamlet_x, int beamlet_y, float dx, float dy, float dz) {
    // Map the buffer object that we want to write the radiological depth to.
    float* radiologicalDepth;
    cudaGLMapBufferObject( (void**)&radiologicalDepth, pbo);

    dim3 blockDim(512,1,1);
    double entries = dimensions.x * dimensions.y * dimensions.z;
	dim3 gridDim((uint)(ceil(entries/blockDim.x)), 1, 1);

    radioDepth<<< gridDim, blockDim >>>(radiologicalDepth, 
                                        dimensions);

    cudaGLUnmapBufferObject(pbo);    
}
