#include "RayCaster.h"
#include <Meta/CUDA.h>

typedef unsigned char uchar;

texture<uchar, 3, cudaReadModeNormalizedFloat> tex;
cudaArray *d_volumeArray = 0;



void SetupRayCaster(int pbo, const float* data, int w, int h, int d) {
    
    cudaGLRegisterBufferObject(pbo);
    CHECK_FOR_CUDA_ERROR();


    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    
    cudaExtent ext = make_cudaExtent(h,w,d);

    cudaMalloc3DArray(&d_volumeArray, &channelDesc, ext);
    CHECK_FOR_CUDA_ERROR();

    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_cudaPitchedPtr((void*)data, 
                                            ext.width*sizeof(float),
                                            ext.width,
                                            ext.height);
    copyParams.dstArray = d_volumeArray;
    copyParams.extent = ext;
    copyParams.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParams);
    CHECK_FOR_CUDA_ERROR();
    
    tex.normalized = true;
    tex.filterMode = cudaFilterModeLinear;
    tex.addressMode[0] = cudaAddressModeWrap;
    tex.addressMode[1] = cudaAddressModeWrap;
    tex.addressMode[2] = cudaAddressModeWrap;

    cudaBindTextureToArray(tex, d_volumeArray, channelDesc);
    CHECK_FOR_CUDA_ERROR();

}

__global__ void rayCaster(uint* p_out, uint imageW, uint imageH) {
    uint x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;

    float u = x / (float) imageW;
    float v = y / (float) imageH;
    // read from 3D texture
    float voxel = tex3D(tex, u, v, 0.0f);

    if ((x < imageW) && (y < imageH)) {
        // write output color
        uint i = __umul24(y, imageW) + x;
        p_out[i] = voxel*255;
        
    }

}

void RenderToPBO(int pbo, int width, int height) {
    uint* p;
    cudaGLMapBufferObject((void**)&p,pbo);
    CHECK_FOR_CUDA_ERROR();
    
    const dim3 blockSize(16, 16, 1);
    const dim3 gridSize(width / blockSize.x, height / blockSize.y);


    rayCaster<<<gridSize, blockSize>>>(p,width,height);

    CHECK_FOR_CUDA_ERROR();

    cudaGLUnmapBufferObject(pbo);
    CHECK_FOR_CUDA_ERROR();

}
