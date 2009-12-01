#include "RayCaster.h"
#include <Meta/CUDA.h>

typedef unsigned char uchar;

texture<float, 3, cudaReadModeElementType> tex;
cudaArray *d_volumeArray = 0;

cudaExtent size;

void SetupRayCaster(int pbo, const float* data, int w, int h, int d) {
    
    cudaGLRegisterBufferObject(pbo);
    CHECK_FOR_CUDA_ERROR();


    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    
    cudaExtent ext = make_cudaExtent(w,h,d);
    size = ext;

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
    tex.addressMode[0] = cudaAddressModeClamp;
    tex.addressMode[1] = cudaAddressModeClamp;
    tex.addressMode[2] = cudaAddressModeClamp;

    cudaBindTextureToArray(tex, d_volumeArray, channelDesc);
    CHECK_FOR_CUDA_ERROR();

}

__device__ uint rgbaFloatToInt(float4 rgba)
{
    rgba.x = __saturatef(rgba.x);   // clamp to [0.0, 1.0]
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    return (uint(rgba.w*255)<<24) | (uint(rgba.z*255)<<16) | (uint(rgba.y*255)<<8) | uint(rgba.x*255);
}


__global__ void rayCaster(uint* p_out, uint imageW, uint imageH, cudaExtent ext) {
    uint x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;

    float u = x / (float) imageW;
    float v = y / (float) imageH;
    // read from 3D texture
    float voxel = tex3D(tex, u, v, 0.0f);

    if ((x < imageW) && (y < imageH)) {
        // write output color
        uint i = __umul24(y, imageW) + x;
        float4 c = make_float4(voxel);
        //c.x = voxel;

        p_out[i] = rgbaFloatToInt(c);
        
    }

}

void RenderToPBO(int pbo, int width, int height) {
    uint* p;
    cudaGLMapBufferObject((void**)&p,pbo);
    CHECK_FOR_CUDA_ERROR();
    
    const dim3 blockSize(16, 16, 1);
    const dim3 gridSize(size.width / blockSize.x, size.height / blockSize.y);

    
    rayCaster<<<gridSize, blockSize>>>(p,width,height,size);

    CHECK_FOR_CUDA_ERROR();

    cudaGLUnmapBufferObject(pbo);
    CHECK_FOR_CUDA_ERROR();

}
