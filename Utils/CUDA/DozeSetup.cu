#include <Meta/CUDA.h>
#include <Utils/CUDA/Doze.h>
// Lets do Teh Setup!


cudaArray *d_volumeArray = 0;

texture<float, 3, cudaReadModeElementType> tex;

cudaArray* GetVolumeArray() {
    return d_volumeArray;
}

void SetupDoze(const float* data, int w, int h, int d) {
    
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();    
    cudaExtent ext = make_cudaExtent(w,h,d);


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

    printf("Doze Are SETUP!\n");
    
}
