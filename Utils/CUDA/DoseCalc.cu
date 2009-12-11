#include <Meta/CUDA.h>

#include <Utils/CUDA/DoseCalc.h>
#include <Utils/CUDA/uint_util.hcu>
#include <Utils/CUDA/DozeCuda.h>

#include <stdlib.h>

typedef unsigned char uchar;
typedef unsigned int  uint;

texture<float, 3, cudaReadModeElementType> tex;
uint3 dimensions; // should be placd in constant memory
float3 scale; // should be placed in constant memory

void SetupDoseCalc(unsigned int pbo, 
                   int w, int h, int d, // dimensions
                   float sw, float sh, float sd) // scale
{ 
    cudaGLRegisterBufferObject(pbo);
    CHECK_FOR_CUDA_ERROR();

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    tex.normalized = true;
    tex.filterMode = cudaFilterModeLinear;
    tex.addressMode[0] = cudaAddressModeClamp;
    tex.addressMode[1] = cudaAddressModeClamp;
    tex.addressMode[2] = cudaAddressModeClamp;

    cudaBindTextureToArray(tex, GetVolumeArray(), channelDesc);

    printf("SetupDoseCalc done: %i\n",pbo);

    CHECK_FOR_CUDA_ERROR();
    dimensions = make_uint3(w, h, d);
    scale = make_float3(sw, sh, sd);
}

__device__ float GetRadiologicalDepth(float3 coordinate, float3 source, float3 dimensions, float3 scale){
    // The vector from the coordinate to the source
    float3 vec = source - coordinate;

    float dist = length(vec);

    // Instead of alpha between [0; 1] use the length of the
    // vector. (and in the future scale the length to make it match
    // texcoords?)

    // delta.x is the distance the beam has to travel between crossing
    // zy-planes.
    float3 delta = dist * scale / vec;

    float3 texDelta = make_float3(1.0f, 1.0f, 1.0f) / dimensions;

    float3 texCoord = coordinate * texDelta;

    texDelta.x = (vec.x > 0) ? texDelta.x : -texDelta.x;
    texDelta.y = (vec.y > 0) ? texDelta.y : -texDelta.y;
    texDelta.z = (vec.z > 0) ? texDelta.z : -texDelta.z;

    // The border texcoords (@TODO: Doesn't have to be calculated for
    // every voxel, move outside later.)
    float3 border = make_float3((vec.x > 0) ? 1 : 0,
                                (vec.y > 0) ? 1 : 0,
                                (vec.z > 0) ? 1 : 0);

    // The remaining distance to the next crossing.
    float3 alpha = delta;

    float radiologicalDepth = 0;
    while (texCoord.x != border.x ||
           texCoord.y != border.y ||
           texCoord.z != border.z){

        // Replace float3 with float[3] so we only need to branch for
        // the index and can then do the calculations?
        float alphaInc;
        if (alpha.x < alpha.y){
            if (alpha.x < alpha.z) {
                // x is smallest
                alphaInc = alpha.x;
                alpha.x += delta.x;
                texCoord.x += texDelta.x;
            } else {
                // z is smalles
                alphaInc = alpha.z;
                alpha.z += delta.z;
                texCoord.z += texDelta.z;
            }
        }else{
            if (alpha.y < alpha.z) {
                // y is smallest
                alphaInc = alpha.y;
                alpha.y += delta.y;
                texCoord.y += texDelta.y;                
            } else {
                // z is smalles
                alphaInc = alpha.z;
                alpha.z += delta.z;
                texCoord.z += texDelta.z;
            }
        }
        alpha.x -= alphaInc;
        alpha.y -= alphaInc;
        alpha.z -= alphaInc;

        radiologicalDepth += tex3D(tex, texCoord.x, texCoord.y, texCoord.z);
    }

    return radiologicalDepth;
}

__global__ void radioDepth(float* output, uint3 dims, float3 scale, Beam beam) {
    const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    // lookup via tex3D(...);

    uint3 coordinate = idx_to_co(idx, dims);

    output[idx] = (coordinate.x / dims.x + coordinate.y / dims.y + coordinate.z / dims.z) * 0.25f;
}

__global__ void doseCalc(uint *d_output) {

}

void RunDoseCalc(unsigned int pbo, Beam beam, int beamlet_x, int beamlet_y, float dx, float dy, float dz) {
    // Map the buffer object that we want to write the radiological depth to.
    float* radiologicalDepth;
    cudaGLMapBufferObject( (void**)&radiologicalDepth, pbo);

    dim3 blockDim(512,1,1);
    double entries = dimensions.x * dimensions.y * dimensions.z;
	dim3 gridDim((uint)(ceil(entries/blockDim.x)), 1, 1);

    radioDepth<<< gridDim, blockDim >>>(radiologicalDepth, 
                                        dimensions,
                                        scale,
                                        beam);

    cudaGLUnmapBufferObject(pbo);    
}
