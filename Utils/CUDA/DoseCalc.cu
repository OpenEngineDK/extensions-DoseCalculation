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

void SetupDoseCalc(float** cuDoseArr, 
                   int w, int h, int d, // dimensions
                   float sw, float sh, float sd) // scale
{ 
    
    cudaMalloc((void**)cuDoseArr, sizeof(float)*w*h*d);
    CHECK_FOR_CUDA_ERROR();

    printf("malloc: %d,%d,%d\n",w,h,d);

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    tex.filterMode = cudaFilterModeLinear;
    tex.addressMode[0] = cudaAddressModeClamp;
    tex.addressMode[1] = cudaAddressModeClamp;
    tex.addressMode[2] = cudaAddressModeClamp;

    cudaBindTextureToArray(tex, GetVolumeArray(), channelDesc);

    printf("SetupDoseCalc done\n");

    CHECK_FOR_CUDA_ERROR();
    dimensions = make_uint3(w, h, d);
    scale = make_float3(sw, sh, sd);
}

__device__ float GetRadiologicalDepth(uint3 coordinate, float3 source, uint3 dimensions, float3 scale){
    // The vector from the coordinate to the source
    const float3 vec = source - coordinate;

    const float dist = length(vec);

    // Instead of alpha between [0; 1] use the length of the
    // vector. This is usefull for when we need the length travelede
    // when accumulating radiological depth.

    // delta.x is the distance the beam has to travel between crossing
    // zy-planes.
    const float delta[3] = {dist * scale.x / vec.x,
                      dist * scale.y / vec.y,
                      dist * scale.z / vec.z};

    int texCoord[3] = {coordinate.x, coordinate.y, coordinate.z};

    const int texDelta[3] = {(vec.x > 0) ? 1 : -1,
                       (vec.y > 0) ? 1 : -1,
                       (vec.z > 0) ? 1 : -1};

    // The border texcoords (@TODO: Doesn't have to be calculated for
    // every voxel, move outside later.)
    const int border[3] = {(vec.x > 0) ? dimensions.x : -1,
                           (vec.y > 0) ? dimensions.y : -1,
                           (vec.z > 0) ? dimensions.z : -1};
    
    // The remaining distance to the next crossing.
    //float3 alpha = delta;
    float alpha[3] = {delta[0], delta[1], delta[2]};

    const int maxItr = 10;

    float radiologicalDepth = 0;
    int itr = 0;

    while (itr < maxItr){
        itr++;

        // is x less then y?
        int minIndex = (alpha[0] < alpha[1]) ? alpha[0] : alpha[1];
        // is the above min less then z?
        minIndex = (minIndex < alpha[2]) ? minIndex : alpha[2];

        // We need to store the smallest alpha value so we can advance
        // the alpha with that value.
        float advance = alpha[minIndex];

        // Add the delta value of the crossing dimension to prepare
        // for the next crossing.
        alpha[minIndex] += delta[minIndex];

        // Advance the alpha values.
        alpha[0] -= advance;
        alpha[1] -= advance;
        alpha[2] -= advance;

        // Advance the texture coordinates
        texCoord[minIndex] += texDelta[minIndex];

        // Add the radiological length for this step to the overall
        // depth.
        radiologicalDepth = tex3D(tex, texCoord[0], texCoord[1], texCoord[2]);
    }

    /*
    while (texCoord[0] != border.x ||
           texCoord[1] != border.y ||
           texCoord[2] != border.z ||
           itr < maxItr){
    
        itr++;

        // is x less then y?
        int minIndex = (alpha[0] < alpha[1]) ? alpha[0] : alpha[1];
        // is the above min less then z?
        minIndex = (minIndex < alpha[2]) ? minIndex : alpha[2];
        
        // We need to store the smallest alpha value so we can advance
        // the alpha with that value.
        float advance = alpha[minIndex];

        // Add the delta value of the crossing dimension to prepare
        // for the next crossing.
        alpha[minIndex] += delta[minIndex];

        // Advance the alpha values.
        alpha[0] -= advance;
        alpha[1] -= advance;
        alpha[2] -= advance;

        // Advance the texture coordinates
        texCoord[minIndex] += texDelta[minIndex];

        // Add the radiological length for this step to the overall
        // depth.
        radiologicalDepth = tex3D(tex, texCoord[0], texCoord[1], texCoord[2]);
    }
    */
    
    return radiologicalDepth;
}

__global__ void radioDepth(float* output, uint3 dims, float3 scale, float3 source) {
    const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    const uint3 coordinate = idx_to_co(idx, dims);

    float rDepth = GetRadiologicalDepth(coordinate, source, dims, scale);

    output[idx] = (float(coordinate.x) / float(dims.x)); // + coordinate.y / dims.y + coordinate.z / dims.z) * 0.25f;
}

__global__ void doseCalc(uint *d_output) {

}

void RunDoseCalc(float* cuDoseArr, Beam beam, int beamlet_x, int beamlet_y, float dx, float dy, float dz) {
    float3 source = make_float3(beam.src[0], beam.src[1], beam.src[2]);

    /*
    const unsigned int blockDimX = 512;
    const dim3 blockSize(blockDimX,1,1);
    const float entries = dimensions.x * dimensions.y * dimensions.z;
    const dim3 gridSize(ceil(entries/(float)blockDimX), 1, 1);
    */

    const dim3 blockSize(16, 16, 1);
    const dim3 gridSize(dimensions.x * dimensions.z / blockSize.x, dimensions.y / blockSize.y);

    radioDepth<<< gridSize, blockSize >>>((float*)cuDoseArr, 
                                        dimensions,
                                        scale,
                                        source);
    CHECK_FOR_CUDA_ERROR();
    printf("Hurray\n");

}
