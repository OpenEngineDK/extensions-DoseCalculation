#include <Meta/CUDA.h>

#include <Utils/CUDA/DoseCalc.h>
#include <Utils/CUDA/uint_util.hcu>
#include <Utils/CUDA/float_util.hcu>
#include <Utils/CUDA/DozeCuda.h>

#include <stdlib.h>

struct Matrix3x3 {
    float3 e[3];

    __device__ float3 mul(float3 m){
        return make_float3(dot(m, e[0]),
                           dot(m, e[1]),
                           dot(m, e[2]));
    }
};

struct CudaBeam {
    float3 source;
    Matrix3x3 invCone1;
    Matrix3x3 invCone2;
};

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

    // Setup texture
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    tex.normalized = false;
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

__device__ bool VoxelInsiceBeam(Matrix3x3 invCone1, Matrix3x3 invCone2, float3 point){
    return (invCone1.mul(point) >= 0.0f && invCone2.mul(point) >= 0.0f);
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

    int texCoord[3] = {coordinate.x, coordinate.y, coordinate.z};

    const int maxItr = 100;

    float radiologicalDepth = 0;
    int itr = 0;

    while ((texCoord[0] != border[0] &&
            texCoord[1] != border[1] &&
            texCoord[2] != border[2]) &&
           /*0 <= texCoord[0] && texCoord[0] < dimensions.x &&
           0 <= texCoord[1] && texCoord[1] < dimensions.y &&
           0 <= texCoord[2] && texCoord[2] < dimensions.z && */
           itr < maxItr){
        itr++;

        // is x less then y?
        int minIndex = (alpha[0] < alpha[1]) ? 0 : 1;
        // is the above min less then z?
        minIndex = (alpha[minIndex] < alpha[2]) ? minIndex : 2;

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
        radiologicalDepth = advance * tex3D(tex, texCoord[0], texCoord[1], texCoord[2]);
    }

    return radiologicalDepth;
}

__global__ void radioDepth(float* output, uint3 dims, float3 scale, float3 source) {
    const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    const uint3 coordinate = idx_to_co(idx, dims);

    float rDepth = GetRadiologicalDepth(coordinate, source, dims, scale);

    if (idx < dims.x * dims.y * dims.z)
        output[idx] = rDepth;
}

__global__ void doseCalc(uint *d_output) {

}

void RunDoseCalc(float* cuDoseArr, Beam beam, int beamlet_x, int beamlet_y, float dx, float dy, float dz) {
    float3 source = make_float3(beam.src[0], beam.src[1], beam.src[2]);

    const dim3 blockSize(16, 16, 1);
    const dim3 gridSize(dimensions.x * dimensions.z / blockSize.x, dimensions.y / blockSize.y);

    radioDepth<<< gridSize, blockSize >>>((float*)cuDoseArr, 
                                        dimensions,
                                        scale,
                                        source);
    CHECK_FOR_CUDA_ERROR();
    printf("Hurray\n");

}
