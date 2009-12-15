#include <Meta/CUDA.h>

#include <Utils/CUDA/DoseCalc.h>
#include <Utils/CUDA/uint_util.hcu>
#include <Utils/CUDA/float_util.hcu>
#include <Utils/CUDA/DozeCuda.h>

#include <stdlib.h>

#include <Utils/CUDA/Matrix3x3.h>

struct CudaBeam {
    float3 src;
    Matrix3x3 cone1;
    Matrix3x3 invCone1;
    Matrix3x3 cone2;
    Matrix3x3 invCone2;

    __host__ void operator() (Beam b){
        src.x = b.src[0];
        src.y = b.src[1];
        src.z = b.src[2];

        cone1(b.p1 - b.src, b.p2 - b.src, b.p3 - b.src);
        invCone1 = cone1.getInverse();

        cone2(b.p1 - b.src, b.p4 - b.src, b.p3 - b.src);
        invCone2 = cone2.getInverse();
    }
};

typedef unsigned char uchar;
typedef unsigned int  uint;

texture<float, 3, cudaReadModeElementType> tex;

uint3 dimensions;
__constant__ uint3 dims;
__constant__ float3 scale;
__constant__ CudaBeam beam;

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
    cudaMemcpyToSymbol(dims, &dimensions, sizeof(uint3));
    CHECK_FOR_CUDA_ERROR();
    //scale = make_float3(sw, sh, sd);
    cudaMemcpyToSymbol(scale, &make_float3(sw, sh, sd), sizeof(float3));
    CHECK_FOR_CUDA_ERROR();
}

__device__ bool VoxelInsideBeam(float3 point){
    // __constant__ CudaBeam beam
    return beam.invCone1.mul(point - beam.src) >= 0
        && beam.invCone1.mul(point - beam.src) >= 0;
}

__device__ float GetRadiologicalDepth(const uint3 coordinate){
    // __constant__ uint3 dims
    // __constant__ float3 scale
    // __constant__ CudaBeam beam;

    // The vector from the coordinate to the source
    const float3 vec = beam.src - coordinate;

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
    const int border[3] = {(vec.x > 0) ? dims.x : -1,
                           (vec.y > 0) ? dims.y : -1,
                           (vec.z > 0) ? dims.z : -1};
    
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

__global__ void radioDepth(float* output) {
    // __constant__ uint3 dims
    // __constant__ float3 scale
    // __constant__ CudaBeam beam;

    const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    const uint3 coordinate = idx_to_co(idx, dims);
   
    float rDepth = GetRadiologicalDepth(coordinate);

    if (idx < dims.x * dims.y * dims.z)
        output[idx] = rDepth;
}

__global__ void voxelsOfInterest(float* output) {
    // __constant__ uint3 dims
    // __constant__ float3 scale
    // __constant__ CudaBeam beam;

    const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    const uint3 coordinate = idx_to_co(idx, dims);

    const float3 fcoord = make_float3(coordinate.x * scale.x,
                                      coordinate.y * scale.y,
                                      coordinate.z * scale.z);
   
    if (idx < dims.x * dims.y * dims.z)
        output[idx] = (VoxelInsideBeam(fcoord)) ? 1 : 0;
}

__global__ void doseCalc(uint *d_output) {
    // __constant__ uint3 dims
    // __constant__ float3 scale
    // __constant__ CudaBeam beam;
    
}

void RunDoseCalc(float* cuDoseArr, Beam oeBeam, int beamlet_x, int beamlet_y, float dx, float dy, float dz) {
    float3 source = make_float3(oeBeam.src[0], oeBeam.src[1], oeBeam.src[2]);

    CudaBeam _beam;
    _beam(oeBeam);

    cudaMemcpyToSymbol(beam, &_beam, sizeof(CudaBeam));
    CHECK_FOR_CUDA_ERROR();

    /* const dim3 blockSize(16, 16, 1); */
    /* const dim3 gridSize(dimensions.x * dimensions.z / blockSize.x, dimensions.y / blockSize.y); */

    const dim3 blockSize(512, 1, 1);
    const dim3 gridSize(dimensions.x * dimensions.z * dimensions.y / blockSize.x, 1);

    //const dim3 blockSize(4, 4, 4);
    //const dim3 gridSize(dimensions.x / 4, dimensions.y / 4, dimensions.z / 4);
    
    radioDepth<<< gridSize, blockSize >>>(cuDoseArr);

    CHECK_FOR_CUDA_ERROR();
    printf("Hurray\n");

    
}
