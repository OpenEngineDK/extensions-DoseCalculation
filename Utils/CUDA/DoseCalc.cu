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

    // Setup texture
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    tex.normalized = false;
    tex.filterMode = cudaFilterModeLinear;
    tex.addressMode[0] = cudaAddressModeClamp;
    tex.addressMode[1] = cudaAddressModeClamp;
    tex.addressMode[2] = cudaAddressModeClamp;
    cudaBindTextureToArray(tex, GetVolumeArray(), channelDesc);
    CHECK_FOR_CUDA_ERROR();

    printf("Dimmensions: %d,%d,%d\n",w,h,d);
    dimensions = make_uint3(w, h, d);
    cudaMemcpyToSymbol(dims, &dimensions, sizeof(uint3));
    CHECK_FOR_CUDA_ERROR();

    printf("Scale: %f,%f,%f\n", sw, sh, sd);
    cudaMemcpyToSymbol(scale, &make_float3(sw, sh, sd), sizeof(float3));
    CHECK_FOR_CUDA_ERROR();

    printf("SetupDoseCalc done\n");
}

__device__ bool VoxelInsideBeam(float3 point){
    // __constant__ CudaBeam beam
    float3 translatedPoint = point - beam.src;
    return beam.invCone1.mul(translatedPoint) >= 0
        || beam.invCone2.mul(translatedPoint) >= 0;
}

__device__ float GetRadiologicalDepth(const uint3 textureCoord){
    // __constant__ uint3 dims
    // __constant__ float3 scale
    // __constant__ CudaBeam beam;

    // The texture coordinate is in buffer space. Not yet scaled.

    // Coordinate in world space.
    const float3 coordinate = make_float3(textureCoord.x * scale.x, 
                                          textureCoord.y * scale.y, 
                                          textureCoord.z * scale.z);

    // The vector from the coordinate to the source
    const float3 vec = beam.src - coordinate;

    const float dist = length(vec);

    // delta.x is the distance the beam has to travel between crossing
    // zy-planes. The distance is always positive.
    const float delta[3] = {abs(dist / vec.x),
                            abs(dist / vec.y),
                            abs(dist / vec.z)};
    
    const int texDelta[3] = {(vec.x > 0) ? 1 : -1,
                             (vec.y > 0) ? 1 : -1,
                             (vec.z > 0) ? 1 : -1};

    // The border texcoords (@TODO: Doesn't have to be calculated for
    // every voxel, make __constant__ and update before each run.)
    const int border[3] = {(vec.x > 0) ? dims.x : -1,
                           (vec.y > 0) ? dims.y : -1,
                           (vec.z > 0) ? dims.z : -1};
    
    // The remaining distance to the next crossing.
    float alpha[3] = {delta[0], delta[1], delta[2]};

    int texCoord[3] = {textureCoord.x, textureCoord.y, textureCoord.z};

    float radiologicalDepth = 0;

    while (0 <= texCoord[0] && texCoord[0] < dims.x &&
           0 <= texCoord[1] && texCoord[1] < dims.y &&
           0 <= texCoord[2] && texCoord[2] < dims.z
           /*texCoord[0] != border[0] &&
           texCoord[1] != border[1] &&
           texCoord[2] != border[2]*/){
        
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
        radiologicalDepth += advance * tex3D(tex, texCoord[0], texCoord[1], texCoord[2]);
    }

    return radiologicalDepth;
}

/**
 * Calculates the radiological depth of each voxel and stores it in
 * the output array.
 */
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

/**
 * Calculates for each voxel wether it is inside the beam or not.
 *
 * param output An array of the voxels interest. Contains 1.0 if the
 * voxel is in the beam otherwise 0.0f.
 */
__global__ void voxelsOfInterest(float* output) {
    // __constant__ uint3 dims
    // __constant__ float3 scale
    // __constant__ CudaBeam beam;

    const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    const uint3 coordinate = idx_to_co(idx, dims);

    // @todo multiply by scale or divide?
    const float3 fcoord = make_float3(coordinate.x * scale.x,
                                      coordinate.y * scale.y,
                                      coordinate.z * scale.z);
   
    if (idx < dims.x * dims.y * dims.z)
        output[idx] = (VoxelInsideBeam(fcoord)) ? 1.0f : 0.0f;
}

/**
 * Calculate the score of each beamlet, dependent on the voxels it hits.
 *
 * param input An array of radiological depths for each voxel.
 * param output An boolean array of how each beamlet performed.
 */
__global__ void doseCalc(float* input, uint *output) {
    // __constant__ uint3 dims
    // __constant__ float3 scale
    // __constant__ CudaBeam beam;

    // Calculate the inverse matrix of the beams 2 convex cones.

    // For each plane calculate wether the beam hits and in which
    // voxels it does.

    // Then rate the beam based on each voxel it hits.
}

void RunDoseCalc(float* cuDoseArr, Beam oeBeam, int beamlet_x, int beamlet_y, int kernel) {
    CudaBeam _beam;
    _beam(oeBeam);

    cudaMemcpyToSymbol(beam, &_beam, sizeof(CudaBeam));
    CHECK_FOR_CUDA_ERROR();

    /* const dim3 blockSize(16, 16, 1); */
    /* const dim3 gridSize(dimensions.x * dimensions.z / blockSize.x, dimensions.y / blockSize.y); */

    const dim3 blockSize(512, 1, 1);
    const dim3 gridSize(dimensions.x * dimensions.z * dimensions.y / blockSize.x, 1);

    switch(kernel){
    case 0:
        radioDepth<<< gridSize, blockSize >>>(cuDoseArr);
        break;
    case 1:
        voxelsOfInterest<<< gridSize, blockSize >>>(cuDoseArr);
        break;
    default:
        voxelsOfInterest<<< gridSize, blockSize >>>(cuDoseArr);
    }

    /*
      // Voxel of interest debug print.
    printf("Source\n");
    printf("[%f, %f, %f]\n", _beam.src.x, _beam.src.y, _beam.src.z);

    printf("\nCone 1\n");
    _beam.cone1.print();

    printf("\nCone 1 inverse\n");
    _beam.invCone1.print();

    printf("\nCone 1 inverse * ((0, 0, 0) - source)\n");
    float3 res = _beam.invCone1.mul(make_float3(0.0f) - _beam.src);
    printf("[%f, %f, %f]\n", res.x, res.y, res.z);

    printf("\nCone 2\n");
    _beam.cone2.print();

    printf("\nCone 2 inverse * ((0, 0, 0) - source)\n");
    res = _beam.invCone2.mul(make_float3(0.0f) - _beam.src);
    printf("[%f, %f, %f]\n", res.x, res.y, res.z);
    */

    CHECK_FOR_CUDA_ERROR();
}
