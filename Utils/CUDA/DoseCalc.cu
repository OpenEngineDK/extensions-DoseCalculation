#include <Meta/CUDA.h>

#include <Utils/CUDA/DoseCalc.h>
#include <Utils/CUDA/uint_util.hcu>
#include <Utils/CUDA/float_util.hcu>
#include <Utils/CUDA/DozeCuda.h>

#include <stdlib.h>

#include <Utils/CUDA/CudaBeam.h>

#define _DOSE_DEVICE_BORDER

typedef unsigned char uchar;
typedef unsigned int  uint;

texture<float, 3, cudaReadModeElementType> tex;

unsigned int timer = 0;
uint3 dimensions;
float3 scaling;
__constant__ uint3 dims;
__constant__ uint2 beamletDims;
__constant__ float3 scale;
__constant__ CudaBeam beam;
#ifdef _DOSE_DEVICE_BORDER
__constant__ uint3 border;
#endif

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

    // Setup graphic card constants
    printf("Dimmensions: %d,%d,%d\n",w,h,d);
    dimensions = make_uint3(w, h, d);
    cudaMemcpyToSymbol(dims, &dimensions, sizeof(uint3));
    CHECK_FOR_CUDA_ERROR();

    printf("Scale: %f,%f,%f\n", sw, sh, sd);
    scaling = make_float3(sw, sh, sd);
    cudaMemcpyToSymbol(scale, &scaling, sizeof(float3));
    CHECK_FOR_CUDA_ERROR();

    //cutCreateTimer( &timer);

    printf("SetupDoseCalc done\n");
}

__device__ float3 GetWorldCoord(uint3 textureCoord){
    return make_float3(textureCoord.x * scale.x, 
                       textureCoord.y * scale.y, 
                       textureCoord.z * scale.z);
}

__device__ bool VoxelInsideBeamlet(float3 point, Matrix3x3 cone1, Matrix3x3 cone2){
    // __constant__ CudaBeam beam
    float3 translatedPoint = point - beam.src;
    return cone1.mul(translatedPoint) >= 0
        || cone2.mul(translatedPoint) >= 0;
}

__device__ float GetRadiologicalDepth(const uint3 textureCoord, const float3 coordinate){
    // __constant__ uint3 dims
    // __constant__ float3 scale
    // __constant__ CudaBeam beam;

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

#ifndef _DOSE_DEVICE_BORDER
    // The border texcoords
    const int border[3] = {(vec.x > 0) ? dims.x : -1,
                           (vec.y > 0) ? dims.y : -1,
                           (vec.z > 0) ? dims.z : -1};
#endif    

    // The remaining distance to the next crossing.
    float alpha[3] = {delta[0], delta[1], delta[2]};

    int texCoord[3] = {textureCoord.x, textureCoord.y, textureCoord.z};

    float radiologicalDepth = 0;

    while (
#ifdef _DOSE_DEVICE_BORDER
           texCoord[0] != border.x &&
           texCoord[1] != border.y &&
           texCoord[2] != border.z
#else
           texCoord[0] != border[0] &&
           texCoord[1] != border[1] &&
           texCoord[2] != border[2]
#endif
           ){
        
        // is x less then y?
        int minIndex = (alpha[0] < alpha[1]) ? 0 : 1;
        // is the above min less then z?
        minIndex = (alpha[minIndex] < alpha[2]) ? minIndex : 2;

        // We need to store the smallest alpha value so we can advance
        // the alpha with that value.
        float advance = alpha[minIndex];

        // Advance the alpha values.
        alpha[0] -= advance;
        alpha[1] -= advance;
        alpha[2] -= advance;

        // Add the delta value of the crossing dimension to prepare
        // for the next crossing.
        alpha[minIndex] += delta[minIndex];

        // Advance the texture coordinates
        texCoord[minIndex] += texDelta[minIndex];

        // Add the radiological length for this step to the overall
        // depth.
        radiologicalDepth += advance * tex3D(tex, texCoord[0], texCoord[1], texCoord[2]);
    }

    return radiologicalDepth;
}

/**
 * Rate each individual voxel based on the intensity, critical mass
 * and tumor found in it, and weighted by the length of the beam
 * passing through plus the amount of fotons deposited.
 *
 * return The rating.
 */
__device__ float RateVoxel(float delta, uint3 coord){
    return 1.0;
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

    const uint3 texCoord = idx_to_co(idx, dims);

    const float3 worldCoord = GetWorldCoord(texCoord);
   
    float rDepth = VoxelInsideBeamlet(worldCoord, beam.invCone1, beam.invCone2) ? GetRadiologicalDepth(texCoord, worldCoord) : 0.0f;

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

    const uint3 texCoord = idx_to_co(idx, dims);

    const float3 worldCoord = GetWorldCoord(texCoord);
   
    if (idx < dims.x * dims.y * dims.z)
        output[idx] = VoxelInsideBeamlet(worldCoord, beam.invCone1, beam.invCone2) ? 1.0f : 0.0f;
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

    

    // Calculate the local beamlet info, vectors are in texture coordinates.
    /*
    float3 v1;
    float3 v2;
    float3 v3;
    float3 v4;
    Matrix3x3 invCone1;
    invCone1(v1, v2, v3);
    invCone1 = invCone1.getInverse();
    Matrix3x3 invCone2;
    invCone2(v1, v2, v3);
    invCone2 = invCone2.getInverse();
    */

    /*
    // For each plane calculate wether the beam hits and in which
    // voxels it does.
    float rating = 0;
    uint2 from, to;
    uint3 coord;
    for (coord.z = 0; coord.z < dims.z; ++coord.z){
        if (BeamletPlaneIntersection(coord.z, from, to, v1, v2, v3, v4)){
            // Our beamlet intersect the plane. Lets see which voxels it hits.
            for (coord.x = from.x; coord.x < to.x; ++coord.x){
                for (coord.y = from.y; coord.y < to.y; ++coord.y){
                    float3 c = make_float3(coord.x, coord.y, coord.z);
                    if (VoxelInsideBeamlet(c, invCone1, invCone2)){
                        // Then rate the beam based on each voxel it hits.
                        
                    }
                }
            }
        }
    }
    */
}

void RunDoseCalc(float* cuDoseArr, Beam oeBeam, int beamlet_x, int beamlet_y, int kernel) {

    // Copy beam to device
    CudaBeam _beam;
    _beam(oeBeam, scaling);
    cudaMemcpyToSymbol(beam, &_beam, sizeof(CudaBeam));
    CHECK_FOR_CUDA_ERROR();

#ifdef _DOSE_DEVICE_BORDER
    // Copy texture borders to device. (borders closest to the
    // radioation source)
    uint3 _border;
    _border.x = abs(_beam.src.x) < abs(_beam.src.x - dimensions.x * scaling.x) ? 0 : dimensions.x;
    _border.y = abs(_beam.src.y) < abs(_beam.src.y - dimensions.y * scaling.y) ? 0 : dimensions.y;
    _border.z = abs(_beam.src.z) < abs(_beam.src.z - dimensions.z * scaling.z) ? 0 : dimensions.z;
    cudaMemcpyToSymbol(border, &_border, sizeof(uint3));
#endif
    
    /* const dim3 blockSize(16, 16, 1); */
    /* const dim3 gridSize(dimensions.x * dimensions.z / blockSize.x, dimensions.y / blockSize.y); */

    const dim3 voxelBlockSize(512, 1, 1);
    const dim3 voxelGridSize(dimensions.x * dimensions.z * dimensions.y / voxelBlockSize.x, 1);

    const dim3 beamletBlockSize(16,16,1);
    const dim3 beamletGridSize();

    // start timer
    //cutResetTimer(timer);
	//cutStartTimer(timer);

    // @TODO bind the radiological depth array as a texture after
    // being filled? Might yield nothing.

    switch(kernel){
    case 0:
        radioDepth<<< voxelGridSize, voxelBlockSize >>>(cuDoseArr);
        break;
    case 1:
        voxelsOfInterest<<< voxelGridSize, voxelBlockSize >>>(cuDoseArr);
        break;
    default:
        radioDepth<<< voxelGridSize, voxelBlockSize >>>(cuDoseArr);
        //doseCalc<<< >>>();
    }

	// Report timing
    /*
	cudaThreadSynchronize();
	cutStopTimer(timer);  
	double time = cutGetTimerValue( timer ); 
	printf("time: %.4f ms.\n", time );
    */

    CHECK_FOR_CUDA_ERROR();
}
