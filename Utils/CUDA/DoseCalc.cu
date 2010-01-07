#include <Meta/CUDA.h>

#include <Utils/CUDA/DoseCalc.h>
#include <Utils/CUDA/uint_util.hcu>
#include <Utils/CUDA/float_util.hcu>
#include <Utils/CUDA/DozeCuda.h>

#include <stdlib.h>

#include <Utils/CUDA/CudaBeam.h>

#include <Logging/Logger.h>
#define _DOSE_DEVICE_BORDER
#define PIXEL_UNIT 0.01f // one pixel = 1 cm^3
#define MU_WATER 0.135f  // lets just use 0.135 for mu_water(E=171keV) ...

#define HALFSIZE 4

texture<float, 3, cudaReadModeElementType> intensityTex;
texture<float, 3, cudaReadModeElementType> termaTex;

unsigned int timer = 0;

uint3 dimensions;
uint2 beamletDimensions;
float3 scaling;
uint size;
__constant__ uint3 dims;
__constant__ uint2 beamletDims;
__constant__ float2 invBeamletDims;
__constant__ float3 scale;
__constant__ CudaBeam beam;
#ifdef _DOSE_DEVICE_BORDER
__constant__ uint3 border;
#endif

__constant__ float3 ssd;      // source to surface distance
__constant__ float energy;    // average energy of the beam i.e 15 MeV


// Dose kernel measurements of 15MeV mono-energetic beam through a
// homogenious phantom (Mackie et. al. 1984).
__constant__ float kern[7][3] = { {.0001f, .0000f, .0000f},
                                  {.3250f, .0110f, .0000f},
                                  {.2340f, .0239f, .0004f},
                                  {.0697f, .0190f, .0309f},
                                  {.0179f, .0089f, .0007f},
                                  {.0047f, .0027f, .0007f},
                                  {.0009f, .0006f, .0004f} };
__constant__ float kscale = 0.01f;
__constant__ float kdensity = 0.001f;
__constant__ int halfsize;

void print(float3 v){
    printf("(%f, %f, %f)", v.x, v.y, v.z);
}

void SetupDoseCalc(float** cuDoseArr, 
                   int w, int h, int d, // dimensions
                   float sw, float sh, float sd) // scale
{ 
    cudaMalloc((void**)cuDoseArr, sizeof(float)*w*h*d);
    CHECK_FOR_CUDA_ERROR();

    cudaMemset(*cuDoseArr, 0, sizeof(float)*w*h*d);

    // Setup texture
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    intensityTex.normalized = false;
    intensityTex.filterMode = cudaFilterModeLinear;
    intensityTex.addressMode[0] = cudaAddressModeClamp;
    intensityTex.addressMode[1] = cudaAddressModeClamp;
    intensityTex.addressMode[2] = cudaAddressModeClamp;
    cudaBindTextureToArray(intensityTex, GetVolumeArray(), channelDesc);
    CHECK_FOR_CUDA_ERROR();

    // Setup graphic card constants
    printf("Dimmensions: %d,%d,%d\n",w,h,d);
    dimensions = make_uint3(w, h, d);
    size = w * h * d;
    cudaMemcpyToSymbol(dims, &dimensions, sizeof(uint3));
    CHECK_FOR_CUDA_ERROR();

    printf("Scale: %f,%f,%f\n", sw, sh, sd);
    scaling = make_float3(sw, sh, sd);
    cudaMemcpyToSymbol(scale, &scaling, sizeof(float3));
    CHECK_FOR_CUDA_ERROR();

    cutCreateTimer( &timer);

    printf("SetupDoseCalc done\n");
}

// utility functions

/**
 * return the central position of a voxel.
 */
__device__ float3 texToVec(uint3 tex) {
    return make_float3(tex.x * scale.x + 0.5f * scale.x,
                       tex.y * scale.y + 0.5f * scale.y,
                       tex.z * scale.z + 0.5f * scale.z);
}

/**
 * Determine the voxel coordinate containing the point vec.
 */
__device__ uint3 vecToTex(float3 vec) {
    return make_uint3(vec.x / scale.x, vec.y / scale.y, vec.z / scale.z);
}

// theoretical utility functions

/**
 * Account for any prior changes made to the voxel data
 */
__device__ float hounsfield(uint3 r) {
    return (tex3D(intensityTex, r.x, r.y, r.z) * 2000.0f - 1000.0f);
}

/**
 * Get linear attenuation coeffecient (mu).
 *
 * HU = ((mu - mu_water(E)) / mu_water(E)) * 1000
 *
 * where mu_water and the resulting HU are dependent on the energy E of
 * the CT scanner (Brown et. al. 2006).
 * 
 */
__device__ float attenuation(uint3 r) {
    return (hounsfield(r) / 1000.0f) * MU_WATER + MU_WATER;
}

/**
 * Du to the way we preprocess the texture and the definition of
 * hounsfield units and attenuation. We can optimize the attenuation
 * like this.
 */
__device__ float attenuationOpt(uint3 r) {
    return tex3D(intensityTex, r.x, r.y, r.z) * 2 * MU_WATER;
}

/**
 * Get density relative to water.
 */
__device__ float density(uint3 r) {
    // not quite sure how relative density relates to the HU. 
    return (attenuation(r) / MU_WATER) * PIXEL_UNIT ;
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

/**
 * Get the attenuation of the radiological depth for the voxel at
 * textureCoord and traced back to coordinate.
 */
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
    
    const uint texDelta[3] = {(vec.x > 0) ? 1 : -1,
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
           /*
#ifdef _DOSE_DEVICE_BORDER
           texCoord[0] != border.x &&
           texCoord[1] != border.y &&
           texCoord[2] != border.z
#else
           texCoord[0] != border[0] &&
           texCoord[1] != border[1] &&
           texCoord[2] != border[2]
#endif
           */
           texCoord[0] < dims.x &&
           texCoord[1] < dims.y &&
           texCoord[2] < dims.z
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

        // Add the radiological length for this step to the overall
        // depth.
        //radiologicalDepth += advance * tex3D(intensityTex, texCoord[0], texCoord[1], texCoord[2]);
        radiologicalDepth += advance * attenuation(make_uint3(texCoord[0], texCoord[1], texCoord[2]));

        // Advance the texture coordinates
        texCoord[minIndex] += texDelta[minIndex];
    }

    return radiologicalDepth;
}

/**
 * Summation of the weighted attenuation coefficients from the point r
 * back to the src point.
 */
__device__ float sumAtt(float3 r, uint3 tc) {
    float3 dir = beam.src - r;
    float3 invDir = make_float3(1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z);
    float l = length(dir) * PIXEL_UNIT;
    
    // delta tc is determined by the sign of the direction.
    const int3 dtc = make_int3((dir.x > 0) ? 1 : -1,
                               (dir.y > 0) ? 1 : -1,
                               (dir.z > 0) ? 1 : -1);
                               
    // the initial offsets of the planes closest to the point r in the
    // direction of dir.
    float3 planes = make_float3(scale.x * (tc.x + dtc.x * 0.5f + 0.5f),
                                scale.y * (tc.y + dtc.y * 0.5f + 0.5f),
                                scale.z * (tc.z + dtc.z * 0.5f + 0.5f));
    
    float prevAlpha = 0.0f;
    float sum = 0.0f;

    // while we are still inside the voxel boundaries ... 

    // Using major hack here, when the coord gets below 0 it will wrap
    // around to max_uint and then be outside the texture (unless you
    // use a huuuuuuuuuuuuuuuuuuuuu ... uuuuuuuuuuge texture, but
    // c'mon!)
    while (tc.x < dims.x &&
           tc.y < dims.y &&
           tc.z < dims.z) {

        // Determine the scaling factors. Result is always positive
        // since the signs are the same on both sides of the division.
        // (planes - r) can never be zero since we advance the plane
        // offset away from r.
        float3 alphas = ( planes - r ) * invDir;
        // if dir is zero then the result will be infty or -infty.
        // this is a dirty hack to ensure that we only get positive infty.
        alphas = make_float3(fabs(alphas.x),
                             fabs(alphas.y),
                             fabs(alphas.z));
        float alpha = fmin(alphas.x, alphas.y);
        alpha = fmin(alpha, alphas.z);
        
        sum += attenuationOpt(make_uint3(tc.x, tc.y, tc.z)) * (alpha - prevAlpha) * l;
        prevAlpha = alpha;

        // Find minimal coordinates. Note that several coordinates
        // could be minimal (equal).
        uint3 min = make_uint3( (alphas.x == alpha) ? dtc.x : 0,
                          (alphas.y == alpha) ? dtc.y : 0,
                          (alphas.z == alpha) ? dtc.z : 0 );
        
        // advance the texture coordinates (termination is based on dot(min, min) != 0)
        tc += min;
        
        // advance the planes.
        planes = make_float3(planes.x + scale.x * min.x,
                             planes.y + scale.y * min.y,
                             planes.z + scale.z * min.z);
        
    }
    return sum;
}

/**
 * incident fluence in the direction of r.
 */
__device__ float initFluence(float3 r) {
    // simply scale the average energy with respect to off-axis effect.
    // This is a nasty simplification!!! 
    return energy * (dot(normalize(r - beam.src), beam.axis));
}

__device__ float fluence(float3 r, uint3 tc) {
    float d = length(r - beam.src) * PIXEL_UNIT;
    return (initFluence(r)/(d*d)) * expf(-sumAtt(r, tc));
    //return (initFluence(r)/(d*d)) * expf(-GetRadiologicalDepth(tc, r));
}

/**
 * Calculate Total Energy Released per unit MAss (TERMA) for each
 * voxel. This is the energy deposited by the incident fluence
 * e.g. the initial collision of photons. TERMA is determined by
 * incident fluence, linear attenuation, and
 * off-axis effects (intensity has an inverse square relation with the
 * angle to the beam axis (Jacques et. al. 2009).
 */
__global__ void terma(uint offset, float* out, uchar* fmap) {
    // const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    // const uint idx = blockIdx.x * blockDim.x + blockIdx.y * gridDim.x*blockDim.x + threadIdx.x;
    const uint idx = offset + blockIdx.x * blockDim.x + blockIdx.y * gridDim.x*blockDim.x + threadIdx.x;
    const uint3 tc = idx_to_co(idx, dims);
    const float3 vec = texToVec(tc);
    out[idx] = VoxelInsideBeamlet(vec, beam.invCone1, beam.invCone2) ? 
        ((attenuation(tc) / density(tc)) * fluence(vec, tc)) : 0.0f;
    // out[idx] = (attenuation(tc) / density(tc)) * fluence(vec, tc)/(energy);
    // out[idx] = VoxelInsideBeamlet(vec, beam.invCone1, beam.invCone2) ? sumAtt(vec, tc) : 0.0f;
    // out[idx] = sumAtt(vec, tc);
}

__device__ float rad(uint3 tc1, float3 vec1, uint3 tc2, float3 vec2) {
    float3 dir = vec2 - vec1;
    float3 invDir = make_float3(1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z);
    uint3 tc = tc1;

    // delta tc is determined by the sign of the direction.
    const int3 dtc = make_int3((dir.x > 0) ? 1 : -1,
                               (dir.y > 0) ? 1 : -1,
                               (dir.z > 0) ? 1 : -1);
                               
    // the initial offsets of the planes closest to the point r in the
    // direction of dir.
    float3 planes = make_float3(scale.x * (tc.x + dtc.x * 0.5f + 0.5f),
                                scale.y * (tc.y + dtc.y * 0.5f + 0.5f),
                                scale.z * (tc.z + dtc.z * 0.5f + 0.5f));

    float prevAlpha = 0.0f;
    float sum = 0.0f;

    // while we are still inside the voxel boundaries ... 
    while (tc.x != tc2.x &&
           tc.y != tc2.y &&
           tc.z != tc2.z) {

        // Determine the scaling factors. Result is always positive
        // since the signs are the same on both sides of the division.
        // (planes - r) can never be zero since we advance the plane
        // offset away from r.
        float3 alphas = ( planes - vec1 ) * invDir;
        // if dir is zero then the result will be infty or -infty.
        // this is a dirty hack to ensure that we only get positive infty.
        alphas = make_float3(fabs(alphas.x),
                             fabs(alphas.y),
                             fabs(alphas.z));
        float alpha = fmin(alphas.x, alphas.y);
        alpha = fmin(alpha, alphas.z);
        
        sum += density(tc) * length((alpha - prevAlpha) *  dir); 
        prevAlpha = alpha;

        uint3 min = make_uint3( (alphas.x == alpha) ? dtc.x : 0,
                                (alphas.y == alpha) ? dtc.y : 0,
                                (alphas.z == alpha) ? dtc.z : 0 );
        tc += min;
        
        planes = make_float3(planes.x + scale.x * min.x,
                             planes.y + scale.y * min.y,
                             planes.z + scale.z * min.z);
        
    }
    return sum + density(tc) * length((1 - prevAlpha) * dir);

}

__device__ float project(float3 src, float3 onto) {
    return dot(src, onto) / dot(onto,onto);
}

__device__ float klookup(uint3 tcDst, int3 _tcSrc) {
    if (_tcSrc.x >= 0 && _tcSrc.x < dims.x &&
        _tcSrc.y >= 0 && _tcSrc.y < dims.y &&
        _tcSrc.z >= 0 && _tcSrc.z < dims.z) {
        
        uint3 tcSrc = make_uint3(_tcSrc.x, _tcSrc.y, _tcSrc.z);

        float3 vecDst = texToVec(tcDst);
        float3 vecSrc = texToVec(tcSrc);

        float rd = rad(tcDst, vecDst, tcSrc, vecSrc);
        float3 kaxis = vecSrc - beam.src;
        float3 lookupVec = (vecDst - vecSrc) * rd;
        float alpha = project(lookupVec, kaxis);
        kaxis = alpha * kaxis;
        uint y = length(kaxis);
        uint x = length(kaxis - lookupVec);
        return (y <= 3 && x <= 7 && alpha >= 0) ? kern[y][x] * tex3D(termaTex, tcSrc.x, tcSrc.y, tcSrc.z) : 0.0f;
        
    }
    return 0.0f;
}

/**
 * Once the TERMA has been calculated the dose deposition kernel
 * describes how the dose is spread throughout the medium. The primary
 * dose can be described as the effect of the initial electron scatter
 * due to the initial fluence. These electrons flow in a forward
 * motion with small angular deviations. The dose deposition is also
 * affected by secondary electron scatter due to scattering of
 * photons. The flow of these electrons is more difficult to predict.
 *
 * Also the emission of new photons due to positron annihilation and
 * bremsstrahlung affect the resulting dose, but these effects are
 * excluded here.
 */
__global__ void doseDeposition(uint offset, float* out) {
    // const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    // const uint idx = blockIdx.x * blockDim.x + blockIdx.y * gridDim.x*blockDim.x + threadIdx.x;
    const uint idx = offset + blockIdx.x * blockDim.x + blockIdx.y * gridDim.x*blockDim.x + threadIdx.x;
    const uint3 tc = idx_to_co(idx, dims);
    const float3 vec = texToVec(tc);
    out[idx] = 0.0f;

    if (VoxelInsideBeamlet(vec, beam.invCone1, beam.invCone2)) {
        // const int halfsize = 4; 
        for (int i = -halfsize; i <= halfsize; ++i) {
            for (int j = -halfsize; j <= halfsize; ++j) {
                for (int k = -halfsize; k <= halfsize; ++k) {
                    out[idx] += klookup(tc, make_int3(tc.x + i,
                                                      tc.y + j,
                                                      tc.z + k));
                }
            }
        }
    }
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
 * param output An array of the voxels interest. Contains 1.0f if the
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
__global__ void rateBeamlet(float* input, uint *output) {
    // __constant__ uint3 dims
    // __constant__ float3 scale
    // __constant__ CudaBeam beam;
    // __constant__ uint2 beamletDims;
    // __constant__ uint2 invBeamletDims;

    uint coordX = blockIdx.x * blockDim.x + threadIdx.x;
    uint coordY = blockIdx.y * blockDim.y + threadIdx.y;
    /**
     * The beam is constructed by 4 vectors setup like this
     *
     * v1----v2
     *  | \  |
     *  |  \ |
     * v4----v3
     *
     * And weighted along the local x axis like this
     *
     * v1--w1-w2--v2
     *  | \       |
     *  |   \     |
     *  |     \   |
     *  |       \ |
     * v4--w4-w3--v3
     *
     * w1 = (1-xStart/dims) * v1 + xStart/dims * v2
     * w2 = (1-xEnd/dims) * v1 + xEnd/dims * v2
     * and the same for w3 and w4.
     */
    
    // Calculate the local beamlet info, vectors are in texture coordinates.
    uint2 coordStart = make_uint2(coordX, coordY);
    uint2 coordEnd = make_uint2(coordStart.x + 1, coordStart.y + 1);

    float3 w1 = (1 - coordStart.x/beamletDims.x) * beam.v1 + coordStart.x/beamletDims.x * beam.v2;
    float3 w2 = (1 - coordEnd.x/beamletDims.x) * beam.v1 + coordEnd.x/beamletDims.x * beam.v2;
    float3 w3 = (1 - coordStart.x/beamletDims.x) * beam.v3 + coordStart.x/beamletDims.x * beam.v4;
    float3 w4 = (1 - coordEnd.x/beamletDims.x) * beam.v3 + coordEnd.x/beamletDims.x * beam.v4;

    float3 v1 = (1 - coordStart.y/beamletDims.y) * w1 + coordStart.y/beamletDims.y * w2;
    float3 v2 = (1 - coordEnd.y/beamletDims.y) * w1 + coordEnd.y/beamletDims.y * w2;
    float3 v3 = (1 - coordStart.y/beamletDims.y) * w3 + coordStart.y/beamletDims.y * w4;
    float3 v4 = (1 - coordEnd.y/beamletDims.y) * w3 + coordEnd.y/beamletDims.y * w4;
}

void RunDoseCalc(float* cuDoseArr, Beam oeBeam, int beamlet_x, int beamlet_y, int kernel) {

    // Copy beam and beamlet info to device
    CudaBeam _beam;
    _beam(oeBeam, scaling);
    cudaMemcpyToSymbol(beam, &_beam, sizeof(CudaBeam));
    CHECK_FOR_CUDA_ERROR();

    beamletDimensions = make_uint2(beamlet_x, beamlet_y);
    cudaMemcpyToSymbol(beamletDims, &beamletDimensions, sizeof(uint2));
    float2 invBeamDims = make_float2(1.0f / beamletDimensions.x, 1.0f / beamletDimensions.y);
    cudaMemcpyToSymbol(invBeamletDims, &invBeamDims, sizeof(float2));
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
    const dim3 beamletGridSize(beamletDimensions.x / beamletBlockSize.x, beamletDimensions.y / beamletBlockSize.y);

    // start timer
    cutResetTimer(timer);
	cutStartTimer(timer);

    switch(kernel){
    case 0:
        radioDepth<<< voxelGridSize, voxelBlockSize >>>(cuDoseArr);
        break;
    case 1:
        voxelsOfInterest<<< voxelGridSize, voxelBlockSize >>>(cuDoseArr);
        break;
    default:
        radioDepth<<< voxelGridSize, voxelBlockSize >>>(cuDoseArr);
        //rateBeamlet<<< beamletGridSize, beamletBlockSize >>>(cuDoseArray, andet array);
    }

	// Report timing
	cudaThreadSynchronize();
	cutStopTimer(timer);  
	double time = cutGetTimerValue( timer ); 
	printf("time: %.4f ms.\n", time );

    CHECK_FOR_CUDA_ERROR();
}

/**
 * CPU entry function for calculating the dose deposition
 * for a given fluence map.
 **/
__host__ void Dose(float** out,                      // result dose map
                   Beam oebeam,                      // beam
                   Beam voi,                         // voxels of interest
                   uchar* fmap,                      // fluence map
                   uint beamlet_x, uint beamlet_y)     // scale
{
    // allocate TERMA array
    float* _terma = NULL;
    cudaMalloc((void**)&_terma, sizeof(float) * size);
    cudaMemset(_terma, 0, sizeof(float) * size);
    CHECK_FOR_CUDA_ERROR();

    // copy constants
    float e = 1.0f; // initial energy is 1.0 (100%)
    cudaMemcpyToSymbol(energy, &e, sizeof(float));
    CHECK_FOR_CUDA_ERROR();

    // Copy beam and beamlet info to device
    CudaBeam _beam;
    _beam(oebeam, scale);
    cudaMemcpyToSymbol(beam, &_beam, sizeof(CudaBeam));
    CHECK_FOR_CUDA_ERROR();

    beamletDimensions = make_uint2(beamlet_x, beamlet_y);
    cudaMemcpyToSymbol(beamletDims, &beamletDimensions, sizeof(uint2));
    float2 invBeamDims = make_float2(1.0f / beamletDimensions.x, 1.0f / beamletDimensions.y);
    cudaMemcpyToSymbol(invBeamletDims, &invBeamDims, sizeof(float2));
    CHECK_FOR_CUDA_ERROR();
    
    // copy fluence map
    uchar* _fmap = NULL;
    cudaMalloc((void**)&_fmap, beamlet_x * beamlet_y * sizeof(uchar));
    cudaMemcpy((void*)_fmap, (void*)fmap, beamlet_x * beamlet_y * sizeof(uchar), cudaMemcpyHostToDevice);
    CHECK_FOR_CUDA_ERROR();

    // calculate TERMA
    
    // We can run the kernels in multiple iterations to prevent GPU
    // timeouts. This also helps when maximum grid size is exceeded for
    // large 3D textures.
    // !!! When compiler option fpeel_loops is enabled there have been
    // some problems calling the cuda kernels in a for-loop. !!!
    const dim3 blockSize(128, 1);
    // const dim3 gridSize(w * h / blockSize.x + 1, d, 1);
    // const dim3 gridSize(w * h * d  / blockSize.x, 1, 1);
    // const dim3 gridSize(itsz / blockSize.x, 1, 1);
    const dim3 gridSize(100, 1, 1);
    const uint iter = size / ( gridSize.x * blockSize.x);
    logger.info << "gridSize.x = " << gridSize.x << logger.end; 
    logger.info << "blockSize.x = " << blockSize.x << logger.end; 
    logger.info << "w*h*d: " << size << logger.end;
    unsigned int offset = 0;
    logger.info << "Run TERMA kernel in " << iter << " iterations..." << logger.end; 
    
    // start timer
    cutResetTimer(timer);
	cutStartTimer(timer);

    for (unsigned int i = 0; i < iter; ++i) {
        logger.info << "TERMA run #" << i  << logger.end; 
        logger.info << "offset = " << offset << logger.end;
        terma<<< gridSize, blockSize >>>(offset, _terma, _fmap);        
        CHECK_FOR_CUDA_ERROR();
        offset += blockSize.x * gridSize.x;
    }

	// Report timing
	cudaThreadSynchronize();
	cutStopTimer(timer);  
	printf("time to calculate terma: %.4f ms.\n", cutGetTimerValue( timer ) );

    // terma<<< gridSize, blockSize >>>(0, _terma, _fmap);

    float* test = new float[size];
    cudaMemcpy((void*)test, (void*)_terma, size * sizeof(float), cudaMemcpyDeviceToHost);
    CHECK_FOR_CUDA_ERROR();

    // bind the terma array to a texture (expensive memory copies can be optimized away...)
    cudaArray* tarr;
    cudaExtent ext = make_cudaExtent(dimensions.x, dimensions.y, dimensions.z);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaMalloc3DArray(&tarr, &channelDesc, ext);
    CHECK_FOR_CUDA_ERROR();
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_cudaPitchedPtr((void*)test, 
                                            ext.width*sizeof(float),
                                            ext.width,
                                            ext.height);
    copyParams.dstArray = tarr;
    copyParams.extent = ext;
    copyParams.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParams);
    CHECK_FOR_CUDA_ERROR();
    cudaBindTextureToArray(termaTex, tarr, channelDesc);
    CHECK_FOR_CUDA_ERROR();

    // set voxels of interest (this is simply a larger cone beam).
    _beam(voi, scale);
    cudaMemcpyToSymbol(beam, &_beam, sizeof(CudaBeam));
    CHECK_FOR_CUDA_ERROR();
    
    int _halfsize = 4;
    cudaMemcpyToSymbol(halfsize, &_halfsize, sizeof(int));
    CHECK_FOR_CUDA_ERROR();
    logger.info << "Kernel half size: " << _halfsize << logger.end; 
   
    // Run the dose deposition kernel
    //logger.info << "Running dose deposition kernel in " << iter << " iterations..." << logger.end; 
    offset = 0;
  
    // start timer
    cutResetTimer(timer);
	cutStartTimer(timer);
    for (unsigned int i = 0; i < iter; ++i) {
        logger.info << "Dose deposition run #" << i << "/" << iter << logger.end;
        //logger.info << "offset = " << offset << logger.end;
        doseDeposition<<< gridSize, blockSize >>>( offset, *out ); 
        CHECK_FOR_CUDA_ERROR();
        offset += blockSize.x * gridSize.x;
    }

	// Report timing
	cudaThreadSynchronize();
	cutStopTimer(timer);  
	printf("time to calculate deposition: %.4f ms.\n", cutGetTimerValue( timer ) );

    /*
    // print some dose values for debugging purposes.
    int s = 5;
    for (uint i = dimensions.x/2-s/2; i < dimensions.x/2 + s/2; ++i) {
        for (uint j = dimensions.y/2 - s/2; j < dimensions.y/2 + s/2; ++j) {
            for (uint k = dimensions.z/2-s/2; k < dimensions.z/2 + s/2; ++k) {
                logger.info << "terma: " << test[i + j*dimensions.x + k*dimensions.x*dimensions.y] << logger.end; 
            }
            
        }
        
    }
    cudaMemcpy((void*)test, (void*)*out, size * sizeof(float), cudaMemcpyDeviceToHost); 
    CHECK_FOR_CUDA_ERROR();
    for (uint i = dimensions.x/2-s/2; i < dimensions.x/2 + s/2; ++i) {
        for (uint j = dimensions.y/2 - s/2; j < dimensions.y/2 + s/2; ++j) {
            for (uint k = dimensions.z/2-s/2; k < dimensions.z/2 + s/2; ++k) {
                logger.info << "depos: " << test[i + j * dimensions.x + k * dimensions.x * dimensions.y] << logger.end; 
            }
            
        }
        
    }
    */
    
    delete[] test;
    cudaFree(_terma);
}
