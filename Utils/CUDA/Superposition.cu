#include <Meta/CUDA.h>
#include <Utils/CUDA/Superposition.h>
#include <Utils/CUDA/DozeCuda.h>
#include <Utils/CUDA/CudaBeam.h>
#include <Utils/CUDA/Matrix3x3.h>

#include <Utils/CUDA/uint_util.hcu>
#include <Utils/CUDA/float_util.hcu>
#include <stdlib.h>

#include <Logging/Logger.h>

#define PIXEL_UNIT 0.01 // one pixel = 1 cm^3


texture<float, 3, cudaReadModeElementType> intensityTex;
texture<float, 3, cudaReadModeElementType> termaTex;

typedef unsigned char uchar;
typedef unsigned int uint;

__constant__ uint3 dims;      // texture dimensions
__constant__ float3 scale;    // texture scale

__constant__ CudaBeam beam;

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

// utility functions

void SetupDoseCalc(float** cuDoseArr, 
                   int w, int h, int d, // dimensions
                   float sw, float sh, float sd) // scale
{ 
    cudaMalloc((void**)cuDoseArr, sizeof(float)*w*h*d);
    CHECK_FOR_CUDA_ERROR();

    cudaMemset(*cuDoseArr, 0, sizeof(float)*w*h*d);

    printf("SetupDoseCalc done\n");
}


/**
 * return the central position of a voxel.
 */
__device__ float3 texToVec(uint3 tex) {
    return make_float3(tex.x * scale.x + 0.5 * scale.x,
                       tex.y * scale.y + 0.5 * scale.y,
                       tex.z * scale.z + 0.5 * scale.z);
}

/**
 * Determine the voxel coordinate containing the point vec.
 */
__device__ uint3 vecToTex(float3 vec) {
    return make_uint3(vec.x / scale.x, vec.y / scale.y, vec.z / scale.z);
}

__device__ bool VoxelInsideBeamlet(float3 point, Matrix3x3 cone1, Matrix3x3 cone2){
    // __constant__ CudaBeam beam
    float3 translatedPoint = point - beam.src;
    return cone1.mul(translatedPoint) >= 0
        || cone2.mul(translatedPoint) >= 0;
}


// theoretical utility functions

/**
 * Account for any prior changes made to the voxel data
 */
__device__ float hounsfield(uint3 r) {
    return (tex3D(intensityTex, r.x, r.y, r.z) * 2000.0 - 1000.0);
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
    // lets just use 0.135 for mu_water(E=171keV) ...
    const float mu_water = 0.135;
    return (hounsfield(r) / 1000.0) * mu_water + mu_water;
}

/**
 * Get density relative to water.
 */
__device__ float density(uint3 r) {
    // not quite sure how relative density relates to the HU. 
    return (attenuation(r) / 0.135) * PIXEL_UNIT ;
}


/**
 * Summation of the weighted attenuation coefficients from the point r
 * back to the src point.
 */
__device__ float sumAtt(float3 r, uint3 _tc) {
    float3 dir = beam.src - r ; 
    float l = length(dir) * PIXEL_UNIT;
    int3 tc = make_int3(_tc.x, _tc.y, _tc.z);
    
    // delta tc is determined by the sign of the direction.
    const int3 dtc = make_int3(__float2int_rn(dir.x / fabs(dir.x)),
                               __float2int_rn(dir.y / fabs(dir.y)),
                               __float2int_rn(dir.z / fabs(dir.z)));    
                               
    // the initial offsets of the planes closest to the point r in the
    // direction of dir.
    float3 planes = make_float3(scale.x * (tc.x + (dtc.x + 1) / 2),
                                scale.y * (tc.y + (dtc.y + 1) / 2),
                                scale.z * (tc.z + (dtc.z + 1) / 2));
    float prevAlpha = 0.0;
    float sum = 0.0;

    // while we are still inside the voxel boundaries ... 
    while (tc.x >= 0 && tc.x < dims.x &&
           tc.y >= 0 && tc.y < dims.y &&
           tc.z >= 0 && tc.z < dims.z) {

        // Determine the scaling factors. Result is always positive
        // since the signs are the same on both sides of the division.
        // (planes - r) can never be zero since we advance the plane
        // offset away from r.
        float3 alphas = (planes - r) / dir;
        // if dir is zero then the result will be infty or -infty.
        // this is a dirty hack to ensure that we only get positive infty.
        alphas = make_float3(fabs(alphas.x),
                             fabs(alphas.y),
                             fabs(alphas.z));
        float alpha = fmin(alphas.x, alphas.y);
        alpha = fmin(alpha, alphas.z);
        
        sum += attenuation(make_uint3(tc.x, tc.y, tc.z)) * (alpha - prevAlpha) * l; 
        prevAlpha = alpha;

        // Find minimal coordinates. Note that several coordinates
        // could be minimal (equal).
        uint3 min = make_uint3( (alphas.x == alpha) ? 1 : 0,
                                (alphas.y == alpha) ? 1 : 0,
                                (alphas.z == alpha) ? 1 : 0 );
        // uint3 tmp = make_uint3(__float2int_rz(alpha / alphas.x),
        //                        __float2int_rz(alpha / alphas.y),
        //                        __float2int_rz(alpha / alphas.z));
        
        // advance the texture coordinates (termination is based on dot(min, min) != 0)
        tc = make_int3(tc.x + min.x * dtc.x, 
                       tc.y + min.y * dtc.y,
                       tc.z + min.z * dtc.z);
        
        // advance the planes.
        planes = make_float3(planes.x + scale.x * dtc.x * min.x,
                             planes.y + scale.y * dtc.y * min.y,
                             planes.z + scale.z * dtc.z * min.z);
        
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
}


/**
 * Calculate Total Energy Released per unit MAss (TERMA) for each
 * voxel. This is the energy deposited by the incident fluence
 * e.g. the initial collision of photons. TERMA is determined by
 * incident fluence, linear attenuation, and off-axis effects
 * (intensity has a linear relation with the angle to the beam axis
 * and an inverse square relation with the distance to the
 * source. (Jacques et. al. 2009).
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
    uint3 tc = tc1;

    // delta tc is determined by the sign of the direction.
    const int3 dtc = make_int3(__float2int_rn(dir.x / fabs(dir.x)),
                               __float2int_rn(dir.y / fabs(dir.y)),
                               __float2int_rn(dir.z / fabs(dir.z)));    
                               
    // the initial offsets of the planes closest to the point r in the
    // direction of dir.
    float3 planes = make_float3(scale.x * (tc.x + (dtc.x + 1) / 2),
                                scale.y * (tc.y + (dtc.y + 1) / 2),
                                scale.z * (tc.z + (dtc.z + 1) / 2));
    float prevAlpha = 0.0;
    float sum = 0.0;

    // while we are still inside the voxel boundaries ... 
    while (tc.x != tc2.x &&
           tc.y != tc2.y &&
           tc.z != tc2.z) {

        // Determine the scaling factors. Result is always positive
        // since the signs are the same on both sides of the division.
        // (planes - r) can never be zero since we advance the plane
        // offset away from r.
        float3 alphas = ( planes - vec1 ) / dir;
        // if dir is zero then the result will be infty or -infty.
        // this is a dirty hack to ensure that we only get positive infty.
        alphas = make_float3(fabs(alphas.x),
                             fabs(alphas.y),
                             fabs(alphas.z));
        float alpha = fmin(alphas.x, alphas.y);
        alpha = fmin(alpha, alphas.z);
        
        sum += density(tc) * length((alpha - prevAlpha) *  dir); 
        prevAlpha = alpha;

        uint3 min = make_uint3( (alphas.x == alpha) ? 1 : 0,
                                (alphas.y == alpha) ? 1 : 0,
                                (alphas.z == alpha) ? 1 : 0 );
        tc = make_uint3(tc.x + min.x * dtc.x, 
                        tc.y + min.y * dtc.y,
                        tc.z + min.z * dtc.z);
        
        planes = make_float3(planes.x + scale.x * dtc.x * min.x,
                             planes.y + scale.y * dtc.y * min.y,
                             planes.z + scale.z * dtc.z * min.z);
        
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
        // const int halfsize = 7; 
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
    // for (int i = -1; i <= 1; ++i) {
    //     for (int j = -1; j <= 1; ++j) {
    //         for (int k = -1; k <= 1; ++k) {
    //             out[idx] += klookup(tc, make_int3(tc.x + i * halfsize,
    //                                               tc.y + j * halfsize,
    //                                               tc.z + k * halfsize));
    //         }
    //     }
    // }
    // out[idx] = klookup(tc, make_int3(tc.x + halfsize,
    //                                  tc.y + halfsize,
    //                                  tc.z + halfsize));
    // out[idx] = 1.0;
}

/**
 * Determine the fluence map of a beam. 
 * Which beamlets should be closed by the collimators.
 *
 * Strategy: Investigate the dose deposition of each beamlet in
 * critical regions.
 */
__global__ void fluenceMap(uchar* fmap) {

}

/**
 * CPU entry function for initiating the fluence map calculations
 * 
 **/
__host__ void FluenceMap(uchar** fmap,
                            uint beamlets_x, uint beamlets_y, // # beamlets
                            int w, int h, int d, // dimensions
                            float sw, float sh, float sd) // scale
{
    // initialize fmap (fluencemap) to all true values on GPU
    
    // setup bounding volumes (target and critical regions).

    // Setup texture
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    intensityTex.normalized = false;
    intensityTex.filterMode = cudaFilterModeLinear;
    intensityTex.addressMode[0] = cudaAddressModeClamp;
    intensityTex.addressMode[1] = cudaAddressModeClamp;
    intensityTex.addressMode[2] = cudaAddressModeClamp;
    cudaBindTextureToArray(intensityTex, GetVolumeArray(), channelDesc);
    CHECK_FOR_CUDA_ERROR();

    // run fluence map kernel

    // copy resulting fluence map back to fmap
}

/**
 * CPU entry function for calculating the dose deposition
 * for a given fluence map.
 **/
__host__ void Dose(float** out,                      // result dose map
                   Beam oebeam,                      // beam
                   Beam voi,                         // voxels of interest
                   uchar* fmap,                      // fluence map
                   uint beamlets_x, uint beamlets_y, // # beamlets
                   int w, int h, int d,              // dimensions
                   float sw, float sh, float sd)     // scale
{
    // Setup texture
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    intensityTex.normalized = false;
    intensityTex.filterMode = cudaFilterModeLinear;
    intensityTex.addressMode[0] = cudaAddressModeClamp;
    intensityTex.addressMode[1] = cudaAddressModeClamp;
    intensityTex.addressMode[2] = cudaAddressModeClamp;
    cudaBindTextureToArray(intensityTex, GetVolumeArray(), channelDesc);
    CHECK_FOR_CUDA_ERROR();

    // allocate TERMA array
    float* _terma = NULL;
    cudaMalloc((void**)&_terma, sizeof(float)*w*h*d);
    cudaMemset(_terma, 0, sizeof(float)*w*h*d);
    CHECK_FOR_CUDA_ERROR();

    // copy constants
    uint3 dim = make_uint3(w, h, d);
    cudaMemcpyToSymbol(dims, &dim, sizeof(uint3));
    CHECK_FOR_CUDA_ERROR();
    
    float3 scl = make_float3(sw, sh, sd);
    cudaMemcpyToSymbol(scale, &scl, sizeof(float3));
    CHECK_FOR_CUDA_ERROR();

    float e = 1.0f; // initial energy is 1.0 (100%)
    cudaMemcpyToSymbol(energy, &e, sizeof(float));
    CHECK_FOR_CUDA_ERROR();

    CudaBeam _beam;
    _beam(oebeam, scl);
    cudaMemcpyToSymbol(beam, &_beam, sizeof(CudaBeam));
    CHECK_FOR_CUDA_ERROR();
    
    // copy fluence map
    uchar* _fmap = NULL;
    cudaMalloc((void**)&_fmap, beamlets_x * beamlets_y * sizeof(uchar));
    cudaMemcpy((void*)_fmap, (void*)fmap, beamlets_x * beamlets_y * sizeof(uchar), cudaMemcpyHostToDevice);
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
    const uint iter = w*h*d / ( gridSize.x * blockSize.x);
    logger.info << "gridSize.x = " << gridSize.x << logger.end; 
    logger.info << "blockSize.x = " << blockSize.x << logger.end; 
    logger.info << "w*h*d: " << w*h*d << logger.end;
    unsigned int offset = 0;
    logger.info << "Run TERMA kernel in " << iter << " iterations..." << logger.end; 
    for (unsigned int i = 0; i < iter; ++i) {
        logger.info << "TERMA run #" << i  << logger.end; 
        logger.info << "offset = " << offset << logger.end;
        terma<<< gridSize, blockSize >>>(offset, _terma, _fmap);        
        CHECK_FOR_CUDA_ERROR();
        offset += blockSize.x * gridSize.x;
    }
    // terma<<< gridSize, blockSize >>>(0, _terma, _fmap);

    float* test = new float[w*h*d];
    cudaMemcpy((void*)test, (void*)_terma, w * h * d * sizeof(float), cudaMemcpyDeviceToHost);
    CHECK_FOR_CUDA_ERROR();

    // bind the terma array to a texture (expensive memory copies can be optimized away...)
    cudaArray* tarr;
    cudaExtent ext = make_cudaExtent(w,h,d);
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
    _beam(voi, scl);
    cudaMemcpyToSymbol(beam, &_beam, sizeof(CudaBeam));
    CHECK_FOR_CUDA_ERROR();
    
    int _halfsize = 4;
    cudaMemcpyToSymbol(halfsize, &_halfsize, sizeof(int));
    CHECK_FOR_CUDA_ERROR();
    logger.info << "Kernel half size: " << _halfsize << logger.end; 
   
    // Run the dose deposition kernel
    logger.info << "Running dose deposition kernel in " << iter << " iterations..." << logger.end; 
    offset = 0;
  
    for (unsigned int i = 0; i < iter; ++i) {
        logger.info << "Dose deposition run #" << i << logger.end;
        logger.info << "offset = " << offset << logger.end;
        doseDeposition<<< gridSize, blockSize >>>( offset, *out ); 
        CHECK_FOR_CUDA_ERROR();
        offset += blockSize.x * gridSize.x;
    }
    // print some dose values for debugging purposes.
    int s = 5;
    for (int i = w/2-s/2; i < w/2 + s/2; ++i) {
        for (int j = h/2 - s/2; j < h/2 + s/2; ++j) {
            for (int k = d/2-s/2; k < d/2 + s/2; ++k) {
                logger.info << "terma: " << test[i + j*w + k*w*h] << logger.end; 
            }
            
        }
        
    }
    cudaMemcpy((void*)test, (void*)*out, w * h * d * sizeof(float), cudaMemcpyDeviceToHost); 
    CHECK_FOR_CUDA_ERROR();
    for (int i = w/2-s/2; i < w/2 + s/2; ++i) {
        for (int j = h/2 - s/2; j < h/2 + s/2; ++j) {
            for (int k = d/2-s/2; k < d/2 + s/2; ++k) {
                logger.info << "depos: " << test[i + j*w + k*w*h] << logger.end; 
            }
            
        }
        
    }
    
    delete[] test;
    cudaFree(_terma);
}
