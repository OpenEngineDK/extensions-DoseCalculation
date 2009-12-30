#include "RayCaster.h"
#include <Utils/CUDA/DozeCuda.h>
#include <Utils/CUDA/uint_util.hcu>
#include <Meta/CUDA.h>


struct Matrix4x4 {
    float4 e[4]; // rows
    
    // i == row, j == col
    __device__ float get(uint i, uint j) {
        switch (j) {
        case 0: 
            return e[i].x;
        case 1: 
            return e[i].y;       
        case 2: 
            return e[i].z;
        case 3: 
            return e[i].w;
        }
        return 0.0f;
    }

    __device__ float4 mul(float4 v) {
        float4 r;
        
        r.x = dot(v,e[0]);
        r.y = dot(v,e[1]);
        r.z = dot(v,e[2]);
        r.w = dot(v,e[3]);

        return r;
    }
};

struct Ray {
	float3 origin;
	float3 direction;
};


__constant__ Matrix4x4 c_invViewMatrix;
__constant__ float3 scale;
__constant__ float3 boxMin;
__constant__ float3 boxMax;

texture<float, 3, cudaReadModeElementType> tex;

uint3 dimensions1;
float3 scale1;

void SetupRayCaster(int pbo,  const float* data,
                    int w, int h, int d,
                    float sw, float sh, float sd ) {
    
    cudaGLRegisterBufferObject(pbo);
    CHECK_FOR_CUDA_ERROR();

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    
    
    tex.normalized = false;
    tex.filterMode = cudaFilterModeLinear;
    tex.addressMode[0] = cudaAddressModeClamp;
    tex.addressMode[1] = cudaAddressModeClamp;
    tex.addressMode[2] = cudaAddressModeClamp;

    cudaBindTextureToArray(tex, GetVolumeArray(), channelDesc);
    CHECK_FOR_CUDA_ERROR();

    dimensions1 = make_uint3(w, h, d);
    scale1 = make_float3(sw, sh, sd);
    cudaMemcpyToSymbol(scale, &scale1, sizeof(float3));
    cudaMemcpyToSymbol(boxMin, &make_float3(0.0f), sizeof(float3));
    cudaMemcpyToSymbol(boxMax, &make_float3( dimensions1.x * scale1.x, 
                                             dimensions1.y * scale1.y, 
                                             dimensions1.z * scale1.z), sizeof(float3));
}

__device__ uint rgbaFloatToInt(float4 rgba)
{
    rgba.x = __saturatef(rgba.x);   // clamp to [0.0, 1.0]
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    return (uint(rgba.w*255)<<24) | (uint(rgba.z*255)<<16) | (uint(rgba.y*255)<<8) | uint(rgba.x*255);
}

// intersect ray with a box
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm

__device__
int intersectBox(Ray r, float *tnear, float *tfar)
{
    //__constant__ float3 scale;
    //__constant__ float3 boxMin;
    //__constant__ float3 boxMax;

    // compute intersection of ray with all six bbox planes
    float3 invR = make_float3(1.0f) / r.direction;
    float3 tbot = invR * (boxMin - r.origin);
    float3 ttop = invR * (boxMax - r.origin);

    // re-order intersections to find smallest and largest on each axis
    float3 tmin = fminf(ttop, tbot);
    float3 tmax = fmaxf(ttop, tbot);

    // find the largest tmin and the smallest tmax
    float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), tmin.z);
    float smallest_tmax = fminf(fminf(tmax.x, tmax.y), tmax.z);

	*tnear = largest_tmin;
	*tfar = smallest_tmax;

	return smallest_tmax > largest_tmin;
}


__device__ Ray RayForPoint(uint u, uint v, uint width, uint height,float pm00, float pm11) {
    float x = (u / float(width)) * 2.0f-1.0f ;
    float y = (v / float(height)) *2.0f-1.0f;

    float4 projPoint = make_float4(x,y,-1,0);
    projPoint.x = x / pm00;
    projPoint.y = y / pm11;
    
    float4 rDir4 = c_invViewMatrix.mul(projPoint); // c_invViewMatrix is transposed!

    float3 rDir = make_float3(rDir4);
    
    
    Ray rr;
    rr.origin.x = c_invViewMatrix.get(0,3);
    rr.origin.y = c_invViewMatrix.get(1,3);
    rr.origin.z = c_invViewMatrix.get(2,3);

    rr.direction = normalize(rDir);

    return rr;
}

__device__ float4 colorblend(float4 col, float i) {
    float4 icol = make_float4(i, 0.0, col.x, 1.0);
    // float4 icol = make_float4(i, i, i, 1.0);
    return icol;//col + 0.5 * icol; 
}

__global__ void rayCaster(uint *d_output, float* d_intense, uint imageW, uint imageH,
                          float minIt, float maxIt,
                          float transferOffset, float transferScale,
                          float pm00, float pm11,
                          uint3 dims) {
    //__constant__ float3 scale;
    //__constant__ float3 boxMin;
    //__constant__ float3 boxMax;

    float tStep = 1.0f;
    
    float4 col = make_float4(0.0f);
    
    uint x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    Ray r = RayForPoint(x,y,imageW,imageH,pm00,pm11);

    // We got the ray now, lets intersect it with the box..
    float tnear, tfar;
	int hit = intersectBox(r, &tnear, &tfar);

    r.origin = r.origin / scale;
    r.direction = r.direction / scale;

    float3 pos;
    float sample;
    bool hitVoxel = false;
    if (hit) {
        for (float t=tnear+tStep;t<tfar;t+=tStep) {                        
            // descale it
            pos = r.origin + r.direction * t;

            sample = tex3D(tex, pos.x, pos.y, pos.z);
            if (sample > minIt && sample <= minIt+maxIt) {
                hitVoxel = true;
                break;
            }
        }
    }

    if (hitVoxel){
        
        col.x = sample;
        
        uint3 posi = make_uint3(floor(pos));
        int idx = co_to_idx(posi, dims);
        
        if (idx < dims.x * dims.y * dims.z) {
            col.y = 0.2 * d_intense[idx];
        }
    }

    uint i = __umul24(y, imageW) + x;
    d_output[i] = rgbaFloatToInt(col);    
    
    // ----- slice view ray caster code ...   -----
    // float tStep = 1.0f;
    
    // float4 col = make_float4(0.0f);
    
    // uint x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    // uint y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    // Ray r = RayForPoint(x,y,imageW,imageH,pm00,pm11);
    // // optimize by making this constant
    // Ray rforw = RayForPoint(imageW*0.5, imageH*0.5,imageW,imageH,pm00,pm11);

    // // We got the ray now, lets intersect it with the box..
    // float tnear, tfar;
	// int hit = intersectBox(r, &tnear, &tfar);

    // float dist = 800.0 * minIt / ( r.direction.x * rforw.direction.x 
    //                        + r.direction.y * rforw.direction.y 
    //                        + r.direction.z * rforw.direction.z);

    // r.origin = r.origin / scale;
    // r.direction = r.direction / scale;

    // float3 spos = r.origin + r.direction * dist;  
    // if (hit && spos.x >= 0 && spos.x < dims.x &&
    //     spos.y >= 0 && spos.y < dims.y && 
    //     spos.z >= 0 && spos.z < dims.z) {
    //     float sample = tex3D(tex, spos.x, spos.y, spos.z);
    //     col.x = col.y = col.z = sample;
    //     uint3 posi = make_uint3(floor(spos));
    //     int idx = co_to_idx(posi, dims);
    //     col = colorblend(col, d_intense[idx]*maxIt);
    //     // col.y += d_intense[idx];
    // }
    // uint i = __umul24(y, imageW) + x;
    // d_output[i] = rgbaFloatToInt(col);    
}

void RenderToPBO(int pbo, float* cuDoseArr, int width, int height, float* invMat, float pm00, float pm11, float minIt, float maxIt) {
    cudaMemcpyToSymbol(c_invViewMatrix, invMat, sizeof(float4)*4);
    CHECK_FOR_CUDA_ERROR();

    uint* p;
    cudaGLMapBufferObject((void**)&p,pbo);
    CHECK_FOR_CUDA_ERROR();
    
    const dim3 blockSize(16, 16, 1);
    const dim3 gridSize(width / blockSize.x, height / blockSize.y);

    //float3 po = make_float3(100,100,30);
    //uint3 poi = make_uint3(po);
    //int idx = co_to_idx(poi,dimensions1);
    //printf("[%d] %d,%d,%d\n",idx,poi.x,poi.y,poi.z);
    //printf(" %d,%d,%d\n",dimensions1.x,dimensions1.y,dimensions1.z);
    
    //printf("cast: %d,%d,%d\n",dimensions1.x,dimensions1.y,dimensions1.z);
    rayCaster<<<gridSize, blockSize>>>(p,cuDoseArr,width,height,
                                       minIt,maxIt,1,1,pm00,pm11,dimensions1);
    CHECK_FOR_CUDA_ERROR();

    cudaGLUnmapBufferObject(pbo);

    CHECK_FOR_CUDA_ERROR();

}
