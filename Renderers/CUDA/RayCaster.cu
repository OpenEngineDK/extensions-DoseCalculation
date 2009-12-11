#include "RayCaster.h"
#include <Utils/CUDA/DozeCuda.h>
#include <Utils/CUDA/uint_util.hcu>
#include <Meta/CUDA.h>

typedef unsigned char uchar;

// typedef struct {
//     float4 m[3];
// } float3x4;

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

texture<float, 3, cudaReadModeElementType> tex;

void SetupRayCaster(int pbo,  const float* data, int w, int h, int d) {
    
    cudaGLRegisterBufferObject(pbo);
    CHECK_FOR_CUDA_ERROR();

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    
    
    tex.normalized = true;
    tex.filterMode = cudaFilterModeLinear;
    tex.addressMode[0] = cudaAddressModeClamp;
    tex.addressMode[1] = cudaAddressModeClamp;
    tex.addressMode[2] = cudaAddressModeClamp;

    cudaBindTextureToArray(tex, GetVolumeArray(), channelDesc);
    CHECK_FOR_CUDA_ERROR();
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
int intersectBox(Ray r, float3 boxmin, float3 boxmax, float *tnear, float *tfar)
{
    // compute intersection of ray with all six bbox planes
    float3 invR = make_float3(1.0f) / r.direction;
    float3 tbot = invR * (boxmin - r.origin);
    float3 ttop = invR * (boxmax - r.origin);

    // re-order intersections to find smallest and largest on each axis
    float3 tmin = fminf(ttop, tbot);
    float3 tmax = fmaxf(ttop, tbot);

    // find the largest tmin and the smallest tmax
    float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
    float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

	*tnear = largest_tmin;
	*tfar = smallest_tmax;

	return smallest_tmax > largest_tmin;
}


__device__ Ray RayForPoint(uint u, uint v, uint width, uint height,float pm00, float pm11) {
    //float x = ((2*u - width) / float(width));
    //float y = ((2*v - height) / float(height));


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

__global__ void rayCaster(uint *d_output, float* d_intense, uint imageW, uint imageH,
                          float density, float brightness,
                          float transferOffset, float transferScale,
                          float pm00, float pm11,
                          float3 dd) {
    int maxD = 500;
    float tStep = 1.0f;
    
    float4 col = make_float4(0.0f);
    
    float3 boxMin = make_float3(0.0f);
    float3 boxMax = make_float3(dd.x, dd.y, dd.z);


    uint x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    Ray r = RayForPoint(x,y,imageW,imageH,pm00,pm11);
    // We got the ray now, lets intersect it with the box..
    
    float tnear, tfar;
	int hit = intersectBox(r, boxMin, boxMax, &tnear, &tfar);
        
    float3 inversedd = make_float3(1.0f, 1.0f, 1.0f) / dd;

    if (hit) {
        if (tnear < 0.0f) tnear = 0.0f;     // clamp to near plane

        float t = tnear;
        
        for (int i=0;i<maxD;i++) {
            float3 pos = r.origin + r.direction*t;
            // descale it
            pos = pos * inversedd;

            if (pos.x < 0 ||
                pos.y < 0 || 
                pos.z < 0 ||
                pos.x > dd.x ||
                pos.y > dd.y ||
                pos.z > dd.z)
                break;
            
            uint3 posi = make_uint3(pos);
            uint3 ddi = make_uint3(dd);

            int idx = co_to_idx(posi, ddi);

            float inte = d_intense[idx];

            float sample = tex3D(tex, pos.x, pos.y, pos.z);
            if (sample > 0.8f) {
                col = make_float4(sample);
                col.x = sample;
                break;
            }

            t += tStep;
        }
    }



    
    

    // Insert directly in loop instead of break. And can we really
    // calculate it for coords outside the screen? If so then there
    // should probably be a fix.
    if ((x < imageW) && (y < imageH)) {
        // write output color
        uint i = __umul24(y, imageW) + x;
        d_output[i] = rgbaFloatToInt(col);
    }

    
}

void RenderToPBO(int pbo, int pbo2, int width, int height, float* invMat, float pm00, float pm11,float dx, float dy, float dz) {
    cudaMemcpyToSymbol(c_invViewMatrix, invMat, sizeof(float4)*4);
    CHECK_FOR_CUDA_ERROR();

 
    uint* p;
    cudaGLMapBufferObject((void**)&p,pbo);
    CHECK_FOR_CUDA_ERROR();

    float* p2;
    cudaGLMapBufferObject((void**)&p2,pbo2);
    CHECK_FOR_CUDA_ERROR();

    
    const dim3 blockSize(16, 16, 1);
    const dim3 gridSize(width / blockSize.x, height / blockSize.y);

    //
    
    rayCaster<<<gridSize, blockSize>>>(p,p2,width,height,1,1,1,1,pm00,pm11,make_float3(dx,dy,dz));

    CHECK_FOR_CUDA_ERROR();

    cudaGLUnmapBufferObject(pbo);
    cudaGLUnmapBufferObject(pbo2);
    CHECK_FOR_CUDA_ERROR();

}
