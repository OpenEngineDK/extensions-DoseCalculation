#include <Math/Vector.h>

using namespace OpenEngine::Math;

struct Matrix3x3 {
    float3 e[3];

    __host__ void operator() (Vector<3, float> v0, Vector<3, float> v1, Vector<3, float> v2){
        e[0].x = v0[0];
        e[0].y = v1[0];
        e[0].z = v2[0];
        e[1].x = v0[1];
        e[1].y = v1[1];
        e[1].z = v2[1];
        e[2].x = v0[2];
        e[2].y = v1[2];
        e[2].z = v2[2];
    }

    __host__ __device__ void operator() (float3 v0, float3 v1, float3 v2){
        e[0].x = v0.x;
        e[0].y = v1.x;
        e[0].z = v2.x;
        e[1].x = v0.y;
        e[1].y = v1.y;
        e[1].z = v2.y;
        e[2].x = v0.z;
        e[2].y = v1.z;
        e[2].z = v2.z;
    }

    __host__ __device__ float3 mul(float3 m){
        return make_float3(dot(m, e[0]),
                           dot(m, e[1]),
                           dot(m, e[2]));
    }

    __host__ __device__ Matrix3x3 getInverse(){
        Matrix3x3 res;

        float e0112 = e[0].y * e[1].z;
        float e0122 = e[0].y * e[2].z;
        float e0211 = e[0].z * e[1].y;
        float e0221 = e[0].z * e[2].y;
        float e1021 = e[1].x * e[2].y;
        float e1022 = e[1].x * e[2].z;
        float e1122 = e[1].y * e[2].z;
        float e1221 = e[1].z * e[2].y;
        float e1120 = e[1].y * e[2].x;
        float e1220 = e[1].z * e[2].x;

        float determinant = e[0].x * (e1122 - e1221) - e[0].y * (e1022 - e1220) + e[0].z * (e1021 - e1120);
        float invDet = 1.0f / determinant;
        
        res.e[0].x = (e1122 - e1221) * invDet;
        res.e[0].y = (e0221 - e0122) * invDet;
        res.e[0].z = (e0112 - e0211) * invDet;

        res.e[1].x = (e1220 - e1022) * invDet;
        res.e[1].y = (e[0].x * e[2].z - e[0].z * e[2].x) * invDet;
        res.e[1].z = (e[0].z * e[1].x - e[0].x * e[1].z) * invDet;

        res.e[2].x = (e1021 - e1120) * invDet;
        res.e[2].y = (e[0].y * e[2].x - e[0].x * e[2].y) * invDet;
        res.e[2].z = (e[0].x * e[1].y - e[0].y * e[1].x) * invDet;

        return res;
    }

    void print(){
        printf("[[%f, %f, %f]\n", e[0].x, e[0].y, e[0].z);
        printf("[%f, %f, %f]\n", e[1].x, e[1].y, e[1].z);
        printf("[%f, %f, %f]]\n", e[2].x, e[2].y, e[2].z);
    }
};
