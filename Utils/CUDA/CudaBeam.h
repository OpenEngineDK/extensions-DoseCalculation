#ifndef _DOSECALC_CUDA_BEAM_H_
#define _DOSECALC_CUDA_BEAM_H_
#include <Utils/CUDA/Matrix3x3.h>
#include <Utils/CUDA/float_util.hcu>
#include <Scene/Beam.h>

using OpenEngine::Scene::Beam;

struct CudaBeam {
    // World coords.
    float3 src;
    float3 axis;
    float3 v1, v2, v3, v4;
    Matrix3x3 invCone1;
    Matrix3x3 invCone2;

    // World coords scaled into texture space. Precalculated prior to
    // launching the kernel to avoid having all the threads compute
    // the same thing.
    float3 srcTex;
    float3 v1Tex, v2Tex, v3Tex, v4Tex;

    __host__ void operator() (Beam b, float3 scale){
        /**
         * The beam is constructed by 4 vectors setup like this
         *
         * v1----v2
         *  | \  |
         *  |  \ |
         * v4----v3
         *
         * therefore the cones are given by [v1, v2, v3] and 
         * [v1, v4, v3].
         */

        src = make_float3(b.src);

        v1 = make_float3(b.p1 - b.src);
        v2 = make_float3(b.p2 - b.src);
        v3 = make_float3(b.p3 - b.src);
        v4 = make_float3(b.p4 - b.src);

        invCone1(v1, v2, v3);
        invCone1 = invCone1.getInverse();

        invCone2(v1, v4, v3);
        invCone2 = invCone2.getInverse();

        srcTex = src / scale;
        v1Tex = v1 / scale;
        v2Tex = v2 / scale;
        v3Tex = v3 / scale;
        v4Tex = v4 / scale;

        axis = normalize(v1 + (v2 - v1) * 0.5 + v2 + (v3 - v2) * 0.5 - src);
    }
};
#endif
