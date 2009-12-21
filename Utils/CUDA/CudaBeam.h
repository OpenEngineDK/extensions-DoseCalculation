#include <Utils/CUDA/Matrix3x3.h>

struct CudaBeam {
    float3 src;
    float3 v1, v2, v3, v4;
    Matrix3x3 invCone1;
    Matrix3x3 invCone2;

    __host__ void operator() (Beam b){
        /**
         * The beam is constructed by 4 vectorss setup like this
         *
         * v1----v2
         *  | \  |
         *  |  \ |
         * v4----v3
         *
         * therefore the cones are given by [v1, v2, v3] and 
         * [v1, v4, v3].
         */

        src.x = b.src[0];
        src.y = b.src[1];
        src.z = b.src[2];

        v1 = make_float3(b.p1 - b.src);
        v2 = make_float3(b.p2 - b.src);
        v3 = make_float3(b.p3 - b.src);
        v4 = make_float3(b.p4 - b.src);

        invCone1(v1, v2, v3);
        invCone1 = invCone1.getInverse();

        invCone2(v1, v4, v3);
        invCone2 = invCone2.getInverse();
    }
};
