// Dose Calculation - beam node.
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Scene/BeamNode.h>
#include <Scene/SearchTool.h>
#include <Math/Quaternion.h>
#include <Scene/TransformationNode.h>

namespace OpenEngine {
    namespace Scene {

using namespace Math;

BeamNode::BeamNode(int beamlets_x, int beamlets_y)
    : beamlets_x(beamlets_x)
    , beamlets_y(beamlets_y)
{}

BeamNode::~BeamNode() {}

Beam BeamNode::GetBeam(float voiScale) {
    SearchTool st;
    Vector<3,float> pos;
    Quaternion<float> rot;
    float offs = voiScale * 0.5;
    Vector<3,float> src(0,1,0), p1(offs,0,offs), p2(offs,0,-offs), p3(-offs,0,-offs), p4(-offs,0,offs);
    TransformationNode* t = st.AncestorTransformationNode(this);
    if (t) {
        Vector<3,float> scl = t->GetScale();
        src = ScaleVec(scl, src);
        p1 = ScaleVec(scl, p1);
        p2 = ScaleVec(scl, p2);
        p3 = ScaleVec(scl, p3);
        p4 = ScaleVec(scl, p4);

        t->GetAccumulatedTransformations(&pos, &rot);
        src = rot.RotateVector(src) + pos;
        p1  = rot.RotateVector(p1)  + pos;
        p2  = rot.RotateVector(p2)  + pos;
        p3  = rot.RotateVector(p3)  + pos;
        p4  = rot.RotateVector(p4)  + pos;
        
        
    }
    return Beam(src, p1, p2, p3, p4);
}

Vector<3,float> BeamNode::ScaleVec(Vector<3,float> vec, Vector<3,float> scl) {
    return Vector<3,float>(vec[0]*scl[0], vec[1]*scl[1], vec[2]*scl[2]); 
}

} // NS Scene
} // NS OpenEngine
