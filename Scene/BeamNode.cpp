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

Beam BeamNode::GetBeam() {
    SearchTool st;
    Vector<3,float> pos;
    Quaternion<float> rot;
    Vector<3,float> src(0,1,0), p1(0.5,0,0.5), p2(0.5,0,-0.5), p3(-0.5,0,-0.5), p4(-0.5,0,0.5);
    TransformationNode* t = st.AncestorTransformationNode(this);
    if (t) {
        t->GetAccumulatedTransformations(&pos, &rot);
        src = rot.RotateVector(src) + pos;
        p1  = rot.RotateVector(p1)  + pos;
        p2  = rot.RotateVector(p2)  + pos;
        p3  = rot.RotateVector(p3)  + pos;
        p4  = rot.RotateVector(p4)  + pos;
    }
    return Beam(src, p1, p2, p3, p4);
}

} // NS Scene
} // NS OpenEngine
