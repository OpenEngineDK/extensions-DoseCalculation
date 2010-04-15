// Dose Calculation - beam node.
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _DOSE_CALCULATION_BEAM_NODE_H_
#define _DOSE_CALCULATION_BEAM_NODE_H_

#include <Scene/ISceneNode.h>
#include <Renderers/IRenderer.h>
#include <Scene/Beam.h>

namespace OpenEngine {
    namespace Scene {

class BeamNode : public ISceneNode {
    OE_SCENE_NODE(BeamNode, ISceneNode)
public:
    // beam surface
    int beamlets_x, beamlets_y;
public:
    BeamNode() : beamlets_x(0), beamlets_y(0) { }
    BeamNode(int beamlets_x, int beamlets_y);
    virtual ~BeamNode();

    Beam GetBeam(float voiScale);

private:
    Vector<3,float> ScaleVec(Vector<3,float> vec, Vector<3,float> scl);

};
        
} // NS Scene
} // NS OpenEngine

#endif //_DOSE_CALCULATION_BEAM_NODE_H_
