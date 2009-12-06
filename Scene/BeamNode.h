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

namespace OpenEngine {
    namespace Scene {

class BeamNode : public ISceneNode {
    OE_SCENE_NODE(BeamNode, ISceneNode)
public:
    // beam surface
    int beamlets_x, beamlets_y;
public:
    BeamNode(int beamlets_x, int beamlets_y);
    virtual ~BeamNode();

private:
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version) {
        // serialize base class information
        ar & boost::serialization::base_object<ISceneNode>(*this);
    }
};
        
} // NS Scene
} // NS OpenEngine

#endif //_DOSE_CALCULATION_RAY_GUN_NODE_H_
