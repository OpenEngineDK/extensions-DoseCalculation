// Dose Calculation - beam node.
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Scene/BeamNode.h>

namespace OpenEngine {
    namespace Scene {

BeamNode::BeamNode(int beamlets_x, int beamlets_y)
    : beamlets_x(beamlets_x)
    , beamlets_y(beamlets_y)
{}

BeamNode::~BeamNode() {}

} // NS Scene
} // NS OpenEngine
