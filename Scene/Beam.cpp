// Dose Calculation - beam representation in world coordinates.
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Scene/Beam.h>

namespace OpenEngine {
    namespace Scene {

Beam::Beam(Vector<3,float> src, 
           Vector<3,float> p1, 
           Vector<3,float> p2, 
           Vector<3,float> p3, 
           Vector<3,float> p4)
    : src(src)
    , p1(p1)
    , p2(p2)
    , p3(p3)
    , p4(p4)
{}

Beam::~Beam() {}

} // NS Scene
} // NS OpenEngine
