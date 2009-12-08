// Dose Calculation - beam node.
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _DOSE_CALCULATION_BEAM_H_
#define _DOSE_CALCULATION_BEAM_H_

#include <Math/Vector.h>

namespace OpenEngine {
    namespace Scene {

using Math::Vector;

class Beam {
public:
    Vector<3,float> src, p1, p2, p3, p4;
    Beam(Vector<3,float> src, 
         Vector<3,float> p1, 
         Vector<3,float> p2, 
         Vector<3,float> p3, 
         Vector<3,float> p4);
    virtual ~Beam();
};
        
} // NS Scene
} // NS OpenEngine

#endif //_DOSE_CALCULATION_BEAM_H_
