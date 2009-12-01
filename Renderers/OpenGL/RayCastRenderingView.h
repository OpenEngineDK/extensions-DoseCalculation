// Dose calculation rendering view using ray casting.
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _DOSE_CALC_RAY_CAST_RENDERING_VIEW_H_
#define _DOSE_CALC_RAY_CAST_RENDERING_VIEW_H_

#include <Renderers/OpenGL/RenderingView.h>

namespace OpenEngine {
namespace Renderers {
namespace OpenGL {

using namespace OpenEngine::Renderers;
    
    class RayCastRenderingView : public RenderingView {
    public:
        RayCastRenderingView(Viewport& viewport);

        void VisitDoseCalcNode(DoseCalcNode* node);
    };

}
}
}

#endif
