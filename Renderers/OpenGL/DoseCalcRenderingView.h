// Dose calculation rendering view.
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _DOSE_CALC_RENDERING_VIEW_H_
#define _DOSE_CALC_RENDERING_VIEW_H_

#include <Renderers/OpenGL/RenderingView.h>

namespace OpenEngine {
    namespace Scene {
        class BeamNode;
    }
namespace Renderers {
namespace OpenGL {

using Renderers::OpenGL::RenderingView;
using Scene::BeamNode;

class DoseCalcRenderingView : public RenderingView {
public:
    DoseCalcRenderingView(Viewport& viewport);
    
    void VisitDoseCalcNode(DoseCalcNode* node);
    
    void VisitBeamNode(BeamNode* node);
};

}
}
}

#endif
