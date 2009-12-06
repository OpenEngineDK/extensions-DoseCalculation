// Dose calculation rendering view.
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _DOSE_CALCULATION_SELECTION_RENDERER_H_
#define _DOSE_CALCULATION_SELECTION_RENDERER_H_

#include <Utils/GLSceneSelection.h>

namespace OpenEngine {
    namespace Scene {
        class BeamNode;
    }
namespace Renderers {
namespace OpenGL {

using Utils::SelectionRenderer;
using Scene::BeamNode;

class DoseCalcSelectionRenderer : public SelectionRenderer {
public:
    void VisitBeamNode(BeamNode* node);
};

}
}
}

#endif
