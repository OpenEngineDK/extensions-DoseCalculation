// Dose calculation rendering view using ray casting.
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Renderers/OpenGL/RayCastRenderingView.h>

#include <Scene/DoseCalcNode.h>
#include <Resources/ITexture3DResource.h>
#include <Logging/Logger.h>

using namespace OpenEngine::Scene;
using namespace OpenEngine::Resources;

namespace OpenEngine {
    namespace Renderers {
        namespace OpenGL {

            RayCastRenderingView::RayCastRenderingView(Viewport& viewport) : 
                IRenderingView(viewport), 
                RenderingView(viewport) {
                
            }
            
            void RayCastRenderingView::VisitDoseCalcNode(DoseCalcNode* node){
            
            }
        }
    }
}
