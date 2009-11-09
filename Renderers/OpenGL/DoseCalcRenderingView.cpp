// Dose calculation renderingview via OpenGL rendering view.
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Renderers/OpenGL/DoseCalcRenderingView.h>

#include <Scene/DoseCalcNode.h>

using namespace OpenEngine::Scene;

namespace OpenEngine {
    namespace Renderers {
        namespace OpenGL {

            DoseCalcRenderingView::DoseCalcRenderingView(Viewport& viewport) : 
                IRenderingView(viewport), 
                RenderingView(viewport) {
            }

            void DoseCalcRenderingView::VisitDoseCalcNode(DoseCalcNode* node){
                glEnable(GL_TEXTURE_3D);
                glBindTexture(GL_TEXTURE_3D, node->GetImage()->GetID());

                // Draw planes

                glBindTexture(GL_TEXTURE_3D, 0);
                glDisable(GL_TEXTURE_3D);
            }

        }
    }
}
