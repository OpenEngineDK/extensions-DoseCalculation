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
#include <Resources/ITexture3DResource.h>
#include <Logging/Logger.h>

using namespace OpenEngine::Scene;
using namespace OpenEngine::Resources;

namespace OpenEngine {
    namespace Renderers {
        namespace OpenGL {

            DoseCalcRenderingView::DoseCalcRenderingView(Viewport& viewport) : 
                IRenderingView(viewport), 
                RenderingView(viewport) {
            }

            void DoseCalcRenderingView::VisitDoseCalcNode(DoseCalcNode* node){
                glEnable(GL_TEXTURE_3D);
                
                ITexture3DResourcePtr image = node->GetImage();
                glBindTexture(GL_TEXTURE_3D, image->GetID());

                float width = (float) image->GetWidth();
                float height = (float) image->GetHeight();
                float depth = (float) image->GetDepth();

                glBindBuffer(GL_ARRAY_BUFFER, node->GetVerticeId());
                glEnableClientState(GL_VERTEX_ARRAY);
                glVertexPointer(DoseCalcNode::DIMENSIONS, GL_FLOAT, 0, 0);

                // Setup Texture coords
                glBindBuffer(GL_ARRAY_BUFFER, node->GetTexCoordId());
                glEnableClientState(GL_TEXTURE_COORD_ARRAY);
                glTexCoordPointer(DoseCalcNode::TEXCOORDS, GL_FLOAT, 0, 0);
                
                glBegin(GL_QUADS);

                float x = node->GetXPlaneCoord();
                glArrayElement(node->GetIndice(x, 0, 0));
                glArrayElement(node->GetIndice(x, 0, depth-1));
                glArrayElement(node->GetIndice(x, height-1, depth-1));
                glArrayElement(node->GetIndice(x, height-1, 0));

                float y = node->GetYPlaneCoord();
                glArrayElement(node->GetIndice(0, y, 0));
                glArrayElement(node->GetIndice(width-1, y, 0));
                glArrayElement(node->GetIndice(width-1, y, depth-1));
                glArrayElement(node->GetIndice(0, y, depth-1));

                float z = node->GetZPlaneCoord();
                glArrayElement(node->GetIndice(0, 0, z));
                glArrayElement(node->GetIndice(width-1, 0, z));
                glArrayElement(node->GetIndice(width-1, height-1, z));
                glArrayElement(node->GetIndice(0, height-1, z));

                glEnd();


                glDisableClientState(GL_TEXTURE_COORD_ARRAY);
                glDisableClientState(GL_VERTEX_ARRAY);
                glBindBuffer(GL_ARRAY_BUFFER, 0);

                glDisable(GL_TEXTURE_3D);
                glBindTexture(GL_TEXTURE_3D, 0);

            }

        }
    }
}
