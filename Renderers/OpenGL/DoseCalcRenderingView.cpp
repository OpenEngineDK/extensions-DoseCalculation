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
                
                // Draw planes
                /*
                glBegin(GL_QUADS);
                glTexCoord3f(0.5, 0, 0);
                glVertex3f(100, 0, 0);

                glTexCoord3f(0.5, 1, 0);
                glVertex3f(100, 100, 0);

                glTexCoord3f(0.5, 1, 1);
                glVertex3f(100, 100, 100);

                glTexCoord3f(0.5, 0, 1);
                glVertex3f(100, 0, 100);
                glEnd();

                */
                glBegin(GL_QUADS);
                // Draw the xy plane
                float z = node->GetZPlaneCoord();
                float zTex = z / depth;

                glTexCoord3f(0, 0, zTex);
                glVertex3f(0, 0, z);

                glTexCoord3f(1, 0, zTex);
                glVertex3f(width, 0, z);

                glTexCoord3f(1, 1, zTex);
                glVertex3f(width, height, z);

                glTexCoord3f(0, 1, zTex);
                glVertex3f(0, height, z);

                // Draw the xz plane
                float y = node->GetYPlaneCoord();
                float yTex = y / height;

                glTexCoord3f(0, yTex, 0);
                glVertex3f(0, y, 0);

                glTexCoord3f(1, yTex, 0);
                glVertex3f(width, y, 0);

                glTexCoord3f(1, yTex, 1);
                glVertex3f(width, y, depth);

                glTexCoord3f(0, yTex, 1);
                glVertex3f(0, y, depth);

                // Draw the yz plane
                float x = node->GetXPlaneCoord();
                float xTex = x / height;

                glTexCoord3f(xTex, 0, 0);
                glVertex3f(x, 0, 0);

                glTexCoord3f(xTex, 1, 0);
                glVertex3f(x, height, 0);

                glTexCoord3f(xTex, 1, 1);
                glVertex3f(x, height, depth);

                glTexCoord3f(xTex, 0, 1);
                glVertex3f(x, 0, depth);

                glEnd();

                glBindTexture(GL_TEXTURE_3D, 0);
                glDisable(GL_TEXTURE_3D);
            }

        }
    }
}
