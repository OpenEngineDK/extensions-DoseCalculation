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
#include <Resources/ITexture3D.h>
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
                glBindBuffer(GL_ARRAY_BUFFER, node->GetVerticeId());
                glEnableClientState(GL_VERTEX_ARRAY);
                glVertexPointer(DoseCalcNode::DIMENSIONS, GL_FLOAT, 0, 0);

                // Setup textures
                glEnable(GL_TEXTURE_3D);
                glBindTexture(GL_TEXTURE_3D, node->GetIntensityTex()->GetID());

                // Setup Texture coords
                glBindBuffer(GL_ARRAY_BUFFER, node->GetTexCoordId());
                glEnableClientState(GL_TEXTURE_COORD_ARRAY);
                glTexCoordPointer(DoseCalcNode::TEXCOORDS, GL_FLOAT, 0, 0);

                /*
                IShaderResourcePtr shader = node->GetShader();
                if (shader){
                    shader->ApplyShader();
                }
                */

                // Render
                {
                    glPushMatrix();
                    
                    Vector<3, float> scale = node->GetScale();
                    glScalef(scale[0], scale[1], scale[2]);
                    
                    glColor4f(1,1,1,1);
                    glDrawArrays(GL_QUADS, 0, 12);
                    
                    glPopMatrix();
                }

                /*
                if (shader){
                    shader->ReleaseShader();
                }
                */

                glDisableClientState(GL_TEXTURE_COORD_ARRAY);
                glDisableClientState(GL_VERTEX_ARRAY);
                glBindBuffer(GL_ARRAY_BUFFER, 0);

                glDisable(GL_TEXTURE_3D);
                glBindTexture(GL_TEXTURE_3D, 0);

            }

        }
    }
}
