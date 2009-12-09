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
#include <Scene/BeamNode.h>
#include <Resources/ITexture3D.h>

#include <Meta/OpenGL.h>
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

            void DoseCalcRenderingView::VisitBeamNode(BeamNode* node){
                glDisable(GL_DEPTH_TEST);
                glColor4f(0,1,0,1);
                glEnable(GL_LINE_SMOOTH);
                // draw beam surface
                glLineWidth(1.0);
                const float dist = 1.0;
                const float wh = 0.5, hh = 0.5;
                glBegin(GL_LINE_STRIP);
                glVertex3f(-wh,0,-hh);
                glVertex3f(-wh,0,hh);
                glVertex3f(wh,0,hh);
                glVertex3f(wh,0,-hh);
                glVertex3f(-wh,0,-hh);
                glEnd();
            
                // draw beam outlines
                glBegin(GL_LINES);
                glVertex3f(0.0,dist,0.0);               
                glVertex3f(-wh,0,-hh);
                glVertex3f(0.0,dist,0.0);               
                glVertex3f(-wh,0,hh);
                glVertex3f(0.0,dist,0.0);               
                glVertex3f(wh,0,hh);
                glVertex3f(0.0,dist,0.0);               
                glVertex3f(wh,0,-hh);
                glEnd();
               
                glDisable(GL_LINE_SMOOTH);
                glEnable(GL_DEPTH_TEST);
                node->VisitSubNodes(*this);
            }

            void DoseCalcRenderingView::VisitDoseCalcNode(DoseCalcNode* node){
                Vector<3,float> zero;
                // node->CalculateDose(Beam(zero, zero, zero, zero, zero), 1, 1);

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
                
                node->VisitSubNodes(*this);
            }
        }
    }
}
