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
#include <Resources/ITexture3D.h>
#include <Logging/Logger.h>
#include <Renderers/CUDA/RayCaster.h>
#include <Meta/CUDA.h>
#include <Display/IViewingVolume.h>


namespace OpenEngine {
    namespace Renderers {
        namespace OpenGL {

using namespace OpenEngine::Scene;
using namespace OpenEngine::Resources;
using namespace OpenEngine::Display;


            RayCastRenderingView::RayCastRenderingView(Viewport& viewport) :
                IRenderingView(viewport),
                RenderingView(viewport) {
                isSetup = false;

            }

            void RayCastRenderingView::VisitDoseCalcNode(DoseCalcNode* node){
                Vector<4,int> d = GetViewport().GetDimension();
                int w = d[2], h = d[3];
                if (!isSetup) {
                    ITexture3DPtr(float) tex = node->GetIntensityTex();
                    // Make PBO
                    glGenBuffers(1,&pbo);
                    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
                    glBufferData(GL_PIXEL_UNPACK_BUFFER, 
                                 w*h*sizeof(GLubyte)*4,
                                 0, 
                                 GL_STATIC_DRAW);

                    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

                    SetupRayCaster(pbo,
                                   tex->GetData(),
                                   tex->GetWidth(),
                                   tex->GetHeight(),
                                   tex->GetDepth());

                    logger.info << "Ray Caster set up" << logger.end;
                    isSetup = true;
                }
                

                IViewingVolume *vol = GetViewport().GetViewingVolume();
                Matrix<4,4,float> IV = vol->GetViewMatrix().GetInverse();
                Matrix<4,4,float> pm = vol->GetProjectionMatrix();
                float iva[16];
                IV.Transpose();
                IV.ToArray(iva);
    
    
                float dx = node->GetDoseTex()->GetWidth();
                float dy = node->GetDoseTex()->GetHeight();
                float dz = node->GetDoseTex()->GetDepth();
                
                
                RenderToPBO(pbo,node->cuDoseArr,w,h,iva,pm(0,0),pm(1,1),dx,dy,dz);

                glMatrixMode(GL_MODELVIEW);
                glLoadIdentity();


                glMatrixMode(GL_PROJECTION);
                glLoadIdentity();
                glOrtho(0,1,0,1,0,1);
                

                //glClear(GL_COLOR_BUFFER_BIT);
                
                // draw image from PBO
                glDisable(GL_DEPTH_TEST);
                glRasterPos2i(0, 0);
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
                glDrawPixels(w, h, GL_RGBA, GL_UNSIGNED_BYTE, 0);
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
            }
        }
    }
}
