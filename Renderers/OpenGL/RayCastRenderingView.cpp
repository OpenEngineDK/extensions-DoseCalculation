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
#include <Renderers/CUDA/RayCaster.h>
#include <Meta/CUDA.h>

using namespace OpenEngine::Scene;
using namespace OpenEngine::Resources;

namespace OpenEngine {
    namespace Renderers {
        namespace OpenGL {

            RayCastRenderingView::RayCastRenderingView(Viewport& viewport) :
                IRenderingView(viewport),
                RenderingView(viewport) {
                isSetup = false;

            }

            void RayCastRenderingView::VisitDoseCalcNode(DoseCalcNode* node){
                int w = 480,h=480;
                if (!isSetup) {
                    ITexture3DResourcePtr tex = node->GetIntensityTex();
                                        // Make PBO


                    glGenBuffers(1,&pbo);
                    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
                    glBufferData(GL_PIXEL_UNPACK_BUFFER, w*h*sizeof(GLubyte)*4, 0, GL_STREAM_DRAW);

                    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

                    SetupRayCaster(pbo,
                                   tex->GetData(),
                                   tex->GetWidth(),
                                   tex->GetHeight(),
                                   tex->GetDepth());

                    logger.info << "Ray Caster set up" << logger.end;
                    isSetup = true;
                }
                RenderToPBO(pbo,w,h);

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
