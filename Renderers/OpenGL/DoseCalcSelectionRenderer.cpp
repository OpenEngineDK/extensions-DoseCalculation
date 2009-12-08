// Dose calculation renderingview via OpenGL rendering view.
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Renderers/OpenGL/DoseCalcSelectionRenderer.h>
#include <Meta/OpenGL.h>
#include <Scene/BeamNode.h>

#include <Logging/Logger.h>

using namespace OpenEngine::Scene;

namespace OpenEngine {
namespace Renderers {
namespace OpenGL {

void DoseCalcSelectionRenderer::VisitBeamNode(BeamNode* node){
    glPushName(count++);
    names.push_back(node);
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
    glVertex3f(0.0,-dist,0.0);               
    glVertex3f(-wh,0,-hh);
    glVertex3f(0.0,-dist,0.0);               
    glVertex3f(-wh,0,hh);
    glVertex3f(0.0,-dist,0.0);               
    glVertex3f(wh,0,hh);
    glVertex3f(0.0,-dist,0.0);               
    glVertex3f(wh,0,-hh);
    glEnd();
               
    glDisable(GL_LINE_SMOOTH);
    glEnable(GL_DEPTH_TEST);
    node->VisitSubNodes(*this);
    CHECK_FOR_GL_ERROR();
    glPopName();
}

} //NS OpenGL
} //NS Renderers
} //NS OpenEngine
