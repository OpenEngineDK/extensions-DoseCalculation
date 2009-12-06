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
    GLUquadricObj* quad = gluNewQuadric();
    gluQuadricNormals(quad, GLU_SMOOTH);
    glColor4f(0,1,0,1);
    glPushMatrix();
    glRotatef(90,1.0,0.0,0.0);
    gluCylinder(quad, /*base*/5.0f, /*top*/5.0f, /*height*/100.0f, /*slices*/10, /*stacks*/10);
    glPushMatrix();
    glTranslatef(0,0,100);
    gluCylinder(quad, /*base*/20.0f, /*top*/0.0f, /*height*/20.0f, /*slices*/10, /*stacks*/10);
    glPopMatrix();
    glPopMatrix();
    gluDeleteQuadric(quad);
    node->VisitSubNodes(*this);
    CHECK_FOR_GL_ERROR();
    glPopName();
}

} //NS OpenGL
} //NS Renderers
} //NS OpenEngine
