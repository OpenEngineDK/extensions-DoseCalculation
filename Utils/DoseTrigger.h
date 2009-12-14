#ifndef _DOSE_TRIGGER_H_
#define _DOSE_TRIGGER_H_
// 
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Scene/DoseCalcNode.h>
#include <Scene/BeamNode.h>
#include <Scene/TransformationNode.h>
#include <Widgets/Widgifier.h>

#include <Logging/Logger.h>

namespace OpenEngine {
namespace Utils {

using namespace Scene;

/**
 * Short description.
 *
 * @class DoseTrigger DoseTrigger.h ons/DoseCalculation/Utils/DoseTrigger.h
 */
class DoseTrigger {
private:
    DoseCalcNode* dnode;
    BeamNode* bnode;
    TransformationNode *pivot, *beamTrans;
    float dist;
public:
    DoseTrigger(DoseCalcNode* dnode) 
  : dnode(dnode)
  , bnode(new BeamNode(1,1))
  , pivot(new TransformationNode()) 
  , beamTrans(new TransformationNode())
  , dist(250)
{
    beamTrans->SetScale(Vector<3,float>(100,200,100));
    ITexture3DBasePtr tex = dnode->GetIntensityTex();
    Vector<3,float> scale = dnode->GetScale();
    pivot->SetPosition(Vector<3,float>(tex->GetWidth()*scale[0]*0.5, tex->GetHeight()*scale[1]*0.5, tex->GetDepth()*scale[2]*0.5));
    
    beamTrans->AddNode(bnode);
    pivot->AddNode(beamTrans);
    SetAngle(0);
}
    virtual ~DoseTrigger() {}
    void DoCalc() { 
        Beam b = bnode->GetBeam();
        logger.info << "Beam: srcpos = " << b.src 
                    << ",\nrect = (" << b.p1 
                    << ", " << b.p2 << ", "
                    << b.p3 << ", " 
                    << b.p4 << ")" 
                    << logger.end;
        dnode->CalculateDose(bnode->GetBeam(), 1, 1); 
    }
    TransformationNode* GetPivotNode() {
        return pivot;
    }

    void SetAngle(float angle) {
        Quaternion<float> q(angle, Vector<3,float>(1.0,0.0,0.0));
        beamTrans->SetRotation(q);
        beamTrans->SetPosition(q.RotateVector(Vector<3,float>(0.0,1.0,0.0))* dist);
    }
    
    float GetAngle() {
        return beamTrans->GetRotation().GetReal();
    }
};

WIDGET_START(DoseTrigger);
WIDGET_BUTTON(Calc, 0, DoCalc, TRIGGER);
WIDGET_SLIDER(BeamAngle, GetAngle, SetAngle, CONST, 0, CONST, 2*PI);
WIDGET_STOP();

} // NS Utils
} // NS OpenEngine

#endif
