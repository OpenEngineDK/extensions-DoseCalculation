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

#include <Display/Camera.h>
#include <Display/ViewingVolume.h>
#include <Display/Viewport.h>

#include <Logging/Logger.h>

namespace OpenEngine {
namespace Utils {

using namespace Scene;
using namespace Display;

/**
 * Short description.
 *
 * @class DoseTrigger DoseTrigger.h ons/DoseCalculation/Utils/DoseTrigger.h
 */
class DoseTrigger {
private:
    DoseCalcNode* dnode;
    BeamNode* bnode;
    TransformationNode *pivot, *beamTrans;//, *beamCalibrate;
    float dist;
    Viewport* vp;
    Vector<3,float> vv_pos;
    Quaternion<float> vv_rot;
    bool bav;
    float scale;
    float srcDist;
public:
    DoseTrigger(DoseCalcNode* dnode, Viewport* vp) 
  : dnode(dnode)
  , bnode(new BeamNode(1,1))
  , pivot(new TransformationNode()) 
  , beamTrans(new TransformationNode())
  // , beamCalibrate(new TransformationNode())
  , dist(500)
  , vp(vp)
  , bav(false)
  , scale(20.0)
  , srcDist(200.0)
{
    beamTrans->SetScale(Vector<3,float>(scale,srcDist,scale));
    ITexture3DBasePtr tex = dnode->GetIntensityTex();
    Vector<3,float> scale = dnode->GetScale();
    pivot->SetPosition(Vector<3,float>(tex->GetWidth()*scale[0]*0.5, tex->GetHeight()*scale[1]*0.5, tex->GetDepth()*scale[2]*0.5));
    
    pivot->AddNode(beamTrans);
    beamTrans->AddNode(bnode);
    // beamTrans->AddNode(beamCalibrate);
    // beamCalibrate->AddNode(bnode);
    SetAngle(0);
}
    virtual ~DoseTrigger() {}
    void DoCalc() { 
        // Beam b = bnode->GetBeam();
        // logger.info << "Beam: srcpos = " << b.src 
        //             << ",\nrect = (" << b.p1 
        //             << ", " << b.p2 << ", "
        //             << b.p3 << ", " 
        //             << b.p4 << ")" 
        //             << logger.end;
        // dnode->CalculateDose(bnode->GetBeam(), 1, 1); 
        dnode->CalculateDose(bnode, 1, 1); 
    }
    TransformationNode* GetPivotNode() {
        return pivot;
    }

    void SetAngle(float angle) {
        Quaternion<float> q(angle, Vector<3,float>(0.0,0.0,1.0));
        beamTrans->SetRotation(q);
        beamTrans->SetPosition(q.RotateVector(Vector<3,float>(0.0,1.0,0.0))* dist);
        if (bav) UpdateViewingVolume();
    }
  
    float GetScale() {
        return scale;
    }

    void SetScale(float scale) {
        beamTrans->SetScale(Vector<3,float>(scale,srcDist,scale));
        this->scale = scale;
    }    

    float GetBeamDist() {
        return dist;
    }

    void SetBeamDist(float dist) {
        this->dist = dist;
        Quaternion<float> q = beamTrans->GetRotation();
        beamTrans->SetPosition(q.RotateVector(Vector<3,float>(0.0,1.0,0.0))* dist);
        if (bav) UpdateViewingVolume();
    }    

    float GetSrcDist() {
        return srcDist;
    }

    void SetSrcDist(float srcDist) {
        beamTrans->SetScale(Vector<3,float>(scale,srcDist,scale));
        this->srcDist = srcDist;
        if (bav) UpdateViewingVolume();
    }    
  
    float GetAngle() {
        return beamTrans->GetRotation().GetReal();
    }

    bool GetBeamAngleView() {
        return bav;
    }

    void SetBeamAngleView(bool bav) {
        bool changed = (this->bav != bav);
        this->bav = bav;
        if (!changed) return;
        if (bav) {
            vv_pos = vp->GetViewingVolume()->GetPosition();
            vv_rot = vp->GetViewingVolume()->GetDirection();
            UpdateViewingVolume();
        }
        else {
            vp->GetViewingVolume()->SetPosition(vv_pos);
            vp->GetViewingVolume()->SetDirection(vv_rot);
        }
    }

private:
    inline void UpdateViewingVolume() {
        // logger.info << "Update BAV vv" << logger.end;
        Quaternion<float> q;
        Vector<3,float> p;
        beamTrans->GetAccumulatedTransformations(&p, &q);
        Quaternion<float> q1(PI, Vector<3,float>(0.0,1.0,0.0));
        Quaternion<float> q2(0.5*PI, Vector<3,float>(-1.0,0.0,0.0));
        vp->GetViewingVolume()->SetPosition(p - q.RotateVector(Vector<3,float>(0.0,-srcDist,0.0)));
        vp->GetViewingVolume()->SetDirection(q*q1*q2);
    }

};

WIDGET_START(DoseTrigger, DoseTriggerWidget);
WIDGET_BUTTON("Calculate dose", 0, DoCalc, TRIGGER);
WIDGET_BUTTON("Beam view", GetBeamAngleView, SetBeamAngleView, TOGGLE);
WIDGET_SLIDER("Beam angle", GetAngle, SetAngle, CONST, 0, CONST, 2*PI);
WIDGET_SLIDER("Beam scale", GetScale, SetScale, CONST, 20, CONST, 200);
WIDGET_SLIDER("Source distance", GetSrcDist, SetSrcDist, CONST, 20, CONST, 400);
WIDGET_SLIDER("Beam distance", GetBeamDist, SetBeamDist, CONST, 100, CONST, 1000);
WIDGET_STOP();

} // NS Utils
} // NS OpenEngine

#endif
