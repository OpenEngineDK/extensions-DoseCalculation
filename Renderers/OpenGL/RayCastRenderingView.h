// Dose calculation rendering view using ray casting.
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _DOSE_CALC_RAY_CAST_RENDERING_VIEW_H_
#define _DOSE_CALC_RAY_CAST_RENDERING_VIEW_H_

#include <Renderers/OpenGL/RenderingView.h>
#include <Widgets/Widgifier.h>


namespace OpenEngine {
namespace Renderers {
namespace OpenGL {

    using namespace Renderers;
    using namespace Widgets;
    
    class RayCastRenderingView : public RenderingView {

        bool isSetup;
        GLuint pbo;
        
        float minIntensity;
        float maxIntensity;

    public:
        RayCastRenderingView(Viewport& viewport);

        void VisitDoseCalcNode(DoseCalcNode* node);
        float GetMinIntensity() {return minIntensity;}
        void SetMinIntensity(float i) {minIntensity = i;}

        float GetMaxIntensity() {return maxIntensity;}
        void SetMaxIntensity(float i) {maxIntensity = i;}


    };

    WIDGET_START(RayCastRenderingView);
    WIDGET_SLIDER("MIN", 
                  GetMinIntensity, 
                  SetMinIntensity,
                  CONST, 0,
                  CONST, 1);
    WIDGET_SLIDER("MAX", 
                  GetMaxIntensity, 
                  SetMaxIntensity,
                  CONST, 0,
                  CONST, 1);
    WIDGET_STOP();


}
}
}

#endif
