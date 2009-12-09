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
#include <Widgets/Widgifier.h>

#include <Logging/Logger.h>

namespace OpenEngine {
namespace Utils {

/**
 * Short description.
 *
 * @class DoseTrigger DoseTrigger.h ons/DoseCalculation/Utils/DoseTrigger.h
 */
class DoseTrigger {
private:
    DoseCalcNode* dnode;
    BeamNode* bnode;
public:
    DoseTrigger(DoseCalcNode* dnode, BeamNode* bnode) : dnode(dnode), bnode(bnode) {}
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
};

WIDGET_START(DoseTrigger);
WIDGET_BUTTON(Calc, 0, DoCalc, TRIGGER);
WIDGET_STOP();

} // NS Utils
} // NS OpenEngine

#endif
