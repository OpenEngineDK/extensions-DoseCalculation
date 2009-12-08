#ifndef _DOZE_SETUP_H_
#define _DOZE_SETUP_H_
// 
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Core/IListener.h>
#include <Renderers/IRenderer.h>
#include <Utils/CUDA/Doze.h>

namespace OpenEngine {
namespace Utils {

    using namespace Core;
    using namespace Resources;
    using namespace Renderers;


/**
 * Short description.
 *
 * @class DozeSetup DozeSetup.h ons/DoseCalculation/Utils/DozeSetup.h
 */
    class DozeSetup : public IListener<RenderingEventArg>{
private:
        DoseCalcNode* node;
public:
        DozeSetup(DoseCalcNode* );

        void Handle(RenderingEventArg arg) {

        }
};

} // NS Utils
} // NS OpenEngine

#endif
