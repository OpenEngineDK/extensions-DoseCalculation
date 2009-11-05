// Dose Calculation node.
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _HEIGHTFIELD_NODE_H_
#define _HEIGHTFIELD_NODE_H_

#include <Scene/ISceneNode.h>

namespace OpenEngine {
    namespace Scene {

        class DoseCalcNode : public ISceneNode {
            OE_SCENE_NODE(DoseCalcNode, ISceneNode)

        public:
            DoseCalcNode();
            ~DoseCalcNode();

            void VisitSubNodes(ISceneNodeVisitor& visitor);
        };

    }
}

#endif
