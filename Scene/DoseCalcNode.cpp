// Dose Calculation node.
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Scene/DoseCalcNode.h>

namespace OpenEngine {
    namespace Scene {

        DoseCalcNode::DoseCalcNode(){

        }

        DoseCalcNode::~DoseCalcNode(){

        }

        void DoseCalcNode::VisitSubNodes(ISceneNodeVisitor& visitor){
            std::list<ISceneNode*>::iterator itr;
            for (itr = subNodes.begin(); itr != subNodes.end(); ++itr){
                (*itr)->Accept(visitor);
            }
        }

    }
}
