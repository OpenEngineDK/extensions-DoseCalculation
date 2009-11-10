// Dose Calculation node.
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Scene/DoseCalcNode.h>
#include <Meta/OpenGL.h>

namespace OpenEngine {
    namespace Scene {

        DoseCalcNode::DoseCalcNode()
            : image(ITexture3DResourcePtr()), xPlaneCoord(0), yPlaneCoord(0), zPlaneCoord(0) {
        }

        DoseCalcNode::DoseCalcNode(ITexture3DResourcePtr i)
            : image(i), xPlaneCoord(0), yPlaneCoord(0), zPlaneCoord(0) {
        }

        DoseCalcNode::~DoseCalcNode(){

        }

        void DoseCalcNode::VisitSubNodes(ISceneNodeVisitor& visitor){
            std::list<ISceneNode*>::iterator itr;
            for (itr = subNodes.begin(); itr != subNodes.end(); ++itr){
                (*itr)->Accept(visitor);
            }
        }
        void DoseCalcNode::Handle(RenderingEventArg arg){
            if (image != NULL){
                // Load the texture
                image->Load();

                GLuint texId;
                glGenTextures(1, &texId);
                glBindTexture(GL_TEXTURE_3D, texId);
                glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S, GL_CLAMP);
                glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T, GL_CLAMP);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB, 
                             image->GetWidth(), image->GetHeight(), image->GetDepth(),
                             0, GL_RGBA, GL_UNSIGNED_BYTE, image->GetData());

                image->SetID(texId);

                glBindTexture(GL_TEXTURE_3D, 0);
            }
            
        }
    }
}
