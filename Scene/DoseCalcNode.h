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
#include <Core/IListener.h>
#include <Renderers/IRenderer.h>
#include <Resources/ITexture3DResource.h>

using namespace OpenEngine::Core;
using namespace OpenEngine::Resources;
using namespace OpenEngine::Renderers;

namespace OpenEngine {
    namespace Scene {
        
        class DoseCalcNode : public ISceneNode, public IListener<RenderingEventArg>  {
            OE_SCENE_NODE(DoseCalcNode, ISceneNode)
        
            ITexture3DResourcePtr image;

            float xPlaneCoord, yPlaneCoord, zPlaneCoord;

        public:
            DoseCalcNode();
            DoseCalcNode(ITexture3DResourcePtr i);
            ~DoseCalcNode();

            void VisitSubNodes(ISceneNodeVisitor& visitor);

            void Handle(RenderingEventArg arg);

            // **** Get/Set ****
            
            ITexture3DResourcePtr GetImage() const { return image; }
            float GetXPlaneCoord() const { return xPlaneCoord; }
            float GetYPlaneCoord() const { return yPlaneCoord; }
            float GetZPlaneCoord() const { return zPlaneCoord; }

            void SetXPlaneCoord(float x) { xPlaneCoord = x; }
            void SetYPlaneCoord(float y) { yPlaneCoord = y; }
            void SetZPlaneCoord(float z) { zPlaneCoord = z; }
        };

    }
}

#endif
