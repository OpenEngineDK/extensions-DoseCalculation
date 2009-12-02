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
#include <Resources/TemplatedMHDResource.h>
#include <Resources/IShaderResource.h>
#include <Widgets/Widgifier.h>

using namespace OpenEngine::Core;
using namespace OpenEngine::Resources;
using namespace OpenEngine::Renderers;
using namespace OpenEngine::Widgets;

namespace OpenEngine {
    namespace Geometry {
        class Ray;
    }
    namespace Scene {
        
        class DoseCalcNode : public ISceneNode, public IListener<RenderingEventArg>  {
            OE_SCENE_NODE(DoseCalcNode, ISceneNode)
            WIDGET_INIT();

        public:
            static const int DIMENSIONS = 3;
            static const int TEXCOORDS = 3;
        
        private:
            MHDPtr(float) intensityTex;
            ITexture3DResourcePtr doseTex;
                
            int width, height, depth;
            Vector<3, float> scale;
            int numberOfVertices;

            float* vertices;
            unsigned int verticeId;
            float* texCoords;
            unsigned int texCoordId;

            int xPlaneCoord, yPlaneCoord, zPlaneCoord;
                
            IShaderResourcePtr shader;

        public:
            DoseCalcNode();
            DoseCalcNode(MHDPtr(float) i);
            ~DoseCalcNode();

            void VisitSubNodes(ISceneNodeVisitor& visitor);

            void Handle(RenderingEventArg arg);

            // **** Get/Set ****
            
            ITexture3DPtr(float) GetIntensityTex() const { return intensityTex; }
            ITexture3DResourcePtr GetDoseTex() const { return doseTex; }
            unsigned int GetVerticeId() const { return verticeId; }
            unsigned int GetTexCoordId() const { return texCoordId; }
            int GetIndice(int x, int y, int z);
            int GetXPlaneCoord() const { return xPlaneCoord; }
            int GetYPlaneCoord() const { return yPlaneCoord; }
            int GetZPlaneCoord() const { return zPlaneCoord; }
            
            float GetWidth() { return width * scale[0]; }
            float GetHeight() { return height * scale[1]; }
            float GetDepth() { return depth * scale[2]; }
            Vector<3, float> GetScale() const { return scale; }
            IShaderResourcePtr GetShader() { return shader; }
            void SetXPlaneCoord(int x);
            void SetYPlaneCoord(int y);
            void SetZPlaneCoord(int z);
            void SetShader(IShaderResourcePtr doseShader) { shader = doseShader; }

            void AddRay(Geometry::Ray* ray);

        private:
            inline void Init();
            inline void SetupVertices();
            inline void SetupXPlane();
            inline void SetupYPlane();
            inline void SetupZPlane();
            inline int GetIndex(int x, int y, int z);
        };

    }
}

#endif
