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
            : image(ITexture3DResourcePtr()) {
            Init();
        }

        DoseCalcNode::DoseCalcNode(ITexture3DResourcePtr i)
            : image(i){
            Init();
        }

        void DoseCalcNode::Init(){
            xPlaneCoord = yPlaneCoord = zPlaneCoord = 0;
            vertices = texCoords = NULL;
            if (image != NULL){
                image->Load();
                width = image->GetWidth();
                height = image->GetHeight();
                depth = image->GetDepth();
                widthScale = image->GetWidthScale();
                heightScale = image->GetHeightScale();
                depthScale = image->GetDepthScale();
            }else{
                width = height = depth = 0;
                widthScale = heightScale = depthScale = 0;
            }
        }

        DoseCalcNode::~DoseCalcNode(){
            delete [] vertices;
            delete [] texCoords;
        }

        void DoseCalcNode::VisitSubNodes(ISceneNodeVisitor& visitor){
            std::list<ISceneNode*>::iterator itr;
            for (itr = subNodes.begin(); itr != subNodes.end(); ++itr){
                (*itr)->Accept(visitor);
            }
        }

        void DoseCalcNode::Handle(RenderingEventArg arg){
            SetupVertices();
            SetupTexCoords();

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

                GLuint colorDepth;
                switch (image->GetColorFormat()) {
                case LUMINANCE: colorDepth = GL_LUMINANCE; break;
                case RGB: colorDepth = GL_RGB; break;
                case RGBA: colorDepth = GL_RGBA; break;
                default:
                    colorDepth = GL_BGRA;
                }
                
                glTexImage3D(GL_TEXTURE_3D, 0, colorDepth, 
                             image->GetWidth(), image->GetHeight(), image->GetDepth(),
                             0, GL_RGBA, GL_FLOAT, image->GetData());

                image->SetID(texId);

                glBindTexture(GL_TEXTURE_3D, 0);
            }
        }

        void DoseCalcNode::SetupVertices(){
            for (int x = 0; x < width; ++x)
                for (int y = 0; y < height; ++y)
                    for (int z = 0; z < depth; ++z){
                        float* vertex = GetVertex(x, y, z);
                        vertex[0] = x * widthScale;
                        vertex[1] = y * heightScale;
                        vertex[2] = z * depthScale;
                    }
        }
        
        void DoseCalcNode::SetupTexCoords(){
            for (int x = 0; x < width; ++x)
                for (int y = 0; y < height; ++y)
                    for (int z = 0; z < depth; ++z){
                        float* coord = GetTexCoord(x, y, z);
                        coord[0] = (x + 0.5f) / (float) width;
                        coord[1] = (y + 0.5f) / (float) height;
                        coord[2] = (z + 0.5f) / (float) depth;
                    }
        }

        int DoseCalcNode::GetIndex(int x, int y, int z){
            return z + x * depth + y * width * depth;
        }        
        
        float* DoseCalcNode::GetVertex(int x, int y, int z){
            int index = GetIndex(x, y, z);
            return vertices + index * DIMENSIONS;
        }
        
        float* DoseCalcNode::GetTexCoord(int x, int y, int z){
            int index = GetIndex(x, y, z);
            return texCoords + index * TEXCOORDS;
        }
    }
}
