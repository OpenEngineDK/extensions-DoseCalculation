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
#include <Logging/Logger.h>

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
            vertices = texCoords = NULL;
            if (image != NULL){
                image->Load();
                width = image->GetWidth();
                height = image->GetHeight();
                depth = image->GetDepth();
                widthScale = image->GetWidthScale();
                heightScale = image->GetHeightScale();
                depthScale = image->GetDepthScale();
                xPlaneCoord = width / 2.0f;
                yPlaneCoord = height / 2.0f;
                zPlaneCoord = depth / 2.0f;
            }else{
                width = height = depth = 0;
                widthScale = heightScale = depthScale = 0;
                xPlaneCoord = yPlaneCoord = zPlaneCoord = 0;
            }

            numberOfVertices = width * height * depth;
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
                glTexParameteri(GL_TEXTURE_3D,GL_TEXTURE_WRAP_S, GL_REPEAT);
                glTexParameteri(GL_TEXTURE_3D,GL_TEXTURE_WRAP_T, GL_REPEAT);
                glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
                
                GLuint colorDepth;
                switch (image->GetColorFormat()) {
                case LUMINANCE: colorDepth = GL_LUMINANCE; break;
                case RGB: colorDepth = GL_RGB; break;
                case RGBA: colorDepth = GL_RGBA; break;
                default:
                    colorDepth = GL_BGRA;
                }
                
                glTexImage3D(GL_TEXTURE_3D, 0, colorDepth, 
                             width, height, depth,
                             0, colorDepth, GL_FLOAT, image->GetData());

                image->SetID(texId);

                glBindTexture(GL_TEXTURE_3D, 0);
            }

            // load the vbo's
            
            // Vertice buffer object
            glGenBuffers(1, &verticeId);
            glBindBuffer(GL_ARRAY_BUFFER, verticeId);
            glBufferData(GL_ARRAY_BUFFER,
                         sizeof(GLfloat) * numberOfVertices * DIMENSIONS,
                         vertices, GL_STATIC_DRAW);
            
            // Tex Coord buffer object
            glGenBuffers(1, &texCoordId);
            glBindBuffer(GL_ARRAY_BUFFER, texCoordId);
            glBufferData(GL_ARRAY_BUFFER, 
                         sizeof(GLfloat) * numberOfVertices * TEXCOORDS,
                         texCoords, GL_STATIC_DRAW);

            glBindBuffer(GL_ARRAY_BUFFER, 0);
        }

        // **** Get/Set ****

        int DoseCalcNode::GetIndice(int x, int y, int z){
            return GetIndex(x, y, z);
        }

        void DoseCalcNode::SetXPlaneCoord(int x) { 
            if (x < 0)
                xPlaneCoord = 0;
            else if (x > width-1)
                xPlaneCoord = width-1;
            else
                xPlaneCoord = x; 
        }

        void DoseCalcNode::SetYPlaneCoord(int y) { 
            if (y < 0)
                yPlaneCoord = 0;
            else if (y > height - 1)
                yPlaneCoord = height - 1;
            else
                yPlaneCoord = y; 
        }

        void DoseCalcNode::SetZPlaneCoord(int z) { 
            if (z < 0)
                zPlaneCoord = 0;
            else if (z > depth-1)
                zPlaneCoord = depth-1;
            else
                zPlaneCoord = z; 
        }

        // *** inline methods ***

        void DoseCalcNode::SetupVertices(){
            vertices = new float[numberOfVertices * DIMENSIONS];
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
            texCoords = new float[numberOfVertices * TEXCOORDS];
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
