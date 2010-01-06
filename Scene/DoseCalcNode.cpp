// Dose Calculation node.
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Scene/DoseCalcNode.h>
#include <Resources/Texture3D.h>
#include <Logging/Logger.h>

#include <Utils/CUDA/DoseCalc.h>
//#include <Utils/CUDA/Superposition.h>
#include <Utils/CUDA/Doze.h>
#include <Scene/BeamNode.h>

using namespace OpenEngine::Geometry;

namespace OpenEngine {
    namespace Scene {

        DoseCalcNode::DoseCalcNode()
            : intensityTex(MHDPtr(float)()) {
            Init();
        }

        DoseCalcNode::DoseCalcNode(MHDPtr(float) i)
            : intensityTex(i){
            Init();
        }

        void DoseCalcNode::Init(){
            vertices = texCoords = NULL;
            if (intensityTex != NULL){
                intensityTex->Load();
                width = intensityTex->GetWidth();
                height = intensityTex->GetHeight();
                depth = intensityTex->GetDepth();
                scale = Vector<3, float>(intensityTex->GetWidthScale(), intensityTex->GetHeightScale(), intensityTex->GetDepthScale());
                xPlaneCoord = width / 2.0f;
                yPlaneCoord = height / 2.0f;
                zPlaneCoord = depth / 2.0f;
            }else{
                width = height = depth = 0;
                scale = Vector<3, float>();
                xPlaneCoord = yPlaneCoord = zPlaneCoord = 0;
            }
            doseTex = ITexture3DPtr(float)(new Texture3D<float>(width, height, depth, OE_RGB));
            doseTex->Load();

            float* data = doseTex->GetData();
            for (int i = 0; i < numberOfVertices; ++i){
                data[0] = 1.0f;
                data[1] = 0.0f;
                data[2] = 0.0f;
            }

            numberOfVertices = 12;

        }

        DoseCalcNode::~DoseCalcNode(){
            delete [] vertices;
            delete [] texCoords;
        }

        void DoseCalcNode::CalculateDose(BeamNode* beam, int beamlet_x, int beamlet_y) {
            //RunDoseCalc(cuDoseArr, beam->GetBeam(1.0), beamlet_x, beamlet_y, 0);
            unsigned char fmap = 1;
            Dose(&cuDoseArr, 
                 beam->GetBeam(1.0), 
                 beam->GetBeam(12.0),
                 &fmap, 
                 1,//beamlet_x, 
                 1);//beamlet_y, 
            logger.info << "RUN DONE" << logger.end;
        }

        void DoseCalcNode::Handle(RenderingEventArg arg){
            // Alloc memory for the buffer objects
            
            // Vertice buffer object
            GLuint bufId;
            glGenBuffers(1, &bufId);
            glBindBuffer(GL_ARRAY_BUFFER, bufId);
            glBufferData(GL_ARRAY_BUFFER,
                         sizeof(GLfloat) * numberOfVertices * DIMENSIONS,
                         NULL, GL_STATIC_DRAW);
            verticeId = bufId;
            
            // Tex Coord buffer object
            glGenBuffers(1, &bufId);
            glBindBuffer(GL_ARRAY_BUFFER, bufId);
            glBufferData(GL_ARRAY_BUFFER, 
                         sizeof(GLfloat) * numberOfVertices * TEXCOORDS,
                         NULL, GL_STATIC_DRAW);
            texCoordId = bufId;

            glBindBuffer(GL_ARRAY_BUFFER, 0);


            SetupVertices();

            if (intensityTex != NULL){
                // Load the texture
                intensityTex->Load();

                GLuint texId;
                glGenTextures(1, &texId);
                glBindTexture(GL_TEXTURE_3D, texId);
                glTexParameteri(GL_TEXTURE_3D,GL_TEXTURE_WRAP_S, GL_REPEAT);
                glTexParameteri(GL_TEXTURE_3D,GL_TEXTURE_WRAP_T, GL_REPEAT);
                glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                // glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
                
                GLuint colorDepth;
                switch (intensityTex->GetColorFormat()) {
                case LUMINANCE: colorDepth = GL_LUMINANCE; break;
                case RGB: colorDepth = GL_RGB; break;
                case RGBA: colorDepth = GL_RGBA; break;
                default:
                    colorDepth = GL_BGRA;
                }
                
                glTexImage3D(GL_TEXTURE_3D, 0, colorDepth, 
                             width, height, depth,
                             0, colorDepth, GL_FLOAT, intensityTex->GetData());

                intensityTex->SetID(texId);

                glBindTexture(GL_TEXTURE_3D, 0);

                // Upload to CUDA
                SetupDoze(intensityTex->GetData(),
                          intensityTex->GetWidth(),
                          intensityTex->GetHeight(),
                          intensityTex->GetDepth());
            }

            // glGenBuffers(1, &dosePbo);
            // glBindBuffer(GL_ARRAY_BUFFER, dosePbo);
            // glBufferData(GL_ARRAY_BUFFER,
            //              sizeof(GLfloat) * width * height * depth,
            //              NULL, GL_DYNAMIC_DRAW);
            // glBindBuffer(GL_ARRAY_BUFFER, 0);
            // CHECK_FOR_GL_ERROR();

            SetupDoseCalc(&cuDoseArr, 
                          width, height, depth,
                          scale[0], scale[1], scale[2]);
            
            // Setup shader
            /*
            if (shader != NULL){
                shader->Load();

                shader->ApplyShader();

                //shader->SetTexture("intensityTex", intensityTex);
                //shader->SetTexture("doseTex", doseTex);

                shader->ReleaseShader();
            }
            */

            // print buffer objects
            /*
            logger.info << "Vertices:" << logger.end;
            for(int i = 0; i < numberOfVertices; ++i){
                int index = i * DIMENSIONS;
                Vector<3, float> hat = Vector<3, float>(vertices + index);
                logger.info << "Vertice " << i << " is << " << hat.ToString() << logger.end;
            }

            logger.info << "TexCoords:" << logger.end;
            for(int i = 0; i < numberOfVertices; ++i){
                int index = i * TEXCOORDS;
                Vector<3, float> hat = Vector<3, float>(texCoords + index);
                logger.info << "TexCoord " << i << " is << " << hat.ToString() << logger.end;
            }
            */
                
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
            
            SetupXPlane();
        }

        void DoseCalcNode::SetYPlaneCoord(int y) {
            if (y < 0)
                yPlaneCoord = 0;
            else if (y > height - 1)
                yPlaneCoord = height - 1;
            else
                yPlaneCoord = y;

            SetupYPlane();
        }

        void DoseCalcNode::SetZPlaneCoord(int z) {
            if (z < 0)
                zPlaneCoord = 0;
            else if (z > depth-1)
                zPlaneCoord = depth-1;
            else
                zPlaneCoord = z;

            SetupZPlane();
        }

        void DoseCalcNode::AddBeam(Beam beam){
            beams.push_back(beam);
        }

        // *** inline methods ***

        void DoseCalcNode::SetupVertices(){
            vertices = new float[numberOfVertices * DIMENSIONS];
            texCoords = new float[numberOfVertices * TEXCOORDS];
            
            SetupXPlane();
            SetupYPlane();
            SetupZPlane();
        }
        
        void DoseCalcNode::SetupXPlane(){
            glBindBuffer(GL_ARRAY_BUFFER, verticeId);
            float* vbo = (float*) glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);

            // Setup vertices
            int offset = 0 * DIMENSIONS;
            vbo[offset + 0] = vertices[offset + 0] = xPlaneCoord;
            vbo[offset + 1] = vertices[offset + 1] = 0;
            vbo[offset + 2] = vertices[offset + 2] = 0;

            offset += DIMENSIONS;
            vbo[offset + 0] = vertices[offset + 0] = xPlaneCoord;
            vbo[offset + 1] = vertices[offset + 1] = 0;
            vbo[offset + 2] = vertices[offset + 2] = depth-1;

            offset += DIMENSIONS;
            vbo[offset + 0] = vertices[offset + 0] = xPlaneCoord;
            vbo[offset + 1] = vertices[offset + 1] = height-1;
            vbo[offset + 2] = vertices[offset + 2] = depth-1;

            offset += DIMENSIONS;
            vbo[offset + 0] = vertices[offset + 0] = xPlaneCoord;
            vbo[offset + 1] = vertices[offset + 1] = height-1;
            vbo[offset + 2] = vertices[offset + 2] = 0;

            glUnmapBuffer(GL_ARRAY_BUFFER);

            // Setup texture coords
            glBindBuffer(GL_ARRAY_BUFFER, texCoordId);
            vbo = (float*) glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
            for (int i = 0 * TEXCOORDS; i < (0 + 4) * TEXCOORDS; i += TEXCOORDS){
                vbo[i] = texCoords[i] = (vertices[i] + 0.5f) / width;
                vbo[i+1] = texCoords[i+1] = (vertices[i+1] + 0.5f) / height;
                vbo[i+2] = texCoords[i+2] = (vertices[i+2] + 0.5f) / depth;
            }

            glUnmapBuffer(GL_ARRAY_BUFFER);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
        }

        void DoseCalcNode::SetupYPlane(){
            glBindBuffer(GL_ARRAY_BUFFER, verticeId);
            float* vbo = (float*) glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);

            // Setup vertices
            int offset = 4 * DIMENSIONS;
            vbo[offset + 0] = vertices[offset + 0] = 0;
            vbo[offset + 1] = vertices[offset + 1] = yPlaneCoord;
            vbo[offset + 2] = vertices[offset + 2] = 0;

            offset += DIMENSIONS;
            vbo[offset + 0] = vertices[offset + 0] = width-1;
            vbo[offset + 1] = vertices[offset + 1] = yPlaneCoord;
            vbo[offset + 2] = vertices[offset + 2] = 0;

            offset += DIMENSIONS;
            vbo[offset + 0] = vertices[offset + 0] = width-1;
            vbo[offset + 1] = vertices[offset + 1] = yPlaneCoord;
            vbo[offset + 2] = vertices[offset + 2] = depth-1;

            offset += DIMENSIONS;
            vbo[offset + 0] = vertices[offset + 0] = 0;
            vbo[offset + 1] = vertices[offset + 1] = yPlaneCoord;
            vbo[offset + 2] = vertices[offset + 2] = depth-1;

            glUnmapBuffer(GL_ARRAY_BUFFER);

            // Setup texture coords
            glBindBuffer(GL_ARRAY_BUFFER, texCoordId);
            vbo = (float*) glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
            for (int i = 4 * TEXCOORDS; i < (4 + 4) * TEXCOORDS; i += TEXCOORDS){
                vbo[i] = texCoords[i] = (vertices[i] + 0.5f) / width;
                vbo[i+1] = texCoords[i+1] = (vertices[i+1] + 0.5f) / height;
                vbo[i+2] = texCoords[i+2] = (vertices[i+2] + 0.5f) / depth;
            }

            glUnmapBuffer(GL_ARRAY_BUFFER);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
        }

        void DoseCalcNode::SetupZPlane(){
            glBindBuffer(GL_ARRAY_BUFFER, verticeId);
            float* vbo = (float*) glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);

            // Setup vertices
            int offset = 8 * DIMENSIONS;
            vbo[offset + 0] = vertices[offset + 0] = 0;
            vbo[offset + 1] = vertices[offset + 1] = 0;
            vbo[offset + 2] = vertices[offset + 2] = zPlaneCoord;

            offset += DIMENSIONS;
            vbo[offset + 0] = vertices[offset + 0] = width-1;
            vbo[offset + 1] = vertices[offset + 1] = 0;
            vbo[offset + 2] = vertices[offset + 2] = zPlaneCoord;

            offset += DIMENSIONS;
            vbo[offset + 0] = vertices[offset + 0] = width-1;
            vbo[offset + 1] = vertices[offset + 1] = height-1;
            vbo[offset + 2] = vertices[offset + 2] = zPlaneCoord;

            offset += DIMENSIONS;
            vbo[offset + 0] = vertices[offset + 0] = 0;
            vbo[offset + 1] = vertices[offset + 1] = height-1;
            vbo[offset + 2] = vertices[offset + 2] = zPlaneCoord;

            glUnmapBuffer(GL_ARRAY_BUFFER);

            // Setup texture coords
            glBindBuffer(GL_ARRAY_BUFFER, texCoordId);
            vbo = (float*) glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
            for (int i = 8 * TEXCOORDS; i < (8 + 4) * TEXCOORDS; i += TEXCOORDS){
                vbo[i] = texCoords[i] = (vertices[i] + 0.5f) / width;
                vbo[i+1] = texCoords[i+1] = (vertices[i+1] + 0.5f) / height;
                vbo[i+2] = texCoords[i+2] = (vertices[i+2] + 0.5f) / depth;
            }

            glUnmapBuffer(GL_ARRAY_BUFFER);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
        }

        int DoseCalcNode::GetIndex(int x, int y, int z){
            return z + x * depth + y * width * depth;
        }        
    }
}
