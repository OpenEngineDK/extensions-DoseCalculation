// Empty 3D Texture resource.
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _EMPTY_TEXTURE_3D_RESOURCE_H_
#define _EMPTY_TEXTURE_3D_RESOURCE_H_

#include <Resources/ITexture3DResource.h>

namespace OpenEngine {
    namespace Resources {

        class EmptyTexture3DResource : public ITexture3DResource {
        private:
            unsigned int width, height, depth, colorDepth;
            //float widthScale, heightScale, depthScale;
            float* data;
            int id;
            ColorFormat format;
        public:
            EmptyTexture3DResource(unsigned int w, unsigned int h, unsigned int d, ColorFormat f) 
                : width(w), height(h), depth(d), data(NULL), id(0), format(f) {
                switch(format){
                case LUMINANCE:
                    colorDepth = 1;
                    break;
                case RGB:
                case BGR:
                    colorDepth = 3;
                    break;
                case RGBA:
                case BGRA:
                    colorDepth = 4;
                    break;
                }
            }
            ~EmptyTexture3DResource() {delete [] data; }
            void Load() { if (!data) data = new float[width * height * depth * colorDepth / 8]; }
            void Unload() { delete [] data; }
            int GetID() { return id; }
            void SetID(int id) { this->id = id; }
            unsigned int GetWidth() { return width; }            
            unsigned int GetHeight() { return height; }
            unsigned int GetDepth() { return depth; }
            float GetWidthScale() { return 1; }
            float GetHeightScale() { return 1; }
            float GetDepthScale() { return 1; }
            unsigned int GetColorDepth() { return colorDepth; }
            float* GetData() { return data; }
            ColorFormat GetColorFormat() { return format; }
        };

    }
}

#endif
