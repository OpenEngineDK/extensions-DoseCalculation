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
            unsigned char* data;
            int id;
        public:
            EmptyTexture3DResource(unsigned int w, unsigned int h, unsigned int d, unsigned int cd) 
                : width(w), height(h), depth(d), colorDepth(cd), data(NULL), id(0) {
            }
            ~EmptyTexture3DResource() {delete [] data; }
            void Load() { if (!data) data = new unsigned char[width * height * depth * colorDepth / 8]; }
            void Unload() { delete data; }
            int GetID() { return id; }
            void SetID(int id) { this->id = id; }
            unsigned int GetWidth() { return width; }
            unsigned int GetHeight() { return height; }
            unsigned int GetDepth() { return depth; }
            unsigned int GetColorDepth() { return colorDepth; }
            unsigned char* GetData() { return data; }
            ColorFormat GetColorFormat() { 
                switch (depth){
                case 8: return LUMINANCE;
                case 24: return RGB;
                    //case 32: return RGBA;
                default: return RGBA;
                }
            }
        };

    }
}

#endif
