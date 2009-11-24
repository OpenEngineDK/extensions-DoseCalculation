// 3D Texture resource.
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _TEXTURE_3D_RESOURCE_H_
#define _TEXTURE_3D_RESOURCE_H_

#include <Resources/ITexture3DResource.h>

namespace OpenEngine {
    namespace Resources {
        
        class CopyTexture3DResource : public ITexture3DResource {
        private:
            ITexture3DResourcePtr src;
            unsigned int width, height, depth, colorDepth;
            float widthScale, heightScale, depthScale;
            float* data;
            int id;
            ColorFormat format;
        public:
            CopyTexture3DResource(ITexture3DResourcePtr src, ColorFormat format);
            ~CopyTexture3DResource() { delete [] data; }
            void Load();
            void Unload() { delete [] data; }
            int GetID() { return id; }
            void SetID(int id) { this->id = id; }
            unsigned int GetWidth() { return width; }            
            unsigned int GetHeight() { return height; }
            unsigned int GetDepth() { return depth; }
            float GetWidthScale() { return widthScale; }
            float GetHeightScale() { return heightScale; }
            float GetDepthScale() { return depthScale; }
            unsigned int GetColorDepth() { return colorDepth; }
            float* GetData() { return data; }
            ColorFormat GetColorFormat() { return format; }
        };
    }
}

#endif
