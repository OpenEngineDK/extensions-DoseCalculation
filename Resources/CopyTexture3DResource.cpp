// 3D Texture resource.
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Resources/CopyTexture3DResource.h>

namespace OpenEngine {
    namespace Resources {
        
        CopyTexture3DResource::CopyTexture3DResource(ITexture3DResourcePtr src, ColorFormat format)
            : src(src), width(src->GetWidth()), height(src->GetHeight()), depth(src->GetDepth()), 
              widthScale(src->GetWidthScale()), heightScale(src->GetHeightScale()), 
              depthScale(src->GetDepthScale()), format(format) {
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
        
        void CopyTexture3DResource::Load(){
            data = new float[width * height * depth * colorDepth];
            
            // from luminance to rgb
            if (src->GetColorFormat() == LUMINANCE && format == RGB){
                for (unsigned int i = 0; i < width * height * depth; ++i){
                    int fromI = i;
                    int toI = i * colorDepth;
                    data[toI] = src->GetData()[fromI];
                    data[toI+1] = src->GetData()[fromI];
                    data[toI+2] = src->GetData()[fromI];
                }
            }
        }
        
    }
}
