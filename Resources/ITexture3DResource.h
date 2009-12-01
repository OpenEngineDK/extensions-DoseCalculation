// 3D Texture resource interface.
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _I_TEXTURE_3D_RESOURCE_H_
#define _I_TEXTURE_3D_RESOURCE_H_

#include <Resources/IResource.h>
#include <Resources/ITextureResource.h>
#include <Resources/ITexture3D.h>

namespace OpenEngine {
    namespace Resources {

        // Forward decleration
        class ITexture3DResource;

        /**
         * Texture resource smart pointer.
         */
        typedef boost::shared_ptr<ITexture3DResource> ITexture3DResourcePtr;

        /**
         * Texture change event argument.
         * Contains a pointer to the texture that changed.
         *
         * @class Texture3DChangedEventArg ITexture3DResource.h Resource/ITexture3DResource.h
         */
        class DeprTexture3DChangedEventArg {
        public:
            DeprTexture3DChangedEventArg(ITexture3DResourcePtr resource) : resource(resource) {}
            ITexture3DResourcePtr resource;
        };

        /**
         * Texture resource interface.
         *
         * @class ITexture3DResource ITexture3DResource.h Resources/ITexture3DResource.h
         */
        class ITexture3DResource : public IResource<DeprTexture3DChangedEventArg> {
        public:
            
            /**
             * Get texture id.
             *
             * @return Texture id.
             */
            virtual int GetID() = 0;
	
            /**
             * Set texture id.
             *
             * @param id Texture id.
             */
            virtual void SetID(int id) = 0;

            /**
             * Get width in pixels on loaded texture.
             *
             * @return width in pixels.
             */
            virtual unsigned int GetWidth() = 0;

            /**
             * Get the scale of the width.
             *
             * @return scale of the width.
             */
            virtual float GetWidthScale() = 0;

            /**
             * Get height in pixels on loaded texture.
             *
             * @return height in pixels.
             */
            virtual unsigned int GetHeight() = 0;

            /**
             * Get the scale of the height.
             *
             * @return scale of the height.
             */
            virtual float GetHeightScale() = 0;

            /**
             * Get depth in pixels on loaded texture.
             *
             * @return depth in pixels.
             */
            virtual unsigned int GetDepth() = 0;

            /**
             * Get the scale of the depth.
             *
             * @return scale of the depth.
             */
            virtual float GetDepthScale() = 0;

            /**
             * Get color depth on loaded texture.
             *
             * @return Color depth in pixels.
             */
            virtual unsigned int GetColorDepth() = 0;

            /**
             * Get pointer to loaded texture.
             *
             * @return Char pointer to loaded data.
             */
            virtual float* GetData() = 0;

            /**
             * Get color format of the texture.
             *
             * @return ColorFormat representing the way colors are stored
             */
            virtual ColorFormat GetColorFormat() = 0;
        
            //! Serialization support
            template<class Archive>
            void serialize(Archive& ar, const unsigned int version) {
                ;
            }
        };

    }
}

#endif
