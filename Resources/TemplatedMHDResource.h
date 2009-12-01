// Templated MHD medical 3d texture image resource.
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// Modified by Anders Bach Nielsen <abachn@daimi.au.dk> - 21. Nov 2007
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _TEMPLATED_MHD_RESOURCE_H_
#define _TEMPLATED_MHD_RESOURCE_H_

#include <Resources/ITexture3D.h>
#include <Resources/IResourcePlugin.h>
#include <Resources/File.h>
#include <string>
#include <iostream>
#include <fstream>

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/split_member.hpp>
#include <Logging/Logger.h>

namespace OpenEngine {
    namespace Resources {

        using namespace std;

        /**
         * MHD 3D texture medical resource.
         *
         * @class TemplatedMHDResource TemplatedMHDResource.h Resources/TemplatedMHDResource.h
         */
        template <class T> class TemplatedMHDResource : public ITexture3D<T> {
        private:
            string filename;            //!< file name
            float space_x, space_y, space_z;
            void Error(int line, string msg){
                logger.warning << filename << " line[" << line << "] " << msg << "." << logger.end;
            }
        public:

            explicit TemplatedMHDResource() 
                : ITexture3D<T>() {
                space_x = space_y = space_z = 0;
            };

            TemplatedMHDResource(string file)
                : ITexture3D<T>() {
                filename = file;
                space_x = space_y = space_z = 0;
            }

            ~TemplatedMHDResource() {}

            // resource methods
            void Load(){
                // Return if data is already loaded.
                if (ITexture<T>::data) return;
                
                string raw_dir = File::Parent(this->filename);
                char buf[255], tmp[255];
                string rawfile;
                unsigned int line = 0;
                this->width = this->height = this->depth = 0;

                ifstream* in = File::Open(filename);
                while (!in->eof()) {
                    ++line;
                    in->getline(buf,255);

                    if (string(buf,7) == "DimSize") {
                        if (sscanf(buf, "DimSize = %d %d %d", &this->width, &this->height, &this->depth) != 3)
                            Error(line, "Invalid DimSize");
                    }
                    else if (string(buf,14) == "ElementSpacing") {
                        if (sscanf(buf, "ElementSpacing = %f %f %f", &space_x, &space_y, &space_z) != 3)
                            Error(line, "Invalid ElementSpacing");
                    }
                    else if (string(buf,11) == "ElementType") {
                        if (sscanf(buf, "ElementType = %s ", tmp) != 1)
                            Error(line, "Invalid ElementType");
                        else {
                            if (string(tmp) == "MET_SHORT") {}
                            else Error(line, "Unsupported element type.");
                        }
                    }
                    else if (string(buf,15) == "ElementDataFile") {
                        if (sscanf(buf, "ElementDataFile = %s ", tmp) != 1)
                            Error(line, "Invalid ElementDataFile");
                        else rawfile = raw_dir + string(tmp);
                    }
                }
                in->close();

                this->size = this->width * this->height * this->depth;

                if (this->size == 0) 
                    throw new ResourceException("Dimensions missing.");
                if (rawfile.empty()) 
                    throw new ResourceException("Raw file missing.");
    
                this->data = new float[this->size];
                short* s_data = new short[this->size];
                FILE* pFile = fopen (rawfile.c_str(), "rb");
                if (pFile == NULL) throw Exception("Raw file not found.");
                size_t count = fread (s_data, 2, this->size, pFile);
                fclose(pFile);
                if (count != ITexture<T>::size) throw new ResourceException("Raw file read error."); 
                for(unsigned int i=0; i < this->size; i++)
                    //data[i] = (float)s_data[i];
                    this->data[i] = (((T)s_data[i]) + 1000.0f) / 2000.0f;
                delete [] s_data;
                
                this->format = OE_LUMINANCE;
            }

            void Unload(){
                if (this->data) delete [] ITexture<T>::data;
                ITexture3D<T>::width = ITexture3D<T>::height = ITexture3D<T>::depth = space_x = space_y = space_z = 0;
            }

            // texture resource methods
            float GetWidthScale() const { return space_x; }
            float GetHeightScale() const { return space_y; }
            float GetDepthScale() const { return space_z; }

            // Inherited virtual functions
            unsigned char GetChannels() const { return 1; }
            bool UseMipmapping() const { return false; }


            template<class Archive>
            void serialize(Archive& ar, const unsigned int version) {
                ar & boost::serialization::base_object<ITextureResource>(*this);
                ar & filename;
            }

        };

        /**
         * MHD texture resource plug-in.
         *
         * @class TemplatedMHDResourcePlugin TemplatedMHDResource.h Resources/TemplatedMHDResource.h
         */
        template <class T> class TemplatedMHDResourcePlugin : public IResourcePlugin<ITexture3D<T> > {
        public:
            TemplatedMHDResourcePlugin(){
                this->AddExtension("mhd");
            }
            
            boost::shared_ptr<ITexture3D<T> > CreateResource(string file){
            //ITexture3D<T>::Ptr CreateResource(string file){
                return boost::shared_ptr<ITexture3D<T> >(new TemplatedMHDResource<T>(file));
            }
        };

    }
}

BOOST_CLASS_EXPORT(OpenEngine::Resources::MHDResources)

#endif
