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

#include <Resources/Texture3D.h>
#include <Resources/IResourcePlugin.h>
#include <Resources/Exceptions.h>
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
        template <class T> class TemplatedMHDResource : public Texture3D<T> {
        private:
            string filename;            //!< file name
            float space_x, space_y, space_z;
            void Error(int line, string msg){
                logger.warning << filename << " line[" << line << "] " << msg << "." << logger.end;
            }
        public:

            explicit TemplatedMHDResource() 
                : Texture3D<T>() {
                space_x = space_y = space_z = 0;
            };

            TemplatedMHDResource(string file)
                : Texture3D<T>() {
                filename = file;
                space_x = space_y = space_z = 0;
            }

            ~TemplatedMHDResource() {}

            // resource methods
            void Load(){
                // Return if data is already loaded.
                if (this->data) return;

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

                unsigned int b_width = this->width;
                unsigned int b_height = this->height;
                unsigned int b_depth = this->depth;
                unsigned int b_size = b_width * b_height * b_depth;
               
                const int scl = 5;
                
                this->space_x *= scl;
                this->space_y *= scl;
                this->space_z *= scl;
                this->width /= scl;
                this->height /= scl;
                this->depth /= scl;
                unsigned int size = this->width * this->height * this->depth;
                
                if (size == 0) 
                    throw ResourceException("Dimensions missing.");
                if (rawfile.empty()) 
                    throw ResourceException("Raw file missing.");
    
                this->data = new float[size];
                T* data = (T*) this->data;
                short* s_data = new short[b_size];
                FILE* pFile = fopen (rawfile.c_str(), "rb");
                if (pFile == NULL) throw Exception("Raw file not found.");
                size_t count = fread (s_data, 2, b_size, pFile);
                fclose(pFile);
                if (count != b_size) throw new ResourceException("Raw file read error."); 
                for(unsigned int i=0; i < this->width; i++) {
                    for(unsigned int j=0; j < this->height; j++) {
                        for(unsigned int k=0; k < this->depth; k++) {
                            unsigned int dataEntry = k * this->width * this->height + j * this->width + i;
                            unsigned int srcEntry = k * scl * b_width * b_height + j * scl * b_width + i * scl;
                            data[dataEntry] = (((T)s_data[srcEntry]) + 1000.0f)/2000.0f;
                        }
                    }
                }
                delete[] s_data;
                this->format = LUMINANCE;
            }

            void Unload(){
                if (this->data) {
                    T* data = (T*) this->data;
                    delete [] data;
                }
            }

            // texture resource methods
            float GetWidthScale() const { return space_x; }
            float GetHeightScale() const { return space_y; }
            float GetDepthScale() const { return space_z; }

            // Inherited virtual functions
            bool UseMipmapping() const { return false; }
        };

#define MHD(T) TemplatedMHDResource<T>
#define MHDPtr(T) boost::shared_ptr<TemplatedMHDResource<T> >

        /**
         * MHD texture resource plug-in.
         *
         * @class TemplatedMHDResourcePlugin TemplatedMHDResource.h Resources/TemplatedMHDResource.h
         */
        template <class T> class TemplatedMHDResourcePlugin : public IResourcePlugin<MHD(T) > {
        public:
            TemplatedMHDResourcePlugin(){
                this->AddExtension("mhd");
            }
            
            MHDPtr(T) CreateResource(string file){
                return MHDPtr(T)(new MHD(T)(file));
            }
        };

    }
}

#endif
