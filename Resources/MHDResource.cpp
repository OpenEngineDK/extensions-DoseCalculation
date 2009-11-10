// MHD medical 3d texture image resource.
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// Modified by Anders Bach Nielsen <abachn@daimi.au.dk> - 21. Nov 2007
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Resources/MHDResource.h>
#include <Resources/Exceptions.h>
#include <Resources/File.h>
#include <Utils/Convert.h>

#include <stdio.h>

namespace OpenEngine {
namespace Resources {

using namespace std;

MHDResourcePlugin::MHDResourcePlugin() {
    this->AddExtension("mhd");
}

ITexture3DResourcePtr MHDResourcePlugin::CreateResource(string file) {
    return ITexture3DResourcePtr(new MHDResource(file));
}

MHDResource::MHDResource(string filename)
    : loaded(false),
      filename(filename),
      data(NULL) {
    space_x = space_y = space_z = dim_x = dim_y = dim_z = id = 0;
}

MHDResource::~MHDResource() {
    Unload();
}

void MHDResource::Error(int line, string msg) {
    logger.warning << filename << " line[" << line << "] " << msg << "." << logger.end;
}

void MHDResource::Load() {
    if (loaded) return;
    string raw_dir = File::Parent(this->filename);
    char buf[255], tmp[255];
    string rawfile;
    unsigned int line;
    short* s_data;
    col_depth = sizeof(float);
    line = dim_x = dim_y = dim_z = 0;
    ifstream* in = File::Open(filename);

    while (!in->eof()) {
        ++line;
        in->getline(buf,255);

        if (string(buf,7) == "DimSize") {
            if (sscanf(buf, "DimSize = %d %d %d", &dim_x, &dim_y, &dim_z) != 3)
                Error(line, "Invalid DimSize");
        }
        else if (string(buf,14) == "ElementSpacing") {
            if (sscanf(buf, "ElementSpacing = %f %f %f", &space_x, &space_y, &space_z) != 3)
                Error(line, "Invalid ElementSpacing");
        }
        else if (string(buf,11) == "ElementType") {
            if (sscanf(buf, "ElementutType = %s ", tmp) != 1)
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
    if (dim_x == 0 || dim_y == 0 || dim_z == 0) 
        throw new ResourceException("Dimensions missing.");
    if (rawfile.empty()) 
        throw new ResourceException("Raw file missing.");
    
    data = new float[dim_x * dim_y * dim_z],
    s_data = new short[dim_x * dim_y * dim_z];
    FILE* pFile = fopen (rawfile.c_str(), "rb");
    if (pFile == NULL) throw Exception("Raw file not found.");
	size_t count = fread (s_data, 2, dim_x * dim_y * dim_z, pFile);
    fclose(pFile);
    if (count != dim_x * dim_y * dim_z) throw new ResourceException("Raw file read error."); 
    
    for(unsigned int i=0; i < dim_x * dim_y * dim_z; i++)
		data[i] = (float)s_data[i];
	delete [] s_data;
    loaded = true;
}

void MHDResource::Unload() {
    if (loaded) {
        if (data) {
            delete[] data;
            data = NULL;
        }
        loaded = false;
    }
}

int MHDResource::GetID() {
    return id;
}

void MHDResource::SetID(int id) {
    this->id = id;
}	

unsigned int MHDResource::GetWidth() {
  return dim_x;
}

unsigned int MHDResource::GetHeight() {
    return dim_y;
}

unsigned int MHDResource::GetDepth() {
  return dim_z;
}

unsigned int MHDResource::GetColorDepth() {
  return col_depth;
}

float* MHDResource::GetData() {
  return data;
}

float MHDResource::GetWidthScale() {
    return space_x;
}

float MHDResource::GetHeightScale() {
    return space_y;
}

float MHDResource::GetDepthScale() {
    return space_z;
}

ColorFormat MHDResource::GetColorFormat() {
    if (col_depth==32)
        return RGBA;
    else if (col_depth==24)
        return RGB;
    else if (col_depth==8)
        return LUMINANCE;
    else
        throw Exception("unknown color depth");
}

} //NS Resources
} //NS OpenEngine
