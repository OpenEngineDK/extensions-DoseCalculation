// MHD medical 3d texture image resource.
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// Modified by Anders Bach Nielsen <abachn@daimi.au.dk> - 21. Nov 2007
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _MHD_RESOURCE_H_
#define _MHD_RESOURCE_H_

#include <Resources/ITexture3DResource.h>
#include <Resources/IResourcePlugin.h>
#include <string>
#include <iostream>
#include <fstream>

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/extended_type_info.hpp>
#include <Logging/Logger.h>

namespace OpenEngine {
namespace Resources {

using namespace std;

/**
 * MHD 3D texture medical resource.
 *
 * @class MHDResource MHDResource.h Resources/MHDResource.h
 */
class MHDResource : public ITexture3DResource {
private:
    bool loaded;
    int id;                     //!< material identifier
    string filename;            //!< file name
    float* data;        //!< binary material data
    unsigned int dim_x;         //!< texture width
    unsigned int dim_y;        //!< texture height
    unsigned int dim_z;         //!< texture depth/bits
    unsigned int col_depth;         //!< texture depth/bits
    float space_x, space_y, space_z;
    void Error(int line, string msg);
public:

    //    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive& ar, const unsigned int version) {
        ar & boost::serialization::base_object<ITextureResource>(*this);
        ar & filename;
    }

    explicit MHDResource() : loaded(false),data(NULL) {
        space_x = space_y = space_z = dim_x = dim_y = dim_z = id = 0;
    };

    MHDResource(string file);
    ~MHDResource();

    // resource methods
    void Load();
    void Unload();

    // texture resource methods
    int GetID();
    void SetID(int id);   
    unsigned int GetWidth();
    unsigned int GetHeight();
    unsigned int GetDepth();
    float GetWidthScale();
    float GetHeightScale();
    float GetDepthScale();
    unsigned int GetColorDepth();
    float* GetData();
    ColorFormat GetColorFormat();
};

/**
 * MHD texture resource plug-in.
 *
 * @class MHDResourcePlugin MHDResource.h Resources/MHDResource.h
 */
class MHDResourcePlugin : public IResourcePlugin<ITexture3DResource> {
public:
    MHDResourcePlugin();
    ITexture3DResourcePtr CreateResource(string file);
};

} //NS Resources
} //NS OpenEngine

BOOST_CLASS_EXPORT(OpenEngine::Resources::MHDResource)

#endif // _MHD_RESOURCE_H_
