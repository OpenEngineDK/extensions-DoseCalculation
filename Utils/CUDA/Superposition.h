#ifndef _DOSE_SUPERPOSITION_H_
#define _DOSE_SUPERPOSITION_H_

#include <Scene/Beam.h>

typedef unsigned int uint;

using OpenEngine::Scene::Beam;

void Dose(float** out,
          Beam oebeam,
          Beam voi,
          unsigned char* fmap, 
          uint beamlets_x, uint beamlets_y,
          int w, int h, int d,
          float sw, float sh, float sd);

#endif
