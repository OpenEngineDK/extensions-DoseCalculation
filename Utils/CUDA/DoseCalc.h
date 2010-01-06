#ifndef _DOSE_RAY_CASTER_H_
#define _DOSE_RAY_CASTER_H_

#include <Scene/Beam.h>

using OpenEngine::Scene::Beam;

typedef unsigned char uchar;
typedef unsigned int  uint;

void SetupDoseCalc(float** cuDoseArr, 
                   int w, int h, int d, 
                   float sw, float sh, float sd,
                   uint bw = 1, uint bh = 1);

void RunDoseCalc(float* cuDoseArr, Beam beam, int beamlet_x, int beamlet_y, int kernel = 1);

void Dose(float** out,
          Beam oebeam,
          Beam voi,
          unsigned char* fmap, 
          uint beamlets_x, uint beamlets_y,
          int w, int h, int d,
          float sw, float sh, float sd);


#endif
