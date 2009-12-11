#ifndef _DOSE_RAY_CASTER_H_
#define _DOSE_RAY_CASTER_H_

#include <Scene/Beam.h>

using OpenEngine::Scene::Beam;

void SetupDoseCalc(float** cuDoseArr, int w, int h, int d, float sw, float sh, float sd);

void RunDoseCalc(float* cuDoseArr, Beam beam, int beamlet_x, int beamlet_y, float dx, float dy, float dz);

#endif
