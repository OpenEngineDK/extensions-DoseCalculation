#ifndef _DOSE_RAY_CASTER_H_
#define _DOSE_RAY_CASTER_H_

#include <Scene/Beam.h>

using OpenEngine::Scene::Beam;

void SetupDoseCalc(GLuint pbo, int w, int h, int d);

void RunDoseCalc(GLuint pbo, Beam beam, int beamlet_x, int beamlet_y, float dx, float dy, float dz);

#endif
