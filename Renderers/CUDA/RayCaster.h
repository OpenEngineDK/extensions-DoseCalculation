#ifndef _DOSE_RAY_CASTER_H_
#define _DOSE_RAY_CASTER_H_



void SetupRayCaster(int pbo, const float* data, int w, int h, int d);

void RenderToPBO(int pbo, int width, int height);

#endif
