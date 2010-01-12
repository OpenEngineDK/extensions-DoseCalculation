#ifndef _DOSE_RAY_CASTER_H_
#define _DOSE_RAY_CASTER_H_



void SetupRayCaster(int pbo, const float* data, int w, int h, int d, float sw, float sh, float sd);

void RenderToPBO(int pbo, float* cuDoseArr, int width, int height, float* iva, float pm00, float pm11, float minIt, float maxIt, bool doSlice, float colorScale);

#endif
