SET ( EXTENSION_NAME "Extensions_DoseCalculation")

CUDA_ADD_LIBRARY("CUDA_KERNELS"
  Renderers/CUDA/RayCaster.cu
  Utils/CUDA/DoseCalc.cu
  Utils/CUDA/DozeSetup.cu
#  Utils/CUDA/Superposition.cu
)

TARGET_LINK_LIBRARIES("CUDA_KERNELS"
  ${CUDA_CUT_LIBRARIES} #cutil
)  

MESSAGE( " cut: ${CUDA_CUT_LIBRARIES} ")#cutil

# Create the extension library
ADD_LIBRARY( ${EXTENSION_NAME}
  Renderers/OpenGL/DoseCalcSelectionRenderer.h
  Renderers/OpenGL/DoseCalcSelectionRenderer.cpp
  Renderers/OpenGL/DoseCalcRenderingView.h
  Renderers/OpenGL/DoseCalcRenderingView.cpp
  Renderers/OpenGL/RayCastRenderingView.h
  Renderers/OpenGL/RayCastRenderingView.cpp
  Scene/DoseCalcNode.h
  Scene/DoseCalcNode.cpp
  Scene/BeamNode.h
  Scene/BeamNode.cpp
  Scene/Beam.h
  Scene/Beam.cpp
  Utils/DoseTrigger.h
)

TARGET_LINK_LIBRARIES( ${EXTENSION_NAME}
  OpenEngine_Scene
  OpenEngine_Renderers
  OpenEngine_Resources
  Extensions_OpenGLRenderer
  Extensions_OpenGLSelection
  ${OPENGL_LIBRARY}
  ${GLEW_LIBRARIES}
  ${SDL_LIBRARY}
  "CUDA_KERNELS"
)
