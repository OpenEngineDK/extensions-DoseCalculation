SET ( EXTENSION_NAME "Extensions_DoseCalculation")

CUDA_ADD_LIBRARY("CUDA_KERNELS"
  Renderers/CUDA/RayCaster.cu
)

# Create the extension library
ADD_LIBRARY( ${EXTENSION_NAME}
  Renderers/OpenGL/DoseCalcRenderingView.h
  Renderers/OpenGL/DoseCalcRenderingView.cpp
  Renderers/OpenGL/RayCastRenderingView.h
  Renderers/OpenGL/RayCastRenderingView.cpp
  Scene/DoseCalcNode.h
  Scene/DoseCalcNode.cpp
)

TARGET_LINK_LIBRARIES( ${EXTENSION_NAME}
  OpenEngine_Scene
  OpenEngine_Renderers
  OpenEngine_Resources
  Extensions_OpenGLRenderer
  Extensions_OpenGLSelection
#  Extensions_TemplatedTextures
  ${OPENGL_LIBRARY}
  ${GLEW_LIBRARIES}
  ${SDL_LIBRARY}
  "CUDA_KERNELS"
)
