SET ( EXTENSION_NAME "Extensions_DoseCalculation")
# Create the extension library
ADD_LIBRARY( ${EXTENSION_NAME}
  Renderers/OpenGL/DoseCalcRenderingView.cpp
  Scene/DoseCalcNode.cpp
  Resources/MHDResource.h
  Resources/MHDResource.cpp
)

TARGET_LINK_LIBRARIES( ${EXTENSION_NAME}
  OpenEngine_Scene
  OpenEngine_Renderers
  OpenEngine_Resources
  Extensions_OpenGLRenderer
  ${OPENGL_LIBRARY}
  ${GLEW_LIBRARIES}
  ${SDL_LIBRARY}

)
