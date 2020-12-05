cmake_minimum_required(VERSION 3.1)

PROJECT(maskDetector)
set(CMAKE_CXX_STANDARD 14)

cmake_policy(SET CMP0072 NEW) # to use GLVND (Nvidia-vendor OpenGL impl) 

find_package( OpenCV 4.5.0 REQUIRED )
find_package( Threads REQUIRED )
find_package( OpenGL REQUIRED )

include_directories(${OpenCV_INCLUDE_DIRS} ${OPENGL_INCLUDE_DIR})

ADD_EXECUTABLE(${PROJECT_NAME} src/main.cpp src/utilities.cpp)
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
TARGET_LINK_LIBRARIES(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} Threads::Threads OpenGL::GL)
