cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project( TESTFILES )
set(CMAKE_CXX_FLAGS "-std=c++11")


SET(CMAKE_BUILD_TYPE Release)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

find_package( OpenCV 3 REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(CMAKE_PREFIX_PATH /home/chen/DeepLearning/settings/libtorch/)
find_package(Torch REQUIRED)

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_executable(capture tools/capture.cpp)
target_link_libraries( capture ${OpenCV_LIBS})

add_executable(main main.cpp)
target_link_libraries( main ${OpenCV_LIBS} ${TORCH_LIBRARIES})
set_property(TARGET main PROPERTY CXX_STANDARD 14)
