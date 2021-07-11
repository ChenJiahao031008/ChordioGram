cmake_minimum_required(VERSION 2.8)
project( TESTFILES )
set(CMAKE_CXX_FLAGS "-std=c++11")

set(CMAKE_BUILD_TYPE Release)

find_package( OpenCV 3.4 REQUIRED)

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
add_executable(main main.cpp)
target_link_libraries( main ${OpenCV_LIBS})
