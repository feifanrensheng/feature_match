cmake_minimum_required( VERSION 2.8 )
project( vo1 )

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )
set( OpenCV_DIR "/usr/local/opencv-3.1.0/share/OpenCV")


find_package( OpenCV REQUIRED )
# find_package( OpenCV REQUIRED ) # use this if in OpenCV2 


include_directories( 
    ${OpenCV_INCLUDE_DIRS} 
    "/usr/include/eigen3/"
)

add_executable( main main.cpp )
target_link_libraries( main ${OpenCV_LIBS})
