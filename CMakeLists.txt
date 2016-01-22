cmake_minimum_required(VERSION 3.3)
project(pcl_extract_dominant_plane)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/bin)

find_package(PCL 1.7 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

include_directories(${CMAKE_HOME_DIRECTORY}/include)

set(SOURCE_FILES main.cpp)

target_link_libraries(pcl_extract_dominant_plane ${PCL_LIBRARIES})

add_executable(pcl_extract_dominant_plane ${SOURCE_FILES})