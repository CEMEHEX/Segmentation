cmake_minimum_required(VERSION 2.8.12)
set (CMAKE_CXX_STANDARD 11)

project(watershed)

file(GLOB_RECURSE watershed_SOURCES "src/*.cpp")
file(GLOB_RECURSE watershed_HEADERS "src/*.h")

set (watershed_INCLUDE_DIRS "")
foreach (_headerFile ${watershed_HEADERS})
    get_filename_component(_dir ${_headerFile} PATH)
    list (APPEND watershed_INCLUDE_DIRS ${_dir})
endforeach()
list(REMOVE_DUPLICATES watershed_INCLUDE_DIRS)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(watershed ${watershed_SOURCES})
target_include_directories(watershed PRIVATE ${watershed_INCLUDE_DIRS})
target_link_libraries(watershed ${OpenCV_LIBS})
