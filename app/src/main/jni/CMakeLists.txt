#
# Copyright (C) 2016 The Android Open Source Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

cmake_minimum_required(VERSION 3.10.2)
set(CMAKE_VERBOSE_MAKEFILE on)

project(FACERECOGNIZER)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/opencv_arm64/include)
add_library(cvcore_arm64 SHARED IMPORTED )
set_target_properties(cvcore_arm64 PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/opencv_arm64/lib/libopencv_core.so)
add_library(cvimproc_arm64 SHARED IMPORTED )
set_target_properties(cvimproc_arm64 PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/opencv_arm64/lib/libopencv_imgproc.so)
add_library(cvvideo_arm64 SHARED IMPORTED )
set_target_properties(cvvideo_arm64 PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/opencv_arm64/lib/libopencv_video.so)
add_library(cvtracking_arm64 SHARED IMPORTED )
set_target_properties(cvtracking_arm64 PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/opencv_arm64/lib/libopencv_tracking.so)

#file(GLOB FR_ALIGN_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB FR_ALIGN_FILES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.hh"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    )

add_library( # Sets the name of the library.
        Fr-align-lib

        # Sets the library as a shared library.
        SHARED

        # Provides a relative path to your source file(s).
        ${FR_ALIGN_FILES})


target_link_libraries( # Specifies the target library.
        Fr-align-lib
        android
        log
        stdc++
        cvcore_arm64
        cvimproc_arm64
        cvvideo_arm64 cvtracking_arm64)

#add_subdirectory(tensorflow_demo)