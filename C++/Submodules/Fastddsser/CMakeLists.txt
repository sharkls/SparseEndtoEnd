cmake_minimum_required(VERSION 3.10)

project(FastddsSer VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/Output/Lib)

aux_source_directory(./data SRC_FILES)
aux_source_directory(./param SRC_FILES)
add_library(${PROJECT_NAME} SHARED ${SRC_FILES})

target_link_libraries(${PROJECT_NAME} fastcdr)
target_link_libraries(${PROJECT_NAME} fastrtps)
target_link_libraries(${PROJECT_NAME} tinyxml2)