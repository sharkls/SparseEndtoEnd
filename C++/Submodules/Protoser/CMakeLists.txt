cmake_minimum_required(VERSION 3.10)

project(ProtoSer VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/Output/Lib)

# aux_source_directory(./data SRC_FILES)
aux_source_directory(./param SRC_FILES)
add_library(${PROJECT_NAME} SHARED ${SRC_FILES})

target_link_libraries(${PROJECT_NAME} protobuf)
target_link_libraries(${PROJECT_NAME} absl_log_internal_check_op)
target_link_libraries(${PROJECT_NAME} absl_log_internal_message)