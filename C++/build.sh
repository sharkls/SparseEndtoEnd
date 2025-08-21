#!/bin/bash

# 删除旧的 build 文件夹
rm -rf build

# 创建新的 build 文件夹
mkdir build

# 进入 build 文件夹
cd build

# 运行 cmake
cmake ..

# 多线程编译
make

# cd ../Output/
# ./testAlgLib