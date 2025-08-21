#! /bin/bash

# 创建data目录（如果不存在）
mkdir -p ../data

# 清理data目录中的文件
rm -f ../data/*

# 按依赖顺序生成IDL文件
# 1. 首先生成基础类CDataBase
echo "Generating CDataBase.idl..."
fastddsgen CDataBase.idl -d ../data -replace

# 2. 然后生成依赖CDataBase的文件
echo "Generating CTimeMatchSrcData.idl..."
fastddsgen CTimeMatchSrcData.idl -d ../data -replace

echo "Generating CAlgResult.idl..."
fastddsgen CAlgResult.idl -d ../data -replace

echo "IDL generation completed!"

