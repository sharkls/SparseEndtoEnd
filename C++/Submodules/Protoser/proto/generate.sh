#! /bin/bash
rm -f ../param/*

src_path="."  # 设置 src_path 为当前目录

# 遍历当前目录下的所有 .proto 文件
for proto_file in *.proto; do
    protoc -I${src_path}/ --cpp_out=../param "$proto_file"
done

# 获取 ../param 的绝对路径
output_path=$(realpath ../param)
echo "Generation successful, results saved in ${output_path}" 