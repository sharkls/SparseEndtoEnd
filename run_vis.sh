#!/bin/bash
# 自动设置环境并运行可视化脚本

# 设置颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${GREEN}Running Visualization Script...${NC}"

# 运行脚本
python3 script/tutorial/040.visualize_inference_result.py \
  --data_dir /share/Code/Sparse4dE2E/deploy/val_data_e2e_fp32 \
  --plugin_dir /share/Code/Sparse4dE2E/deploy

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Success! Output files: vis_frame0_bev.png, vis_frame1_bev.png${NC}"
else
    echo -e "${RED}Failed to run script.${NC}"
    exit 1
fi

