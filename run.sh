# 重新编译所有插件
bash ./build_plugin.sh

# 生成插件的输入输出数据
cd /share/Code/Sparse4dE2E
/bin/python3 script/tutorial/012.export_plugin_io_real_data.py 
    --samples 1 
    --save-dir deploy/val_data_plugin/real_data

# 生成e2e_fp32数据
cd /share/Code/Sparse4dE2E
/bin/python3 script/tutorial/050.generate_verification_data.py \
    --config dataset/config/sparse4d_temporal_r50_1x1_bs1_256x704_mini.py \
    --checkpoint ckpt/sparse4dv3_r50.pth \
    --save-dir deploy/val_data_e2e_fp32

# 验证backbone 在FP16和FP32精度下的与pytorch的推理误差
cd /share/Code/Sparse4dE2E/deploy
/bin/python3 validate_backbone_engine.py --precision all --data_dir val_data_e2e_fp32

# 验证head1 和 head2 在FP16和FP32精度下的与pytorch的推理误差
cd /share/Code/Sparse4dE2E/deploy
/bin/python3 validate_head_engine.py --head all --precision all --data_dir val_data_e2e_fp32  # 测试head1/2 在FP16和FP32精度下的差异

# 验证e2e 在FP16和FP32精度下的与pytorch的推理误差(使用GT Class Score进行TOPK筛选，验证engine推理是否有问题)
cd /share/Code/Sparse4dE2E
/bin/python3 script/tutorial/015.validate_e2e_mixed_precision.py \
    --data_dir deploy/val_data_e2e_fp32 \
    --plugin_dir deploy

# 验证e2e 在FP16和FP32精度下的与pytorch的推理误差(使用上一帧的预测结果进行TOPK筛选，端到端的engine推理是否有问题)
/bin/python3 script/tutorial/016.validate_e2e_pred_score.py \
    --data_dir deploy/val_data_e2e_fp32 \
    --plugin_dir deploy

# 可视化e2e的推理结果与真值
python3 -u script/tutorial/040.visualize_inference_result.py \
    --data_dir /share/Code/Sparse4dE2E/deploy/val_data_e2e_fp32 \
    --plugin_dir /share/Code/Sparse4dE2E/deploy

