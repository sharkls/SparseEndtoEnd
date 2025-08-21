1. 数据预处理阶段 (Preprocessing)
1.1 原始图像数据
文件名: sample_X_ori_imgs_1*6*3*H*W_float32.bin
含义: 原始多视角图像数据
形状: [1, 6, 3, H, W] - [batch_size, num_cameras, channels, height, width]
用途: 输入到预处理模块的原始图像
1.2 预处理后图像数据
文件名: sample_X_imgs_1*6*3*256*704_float32.bin
含义: 经过预处理（归一化、resize等）的图像数据
形状: [1, 6, 3, 256, 704] - 标准化的输入尺寸
用途: 输入到backbone的特征提取模块
2. Backbone特征提取阶段
2.1 输入图像
文件名: sample_X_imgs_1*6*3*256*704_float32.bin
含义: 与预处理输出相同，backbone的输入
2.2 特征图输出
文件名: sample_X_feature_1*1536*256_float32.bin
含义: ResNet+FPN提取的多尺度特征
形状: [1, 1536, 256] - [batch_size, total_spatial_points, feature_dim]
用途: 输入到Sparse4D Head模块
3. Sparse4D Head阶段
3.1 第一帧输入 (First Frame)
spatial_shapes: sample_X_spatial_shapes_4*2_int32.bin
含义: 各尺度特征图的空间形状 [4, 2] (4个尺度，每个尺度的H和W)
level_start_index: sample_X_level_start_index_4_int32.bin
含义: 各尺度特征在展平特征中的起始索引 [4]
instance_feature: sample_X_instance_feature_1*900*256_float32.bin
含义: 实例特征，初始化为零或预训练特征 [1, 900, 256]
anchor: sample_X_anchor_1*900*10_float32.bin
含义: 3D锚框参数 [1, 900, 10] (x,y,z,l,w,h,sin_yaw,cos_yaw,vx,vy)
time_interval: sample_X_time_interval_1_float32.bin
含义: 时间间隔，第一帧通常为0或1
image_wh: sample_X_image_wh_2_float32.bin
含义: 图像宽高 [2] (width, height)
lidar2img: sample_X_lidar2img_6*4*4_float32.bin
含义: 6个相机的lidar到图像的变换矩阵 [6, 4, 4]
3.2 第一帧输出
pred_instance_feature: sample_X_pred_instance_feature_1*900*256_float32.bin
含义: 预测的实例特征 [1, 900, 256]
pred_anchor: sample_X_pred_anchor_1*900*10_float32.bin
含义: 预测的3D锚框参数 [1, 900, 10]
pred_class_score: sample_X_pred_class_score_1*900*10_float32.bin
含义: 分类得分 [1, 900, 10] (10个类别)
pred_quality_score: sample_X_pred_quality_score_1*900*1_float32.bin
含义: 质量得分 [1, 900, 1]
3.3 后续帧输入 (Frame > 1)
temp_instance_feature: sample_X_temp_instance_feature_1*600*256_float32.bin
含义: 上一帧的实例特征 [1, 600, 256]
temp_anchor: sample_X_temp_anchor_1*600*10_float32.bin
含义: 上一帧的锚框参数 [1, 600, 10]
mask: sample_X_mask_1*600_int32.bin
含义: 有效实例的掩码 [1, 600]
track_id: sample_X_track_id_1*600_int32.bin
含义: 跟踪ID [1, 600]
3.4 后续帧输出
pred_track_id: sample_X_pred_track_id_1*900_int32.bin
含义: 预测的跟踪ID [1, 900]
4. InstanceBank阶段
4.1 输入
ibank_timestamp: sample_X_ibank_timestamp_1_float32.bin
含义: 时间戳
ibank_global2lidar: sample_X_ibank_global2lidar_4*4_float32.bin
含义: 全局到lidar的变换矩阵 [4, 4]
4.2 输出
ibank_temp_confidence: sample_X_ibank_temp_confidence_1*600_float32.bin
含义: 临时实例的置信度 [1, 600]
ibank_confidence: sample_X_ibank_confidence_1*900_float32.bin
含义: 当前实例的置信度 [1, 900]
ibank_cached_feature: sample_X_ibank_cached_feature_1*600*256_float32.bin
含义: 缓存的实例特征 [1, 600, 256]
ibank_cached_anchor: sample_X_ibank_cached_anchor_1*600*10_float32.bin
含义: 缓存的锚框参数 [1, 600, 10]
ibank_prev_id: sample_X_ibank_prev_id_1_int32.bin
含义: 前一帧的ID [1]
ibank_updated_cur_track_id: sample_X_ibank_updated_cur_track_id_1*900_int32.bin
含义: 更新的当前跟踪ID [1, 900]
ibank_updated_temp_track_id: sample_X_ibank_updated_temp_track_id_1*600_int32.bin
含义: 更新的临时跟踪ID [1, 600]
5. 后处理阶段 (Postprocessor)
5.1 输出
decoder_boxes_3d: sample_X_decoder_boxes_3d_N*9_float32.bin
含义: 解码后的3D边界框 [N, 9] (x,y,z,l,w,h,sin_yaw,cos_yaw,vel)
decoder_scores_3d: sample_X_decoder_scores_3d_N_float32.bin
含义: 检测得分 [N]
decoder_labels_3d: sample_X_decoder_labels_3d_N_int32.bin
含义: 类别标签 [N]
decoder_cls_scores: sample_X_decoder_cls_scores_N*10_float32.bin
含义: 各类别得分 [N, 10]
decoder_track_ids: sample_X_decoder_track_ids_N_int32.bin
含义: 最终跟踪ID [N]