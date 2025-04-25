import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points

# 初始化 NuScenes 数据集
nusc = NuScenes(version='v1.0-mini', dataroot='/share/Code/SparseEnd2End_v2/data/nuscenes', verbose=True)

# 指定要处理的帧
frame_index = 0  # 替换为您想要的帧索引
sample = nusc.sample[frame_index]
lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
camera_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])

# 加载激光雷达点云
lidar_path = nusc.get_sample_data_path(lidar_data['token'])
pc = LidarPointCloud.from_file(lidar_path)

# 获取相机内外参
camera_sensor = nusc.get('calibrated_sensor', camera_data['calibrated_sensor_token'])
camera_intrinsic = np.array(camera_sensor['camera_intrinsic'])  # 转换为 NumPy 数组
camera_extrinsic = np.array(camera_sensor['translation'])  # 或者使用其他方法获取外参

# 投影激光雷达点云到相机图像平面
points_2d = view_points(pc.points[:3, :], camera_intrinsic, normalize=True)

# 生成 lidar2img_5*6*4*4_float32.bin 文件
output_file = 'lidar2img_5*6*4*4_float32.bin'
with open(output_file, 'wb') as f:
    f.write(points_2d.astype(np.float32).tobytes())

print(f"生成的文件: {output_file}")