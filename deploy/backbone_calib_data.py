import os
import numpy as np

# 这里的类名和函数名将被 polygraphy 调用
class BackboneCalibrator:
    def __init__(self):
        # 采样数据存放目录
        self.data_dir = "deploy/val_data_e2e_fp32"
        # 匹配文件的关键词
        self.node_key = "imgs"
        # 模型的输入节点名称 (请确保与 ONNX 匹配)
        self.input_name = "img"
        # 输入形状
        self.shape = (1, 6, 3, 256, 704)
        
        # 查找所有匹配的 .bin 文件并排序
        if not os.path.exists(self.data_dir):
            print(f"[ERROR] Data directory {self.data_dir} not found!")
            self.all_files = []
        else:
            self.all_files = sorted([
                f for f in os.listdir(self.data_dir) 
                if f.startswith("sample_") and f"_{self.node_key}_" in f 
                and "_ori_imgs_" not in f and f.endswith(".bin")
            ])
        
        # 限制校准样本数量，通常 100 个左右即可
        self.all_files = self.all_files[:100]
        self.count = 0
        print(f"[Calibrator] Found {len(self.all_files)} samples for INT8 calibration.")

    def __iter__(self):
        return self

    def __next__(self):
        if self.count >= len(self.all_files):
            raise StopIteration
        
        file_path = os.path.join(self.data_dir, self.all_files[self.count])
        if self.count % 20 == 0:
            print(f"[Calibrator] Loading batch {self.count}/{len(self.all_files)}: {self.all_files[self.count]}")
            
        # 使用 numpy 加载原始二进制数据
        data = np.fromfile(file_path, dtype=np.float32).reshape(self.shape)
        self.count += 1
        
        # 返回一个字典，Key 是 ONNX 的输入节点名
        return {self.input_name: data}

# Polygraphy 的数据加载工厂函数
def get_data_loader():
    return BackboneCalibrator()
