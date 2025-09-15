# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import torch
import torch.nn as nn
from torchao.quantization.prototype.qat import Int8DynActInt4WeightQATQuantizer
from torch.ao.quantization import QConfig, MinMaxObserver, MovingAverageMinMaxObserver
from torch.ao.quantization.qconfig import default_qconfig

from modules.sparse4d_detector import Sparse4D
from tool.utils.config import read_cfg
from tool.runner.checkpoint import load_checkpoint

def build_module(cfg, default_args=None):
    """构建模块的辅助函数"""
    cfg2 = cfg.copy()
    if default_args is not None:
        for name, value in default_args.items():
            cfg2.setdefault(name, value)
    type = cfg2.pop("type")
    return eval(type)(**cfg2)

def prepare_sparse4d_for_qat(model, config_path, checkpoint_path):
    """
    为Sparse4D模型准备QAT量化
    
    Args:
        model: Sparse4D模型
        config_path: 配置文件路径
        checkpoint_path: 检查点路径
    """
    # 加载预训练权重
    checkpoint = load_checkpoint(model, checkpoint_path, map_location="cpu")
    
    # 设置模型为训练模式
    model.train()
    
    # 创建QAT量化器
    # 方案1: 使用torchao的Int8DynActInt4WeightQATQuantizer
    qat_quantizer = Int8DynActInt4WeightQATQuantizer()
    
    # 方案2: 或者使用PyTorch原生的QAT配置
    # qconfig = QConfig(
    #     activation=MovingAverageMinMaxObserver.with_args(dtype=torch.qint8),
    #     weight=MinMaxObserver.with_args(dtype=torch.qint8)
    # )
    
    # 准备模型进行QAT
    model = qat_quantizer.prepare(model)
    
    print("模型已准备进行QAT量化")
    return model, qat_quantizer

def qat_training_loop(model, dataloader, num_epochs=5, learning_rate=1e-4):
    """
    QAT训练循环
    
    Args:
        model: 已准备QAT的模型
        dataloader: 数据加载器
        num_epochs: 训练轮数
        learning_rate: 学习率
    """
    # 设置优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # 设置损失函数
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, data in enumerate(dataloader):
            optimizer.zero_grad()
            
            try:
                # 前向传播
                results = model(return_loss=True, **data)
                
                # 计算损失
                if isinstance(results, dict) and 'loss' in results:
                    loss = results['loss']
                else:
                    # 如果没有直接返回loss，需要手动计算
                    # 这里需要根据你的具体损失计算逻辑来调整
                    loss = compute_custom_loss(results, data)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                    
            except Exception as e:
                print(f"训练批次 {batch_idx} 出错: {e}")
                continue
        
        avg_loss = total_loss / max(1, num_batches)
        print(f"Epoch {epoch+1}/{num_epochs} 完成, 平均损失: {avg_loss:.4f}")
    
    return model

def compute_custom_loss(results, data):
    """
    计算自定义损失函数
    这里需要根据Sparse4D的具体损失计算逻辑来实现
    """
    # 示例：假设results包含预测结果，data包含标签
    # 你需要根据实际的损失计算逻辑来修改这里
    
    if isinstance(results, list) and len(results) > 0:
        result = results[0]
        
        # 提取预测结果
        if 'img_bbox' in result and result['img_bbox'] is not None:
            bbox_pred = result['img_bbox']
            
            # 这里需要根据你的损失函数来计算
            # 例如：分类损失 + 回归损失 + 其他损失
            loss = 0.0
            
            # 分类损失
            if 'cls_scores' in bbox_pred:
                # 假设有标签数据
                if 'gt_labels' in data:
                    cls_loss = nn.CrossEntropyLoss()(bbox_pred['cls_scores'], data['gt_labels'])
                    loss += cls_loss
            
            # 回归损失
            if 'boxes_3d' in bbox_pred:
                # 假设有GT边界框
                if 'gt_boxes' in data:
                    reg_loss = nn.MSELoss()(bbox_pred['boxes_3d'], data['gt_boxes'])
                    loss += reg_loss
            
            return loss
    
    # 如果没有有效结果，返回一个小的损失值
    return torch.tensor(0.1, requires_grad=True, device=next(model.parameters()).device)

def convert_to_quantized_model(model, qat_quantizer):
    """
    将QAT模型转换为量化模型
    """
    # 设置为评估模式
    model.eval()
    
    # 转换为量化模型
    quantized_model = qat_quantizer.convert(model)
    
    print("模型已转换为量化模型")
    return quantized_model

def export_quantized_onnx(quantized_model, output_path, dummy_inputs):
    """
    导出量化后的ONNX模型
    """
    quantized_model.eval()
    
    with torch.no_grad():
        torch.onnx.export(
            quantized_model,
            dummy_inputs,
            output_path,
            input_names=["img"],
            output_names=["feature"],
            opset_version=15,
            do_constant_folding=True,
            verbose=False,
            dynamic_axes={
                'img': {0: 'batch_size'},
                'feature': {0: 'batch_size'}
            }
        )
    
    print(f"量化ONNX模型已导出到: {output_path}")

def main():
    """主函数：执行QAT量化流程"""
    
    # 配置参数
    config_path = "dataset/config/sparse4d_temporal_r50_1x1_bs1_256x704_mini.py"
    checkpoint_path = "ckpt/sparse4dv3_r50.pth"
    output_dir = "qat_output"
    
    # 创建输出目录
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取配置
    cfg = read_cfg(config_path)
    cfg["model"]["img_backbone"]["init_cfg"] = {}
    
    # 构建模型
    print("构建Sparse4D模型...")
    model = build_module(cfg["model"])
    
    # 准备QAT
    print("准备QAT量化...")
    model, qat_quantizer = prepare_sparse4d_for_qat(model, config_path, checkpoint_path)
    
    # 创建数据加载器（你需要根据实际情况调整）
    from dataset.dataloader_wrapper import dataloader_wrapper_without_dist
    from dataset import NuScenes4DDetTrackDataset
    
    # 构建数据集
    dataset = build_module(cfg["data"]["test"])
    data_loader = dataloader_wrapper_without_dist(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=1,
        dist=False,
        shuffle=False,
    )
    
    # QAT训练
    print("开始QAT训练...")
    model = qat_training_loop(model, data_loader, num_epochs=5, learning_rate=1e-4)
    
    # 转换为量化模型
    print("转换为量化模型...")
    quantized_model = convert_to_quantized_model(model, qat_quantizer)
    
    # 保存量化模型
    quantized_model_path = os.path.join(output_dir, "sparse4d_quantized.pth")
    torch.save(quantized_model.state_dict(), quantized_model_path)
    print(f"量化模型已保存到: {quantized_model_path}")
    
    # 导出量化ONNX（可选）
    print("导出量化ONNX模型...")
    dummy_input = torch.randn(1, 6, 3, 256, 704).cuda()
    onnx_path = os.path.join(output_dir, "sparse4d_quantized.onnx")
    export_quantized_onnx(quantized_model, onnx_path, dummy_input)

if __name__ == "__main__":
    main()