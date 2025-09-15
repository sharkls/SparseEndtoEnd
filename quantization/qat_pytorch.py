# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import torch
import torch.nn as nn
from torch.ao.quantization import QConfig, MinMaxObserver, MovingAverageMinMaxObserver
from torch.ao.quantization.qconfig import default_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

def prepare_sparse4d_native_qat(model):
    """
    使用PyTorch原生QAT准备Sparse4D模型
    """
    # 设置QAT配置
    qconfig = QConfig(
        activation=MovingAverageMinMaxObserver.with_args(dtype=torch.qint8),
        weight=MinMaxObserver.with_args(dtype=torch.qint8)
    )
    
    # 为模型设置量化配置
    model.qconfig = qconfig
    
    # 准备模型进行QAT
    model_prepared = prepare_fx(model, qconfig_dict={'': qconfig})
    
    print("模型已准备进行原生QAT量化")
    return model_prepared

def native_qat_training(model_prepared, dataloader, num_epochs=5):
    """
    原生QAT训练循环
    """
    optimizer = torch.optim.AdamW(model_prepared.parameters(), lr=1e-4)
    
    model_prepared.train()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, data in enumerate(dataloader):
            optimizer.zero_grad()
            
            try:
                results = model_prepared(return_loss=True, **data)
                
                if isinstance(results, dict) and 'loss' in results:
                    loss = results['loss']
                else:
                    loss = compute_custom_loss(results, data)
                
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
    
    return model_prepared

def convert_native_quantized_model(model_prepared):
    """
    转换原生QAT模型为量化模型
    """
    model_prepared.eval()
    
    # 转换为量化模型
    quantized_model = convert_fx(model_prepared)
    
    print("模型已转换为原生量化模型")
    return quantized_model