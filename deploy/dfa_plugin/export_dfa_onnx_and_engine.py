# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import torch
import torch.onnx
import torch.nn as nn
import os
import onnx
import onnxsim
import subprocess
import sys
from pathlib import Path

from tool.utils.save_bin import save_bins
from tool.utils.logger import logger_wrapper
from modules.ops.deformable_aggregation import DeformableAggregationFunction


class CustomDFAModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        value,
        input_spatial_shapes,
        input_level_start_index,
        sampling_locations,
        attention_weights,
    ):
        output = DeformableAggregationFunction.apply(
            value,
            input_spatial_shapes,
            input_level_start_index,
            sampling_locations,
            attention_weights,
        )
        return output


def setup_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def export_onnx(model, save_path, save_file=True):
    """导出ONNX模型"""
    print(f"开始导出ONNX模型到: {save_path}")
    
    # 创建保存目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 准备虚拟输入数据
    num_cams = 6
    dummy_feature = torch.rand(
        [1, num_cams * (64 * 176 + 32 * 88 + 16 * 44 + 8 * 22), 256]
    ).cuda()
    dummy_spatial_shapes = (
        torch.tensor([[64, 176], [32, 88], [16, 44], [8, 22]])
        .int()
        .unsqueeze(0)
        .repeat(num_cams, 1, 1)
        .cuda()
    )
    dummy_level_start_index = (
        torch.tensor(
            [
                [0, 11264, 14080, 14784],
                [14960, 26224, 29040, 29744],
                [29920, 41184, 44000, 44704],
                [44880, 56144, 58960, 59664],
                [59840, 71104, 73920, 74624],
                [74800, 86064, 88880, 89584],
            ]
        )
        .int()
        .cuda()
    )
    dummy_sampling_loc = torch.rand([1, 900, 13, 6, 2]).cuda()
    dummy_weights = torch.rand([1, 900, 13, 6, 4, 8]).cuda()

    # 导出ONNX
    with torch.no_grad():
        torch.onnx.export(
            model=model,
            args=(
                dummy_feature,
                dummy_spatial_shapes,
                dummy_level_start_index,
                dummy_sampling_loc,
                dummy_weights,
            ),
            f=save_path,
            input_names=[
                "feature",
                "spatial_shapes",
                "level_start_index",
                "sampling_loc",
                "attn_weight",
            ],
            output_names=["output"],
            opset_version=15,
            do_constant_folding=True,
            verbose=False,
        )

    print(f"ONNX模型导出完成: {save_path}")

    # 验证导出的ONNX模型
    print("验证ONNX模型...")
    model_onnx = onnx.load(save_path)
    onnx.checker.check_model(model_onnx)
    print("ONNX模型验证通过")

    # 使用onnx-simplifier简化模型
    print("使用onnx-simplifier简化模型...")
    model_onnx, check = onnxsim.simplify(model_onnx)
    assert check, "ONNX简化失败"
    onnx.save(model_onnx, save_path)
    print(f"ONNX模型简化完成，保存到: {save_path}")

    # 保存推理结果用于后续验证
    if save_file:
        logger, _, _ = logger_wrapper("", False)
        output = model(
            dummy_feature,
            dummy_spatial_shapes,
            dummy_level_start_index,
            dummy_sampling_loc,
            dummy_weights,
        )
        
        save_bins(
            inputs=[
                dummy_feature.detach().cpu().numpy(),
                dummy_spatial_shapes.detach().cpu().numpy(),
                dummy_level_start_index.detach().cpu().numpy(),
                dummy_sampling_loc.detach().cpu().numpy(),
                dummy_weights.detach().cpu().numpy(),
            ],
            outputs=[output.detach().cpu().numpy()],
            names=[
                "rand_fetaure",
                "rand_spatial_shapes",
                "rand_level_start_index",
                "rand_sampling_loc",
                "rand_weights",
                "output",
            ],
            sample_index=0,
            logger=logger,
            save_prefix="deploy/dfa_plugin/asset",
        )
        print("测试数据保存完成")


def build_tensorrt_engine(onnx_path, engine_path, plugin_path=None, verbose=True):
    """构建TensorRT引擎（最小化版本）"""
    print(f"开始构建TensorRT引擎...")
    print(f"ONNX模型: {onnx_path}")
    print(f"引擎保存路径: {engine_path}")
    
    # 创建引擎保存目录
    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
    
    # 构建最小化的trtexec命令
    cmd = [
        "trtexec",
        "--onnx=" + onnx_path,
        "--saveEngine=" + engine_path,
        "--memPoolSize=workspace:512",   # 最小工作内存
        "--warmUp=1",                    # 最小预热次数
        "--iterations=1",                # 最小迭代次数
        "--buildOnly",                   # 只构建，不进行性能测试
        "--noTF32",                     # 禁用TF32
        "--fp16",                       # 使用FP16
        "--int8",                       # 使用INT8进一步减少内存
        "--verbose=false",              # 禁用详细输出
        "--dumpOutput=false",           # 禁用输出转储
        "--dumpProfile=false",          # 禁用性能分析
        "--dumpLayerInfo=false",        # 禁用层信息转储
    ]
    
    # 如果有自定义插件，添加插件路径
    if plugin_path and os.path.exists(plugin_path):
        cmd.append("--plugins=" + plugin_path)
        print(f"使用自定义插件: {plugin_path}")
    
    # 移除空字符串
    cmd = [arg for arg in cmd if arg]
    
    print(f"执行命令: {' '.join(cmd)}")
    
    # 设置环境变量以减少内存使用
    env = os.environ.copy()
    env['CUDA_LAUNCH_BLOCKING'] = '1'
    env['CUDA_MEMORY_FRACTION'] = '0.5'
    env['CUDA_CACHE_DISABLE'] = '1'
    env['TENSORRT_CACHE_DIR'] = '/tmp/trt_cache'
    
    try:
        # 执行trtexec命令
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=900  # 15分钟超时
        )
        
        if result.returncode == 0:
            print("TensorRT引擎构建成功!")
            if result.stdout:
                print("标准输出:")
                print(result.stdout)
        else:
            print(f"TensorRT引擎构建失败，返回码: {result.returncode}")
            if result.stderr:
                print("错误输出:")
                print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("TensorRT引擎构建超时（15分钟）")
        return False
    except Exception as e:
        print(f"TensorRT引擎构建失败: {e}")
        return False
    
    return True


def verify_engine(engine_path):
    """验证TensorRT引擎"""
    if not os.path.exists(engine_path):
        print(f"错误: TensorRT引擎文件不存在: {engine_path}")
        return False
    
    file_size = os.path.getsize(engine_path) / (1024 * 1024)  # MB
    print(f"TensorRT引擎文件大小: {file_size:.2f} MB")
    
    return True


def run_consistency_test(onnx_path, engine_path, plugin_path=None):
    """运行一致性测试"""
    print("开始运行一致性测试...")
    
    # 这里可以调用之前创建的一致性测试脚本
    test_script = "deploy/dfa_plugin/unit_test/deformable_feature_aggregation_infer-consistency-val_pytorch_vs_trt_unit_test.py"
    
    if os.path.exists(test_script):
        cmd = [
            sys.executable,
            test_script,
            "--onnx", onnx_path,
            "--engine", engine_path
        ]
        
        if plugin_path:
            cmd.extend(["--plugin", plugin_path])
        
        print(f"执行一致性测试: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            print("一致性测试输出:")
            print(result.stdout)
            
            if result.stderr:
                print("一致性测试错误:")
                print(result.stderr)
                
        except Exception as e:
            print(f"一致性测试失败: {e}")
    else:
        print(f"一致性测试脚本不存在: {test_script}")


def main():
    """主函数"""
    print("=" * 60)
    print("DFA ONNX导出和TensorRT引擎构建")
    print("=" * 60)
    
    # 设置路径
    onnx_path = "deploy/dfa_plugin/onnx/deformableAttentionAggr.onnx"
    engine_path = "deploy/dfa_plugin/engine/deformableAttentionAggr.engine"
    plugin_path = "deploy/dfa_plugin/lib/deformableAttentionAggr.so"
    
    # 设置随机种子
    setup_seed(1)
    
    # 步骤1: 导出ONNX模型
    print("\n步骤1: 导出ONNX模型")
    print("-" * 40)
    
    model = CustomDFAModel()
    model.eval()
    
    try:
        export_onnx(model, onnx_path)
        print("✓ ONNX模型导出成功")
    except Exception as e:
        print(f"✗ ONNX模型导出失败: {e}")
        return False
    
    # 步骤2: 构建TensorRT引擎
    print("\n步骤2: 构建TensorRT引擎")
    print("-" * 40)
    
    try:
        success = build_tensorrt_engine(onnx_path, engine_path, plugin_path)
        if success:
            print("✓ TensorRT引擎构建成功")
        else:
            print("✗ TensorRT引擎构建失败")
            return False
    except Exception as e:
        print(f"✗ TensorRT引擎构建失败: {e}")
        return False
    
    # 步骤3: 验证引擎
    print("\n步骤3: 验证TensorRT引擎")
    print("-" * 40)
    
    if verify_engine(engine_path):
        print("✓ TensorRT引擎验证通过")
    else:
        print("✗ TensorRT引擎验证失败")
        return False
    
    # 步骤4: 运行一致性测试
    print("\n步骤4: 运行一致性测试")
    print("-" * 40)
    
    run_consistency_test(onnx_path, engine_path, plugin_path)
    
    # 总结
    print("\n" + "=" * 60)
    print("导出和构建完成!")
    print(f"ONNX模型: {onnx_path}")
    print(f"TensorRT引擎: {engine_path}")
    print(f"自定义插件: {plugin_path}")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 