# deploy/dfa_plugin/test_dfa_consistency.py

import os
import torch
import numpy as np
import onnx
import onnxruntime as ort
import tensorrt as trt
import ctypes
from typing import Dict, List, Tuple, Any
import logging
from tool.utils.logger import set_logger
from modules.ops.deformable_aggregation import DeformableAggregationFunction


class DFAConsistencyTest:
    """DFA算子一致性测试类"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def generate_test_data(self, seed: int = 42) -> Dict[str, np.ndarray]:
        """生成DFA算子测试数据"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # 根据你的DFA算子参数生成测试数据
        batch_size = 1
        num_cams = 6
        num_levels = 4
        num_query = 900
        num_point = 13
        num_embeds = 256
        num_groups = 8
        
        # 计算spatial_size (根据你的代码逻辑)
        spatial_size = num_cams * (64 * 176 + 32 * 88 + 16 * 44 + 8 * 22)  # 89760
        
        test_data = {
            'value': np.random.rand(batch_size, spatial_size, num_embeds).astype(np.float32),
            'spatial_shapes': np.random.randint(8, 177, (num_cams, num_levels, 2)).astype(np.int32),
            'level_start_index': np.random.randint(0, spatial_size, (num_cams, num_levels)).astype(np.int32),
            'sampling_loc': np.random.rand(batch_size, num_query, num_point, num_cams, 2).astype(np.float32),
            'attn_weight': np.random.rand(batch_size, num_query, num_point, num_cams, num_levels, num_groups).astype(np.float32)
        }
        
        return test_data
    
    def pytorch_inference(self, test_data: Dict[str, np.ndarray]) -> np.ndarray:
        """PyTorch原生DFA推理"""
        # 转换为PyTorch张量
        inputs = {
            'value': torch.from_numpy(test_data['value']).to(self.device),
            'spatial_shapes': torch.from_numpy(test_data['spatial_shapes']).to(self.device),
            'level_start_index': torch.from_numpy(test_data['level_start_index']).to(self.device),
            'sampling_loc': torch.from_numpy(test_data['sampling_loc']).to(self.device),
            'attn_weight': torch.from_numpy(test_data['attn_weight']).to(self.device)
        }
        
        with torch.no_grad():
            output = DeformableAggregationFunction.apply(
                inputs['value'], inputs['spatial_shapes'], inputs['level_start_index'],
                inputs['sampling_loc'], inputs['attn_weight']
            )
        
        return output.cpu().numpy()
    
    def onnx_inference(self, session: ort.InferenceSession, test_data: Dict[str, np.ndarray]) -> np.ndarray:
        """ONNX推理"""
        input_names = [input.name for input in session.get_inputs()]
        input_feed = {
            input_names[0]: test_data['value'],
            input_names[1]: test_data['spatial_shapes'],
            input_names[2]: test_data['level_start_index'],
            input_names[3]: test_data['sampling_loc'],
            input_names[4]: test_data['attn_weight']
        }
        
        outputs = session.run(None, input_feed)
        return outputs[0]
    
    def tensorrt_inference(self, engine: trt.ICudaEngine, test_data: Dict[str, np.ndarray]) -> np.ndarray:
        """TensorRT推理（包含自定义插件）"""
        from cuda import cudart
        
        context = engine.create_execution_context()
        n_io = engine.num_io_tensors
        tensor_names = [engine.get_tensor_name(i) for i in range(n_io)]
        n_inputs = sum(1 for i in range(n_io) if engine.get_tensor_mode(tensor_names[i]) == trt.TensorIOMode.INPUT)
        
        # 准备缓冲区
        buffers = []
        for i in range(n_inputs):
            buffers.append(test_data[tensor_names[i]])
        
        for i in range(n_inputs, n_io):
            shape = context.get_tensor_shape(tensor_names[i])
            dtype = engine.get_tensor_dtype(tensor_names[i])
            buffers.append(np.zeros(shape, dtype=trt.nptype(dtype)))
        
        # GPU内存分配和推理
        gpu_buffers = []
        for buffer in buffers:
            gpu_buffers.append(cudart.cudaMalloc(buffer.nbytes)[1])
        
        for i in range(n_inputs):
            cudart.cudaMemcpy(
                gpu_buffers[i], buffers[i].ctypes.data, buffers[i].nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
            )
        
        for i in range(n_io):
            context.set_tensor_address(tensor_names[i], int(gpu_buffers[i]))
        
        context.execute_async_v3(0)
        
        for i in range(n_inputs, n_io):
            cudart.cudaMemcpy(
                buffers[i].ctypes.data, gpu_buffers[i], buffers[i].nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
            )
        
        for buffer in gpu_buffers:
            cudart.cudaFree(buffer)
        
        return buffers[-1]
    
    def calculate_metrics(self, output1: np.ndarray, output2: np.ndarray, name: str = "") -> Dict[str, float]:
        """计算一致性指标"""
        if output1.shape != output2.shape:
            raise ValueError(f"输出形状不一致: {output1.shape} vs {output2.shape}")
        
        flat1 = output1.flatten()
        flat2 = output2.flatten()
        
        cosine_distance = 1 - np.dot(flat1, flat2) / (np.linalg.norm(flat1) * np.linalg.norm(flat2))
        abs_diff = np.abs(output1 - output2)
        max_abs_error = float(np.max(abs_diff))
        mean_abs_error = float(np.mean(abs_diff))
        relative_error = float(np.mean(abs_diff / (np.abs(output1) + 1e-8)))
        l2_error = float(np.sqrt(np.mean((output1 - output2) ** 2)))
        
        metrics = {
            'cosine_distance': cosine_distance,
            'max_abs_error': max_abs_error,
            'mean_abs_error': mean_abs_error,
            'relative_error': relative_error,
            'l2_error': l2_error
        }
        
        self.logger.info(f"{name} 一致性指标:")
        self.logger.info(f"  余弦距离: {cosine_distance:.6f}")
        self.logger.info(f"  最大绝对误差: {max_abs_error:.6f}")
        self.logger.info(f"  平均绝对误差: {mean_abs_error:.6f}")
        self.logger.info(f"  相对误差: {relative_error:.6f}")
        self.logger.info(f"  L2误差: {l2_error:.6f}")
        
        return metrics
    
    def validate_consistency(self, metrics: Dict[str, float], tolerance: float = 1e-3, max_abs_tolerance: float = 0.1) -> bool:
        """验证一致性"""
        cosine_distance = metrics['cosine_distance']
        max_abs_error = metrics['max_abs_error']
        
        if cosine_distance >= tolerance:
            self.logger.error(f"余弦距离过大: {cosine_distance} >= {tolerance}")
            return False
        
        if max_abs_error >= max_abs_tolerance:
            self.logger.error(f"最大绝对误差过大: {max_abs_error} >= {max_abs_tolerance}")
            return False
        
        return True
    
    def run_consistency_test(self, 
                           onnx_path: str = "deploy/dfa_plugin/onnx/deformableAttentionAggr.onnx",
                           engine_path: str = "deploy/dfa_plugin/engine/deformableAttentionAggr.engine",
                           plugin_path: str = "deploy/dfa_plugin/lib/deformableAttentionAggr.so",
                           num_samples: int = 3,
                           tolerance: float = 1e-3,
                           max_abs_tolerance: float = 0.1) -> bool:
        """运行DFA算子一致性测试"""
        
        self.logger.info("开始DFA算子一致性测试...")
        
        # 尝试加载ONNX模型
        onnx_session = None
        if os.path.exists(onnx_path):
            try:
                onnx_model = onnx.load(onnx_path)
                onnx.checker.check_model(onnx_model)
                onnx_session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
                self.logger.info("ONNX模型加载成功")
            except Exception as e:
                self.logger.warning(f"ONNX模型加载失败: {e}")
        
        # 尝试加载TensorRT引擎
        trt_engine = None
        if os.path.exists(engine_path):
            try:
                if plugin_path and os.path.exists(plugin_path):
                    ctypes.cdll.LoadLibrary(plugin_path)
                    self.logger.info(f"已加载自定义插件: {plugin_path}")
                
                logger = trt.Logger(trt.Logger.INFO)
                trt.init_libnvinfer_plugins(logger, "")
                
                with open(engine_path, 'rb') as f:
                    trt_engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
                
                self.logger.info("TensorRT引擎加载成功")
            except Exception as e:
                self.logger.warning(f"TensorRT引擎加载失败: {e}")
        
        # 运行测试
        success_count = 0
        total_tests = 0
        
        for i in range(num_samples):
            self.logger.info(f"测试样本 {i+1}/{num_samples}")
            
            # 生成测试数据
            test_data = self.generate_test_data(seed=42 + i)
            
            # PyTorch推理（参考标准）
            pytorch_output = self.pytorch_inference(test_data)
            
            # ONNX推理对比
            if onnx_session is not None:
                total_tests += 1
                try:
                    onnx_output = self.onnx_inference(onnx_session, test_data)
                    metrics = self.calculate_metrics(pytorch_output, onnx_output, f"PyTorch vs ONNX (样本{i+1})")
                    
                    if self.validate_consistency(metrics, tolerance, max_abs_tolerance):
                        self.logger.info(f"✓ PyTorch vs ONNX 一致性测试通过 (样本{i+1})")
                        success_count += 1
                    else:
                        self.logger.error(f"✗ PyTorch vs ONNX 一致性测试失败 (样本{i+1})")
                except Exception as e:
                    self.logger.error(f"ONNX推理失败 (样本{i+1}): {e}")
            
            # TensorRT推理对比
            if trt_engine is not None:
                total_tests += 1
                try:
                    trt_output = self.tensorrt_inference(trt_engine, test_data)
                    metrics = self.calculate_metrics(pytorch_output, trt_output, f"PyTorch vs TensorRT (样本{i+1})")
                    
                    if self.validate_consistency(metrics, tolerance, max_abs_tolerance):
                        self.logger.info(f"✓ PyTorch vs TensorRT 一致性测试通过 (样本{i+1})")
                        success_count += 1
                    else:
                        self.logger.error(f"✗ PyTorch vs TensorRT 一致性测试失败 (样本{i+1})")
                except Exception as e:
                    self.logger.error(f"TensorRT推理失败 (样本{i+1}): {e}")
        
        # 输出总结
        self.logger.info(f"\n测试总结:")
        self.logger.info(f"总测试数: {total_tests}")
        self.logger.info(f"成功测试数: {success_count}")
        self.logger.info(f"失败测试数: {total_tests - success_count}")
        self.logger.info(f"成功率: {success_count/total_tests*100:.2f}%" if total_tests > 0 else "无有效测试")
        
        return success_count == total_tests


def main():
    """主函数"""
    logger, _, _ = set_logger("deploy/dfa_plugin/dfa_consistency_test.log", save_file=True)
    logger.setLevel(logging.INFO)
    
    tester = DFAConsistencyTest(logger)
    
    success = tester.run_consistency_test(
        onnx_path="deploy/dfa_plugin/onnx/deformableAttentionAggr.onnx",
        engine_path="deploy/dfa_plugin/engine/deformableAttentionAggr.engine",
        plugin_path="deploy/dfa_plugin/lib/deformableAttentionAggr.so",
        num_samples=3,
        tolerance=1e-3,
        max_abs_tolerance=0.1
    )
    
    if success:
        logger.info("所有DFA算子一致性测试通过！")
    else:
        logger.error("部分DFA算子一致性测试失败！")
    
    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)