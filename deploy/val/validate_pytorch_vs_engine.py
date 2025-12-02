#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.

"""
验证PyTorch模型和TensorRT引擎在FP32精度下的输出差异
"""

import os
import sys
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, List

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from tool.utils.config import read_cfg
from tool.runner.checkpoint import load_checkpoint
from tool.runner.fp16_utils import wrap_fp16_model
from tool.trainer.utils import set_random_seed
from tool.utils.logger import set_logger
from modules.sparse4d_detector import Sparse4D
from dataset import NuScenes4DDetTrackDataset
from dataset.utils.collate import collate_fn

try:
    import tensorrt as trt
    from cuda import cudart
    import ctypes
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    print("[WARNING] TensorRT not available. Engine validation will be skipped.")


def parse_args():
    parser = argparse.ArgumentParser(description="验证PyTorch模型和TensorRT引擎输出差异")
    parser.add_argument(
        "--config",
        type=str,
        default="dataset/config/sparse4d_temporal_r50_1x1_bs1_256x704_mini.py",
        help="配置文件路径"
    )
    parser.add_argument(
        "--use_val_dataset",
        action="store_true",
        help="使用val数据集而不是test数据集（val数据集包含ground truth，test数据集不包含）"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="ckpt/sparse4dv3_r50.pth",
        help="PyTorch模型检查点路径"
    )
    parser.add_argument(
        "--backbone_engine",
        type=str,
        default="deploy/engine/sparse4dbackbone.engine",
        help="Backbone TensorRT引擎路径"
    )
    parser.add_argument(
        "--head_engine",
        type=str,
        default="deploy/engine/sparse4dhead1st.engine",
        help="Head TensorRT引擎路径（第一帧使用head1st，第二帧使用head2nd）"
    )
    parser.add_argument(
        "--head2nd_engine",
        type=str,
        default=None,
        help="Head第二帧TensorRT引擎路径（如果指定，将验证连续帧；否则每个样本都作为第一帧验证）"
    )
    parser.add_argument(
        "--plugin_paths",
        type=str,
        nargs="+",
        default=None,
        help="TensorRT插件库路径列表（可选，默认会自动查找常用插件）"
    )
    parser.add_argument(
        "--validate_as_first_frame",
        action="store_true",
        help="将每个样本都作为第一帧验证（会重置instance_bank状态）。如果不指定--head2nd_engine，这是默认行为"
    )
    parser.add_argument(
        "--validate_continuous_frames",
        action="store_true",
        help="验证连续帧（第一个样本是第一帧，后续是第二帧）。需要同时指定--head2nd_engine"
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=0,
        help="验证的样本索引"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="验证的样本数量"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="deploy/val/validation_results",
        help="验证结果输出目录"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="推理设备"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="是否使用确定性选项"
    )
    parser.add_argument(
        "--log",
        type=str,
        default="deploy/val/validate_pytorch_vs_engine.log",
        help="日志文件路径"
    )
    parser.add_argument(
        "--analyze_plugin_error",
        action="store_true",
        help="详细分析插件可能的误差来源（会记录更多统计信息）"
    )
    parser.add_argument(
        "--capture_keypoints",
        action="store_true",
        help="捕获并对比关键点输出（需要hook PyTorch模型）"
    )
    parser.add_argument(
        "--analyze_error_patterns",
        action="store_true",
        help="分析误差模式（anchor尺寸、角度等与误差的关系）"
    )
    return parser.parse_args()


def build_module(cfg, default_args: Optional[Dict] = None) -> Any:
    """构建模块的辅助函数"""
    cfg2 = cfg.copy()
    if default_args is not None:
        for name, value in default_args.items():
            cfg2.setdefault(name, value)
    type = cfg2.pop("type")
    return eval(type)(**cfg2)


def ensure_img_metas_list_format(metas_dict, context_name="metas", logger=None):
    """
    确保metas字典中的img_metas是列表格式，且每个元素都是字典（不是嵌套列表）
    
    Args:
        metas_dict: 要检查的metas字典
        context_name: 上下文名称，用于日志记录
        logger: 日志记录器（可选）
    """
    if metas_dict is None:
        return
    if not isinstance(metas_dict, dict):
        return
    if "img_metas" not in metas_dict:
        return
    img_metas = metas_dict["img_metas"]
    
    # 如果img_metas是字典，转换为列表
    if isinstance(img_metas, dict):
        metas_dict["img_metas"] = [img_metas]
        if logger:
            logger.debug(f"修复 {context_name}['img_metas']: 字典 -> 列表")
        return
    
    # 如果img_metas不是列表，转换为列表
    if not isinstance(img_metas, list):
        metas_dict["img_metas"] = [img_metas]
        if logger:
            logger.debug(f"修复 {context_name}['img_metas']: 其他类型 -> 列表")
        return
    
    # 如果img_metas是列表，检查是否有嵌套列表的情况
    # 例如：[[dict]] 应该展平为 [dict]
    if len(img_metas) > 0:
        first_item = img_metas[0]
        # 如果第一个元素是列表，说明是嵌套列表，需要展平
        if isinstance(first_item, list):
            # 展平嵌套列表：[[dict]] -> [dict]
            flattened = []
            for item in img_metas:
                if isinstance(item, list):
                    flattened.extend(item)
                else:
                    flattened.append(item)
            metas_dict["img_metas"] = flattened
            if logger:
                logger.debug(f"修复 {context_name}['img_metas']: 嵌套列表 -> 展平列表")
            return
        
        # 确保所有元素都是字典（不是列表）
        # 如果发现列表元素，说明可能是嵌套结构
        has_list_elements = any(isinstance(item, list) for item in img_metas)
        if has_list_elements:
            # 展平所有列表元素
            flattened = []
            for item in img_metas:
                if isinstance(item, list):
                    flattened.extend(item)
                else:
                    flattened.append(item)
            metas_dict["img_metas"] = flattened
            if logger:
                logger.debug(f"修复 {context_name}['img_metas']: 包含列表元素 -> 展平列表 (长度: {len(img_metas)} -> {len(flattened)})")
        
        # 最终验证：确保所有元素都是字典
        final_img_metas = metas_dict["img_metas"]
        if isinstance(final_img_metas, list) and len(final_img_metas) > 0:
            non_dict_items = [i for i, item in enumerate(final_img_metas) if not isinstance(item, dict)]
            if non_dict_items:
                if logger:
                    logger.warning(f"警告 {context_name}['img_metas']: 发现非字典元素在索引 {non_dict_items[:5]} (总共 {len(non_dict_items)} 个)")
                # 移除非字典元素，或者尝试转换
                cleaned = [item for item in final_img_metas if isinstance(item, dict)]
                if len(cleaned) > 0:
                    metas_dict["img_metas"] = cleaned
                    if logger:
                        logger.debug(f"清理 {context_name}['img_metas']: 移除非字典元素 (保留 {len(cleaned)} 个字典)")


class Sparse4DBackboneWrapper(nn.Module):
    """Backbone包装类，用于提取特征"""
    def __init__(self, model):
        super(Sparse4DBackboneWrapper, self).__init__()
        self.model = model

    def forward(self, img):
        feature, spatial_shapes, level_start_index = self.model.extract_feat(img)
        return feature


class Sparse4DHead1stWrapper(nn.Module):
    """Head第一帧包装类，用于提取head输出"""
    def __init__(self, head, capture_keypoints=False):
        super(Sparse4DHead1stWrapper, self).__init__()
        self.head = head
        self.capture_keypoints = capture_keypoints
        self.captured_keypoints = []  # 存储捕获的关键点

    def forward(
        self,
        feature,
        spatial_shapes,
        level_start_index,
        instance_feature,
        anchor,
        time_interval,
        image_wh,
        lidar2img,
    ):
        """Head第一帧的前向传播 - 使用head的原始forward方法"""
        # 构建feature_maps和metas，与head.forward期望的格式一致
        feature_maps = [feature, spatial_shapes, level_start_index]
        
        # 确保metas包含所有必需的字段，包括timestamp
        # 如果instance_bank有cached_anchor，它会需要timestamp
        timestamp = torch.tensor([0.0], device=feature.device)  # 默认timestamp
        
        # 转换tensor到numpy（如果需要）
        image_wh_np = image_wh.cpu().numpy() if isinstance(image_wh, torch.Tensor) else image_wh
        lidar2img_np = lidar2img.cpu().numpy() if isinstance(lidar2img, torch.Tensor) else lidar2img
        timestamp_np = timestamp.cpu().numpy() if isinstance(timestamp, torch.Tensor) else timestamp
        
        # 构建img_metas，包含所有必需的字段
        img_meta = {
            "image_wh": image_wh_np,
            "lidar2img": lidar2img_np,
            "timestamp": timestamp_np,
            "global2lidar": np.eye(4, dtype=np.float32),  # 单位矩阵（第一帧验证）
            "lidar2global": np.eye(4, dtype=np.float32),  # 单位矩阵（第一帧验证）
        }
        
        metas = {
            "timestamp": timestamp,
            "image_wh": image_wh,
            "lidar2img": lidar2img,
            "img_metas": [img_meta],  # head.forward期望的格式
        }
        
        # 使用hook来捕获forward过程中的instance_feature和anchor
        captured_instance_feature = [None]
        captured_anchor = [None]
        
        # 保存原始的cache和get方法
        original_cache = self.head.instance_bank.cache
        original_get = self.head.instance_bank.get
        
        def cache_wrapper(instance_feature, anchor, confidence, metas=None, feature_maps=None):
            # 捕获cache的输入参数
            captured_instance_feature[0] = instance_feature.clone()
            captured_anchor[0] = anchor.clone()
            # 确保metas["img_metas"]是列表格式（instance_bank.get需要）
            if metas is not None:
                # 创建metas的副本，避免修改原始metas
                cache_metas = metas.copy() if isinstance(metas, dict) else metas
                # 使用统一的格式修复函数
                ensure_img_metas_list_format(cache_metas, "cache_metas (in wrapper)")
                # 调用原始cache方法，使用修复后的metas
                return original_cache(instance_feature, anchor, confidence, cache_metas, feature_maps)
            else:
                # 调用原始cache方法
                return original_cache(instance_feature, anchor, confidence, metas, feature_maps)
        
        def get_wrapper(batch_size, metas=None, dn_metas=None):
            # 在调用原始get之前，确保self.metas["img_metas"]是列表格式
            # 这很重要，因为instance_bank.get会访问self.metas["img_metas"][i]
            # 注意：必须在每次调用时都检查，因为self.metas可能在之前的调用中被修改
            instance_bank = self.head.instance_bank
            
            # 修复self.metas["img_metas"]格式
            if hasattr(instance_bank, 'metas') and instance_bank.metas is not None:
                ensure_img_metas_list_format(instance_bank.metas, "instance_bank.metas (in wrapper)")
                # 验证修复后的格式
                if isinstance(instance_bank.metas, dict) and "img_metas" in instance_bank.metas:
                    img_metas = instance_bank.metas["img_metas"]
                    if isinstance(img_metas, list) and len(img_metas) > 0:
                        # 确保每个元素都是字典
                        for i, item in enumerate(img_metas):
                            if not isinstance(item, dict):
                                raise TypeError(
                                    f"instance_bank.metas['img_metas'][{i}] 应该是字典，但得到 {type(item)}: {item}"
                                )
            
            # 也确保传入的metas["img_metas"]是列表格式
            if metas is not None:
                ensure_img_metas_list_format(metas, "metas (in wrapper)")
                # 验证修复后的格式
                if isinstance(metas, dict) and "img_metas" in metas:
                    img_metas = metas["img_metas"]
                    if isinstance(img_metas, list) and len(img_metas) > 0:
                        # 确保每个元素都是字典
                        for i, item in enumerate(img_metas):
                            if not isinstance(item, dict):
                                raise TypeError(
                                    f"metas['img_metas'][{i}] 应该是字典，但得到 {type(item)}: {item}"
                                )
            
            # 调用原始get方法
            return original_get(batch_size, metas, dn_metas)
        
        # 替换cache和get方法
        self.head.instance_bank.cache = cache_wrapper
        self.head.instance_bank.get = get_wrapper
        
        # 如果启用关键点捕获，注册hook
        keypoints_hooks = []
        if self.capture_keypoints:
            self.captured_keypoints = []
            # 遍历head中的所有层，找到kps_generator
            def register_keypoints_hooks(module, prefix=""):
                for name, child in module.named_children():
                    full_name = f"{prefix}.{name}" if prefix else name
                    if hasattr(child, 'kps_generator'):
                        # 注册forward hook来捕获关键点
                        def make_hook(layer_name):
                            def hook_fn(module, input, output):
                                if isinstance(output, torch.Tensor):
                                    self.captured_keypoints.append({
                                        'layer': layer_name,
                                        'keypoints': output.clone().detach(),
                                        'anchor': input[0].clone().detach() if len(input) > 0 and isinstance(input[0], torch.Tensor) else None,
                                        'instance_feature': input[1].clone().detach() if len(input) > 1 and isinstance(input[1], torch.Tensor) else None,
                                    })
                            return hook_fn
                        child.kps_generator.register_forward_hook(make_hook(full_name))
                    # 递归注册子模块
                    register_keypoints_hooks(child, full_name)
            
            register_keypoints_hooks(self.head)
        
        try:
            # 直接调用head的forward方法
            model_outs = self.head(feature_maps, metas)
            
            # 从model_outs中提取输出
            classification = model_outs.get("classification", [])
            prediction = model_outs.get("prediction", [])
            quality = model_outs.get("quality", [])
            
            # 获取最后一个decoder的输出
            if len(prediction) > 0:
                pred_anchor = prediction[-1]
            else:
                pred_anchor = anchor
            
            if len(classification) > 0:
                pred_class_score = classification[-1]
            else:
                pred_class_score = None
            
            if len(quality) > 0:
                pred_quality_score = quality[-1]
            else:
                pred_quality_score = None
            
            # 获取instance_feature和anchor - 从hook捕获的值
            if captured_instance_feature[0] is not None:
                pred_instance_feature = captured_instance_feature[0]
            else:
                # 如果hook没有捕获到（可能cache没有被调用），使用输入值
                pred_instance_feature = instance_feature
            
            # 如果anchor也被捕获，使用捕获的值（更准确）
            if captured_anchor[0] is not None and len(prediction) == 0:
                pred_anchor = captured_anchor[0]
        finally:
            # 恢复原始的cache和get方法
            self.head.instance_bank.cache = original_cache
            self.head.instance_bank.get = original_get
        
        return pred_instance_feature, pred_anchor, pred_class_score, pred_quality_score


class TensorRTEngine:
    """TensorRT引擎封装类"""
    # 类变量：记录已加载的插件，避免重复加载
    _loaded_plugins = set()
    
    def __init__(self, engine_path: str, logger: logging.Logger, plugin_paths: list = None):
        if not TENSORRT_AVAILABLE:
            raise RuntimeError("TensorRT is not available")
        
        self.engine_path = engine_path
        self.logger = logger
        self.engine = None
        self.context = None
        self.input_names = []
        self.output_names = []
        self.bufferH = []
        self.bufferD = []
        self.stream = None
        
        # 加载插件（如果指定）
        if plugin_paths is None:
            plugin_paths = []
        self._load_plugins(plugin_paths)
        
        self._load_engine()
    
    def _load_plugins(self, plugin_paths: list):
        """加载TensorRT插件库"""
        # 获取项目根目录（假设脚本在deploy/val/目录下）
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '../..'))
        
        # 默认插件路径（相对于项目根目录）
        default_plugins = [
            os.path.join(project_root, "deploy/sparsebox_plugin/lib/SparseBox3DKeyPointsPlugin.so"),
            os.path.join(project_root, "deploy/dfa_plugin/lib/deformableAttentionAggr.so"),
            os.path.join(project_root, "deploy/ln_plugin/lib/customLayerNorm.so"),
        ]
        
        # 合并默认插件和用户指定的插件
        all_plugins = list(set(default_plugins + (plugin_paths or [])))
        
        for plugin_path in all_plugins:
            # 转换为绝对路径
            if not os.path.isabs(plugin_path):
                plugin_path = os.path.join(project_root, plugin_path)
            plugin_path = os.path.abspath(plugin_path)
            
            # 检查插件是否已加载
            if plugin_path in self._loaded_plugins:
                continue
            
            # 检查文件是否存在
            if not os.path.exists(plugin_path):
                continue
            
            try:
                # 加载插件库
                ctypes.cdll.LoadLibrary(plugin_path)
                self._loaded_plugins.add(plugin_path)
                self.logger.info(f"Loaded TensorRT plugin: {plugin_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load plugin {plugin_path}: {e}")
        
        # 初始化TensorRT插件注册表
        try:
            trt_logger = trt.Logger(trt.Logger.WARNING)
            trt.init_libnvinfer_plugins(trt_logger, "")
            self.logger.info("Initialized TensorRT plugin registry")
        except Exception as e:
            self.logger.warning(f"Failed to initialize plugin registry: {e}")
    
    def _load_engine(self):
        """加载TensorRT引擎"""
        if not os.path.exists(self.engine_path):
            raise FileNotFoundError(f"Engine file not found: {self.engine_path}")
        
        self.logger.info(f"Loading TensorRT engine: {self.engine_path}")
        
        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)
        
        with open(self.engine_path, "rb") as f:
            engine_data = f.read()
        
        try:
            self.engine = runtime.deserialize_cuda_engine(engine_data)
            if self.engine is None:
                raise RuntimeError("Failed to deserialize engine")
            self.context = self.engine.create_execution_context()
        except Exception as e:
            self.logger.error(f"Failed to load engine: {e}")
            raise
        
        # 获取输入输出信息
        nIO = self.engine.num_io_tensors
        lTensorName = [self.engine.get_tensor_name(i) for i in range(nIO)]
        nInput = [self.engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(
            trt.TensorIOMode.INPUT
        )
        
        self.input_names = lTensorName[:nInput]
        self.output_names = lTensorName[nInput:]
        
        # 检测引擎精度类型（通过检查输出张量的数据类型）
        self.engine_dtype = np.float32  # 默认FP32
        if len(self.output_names) > 0:
            first_output_dtype = self.engine.get_tensor_dtype(self.output_names[0])
            if first_output_dtype == trt.float16:
                self.engine_dtype = np.float16
                self.logger.info("检测到FP16引擎")
            elif first_output_dtype == trt.float32:
                self.engine_dtype = np.float32
                self.logger.info("检测到FP32引擎")
            else:
                self.logger.warning(f"未知的引擎精度类型: {first_output_dtype}，使用FP32")
        
        self.logger.info(f"Engine输入数量: {nInput}")
        self.logger.info(f"Engine输出数量: {nIO - nInput}")
        self.logger.info(f"输入名称: {self.input_names}")
        self.logger.info(f"输出名称: {self.output_names}")
        self.logger.info(f"引擎精度: {self.engine_dtype}")
        
        # 创建CUDA流
        self.stream = cudart.cudaStreamCreate()[1]
    
    def infer(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """执行推理"""
        # 准备输入数据
        bufferH = []
        for name in self.input_names:
            if name not in inputs:
                raise ValueError(f"Missing input: {name}")
            input_data = inputs[name]
            
            # 检查输入张量的数据类型，如果引擎是FP16且输入是FP32，尝试转换为FP16
            # 但对于某些输入（如int32类型的spatial_shapes），保持原类型
            input_tensor_dtype = self.engine.get_tensor_dtype(name)
            if input_tensor_dtype == trt.float16 and input_data.dtype == np.float32:
                # 对于FP16引擎的浮点输入，转换为FP16
                input_data = input_data.astype(np.float16)
            elif input_tensor_dtype == trt.int32 and input_data.dtype != np.int32:
                # 对于int32输入，确保是int32类型
                input_data = input_data.astype(np.int32)
            
            bufferH.append(input_data)
        
        # 准备输出缓冲区 - 根据引擎精度类型设置dtype
        for i, name in enumerate(self.output_names):
            shape = self.context.get_tensor_shape(name)
            # 获取该张量的实际数据类型
            tensor_dtype = self.engine.get_tensor_dtype(name)
            if tensor_dtype == trt.float16:
                output_dtype = np.float16
            elif tensor_dtype == trt.float32:
                output_dtype = np.float32
            elif tensor_dtype == trt.int32:
                output_dtype = np.int32
            else:
                # 默认使用引擎的全局精度类型
                output_dtype = self.engine_dtype
            bufferH.append(np.zeros(shape, dtype=output_dtype))
        
        # 分配GPU内存
        bufferD = []
        for i in range(len(bufferH)):
            bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])
        
        # 复制输入数据到GPU
        for i in range(len(self.input_names)):
            cudart.cudaMemcpy(
                bufferD[i],
                bufferH[i].ctypes.data,
                bufferH[i].nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            )
        
        # 设置张量地址
        for i, name in enumerate(self.input_names + self.output_names):
            self.context.set_tensor_address(name, int(bufferD[i]))
        
        # 执行推理
        self.context.execute_async_v3(stream_handle=self.stream)
        cudart.cudaStreamSynchronize(self.stream)
        
        # 复制输出数据到CPU
        outputs = {}
        for i, name in enumerate(self.output_names):
            idx = len(self.input_names) + i
            cudart.cudaMemcpy(
                bufferH[idx].ctypes.data,
                bufferD[idx],
                bufferH[idx].nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            )
            outputs[name] = bufferH[idx].copy()
        
        # 释放GPU内存
        for b in bufferD:
            cudart.cudaFree(b)
        
        return outputs
    
    def __del__(self):
        """清理资源"""
        if self.stream is not None:
            cudart.cudaStreamDestroy(self.stream)


def compute_metrics(pytorch_output: np.ndarray, engine_output: np.ndarray, name: str, analyze_plugin: bool = False) -> Dict:
    """计算输出差异指标"""
    if pytorch_output.shape != engine_output.shape:
        return {
            "name": name,
            "error": f"Shape mismatch: PyTorch {pytorch_output.shape} vs Engine {engine_output.shape}"
        }
    
    # 检查NaN和Inf值
    pytorch_has_nan = np.isnan(pytorch_output).any()
    pytorch_has_inf = np.isinf(pytorch_output).any()
    engine_has_nan = np.isnan(engine_output).any()
    engine_has_inf = np.isinf(engine_output).any()
    
    if pytorch_has_nan or pytorch_has_inf or engine_has_nan or engine_has_inf:
        return {
            "name": name,
            "error": f"NaN/Inf detected: PyTorch (NaN: {pytorch_has_nan}, Inf: {pytorch_has_inf}), "
                     f"Engine (NaN: {engine_has_nan}, Inf: {engine_has_inf})",
            "pytorch_stats": {
                "min": float(np.nanmin(pytorch_output)) if not pytorch_has_nan else None,
                "max": float(np.nanmax(pytorch_output)) if not pytorch_has_nan else None,
                "nan_count": int(np.isnan(pytorch_output).sum()),
                "inf_count": int(np.isinf(pytorch_output).sum()),
            },
            "engine_stats": {
                "min": float(np.nanmin(engine_output)) if not engine_has_nan else None,
                "max": float(np.nanmax(engine_output)) if not engine_has_nan else None,
                "nan_count": int(np.isnan(engine_output).sum()),
                "inf_count": int(np.isinf(engine_output).sum()),
            }
        }
    
    # 确保数据类型一致（统一转换为float32进行比较，避免FP16精度损失）
    if pytorch_output.dtype != engine_output.dtype:
        # 将两个数组都转换为float32进行比较
        pytorch_output = pytorch_output.astype(np.float32)
        engine_output = engine_output.astype(np.float32)
    
    diff = pytorch_output - engine_output
    abs_diff = np.abs(diff)
    
    metrics = {
        "name": name,
        "shape": pytorch_output.shape,
        "dtype": str(pytorch_output.dtype),
        "mse": float(np.mean(diff ** 2)),
        "mae": float(np.mean(abs_diff)),
        "max_abs_diff": float(np.max(abs_diff)),
        "mean_abs_diff": float(np.mean(abs_diff)),
        "std_abs_diff": float(np.std(abs_diff)),
        "relative_error": float(np.mean(abs_diff / (np.abs(pytorch_output) + 1e-8))),
        "pytorch_stats": {
            "min": float(np.min(pytorch_output)),
            "max": float(np.max(pytorch_output)),
            "mean": float(np.mean(pytorch_output)),
            "std": float(np.std(pytorch_output)),
        },
        "engine_stats": {
            "min": float(np.min(engine_output)),
            "max": float(np.max(engine_output)),
            "mean": float(np.mean(engine_output)),
            "std": float(np.std(engine_output)),
        },
    }
    
    # 如果启用插件误差分析，添加更详细的统计信息
    if analyze_plugin:
        # 计算误差分布
        abs_diff_flat = abs_diff.flatten()
        metrics["error_distribution"] = {
            "p50": float(np.percentile(abs_diff_flat, 50)),
            "p75": float(np.percentile(abs_diff_flat, 75)),
            "p90": float(np.percentile(abs_diff_flat, 90)),
            "p95": float(np.percentile(abs_diff_flat, 95)),
            "p99": float(np.percentile(abs_diff_flat, 99)),
        }
        
        # 计算大误差的比例（超过mean+2*std的误差）
        threshold = np.mean(abs_diff_flat) + 2 * np.std(abs_diff_flat)
        large_error_ratio = float(np.sum(abs_diff_flat > threshold) / len(abs_diff_flat))
        metrics["large_error_ratio"] = large_error_ratio
        metrics["large_error_threshold"] = float(threshold)
        
        # 对于anchor相关的输出，分析各个维度的误差
        if "anchor" in name.lower() and len(pytorch_output.shape) >= 2:
            # anchor通常是 [B, N, 11]，分析每个维度的误差
            if pytorch_output.shape[-1] == 11:
                dim_errors = []
                for dim in range(11):
                    dim_diff = np.abs(pytorch_output[..., dim] - engine_output[..., dim])
                    dim_errors.append({
                        "dim": dim,
                        "mean_abs_diff": float(np.mean(dim_diff)),
                        "max_abs_diff": float(np.max(dim_diff)),
                    })
                metrics["dimension_errors"] = dim_errors
        
        # 对于关键点相关的输出（如果形状匹配 [B, N, num_pts, 3]）
        if len(pytorch_output.shape) == 4 and pytorch_output.shape[-1] == 3:
            # 可能是关键点输出，分析每个点的误差
            point_errors = []
            num_points = pytorch_output.shape[2]
            for pt_idx in range(min(num_points, 13)):  # 通常最多13个点（7固定+6可学习）
                pt_diff = np.abs(pytorch_output[:, :, pt_idx, :] - engine_output[:, :, pt_idx, :])
                point_errors.append({
                    "point_idx": pt_idx,
                    "mean_abs_diff": float(np.mean(pt_diff)),
                    "max_abs_diff": float(np.max(pt_diff)),
                })
            metrics["point_errors"] = point_errors
    
    return metrics


def analyze_error_patterns(
    pytorch_anchor: np.ndarray,
    engine_anchor: np.ndarray,
    logger: logging.Logger
) -> Dict:
    """
    分析anchor误差模式，找出误差与anchor属性（尺寸、角度等）的关系
    
    Args:
        pytorch_anchor: [B, N, 11] PyTorch anchor输出
        engine_anchor: [B, N, 11] TensorRT engine anchor输出
        logger: 日志记录器
    
    Returns:
        包含误差模式分析的字典
    """
    from dataset.config.nusc_std_bbox3d import W, L, H, SIN_YAW, COS_YAW, X, Y, Z
    
    if pytorch_anchor.shape != engine_anchor.shape:
        return {"error": "Shape mismatch"}
    
    # 计算各维度误差
    diff = pytorch_anchor - engine_anchor
    abs_diff = np.abs(diff)
    
    # 提取anchor属性
    anchor_x = pytorch_anchor[:, :, X]
    anchor_y = pytorch_anchor[:, :, Y]
    anchor_z = pytorch_anchor[:, :, Z]
    anchor_w = np.exp(pytorch_anchor[:, :, W])  # 实际宽度
    anchor_l = np.exp(pytorch_anchor[:, :, L])  # 实际长度
    anchor_h = np.exp(pytorch_anchor[:, :, H])  # 实际高度
    anchor_sin_yaw = pytorch_anchor[:, :, SIN_YAW]
    anchor_cos_yaw = pytorch_anchor[:, :, COS_YAW]
    anchor_yaw = np.arctan2(anchor_sin_yaw, anchor_cos_yaw)
    
    # 计算anchor尺寸（对角线长度）
    anchor_size = np.sqrt(anchor_w**2 + anchor_l**2 + anchor_h**2)
    
    # 计算位置误差
    pos_error_x = abs_diff[:, :, X]
    pos_error_y = abs_diff[:, :, Y]
    pos_error_z = abs_diff[:, :, Z]
    pos_error_total = np.sqrt(pos_error_x**2 + pos_error_y**2 + pos_error_z**2)
    
    # 分析误差与anchor属性的关系
    patterns = {}
    
    # 1. 误差与anchor尺寸的关系
    size_bins = np.percentile(anchor_size.flatten(), [0, 25, 50, 75, 100])
    for i in range(len(size_bins) - 1):
        mask = (anchor_size >= size_bins[i]) & (anchor_size < size_bins[i+1])
        if np.sum(mask) > 0:
            patterns[f"size_bin_{i}"] = {
                "size_range": (float(size_bins[i]), float(size_bins[i+1])),
                "count": int(np.sum(mask)),
                "mean_pos_error": float(np.mean(pos_error_total[mask])),
                "mean_x_error": float(np.mean(pos_error_x[mask])),
                "mean_y_error": float(np.mean(pos_error_y[mask])),
            }
    
    # 2. 误差与角度的关系
    angle_bins = np.linspace(-np.pi, np.pi, 9)  # 8个角度区间
    for i in range(len(angle_bins) - 1):
        mask = (anchor_yaw >= angle_bins[i]) & (anchor_yaw < angle_bins[i+1])
        if np.sum(mask) > 0:
            patterns[f"angle_bin_{i}"] = {
                "angle_range": (float(angle_bins[i]), float(angle_bins[i+1])),
                "count": int(np.sum(mask)),
                "mean_pos_error": float(np.mean(pos_error_total[mask])),
                "mean_x_error": float(np.mean(pos_error_x[mask])),
                "mean_y_error": float(np.mean(pos_error_y[mask])),
            }
    
    # 3. 误差与位置的关系（距离原点的距离）
    distance_from_origin = np.sqrt(anchor_x**2 + anchor_y**2 + anchor_z**2)
    dist_bins = np.percentile(distance_from_origin.flatten(), [0, 25, 50, 75, 100])
    for i in range(len(dist_bins) - 1):
        mask = (distance_from_origin >= dist_bins[i]) & (distance_from_origin < dist_bins[i+1])
        if np.sum(mask) > 0:
            patterns[f"distance_bin_{i}"] = {
                "distance_range": (float(dist_bins[i]), float(dist_bins[i+1])),
                "count": int(np.sum(mask)),
                "mean_pos_error": float(np.mean(pos_error_total[mask])),
                "mean_x_error": float(np.mean(pos_error_x[mask])),
                "mean_y_error": float(np.mean(pos_error_y[mask])),
            }
    
    # 4. 找出误差最大的anchor
    max_error_indices = np.unravel_index(np.argmax(pos_error_total), pos_error_total.shape)
    patterns["max_error_anchor"] = {
        "batch_idx": int(max_error_indices[0]),
        "anchor_idx": int(max_error_indices[1]),
        "pos_error": float(pos_error_total[max_error_indices]),
        "x_error": float(pos_error_x[max_error_indices]),
        "y_error": float(pos_error_y[max_error_indices]),
        "z_error": float(pos_error_z[max_error_indices]),
        "anchor_size": float(anchor_size[max_error_indices]),
        "anchor_angle": float(anchor_yaw[max_error_indices]),
        "anchor_distance": float(distance_from_origin[max_error_indices]),
    }
    
    # 输出分析结果
    logger.info("=" * 80)
    logger.info("误差模式分析")
    logger.info("=" * 80)
    logger.info(f"最大位置误差: {patterns['max_error_anchor']['pos_error']:.6f}")
    logger.info(f"  位置: batch={patterns['max_error_anchor']['batch_idx']}, anchor={patterns['max_error_anchor']['anchor_idx']}")
    logger.info(f"  误差分解: X={patterns['max_error_anchor']['x_error']:.6f}, Y={patterns['max_error_anchor']['y_error']:.6f}, Z={patterns['max_error_anchor']['z_error']:.6f}")
    logger.info(f"  Anchor属性: size={patterns['max_error_anchor']['anchor_size']:.2f}, angle={patterns['max_error_anchor']['anchor_angle']:.3f}, distance={patterns['max_error_anchor']['anchor_distance']:.2f}")
    
    logger.info("\n按anchor尺寸分组的误差:")
    for key, value in patterns.items():
        if key.startswith("size_bin_"):
            logger.info(f"  尺寸范围 [{value['size_range'][0]:.2f}, {value['size_range'][1]:.2f}]: "
                       f"平均位置误差={value['mean_pos_error']:.6f}, "
                       f"X误差={value['mean_x_error']:.6f}, Y误差={value['mean_y_error']:.6f}, "
                       f"样本数={value['count']}")
    
    logger.info("\n按角度分组的误差:")
    for key, value in patterns.items():
        if key.startswith("angle_bin_"):
            logger.info(f"  角度范围 [{value['angle_range'][0]:.3f}, {value['angle_range'][1]:.3f}]: "
                       f"平均位置误差={value['mean_pos_error']:.6f}, "
                       f"X误差={value['mean_x_error']:.6f}, Y误差={value['mean_y_error']:.6f}, "
                       f"样本数={value['count']}")
    
    return patterns


def check_plugin_implementation(logger: logging.Logger) -> Dict:
    """
    检查SparseBox3DKeyPointsPlugin的实现，分析可能的FP16精度问题
    
    Args:
        logger: 日志记录器
    
    Returns:
        插件实现分析结果
    """
    import os
    
    logger.info("=" * 80)
    logger.info("插件实现检查")
    logger.info("=" * 80)
    
    analysis = {}
    
    # 1. 检查插件文件是否存在
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '../..'))
    plugin_path = os.path.join(project_root, "deploy/sparsebox_plugin/lib/SparseBox3DKeyPointsPlugin.so")
    kernel_path = os.path.join(project_root, "deploy/sparsebox_plugin/SparseBox3DKeyPointsKernel.cu")
    
    analysis["plugin_file_exists"] = os.path.exists(plugin_path)
    analysis["kernel_file_exists"] = os.path.exists(kernel_path)
    
    if analysis["plugin_file_exists"]:
        logger.info(f"✓ 插件库文件存在: {plugin_path}")
    else:
        logger.warning(f"✗ 插件库文件不存在: {plugin_path}")
    
    if analysis["kernel_file_exists"]:
        logger.info(f"✓ Kernel源文件存在: {kernel_path}")
        
        # 读取kernel文件，检查关键实现
        try:
            with open(kernel_path, 'r', encoding='utf-8') as f:
                kernel_content = f.read()
            
            # 检查关键实现点
            checks = {
                "FP32中间计算": "toFloat" in kernel_content or "float" in kernel_content,
                "旋转矩阵计算": "rotation" in kernel_content.lower() or "matmul" in kernel_content.lower(),
                "中心点累加": "center" in kernel_content.lower() and ("+" in kernel_content or "add" in kernel_content.lower()),
                "数值范围检查": "clamp" in kernel_content.lower() or "fmaxf" in kernel_content or "fminf" in kernel_content,
            }
            
            analysis["implementation_checks"] = checks
            
            logger.info("\n实现检查:")
            for check_name, passed in checks.items():
                status = "✓" if passed else "✗"
                logger.info(f"  {status} {check_name}: {'通过' if passed else '未找到'}")
            
            # 检查FP16精度处理
            if "toFloat" in kernel_content:
                logger.info("\n✓ 检测到FP16到FP32的转换（toFloat函数）")
                logger.info("  建议: 确保所有关键计算（旋转矩阵、中心点累加）都在FP32精度下进行")
            else:
                logger.warning("\n✗ 未检测到FP16到FP32的转换")
                logger.warning("  警告: 如果使用FP16，可能导致精度损失")
            
            # 检查数值稳定性
            if "clamp" in kernel_content.lower() or "fmaxf" in kernel_content:
                logger.info("\n✓ 检测到数值范围检查/裁剪")
                logger.info("  这有助于提高数值稳定性")
            else:
                logger.warning("\n✗ 未检测到数值范围检查")
                logger.warning("  建议: 添加数值范围检查以提高FP16下的稳定性")
            
        except Exception as e:
            logger.warning(f"读取kernel文件失败: {e}")
            analysis["kernel_read_error"] = str(e)
    else:
        logger.warning(f"✗ Kernel源文件不存在: {kernel_path}")
    
    # 2. 检查插件参数
    logger.info("\n插件使用建议:")
    logger.info("  1. 对于FP16引擎，建议在关键计算步骤使用FP32精度")
    logger.info("  2. 旋转矩阵乘法和中心点累加是误差的主要来源，应特别关注")
    logger.info("  3. 考虑使用混合精度：输入输出FP16，中间计算FP32")
    
    return analysis


def compare_keypoints(
    pytorch_keypoints: List[Dict],
    logger: logging.Logger
) -> Dict:
    """
    对比关键点输出（目前只能分析PyTorch端的关键点，因为TensorRT引擎无法输出中间层）
    
    Args:
        pytorch_keypoints: PyTorch捕获的关键点列表
        logger: 日志记录器
    
    Returns:
        关键点分析结果
    """
    if not pytorch_keypoints:
        return {"error": "No keypoints captured"}
    
    logger.info("=" * 80)
    logger.info("关键点输出分析")
    logger.info("=" * 80)
    
    analysis = {}
    for i, kp_data in enumerate(pytorch_keypoints):
        layer_name = kp_data['layer']
        keypoints = kp_data['keypoints']
        anchor = kp_data['anchor']
        
        logger.info(f"\n层 {i}: {layer_name}")
        logger.info(f"  关键点形状: {keypoints.shape}")
        
        if anchor is not None:
            logger.info(f"  Anchor形状: {anchor.shape}")
            # 分析关键点的统计信息
            kp_np = keypoints.detach().cpu().numpy()
            analysis[layer_name] = {
                "shape": list(keypoints.shape),
                "keypoints_stats": {
                    "min": float(np.min(kp_np)),
                    "max": float(np.max(kp_np)),
                    "mean": float(np.mean(kp_np)),
                    "std": float(np.std(kp_np)),
                },
                "keypoints_range": {
                    "x_range": (float(np.min(kp_np[:, :, :, 0])), float(np.max(kp_np[:, :, :, 0]))),
                    "y_range": (float(np.min(kp_np[:, :, :, 1])), float(np.max(kp_np[:, :, :, 1]))),
                    "z_range": (float(np.min(kp_np[:, :, :, 2])), float(np.max(kp_np[:, :, :, 2]))),
                }
            }
            logger.info(f"  关键点范围: X[{analysis[layer_name]['keypoints_range']['x_range'][0]:.2f}, {analysis[layer_name]['keypoints_range']['x_range'][1]:.2f}], "
                       f"Y[{analysis[layer_name]['keypoints_range']['y_range'][0]:.2f}, {analysis[layer_name]['keypoints_range']['y_range'][1]:.2f}], "
                       f"Z[{analysis[layer_name]['keypoints_range']['z_range'][0]:.2f}, {analysis[layer_name]['keypoints_range']['z_range'][1]:.2f}]")
    
    logger.info("\n注意: TensorRT引擎无法直接输出中间层（关键点），需要修改引擎构建才能对比。")
    logger.info("建议: 检查SparseBox3DKeyPointsPlugin的实现，特别是FP16精度下的数值稳定性。")
    
    return analysis


def compute_3d_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    计算两个3D边界框的IoU（简化版本，基于2D投影）
    
    Args:
        box1: [x, y, z, w, l, h, yaw, ...] 或 [x, y, z, w, l, h, sin_yaw, cos_yaw, ...]
        box2: 同上
    
    Returns:
        IoU值 (0-1)
    """
    # 提取位置和尺寸
    x1, y1, z1 = box1[0], box1[1], box1[2]
    w1, l1, h1 = box1[3], box1[4], box1[5]
    
    x2, y2, z2 = box2[0], box2[1], box2[2]
    w2, l2, h2 = box2[3], box2[4], box2[5]
    
    # 计算中心点距离
    dx = x1 - x2
    dy = y1 - y2
    distance = np.sqrt(dx * dx + dy * dy)
    
    # 计算边界框对角线长度的一半作为阈值
    threshold1 = np.sqrt(l1 * l1 + w1 * w1) / 2.0
    threshold2 = np.sqrt(l2 * l2 + w2 * w2) / 2.0
    overlap_threshold = (threshold1 + threshold2) * 0.5
    
    # 如果距离太远，IoU为0
    if distance > overlap_threshold:
        return 0.0
    
    # 简化的IoU计算
    overlap_ratio = 1.0 - (distance / overlap_threshold)
    return max(0.0, overlap_ratio)


def compute_ap(pred_boxes: np.ndarray, pred_scores: np.ndarray, pred_labels: np.ndarray,
               gt_boxes: np.ndarray, gt_labels: np.ndarray,
               iou_threshold: float = 0.5, class_names: List[str] = None) -> Dict:
    """
    计算Average Precision (AP)
    
    Args:
        pred_boxes: [N, 9] 预测的边界框 (x, y, z, w, l, h, sin_yaw, cos_yaw, ...)
        pred_scores: [N] 预测的置信度分数
        pred_labels: [N] 预测的类别标签
        gt_boxes: [M, 9] Ground truth边界框
        gt_labels: [M] Ground truth类别标签
        iou_threshold: IoU阈值
        class_names: 类别名称列表
    
    Returns:
        包含各类别AP和mAP的字典
    """
    if len(pred_boxes) == 0 and len(gt_boxes) == 0:
        return {"mAP": 1.0, "per_class_AP": {}}
    
    if len(pred_boxes) == 0:
        return {"mAP": 0.0, "per_class_AP": {}}
    
    if len(gt_boxes) == 0:
        return {"mAP": 0.0, "per_class_AP": {}}
    
    # 获取所有类别
    all_classes = set()
    if len(pred_labels) > 0:
        all_classes.update(pred_labels.tolist())
    if len(gt_labels) > 0:
        all_classes.update(gt_labels.tolist())
    
    per_class_ap = {}
    
    for cls_id in all_classes:
        # 过滤出当前类别的预测和GT
        pred_mask = pred_labels == cls_id
        gt_mask = gt_labels == cls_id
        
        pred_boxes_cls = pred_boxes[pred_mask]
        pred_scores_cls = pred_scores[pred_mask]
        gt_boxes_cls = gt_boxes[gt_mask]
        
        if len(gt_boxes_cls) == 0:
            # 如果没有GT，所有预测都是FP
            if len(pred_boxes_cls) == 0:
                ap = 1.0
            else:
                ap = 0.0
            per_class_ap[cls_id] = ap
            continue
        
        if len(pred_boxes_cls) == 0:
            # 如果没有预测，AP为0
            per_class_ap[cls_id] = 0.0
            continue
        
        # 按置信度排序
        sorted_indices = np.argsort(pred_scores_cls)[::-1]
        pred_boxes_cls = pred_boxes_cls[sorted_indices]
        pred_scores_cls = pred_scores_cls[sorted_indices]
        
        # 计算TP和FP
        tp = np.zeros(len(pred_boxes_cls), dtype=bool)
        fp = np.zeros(len(pred_boxes_cls), dtype=bool)
        gt_matched = np.zeros(len(gt_boxes_cls), dtype=bool)
        
        for i, pred_box in enumerate(pred_boxes_cls):
            best_iou = 0.0
            best_gt_idx = -1
            
            for j, gt_box in enumerate(gt_boxes_cls):
                if gt_matched[j]:
                    continue
                
                iou = compute_3d_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_iou >= iou_threshold:
                tp[i] = True
                gt_matched[best_gt_idx] = True
            else:
                fp[i] = True
        
        # 计算累积TP和FP
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        # 计算recall和precision
        recalls = tp_cumsum / len(gt_boxes_cls)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
        
        # 计算AP（使用11点插值法）
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11.0
        
        per_class_ap[cls_id] = ap
    
    # 计算mAP
    if len(per_class_ap) > 0:
        map_value = np.mean(list(per_class_ap.values()))
    else:
        map_value = 0.0
    
    result = {
        "mAP": float(map_value),
        "per_class_AP": {int(k): float(v) for k, v in per_class_ap.items()}
    }
    
    # 如果有类别名称，添加类别名称映射
    if class_names is not None:
        result["per_class_AP_with_names"] = {
            class_names[int(k)] if int(k) < len(class_names) else f"class_{int(k)}": float(v)
            for k, v in per_class_ap.items()
        }
    
    return result


def validate_backbone(
    pytorch_model: nn.Module,
    engine: TensorRTEngine,
    img: torch.Tensor,
    logger: logging.Logger,
    analyze_plugin: bool = False
) -> Dict:
    """验证Backbone输出"""
    logger.info("=" * 80)
    logger.info("验证 Backbone")
    logger.info(f"引擎精度类型: {engine.engine_dtype}")
    logger.info("=" * 80)
    
    # 确保img在正确的设备上
    device = next(pytorch_model.parameters()).device
    if img.device != device:
        logger.warning(f"图像数据设备 ({img.device}) 与模型设备 ({device}) 不匹配，正在移动...")
        img = img.to(device)
    
    # PyTorch推理
    logger.info("执行PyTorch推理...")
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(img)
    
    pytorch_output_np = pytorch_output.detach().cpu().numpy()
    logger.info(f"PyTorch输出形状: {pytorch_output_np.shape}")
    
    # TensorRT推理
    logger.info("执行TensorRT推理...")
    img_np = img.detach().cpu().numpy()
    engine_inputs = {"img": img_np}
    engine_outputs = engine.infer(engine_inputs)
    
    # 获取引擎输出（通常只有一个输出，但可能有多个）
    if len(engine.output_names) == 1:
        engine_output_name = engine.output_names[0]
        engine_output_np = engine_outputs[engine_output_name]
    else:
        # 如果有多个输出，尝试找到名为"feature"或"output"的输出
        engine_output_name = None
        for name in ["feature", "output", "backbone_output"]:
            if name in engine_outputs:
                engine_output_name = name
                break
        if engine_output_name is None:
            engine_output_name = engine.output_names[0]
        engine_output_np = engine_outputs[engine_output_name]
    logger.info(f"TensorRT输出名称: {engine_output_name}, 形状: {engine_output_np.shape}")
    logger.info(f"TensorRT输出数据类型: {engine_output_np.dtype}, PyTorch输出数据类型: {pytorch_output_np.dtype}")
    
    # 检查输出中的NaN/Inf
    if np.isnan(engine_output_np).any():
        nan_count = np.isnan(engine_output_np).sum()
        logger.warning(f"TensorRT输出包含 {nan_count} 个NaN值")
    if np.isinf(engine_output_np).any():
        inf_count = np.isinf(engine_output_np).sum()
        logger.warning(f"TensorRT输出包含 {inf_count} 个Inf值")
    if np.isnan(pytorch_output_np).any():
        nan_count = np.isnan(pytorch_output_np).sum()
        logger.warning(f"PyTorch输出包含 {nan_count} 个NaN值")
    if np.isinf(pytorch_output_np).any():
        inf_count = np.isinf(pytorch_output_np).sum()
        logger.warning(f"PyTorch输出包含 {inf_count} 个Inf值")
    
    # 计算差异
    metrics = compute_metrics(pytorch_output_np, engine_output_np, "backbone_output", analyze_plugin=analyze_plugin)
    
    logger.info(f"Backbone验证结果:")
    logger.info(f"  MSE: {metrics['mse']:.6e}")
    logger.info(f"  MAE: {metrics['mae']:.6e}")
    logger.info(f"  最大绝对差异: {metrics['max_abs_diff']:.6e}")
    logger.info(f"  相对误差: {metrics['relative_error']:.6e}")
    
    return metrics


def get_ground_truth(data: Dict, device: torch.device, logger: logging.Logger = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    从数据中提取ground truth边界框和标签
    
    Args:
        data: 数据字典
        device: 设备
        logger: 日志记录器（可选，用于调试）
    
    Returns:
        (gt_boxes, gt_labels) 或 (None, None) 如果不存在
    """
    gt_boxes = None
    gt_labels = None
    
    # 调试：打印数据键
    if logger:
        logger.debug(f"数据字典中的键: {list(data.keys())}")
    
    # 尝试从不同位置获取ground truth
    if 'gt_bboxes_3d' in data:
        gt_boxes_data = data['gt_bboxes_3d']
        if logger:
            logger.debug(f"找到gt_bboxes_3d, 类型: {type(gt_boxes_data)}")
        
        # 处理DataContainer
        if hasattr(gt_boxes_data, 'data'):
            gt_boxes_data = gt_boxes_data.data
            if logger:
                logger.debug(f"从DataContainer提取, 类型: {type(gt_boxes_data)}")
        
        # 处理列表格式（可能嵌套）
        while isinstance(gt_boxes_data, list):
            if logger:
                logger.debug(f"gt_bboxes_3d是列表, 长度: {len(gt_boxes_data)}")
            if len(gt_boxes_data) > 0:
                gt_boxes_data = gt_boxes_data[0]
                if logger:
                    logger.debug(f"取第一个元素, 类型: {type(gt_boxes_data)}")
            else:
                if logger:
                    logger.warning("gt_bboxes_3d列表为空")
                return None, None
        
        # 转换为numpy数组
        if isinstance(gt_boxes_data, torch.Tensor):
            gt_boxes = gt_boxes_data.detach().cpu().numpy()
            if logger:
                logger.debug(f"gt_boxes形状: {gt_boxes.shape}")
        elif isinstance(gt_boxes_data, np.ndarray):
            gt_boxes = gt_boxes_data
            if logger:
                logger.debug(f"gt_boxes形状: {gt_boxes.shape}")
        else:
            if logger:
                logger.warning(f"gt_bboxes_3d类型不支持: {type(gt_boxes_data)}")
                logger.warning(f"gt_bboxes_3d值示例: {gt_boxes_data[:3] if hasattr(gt_boxes_data, '__getitem__') and len(gt_boxes_data) > 0 else gt_boxes_data}")
    else:
        if logger:
            logger.debug("数据字典中未找到'gt_bboxes_3d'键")
    
    if 'gt_labels_3d' in data:
        gt_labels_data = data['gt_labels_3d']
        if logger:
            logger.debug(f"找到gt_labels_3d, 类型: {type(gt_labels_data)}")
        
        # 处理DataContainer
        if hasattr(gt_labels_data, 'data'):
            gt_labels_data = gt_labels_data.data
            if logger:
                logger.debug(f"从DataContainer提取, 类型: {type(gt_labels_data)}")
        
        # 处理列表格式（可能嵌套）
        while isinstance(gt_labels_data, list):
            if logger:
                logger.debug(f"gt_labels_3d是列表, 长度: {len(gt_labels_data)}")
            if len(gt_labels_data) > 0:
                gt_labels_data = gt_labels_data[0]
                if logger:
                    logger.debug(f"取第一个元素, 类型: {type(gt_labels_data)}")
            else:
                if logger:
                    logger.warning("gt_labels_3d列表为空")
                return None, None
        
        # 转换为numpy数组
        if isinstance(gt_labels_data, torch.Tensor):
            gt_labels = gt_labels_data.detach().cpu().numpy()
            if logger:
                logger.debug(f"gt_labels形状: {gt_labels.shape}")
        elif isinstance(gt_labels_data, np.ndarray):
            gt_labels = gt_labels_data
            if logger:
                logger.debug(f"gt_labels形状: {gt_labels.shape}")
        else:
            if logger:
                logger.warning(f"gt_labels_3d类型不支持: {type(gt_labels_data)}")
                logger.warning(f"gt_labels_3d值示例: {gt_labels_data[:10] if hasattr(gt_labels_data, '__getitem__') and len(gt_labels_data) > 0 else gt_labels_data}")
    else:
        if logger:
            logger.debug("数据字典中未找到'gt_labels_3d'键")
    
    # 检查是否成功提取
    if gt_boxes is not None and gt_labels is not None:
        if len(gt_boxes) == 0:
            if logger:
                logger.warning("gt_boxes为空数组")
            return None, None
        if len(gt_labels) == 0:
            if logger:
                logger.warning("gt_labels为空数组")
            return None, None
        if logger:
            logger.info(f"成功提取ground truth: {len(gt_boxes)}个边界框, {len(gt_labels)}个标签")
    
    return gt_boxes, gt_labels


def convert_model_outputs_to_detections(
    model_outs: Dict,
    decoder,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将模型输出转换为检测结果（boxes, scores, labels）
    
    Args:
        model_outs: 模型输出字典，包含classification和prediction列表
        decoder: SparseBox3DDecoder实例
        device: 设备
    
    Returns:
        (boxes, scores, labels) 作为numpy数组
    """
    # 提取最后一个decoder的输出
    classification = model_outs["classification"]
    prediction = model_outs["prediction"]
    
    # 确保是列表格式，取最后一个
    if isinstance(classification, list):
        cls_scores = classification[-1]  # [1, 900, num_classes]
    else:
        cls_scores = classification
    
    if isinstance(prediction, list):
        box_preds = prediction[-1]  # [1, 900, 11]
    else:
        box_preds = prediction

    # 确保张量同时包含decoder维和batch维：[num_decoder, batch, num_query, ...]
    def ensure_decoder_batch_dims(tensor, name: str):
        if tensor is None:
            return None
        if tensor.dim() == 4:
            return tensor
        if tensor.dim() == 3:
            # 缺少decoder维
            return tensor.unsqueeze(0)
        if tensor.dim() == 2:
            # 同时缺少decoder和batch维
            return tensor.unsqueeze(0).unsqueeze(0)
        raise ValueError(
            f"{name} 维度为{tensor.dim()}，无法解析形状: {tensor.shape}. "
            "期望形状为 [num_decoder, batch, num_query, ...]"
        )

    cls_scores = ensure_decoder_batch_dims(cls_scores, "cls_scores")
    box_preds = ensure_decoder_batch_dims(box_preds, "box_preds")

    if cls_scores.dim() == 1:
        raise ValueError(
            f"cls_scores 维度为1，无法解析形状: {cls_scores.shape}. "
            "期望形状为 [batch, num_query, num_class]"
        )

    # 获取可选的track_id和quality
    track_id = model_outs.get("track_id")
    if track_id is not None and isinstance(track_id, list):
        track_id = track_id[-1]
    
    quality = model_outs.get("quality")
    if quality is not None and isinstance(quality, list):
        quality = quality[-1]
        quality = ensure_decoder_batch_dims(quality, "quality")
    
    # 使用decoder进行后处理
    # decoder.decode期望的输入格式：
    # - cls_scores: [1, 900, num_classes] tensor
    # - box_preds: [1, 900, 11] tensor
    # - track_id: [1, 900] tensor (可选)
    # - quality: [1, 900, 2] tensor (可选)
    results = decoder.decode(
        cls_scores,
        box_preds,
        track_id,
        quality,
        output_idx=-1,
    )
    
    if len(results) == 0:
        return np.array([]), np.array([]), np.array([])
    
    result = results[0]
    boxes = result["boxes_3d"].detach().cpu().numpy()
    scores = result["scores_3d"].detach().cpu().numpy()
    labels = result["labels_3d"].detach().cpu().numpy()
    
    return boxes, scores, labels


def convert_engine_outputs_to_detections(
    engine_outputs: Dict,
    decoder,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将TensorRT引擎输出转换为检测结果
    
    Args:
        engine_outputs: 引擎输出字典
        decoder: SparseBox3DDecoder实例
        device: 设备
    
    Returns:
        (boxes, scores, labels) 作为numpy数组
    """
    # 尝试找到引擎输出
    classification = None
    prediction = None
    quality = None
    track_id = None
    
    # 查找输出名称（可能的名称变体）
    output_mapping = {
        "classification": ["classification", "pred_class_score", "cls_score", "output_2"],
        "prediction": ["prediction", "pred_anchor", "anchor", "output_1"],
        "quality": ["quality", "pred_quality_score", "quality_score", "output_3"],
        "track_id": ["track_id", "track_ids", "output_4"],
    }
    
    for key, possible_names in output_mapping.items():
        for name in possible_names:
            if name in engine_outputs:
                if key == "classification":
                    classification = torch.from_numpy(engine_outputs[name]).to(device)
                elif key == "prediction":
                    prediction = torch.from_numpy(engine_outputs[name]).to(device)
                elif key == "quality":
                    quality = torch.from_numpy(engine_outputs[name]).to(device)
                elif key == "track_id":
                    track_id = torch.from_numpy(engine_outputs[name]).to(device)
                break
    
    if classification is None or prediction is None:
        # 如果找不到输出，返回空结果
        return np.array([]), np.array([]), np.array([])
    
    # 构建model_outs字典
    model_outs = {
        "classification": [classification],
        "prediction": [prediction],
    }
    if quality is not None:
        model_outs["quality"] = [quality]
    if track_id is not None:
        model_outs["track_id"] = track_id
    
    # 使用decoder进行后处理
    return convert_model_outputs_to_detections(model_outs, decoder, device)


def validate_head(
    pytorch_head: nn.Module,
    engine: TensorRTEngine,
    feature: torch.Tensor,
    spatial_shapes: torch.Tensor,
    level_start_index: torch.Tensor,
    instance_feature: torch.Tensor,
    anchor: torch.Tensor,
    time_interval: torch.Tensor,
    image_wh: torch.Tensor,
    lidar2img: torch.Tensor,
    logger: logging.Logger,
    analyze_plugin: bool = False,
    head_wrapper=None
) -> Dict:
    """验证Head输出"""
    logger.info("=" * 80)
    logger.info("验证 Head (第一帧)")
    logger.info(f"引擎精度类型: {engine.engine_dtype}")
    logger.info("=" * 80)
    
    # 确保所有输入在正确的设备上
    device = next(pytorch_head.parameters()).device
    feature = feature.to(device)
    spatial_shapes = spatial_shapes.to(device)
    level_start_index = level_start_index.to(device)
    instance_feature = instance_feature.to(device)
    anchor = anchor.to(device)
    time_interval = time_interval.to(device)
    image_wh = image_wh.to(device)
    lidar2img = lidar2img.to(device)
    
    logger.debug(f"输入设备检查: feature={feature.device}, instance_feature={instance_feature.device}, anchor={anchor.device}")
    
    # PyTorch推理
    logger.info("执行PyTorch推理...")
    pytorch_head.eval()
    with torch.no_grad():
        outputs = pytorch_head(
            feature,
            spatial_shapes,
            level_start_index,
            instance_feature,
            anchor,
            time_interval,
            image_wh,
            lidar2img,
        )
        pred_instance_feature, pred_anchor, pred_class_score, pred_quality_score = outputs
    
    # TensorRT推理
    logger.info("执行TensorRT推理...")
    engine_inputs = {
        "feature": feature.detach().cpu().numpy(),
        "spatial_shapes": spatial_shapes.detach().cpu().numpy().astype(np.int32),
        "level_start_index": level_start_index.detach().cpu().numpy().astype(np.int32),
        "instance_feature": instance_feature.detach().cpu().numpy(),
        "anchor": anchor.detach().cpu().numpy(),
        "time_interval": time_interval.detach().cpu().numpy(),
        "image_wh": image_wh.detach().cpu().numpy(),
        "lidar2img": lidar2img.detach().cpu().numpy(),
    }
    
    # 记录输入数据信息（用于调试）
    logger.debug(f"引擎输入数据类型: {[f'{k}: {v.dtype}' for k, v in engine_inputs.items()]}")
    
    engine_outputs = engine.infer(engine_inputs)
    
    # 记录输出数据信息（用于调试）
    logger.debug(f"引擎输出数据类型: {[f'{k}: {v.dtype}' for k, v in engine_outputs.items()]}")
    
    # 比较所有输出
    all_metrics = {}
    
    # 定义输出名称映射（处理可能的名称差异）
    output_mapping = {
        "pred_instance_feature": ["pred_instance_feature", "instance_feature", "output_0"],
        "pred_anchor": ["pred_anchor", "anchor", "output_1"],
        "pred_class_score": ["pred_class_score", "class_score", "cls_score", "output_2"],
        "pred_quality_score": ["pred_quality_score", "quality_score", "output_3"],
    }
    
    # 比较pred_instance_feature
    if pred_instance_feature is not None:
        engine_name = None
        for name in output_mapping["pred_instance_feature"]:
            if name in engine_outputs:
                engine_name = name
                break
        if engine_name:
            pytorch_np = pred_instance_feature.detach().cpu().numpy()
            engine_np = engine_outputs[engine_name]
            metrics = compute_metrics(pytorch_np, engine_np, "pred_instance_feature", analyze_plugin=analyze_plugin)
            all_metrics["pred_instance_feature"] = metrics
            logger.info(f"pred_instance_feature验证结果 (引擎输出名称: {engine_name}):")
            logger.info(f"  MSE: {metrics['mse']:.6e}, MAE: {metrics['mae']:.6e}, Max Diff: {metrics['max_abs_diff']:.6e}")
            if analyze_plugin and "error_distribution" in metrics:
                logger.info(f"  误差分布 - P50: {metrics['error_distribution']['p50']:.6e}, P95: {metrics['error_distribution']['p95']:.6e}, P99: {metrics['error_distribution']['p99']:.6e}")
                logger.info(f"  大误差比例: {metrics.get('large_error_ratio', 0):.2%}")
        else:
            logger.warning(f"未找到pred_instance_feature对应的引擎输出，可用输出: {list(engine_outputs.keys())}")
    
    # 比较pred_anchor
    if pred_anchor is not None:
        engine_name = None
        for name in output_mapping["pred_anchor"]:
            if name in engine_outputs:
                engine_name = name
                break
        if engine_name:
            pytorch_np = pred_anchor.detach().cpu().numpy()
            engine_np = engine_outputs[engine_name]
            metrics = compute_metrics(pytorch_np, engine_np, "pred_anchor", analyze_plugin=analyze_plugin)
            all_metrics["pred_anchor"] = metrics
            logger.info(f"pred_anchor验证结果 (引擎输出名称: {engine_name}):")
            logger.info(f"  MSE: {metrics['mse']:.6e}, MAE: {metrics['mae']:.6e}, Max Diff: {metrics['max_abs_diff']:.6e}")
            if analyze_plugin:
                if "error_distribution" in metrics:
                    logger.info(f"  误差分布 - P50: {metrics['error_distribution']['p50']:.6e}, P95: {metrics['error_distribution']['p95']:.6e}, P99: {metrics['error_distribution']['p99']:.6e}")
                    logger.info(f"  大误差比例: {metrics.get('large_error_ratio', 0):.2%}")
                if "dimension_errors" in metrics:
                    logger.info(f"  各维度误差:")
                    for dim_err in metrics["dimension_errors"]:
                        logger.info(f"    维度 {dim_err['dim']}: MAE={dim_err['mean_abs_diff']:.6e}, Max={dim_err['max_abs_diff']:.6e}")
        else:
            logger.warning(f"未找到pred_anchor对应的引擎输出，可用输出: {list(engine_outputs.keys())}")
    
    # 比较pred_class_score
    if pred_class_score is not None:
        engine_name = None
        for name in output_mapping["pred_class_score"]:
            if name in engine_outputs:
                engine_name = name
                break
        if engine_name:
            pytorch_np = pred_class_score.detach().cpu().numpy()
            engine_np = engine_outputs[engine_name]
            metrics = compute_metrics(pytorch_np, engine_np, "pred_class_score", analyze_plugin=analyze_plugin)
            all_metrics["pred_class_score"] = metrics
            logger.info(f"pred_class_score验证结果 (引擎输出名称: {engine_name}):")
            logger.info(f"  MSE: {metrics['mse']:.6e}, MAE: {metrics['mae']:.6e}, Max Diff: {metrics['max_abs_diff']:.6e}")
            if analyze_plugin and "error_distribution" in metrics:
                logger.info(f"  误差分布 - P50: {metrics['error_distribution']['p50']:.6e}, P95: {metrics['error_distribution']['p95']:.6e}, P99: {metrics['error_distribution']['p99']:.6e}")
                logger.info(f"  大误差比例: {metrics.get('large_error_ratio', 0):.2%}")
        else:
            logger.warning(f"未找到pred_class_score对应的引擎输出，可用输出: {list(engine_outputs.keys())}")
    
    # 比较pred_quality_score
    if pred_quality_score is not None:
        engine_name = None
        for name in output_mapping["pred_quality_score"]:
            if name in engine_outputs:
                engine_name = name
                break
        if engine_name:
            pytorch_np = pred_quality_score.detach().cpu().numpy()
            engine_np = engine_outputs[engine_name]
            metrics = compute_metrics(pytorch_np, engine_np, "pred_quality_score", analyze_plugin=analyze_plugin)
            all_metrics["pred_quality_score"] = metrics
            logger.info(f"pred_quality_score验证结果 (引擎输出名称: {engine_name}):")
            logger.info(f"  MSE: {metrics['mse']:.6e}, MAE: {metrics['mae']:.6e}, Max Diff: {metrics['max_abs_diff']:.6e}")
            if analyze_plugin and "error_distribution" in metrics:
                logger.info(f"  误差分布 - P50: {metrics['error_distribution']['p50']:.6e}, P95: {metrics['error_distribution']['p95']:.6e}, P99: {metrics['error_distribution']['p99']:.6e}")
                logger.info(f"  大误差比例: {metrics.get('large_error_ratio', 0):.2%}")
        else:
            logger.warning(f"未找到pred_quality_score对应的引擎输出，可用输出: {list(engine_outputs.keys())}")
    
    # 如果没有任何匹配的输出，打印所有可用的输出
    if len(all_metrics) == 0:
        logger.warning(f"未找到任何匹配的输出！")
        logger.warning(f"PyTorch输出: pred_instance_feature, pred_anchor, pred_class_score, pred_quality_score")
        logger.warning(f"TensorRT可用输出: {list(engine_outputs.keys())}")
    
    # 如果启用关键点捕获，分析关键点（通过head_wrapper访问）
    if head_wrapper is not None and hasattr(head_wrapper, 'capture_keypoints') and head_wrapper.capture_keypoints:
        if hasattr(head_wrapper, 'captured_keypoints') and head_wrapper.captured_keypoints:
            keypoints_analysis = compare_keypoints(head_wrapper.captured_keypoints, logger)
            all_metrics["keypoints_analysis"] = keypoints_analysis
    
    return all_metrics


def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    
    # 设置日志
    logger, _, _ = set_logger(args.log, save_file=True)
    logger.setLevel(logging.INFO)
    
    logger.info("=" * 80)
    logger.info("PyTorch vs TensorRT引擎验证")
    logger.info("=" * 80)
    logger.info(f"配置文件: {args.config}")
    logger.info(f"检查点: {args.checkpoint}")
    logger.info(f"Backbone引擎: {args.backbone_engine}")
    logger.info(f"Head引擎: {args.head_engine}")
    logger.info(f"样本索引: {args.sample_idx}")
    logger.info(f"样本数量: {args.num_samples}")
    
    # 如果启用插件误差分析，先检查插件实现
    if args.analyze_plugin_error or args.capture_keypoints or args.analyze_error_patterns:
        plugin_analysis = check_plugin_implementation(logger)
        logger.info("")
    
    # 设置随机种子
    set_random_seed(seed=100, deterministic=args.deterministic)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 读取配置
    cfg = read_cfg(args.config)
    cfg["model"]["img_backbone"]["init_cfg"] = {}
    
    # 构建数据集
    logger.info("构建数据集...")
    if args.use_val_dataset:
        logger.info("使用val数据集（包含ground truth）")
        dataset_config = cfg["data"]["val"]
        samples_per_gpu = dataset_config.pop("samples_per_gpu", 1)
        dataset = build_module(dataset_config)
    else:
        logger.info("使用test数据集（注意：test数据集通常不包含ground truth）")
        dataset_config = cfg["data"]["test"]
        samples_per_gpu = dataset_config.pop("samples_per_gpu", 1)
        dataset = build_module(dataset_config)
    logger.info(f"数据集大小: {len(dataset)}")
    
    # 构建模型
    logger.info("构建PyTorch模型...")
    model = build_module(cfg["model"])
    
    # 加载检查点
    logger.info(f"加载检查点: {args.checkpoint}")
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    
    # 设置FP16（如果需要）
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    
    model = model.to(device)
    model.eval()
    
    # 创建包装类
    backbone_wrapper = Sparse4DBackboneWrapper(model).to(device)
    head_wrapper = Sparse4DHead1stWrapper(model.head, capture_keypoints=args.capture_keypoints).to(device)
    
    # 加载TensorRT引擎
    logger.info("加载TensorRT引擎...")
    backbone_engine = TensorRTEngine(args.backbone_engine, logger, args.plugin_paths)
    head_engine = TensorRTEngine(args.head_engine, logger, args.plugin_paths)
    
    # 如果指定了head2nd引擎，也加载它
    head2nd_engine = None
    if args.head2nd_engine is not None:
        logger.info(f"加载Head第二帧引擎: {args.head2nd_engine}")
        head2nd_engine = TensorRTEngine(args.head2nd_engine, logger, args.plugin_paths)
    
    # 确定验证模式
    if args.validate_continuous_frames and args.head2nd_engine is not None:
        validate_mode = "continuous"  # 连续帧验证
    else:
        validate_mode = "independent"  # 独立第一帧验证（默认）
    
    # 验证模式说明
    if validate_mode == "continuous":
        logger.info("=" * 80)
        logger.info("验证模式: 连续帧验证")
        logger.info("  - 第一个样本: 第一帧（使用head1st引擎）")
        logger.info("  - 后续样本: 第二帧（使用head2nd引擎，会使用前一个样本的缓存）")
        logger.info("  - 注意: 后续样本的验证结果可能会与独立第一帧验证有差异")
        logger.info("=" * 80)
    else:
        logger.info("=" * 80)
        logger.info("验证模式: 独立第一帧验证（推荐）")
        logger.info("  - 每个样本都作为独立的第一帧验证（会重置instance_bank状态）")
        logger.info("  - 所有样本都使用head1st引擎")
        logger.info("  - 这样可以确保每个样本的验证结果一致，不会因为时序状态导致差异")
        logger.info("=" * 80)
    
    # 验证结果
    all_results = []
    
    # 处理指定数量的样本
    num_samples_to_process = min(args.num_samples, len(dataset))
    start_idx = args.sample_idx
    
    for sample_idx in range(start_idx, start_idx + num_samples_to_process):
        if sample_idx >= len(dataset):
            break
        
        logger.info("=" * 80)
        logger.info(f"处理样本 {sample_idx}")
        logger.info("=" * 80)
        
        # 获取数据
        single_data = [dataset[sample_idx]]
        data = collate_fn(single_data, samples_per_gpu=1)
        
        # 处理DataContainer格式的数据
        processed_data = {}
        for key, value in data.items():
            if hasattr(value, 'data'):  # DataContainer
                if isinstance(value.data, torch.Tensor):
                    # 确保tensor在正确的设备上
                    processed_data[key] = value.data.to(device)
                elif isinstance(value.data, list):
                    # 如果是列表，处理每个元素
                    processed_data[key] = [v.to(device) if isinstance(v, torch.Tensor) else v for v in value.data]
                else:
                    processed_data[key] = value.data
            elif isinstance(value, torch.Tensor):
                processed_data[key] = value.to(device)
            elif isinstance(value, list):
                # 处理列表中的tensor
                processed_data[key] = [v.to(device) if isinstance(v, torch.Tensor) else v for v in value]
            elif isinstance(value, np.ndarray):
                processed_data[key] = torch.from_numpy(value).to(device)
            else:
                processed_data[key] = value
        
        # 调试：打印数据键
        logger.debug(f"数据键: {list(processed_data.keys())}")
        
        # 提取输入数据 - 尝试多种可能的键名
        img = None
        if 'img' in processed_data:
            img = processed_data['img']
            if isinstance(img, list):
                img = img[0]  # 如果是列表，取第一个
        elif 'img_inputs' in processed_data:
            img_inputs = processed_data['img_inputs']
            if isinstance(img_inputs, list) and len(img_inputs) > 0:
                img = img_inputs[0]
            else:
                img = img_inputs
        else:
            # 尝试查找包含'img'的键
            img_keys = [k for k in processed_data.keys() if 'img' in k.lower()]
            if img_keys:
                logger.warning(f"未找到标准的'img'或'img_inputs'键，尝试使用: {img_keys[0]}")
                img = processed_data[img_keys[0]]
                if isinstance(img, list):
                    img = img[0]
            else:
                raise KeyError(f"无法找到图像数据。可用的键: {list(processed_data.keys())}")
        
        if img is None:
            raise ValueError("无法获取图像数据")
        
        # 确保img在正确的设备上
        if isinstance(img, torch.Tensor):
            img = img.to(device)
            logger.info(f"图像数据形状: {img.shape}, 设备: {img.device}")
        else:
            logger.info(f"图像数据形状: {img.shape if hasattr(img, 'shape') else 'unknown'}")
        
        # 验证Backbone
        backbone_metrics = validate_backbone(backbone_wrapper, backbone_engine, img, logger, analyze_plugin=args.analyze_plugin_error)
        
        # 获取backbone输出（用于head验证）
        with torch.no_grad():
            feature_maps = model.extract_feat(img)
            feature = feature_maps[0]
            spatial_shapes = feature_maps[1]
            level_start_index = feature_maps[2]
        
        # 确保spatial_shapes和level_start_index是正确的格式
        if isinstance(spatial_shapes, torch.Tensor):
            spatial_shapes = spatial_shapes.to(device)
        else:
            spatial_shapes = torch.tensor(spatial_shapes, device=device)
        
        if isinstance(level_start_index, torch.Tensor):
            level_start_index = level_start_index.to(device)
        else:
            level_start_index = torch.tensor(level_start_index, device=device)
        
        # 获取head输入
        instance_bank = model.head.instance_bank
        batch_size = feature.shape[0]
        
        # 获取metas数据
        img_metas_data = processed_data.get("img_metas")
        if hasattr(img_metas_data, 'data'):
            img_metas_list = img_metas_data.data
        else:
            img_metas_list = img_metas_data
        
        # 先获取image_wh和lidar2img（在构建metas之前）
        if isinstance(img_metas_list[0], dict):
            image_wh = img_metas_list[0].get("image_wh")
            lidar2img = img_metas_list[0].get("lidar2img")
        else:
            # 尝试从DataContainer中获取
            image_wh = processed_data.get("image_wh")
            lidar2img = processed_data.get("lidar2img")
        
        # 转换为tensor并移到正确设备
        if image_wh is not None:
            if isinstance(image_wh, np.ndarray):
                image_wh = torch.from_numpy(image_wh).to(device)
            elif isinstance(image_wh, torch.Tensor):
                image_wh = image_wh.to(device)
            else:
                image_wh = torch.tensor(image_wh, device=device)
        else:
            raise ValueError("无法获取image_wh")
        
        if lidar2img is not None:
            if isinstance(lidar2img, np.ndarray):
                lidar2img = torch.from_numpy(lidar2img).to(device)
            elif isinstance(lidar2img, torch.Tensor):
                lidar2img = lidar2img.to(device)
            else:
                lidar2img = torch.tensor(lidar2img, device=device)
        else:
            raise ValueError("无法获取lidar2img")
        
        # 从img_metas_list中提取timestamp
        timestamp = None
        if isinstance(img_metas_list[0], dict):
            timestamp = img_metas_list[0].get("timestamp")
        elif isinstance(img_metas_list, list) and len(img_metas_list) > 0:
            # 尝试从processed_data中获取timestamp
            if "timestamp" in processed_data:
                timestamp_data = processed_data["timestamp"]
                if hasattr(timestamp_data, 'data'):
                    timestamp_data = timestamp_data.data
                if isinstance(timestamp_data, list) and len(timestamp_data) > 0:
                    timestamp = timestamp_data[0]
                elif isinstance(timestamp_data, torch.Tensor):
                    timestamp = timestamp_data
                elif isinstance(timestamp_data, np.ndarray):
                    timestamp = torch.from_numpy(timestamp_data)
        
        # 如果仍然没有timestamp，使用默认值0.0
        if timestamp is None:
            timestamp = torch.tensor([0.0], device=device)
            logger.warning(f"未找到timestamp，使用默认值0.0")
        elif isinstance(timestamp, (int, float)):
            timestamp = torch.tensor([float(timestamp)], device=device)
        elif isinstance(timestamp, np.ndarray):
            timestamp = torch.from_numpy(timestamp).to(device)
        elif isinstance(timestamp, torch.Tensor):
            timestamp = timestamp.to(device)
            if timestamp.dim() == 0:
                timestamp = timestamp.unsqueeze(0)
        
        # 构建完整的metas字典，包含所有必需的字段
        metas = {
            "timestamp": timestamp,
            "img_metas": img_metas_list,
            "image_wh": image_wh,
            "lidar2img": lidar2img,
        }
        
        # 确保img_metas中的每个元素也包含这些字段（如果需要）
        if isinstance(img_metas_list, list) and len(img_metas_list) > 0:
            for i, meta in enumerate(img_metas_list):
                if isinstance(meta, dict):
                    if "image_wh" not in meta:
                        meta["image_wh"] = image_wh.cpu().numpy() if isinstance(image_wh, torch.Tensor) else image_wh
                    if "lidar2img" not in meta:
                        meta["lidar2img"] = lidar2img.cpu().numpy() if isinstance(lidar2img, torch.Tensor) else lidar2img
                    if "timestamp" not in meta and timestamp is not None:
                        meta["timestamp"] = timestamp.cpu().numpy() if isinstance(timestamp, torch.Tensor) else timestamp
                    # 添加可能缺失的字段（instance_bank.get可能需要）
                    if "global2lidar" not in meta:
                        # 如果没有global2lidar，使用单位矩阵（对于第一帧验证，这应该足够）
                        meta["global2lidar"] = np.eye(4, dtype=np.float32)
                    if "lidar2global" not in meta:
                        # 如果没有lidar2global，使用单位矩阵
                        meta["lidar2global"] = np.eye(4, dtype=np.float32)
        
        # 判断是否为第一帧
        if validate_mode == "continuous":
            is_first_frame = (sample_idx == start_idx)
        else:
            is_first_frame = True  # 独立验证模式，每个样本都是第一帧
        
        # 如果是第一帧或独立验证模式，重置instance_bank状态
        if is_first_frame:
            logger.info(f"样本 {sample_idx}: 作为第一帧处理（重置instance_bank状态）")
            instance_bank.reset()
        else:
            logger.info(f"样本 {sample_idx}: 作为第二帧处理（使用前一个样本的缓存）")
        
        # 确保metas包含timestamp（instance_bank.get需要）
        # 注意：即使我们已经构建了metas，也要确保timestamp存在
        if "timestamp" not in metas or metas["timestamp"] is None:
            metas["timestamp"] = timestamp
        
        # 确保img_metas是列表格式（instance_bank.get期望列表）
        if not isinstance(metas["img_metas"], list):
            if isinstance(metas["img_metas"], dict):
                metas["img_metas"] = [metas["img_metas"]]
            else:
                metas["img_metas"] = [metas["img_metas"]]
        
        # 确保instance_bank中的self.metas也是列表格式（如果存在）
        # 这很重要，因为instance_bank.get会访问self.metas["img_metas"][i]
        # 必须在调用get之前修复，因为get方法内部会直接访问self.metas["img_metas"][i]
        
        # 修复instance_bank.metas的格式（如果存在）
        if hasattr(instance_bank, 'metas') and instance_bank.metas is not None:
            ensure_img_metas_list_format(instance_bank.metas, "instance_bank.metas", logger)
        
        # 修复传入的metas格式
        ensure_img_metas_list_format(metas, "metas", logger)
        
        # 使用monkey patch修复instance_bank.get中的问题
        # 保存原始的get方法
        original_get = instance_bank.get
        
        def patched_get(batch_size, metas=None, dn_metas=None):
            # 在调用原始get之前，再次确保self.metas["img_metas"]是列表格式
            # 这很重要，因为instance_bank.get会访问self.metas["img_metas"][i]
            # 必须在每次调用时都检查，因为self.metas可能在之前的调用中被修改
            if hasattr(instance_bank, 'metas') and instance_bank.metas is not None:
                ensure_img_metas_list_format(instance_bank.metas, "instance_bank.metas (in get)", logger)
                # 验证修复后的格式
                if isinstance(instance_bank.metas, dict) and "img_metas" in instance_bank.metas:
                    img_metas = instance_bank.metas["img_metas"]
                    if isinstance(img_metas, list) and len(img_metas) > 0:
                        # 确保每个元素都是字典
                        for i, item in enumerate(img_metas):
                            if not isinstance(item, dict):
                                error_msg = f"instance_bank.metas['img_metas'][{i}] 应该是字典，但得到 {type(item)}"
                                if logger:
                                    logger.error(error_msg)
                                raise TypeError(error_msg)
            # 也确保传入的metas["img_metas"]是列表格式
            if metas is not None:
                ensure_img_metas_list_format(metas, "metas (in get)", logger)
                # 验证修复后的格式
                if isinstance(metas, dict) and "img_metas" in metas:
                    img_metas = metas["img_metas"]
                    if isinstance(img_metas, list) and len(img_metas) > 0:
                        # 确保每个元素都是字典
                        for i, item in enumerate(img_metas):
                            if not isinstance(item, dict):
                                error_msg = f"metas['img_metas'][{i}] 应该是字典，但得到 {type(item)}"
                                if logger:
                                    logger.error(error_msg)
                                raise TypeError(error_msg)
            # 调用原始get方法
            return original_get(batch_size, metas, dn_metas)
        
        # 临时替换get方法
        instance_bank.get = patched_get
        
        try:
            instance_feature, anchor, temp_instance_feature, temp_anchor, time_interval = instance_bank.get(
                batch_size, metas, dn_metas=model.head.sampler.dn_metas if hasattr(model.head, 'sampler') else None
            )
        finally:
            # 恢复原始的get方法
            instance_bank.get = original_get
        
        # 确保instance_feature和anchor在正确的设备上
        if isinstance(instance_feature, torch.Tensor):
            instance_feature = instance_feature.to(device)
        if isinstance(anchor, torch.Tensor):
            anchor = anchor.to(device)
        
        # 确保time_interval是tensor
        if not isinstance(time_interval, torch.Tensor):
            if isinstance(time_interval, np.ndarray):
                time_interval = torch.from_numpy(time_interval).to(device)
            else:
                time_interval = torch.tensor(time_interval, device=device)
        
        # 选择使用哪个head引擎
        current_head_engine = head_engine if is_first_frame else (head2nd_engine if head2nd_engine is not None else head_engine)
        
        if not is_first_frame and head2nd_engine is None:
            logger.warning(f"样本 {sample_idx}: 作为第二帧但未指定head2nd引擎，使用head1st引擎（可能不准确）")
        
        # 验证Head并保存model_outs用于AP计算
        head_model_outs = None
        if is_first_frame:
            # 第一帧验证
            # 先执行PyTorch head推理以获取model_outs
            # 注意：这里需要使用完整的metas，包括timestamp和所有必需的字段
            # 确保img_metas包含所有必需的字段
            for i, meta in enumerate(metas["img_metas"]):
                if isinstance(meta, dict):
                    # 添加可能缺失的字段
                    if "global2lidar" not in meta:
                        # 如果没有global2lidar，尝试从lidar2img计算，或者使用单位矩阵
                        if "lidar2img" in meta:
                            # 简化处理：使用单位矩阵（对于第一帧验证，这应该足够）
                            meta["global2lidar"] = np.eye(4, dtype=np.float32)
                        else:
                            meta["global2lidar"] = np.eye(4, dtype=np.float32)
                    if "lidar2global" not in meta:
                        # 如果没有lidar2global，使用单位矩阵
                        meta["lidar2global"] = np.eye(4, dtype=np.float32)
            
            with torch.no_grad():
                feature_maps_for_head = [feature, spatial_shapes, level_start_index]
                # 使用之前构建的metas（包含timestamp）
                head_model_outs = model.head(feature_maps_for_head, metas)
            
            # 然后执行验证
            head_metrics = validate_head(
                head_wrapper,
                current_head_engine,
                feature,
                spatial_shapes,
                level_start_index,
                instance_feature,
                anchor,
                time_interval,
                image_wh,
                lidar2img,
                logger,
                analyze_plugin=args.analyze_plugin_error,
                head_wrapper=head_wrapper
            )
            
            # 如果启用误差模式分析，分析pred_anchor的误差模式
            if args.analyze_error_patterns and "pred_anchor" in head_metrics:
                pred_anchor_metrics = head_metrics["pred_anchor"]
                if "error" not in pred_anchor_metrics:
                    # 需要重新获取PyTorch和TensorRT的anchor输出
                    with torch.no_grad():
                        outputs = head_wrapper(
                            feature,
                            spatial_shapes,
                            level_start_index,
                            instance_feature,
                            anchor,
                            time_interval,
                            image_wh,
                            lidar2img,
                        )
                        pytorch_pred_anchor = outputs[1].detach().cpu().numpy()
                    
                    # 获取TensorRT的anchor输出
                    engine_inputs = {
                        "feature": feature.detach().cpu().numpy(),
                        "spatial_shapes": spatial_shapes.detach().cpu().numpy().astype(np.int32),
                        "level_start_index": level_start_index.detach().cpu().numpy().astype(np.int32),
                        "instance_feature": instance_feature.detach().cpu().numpy(),
                        "anchor": anchor.detach().cpu().numpy(),
                        "time_interval": time_interval.detach().cpu().numpy(),
                        "image_wh": image_wh.detach().cpu().numpy(),
                        "lidar2img": lidar2img.detach().cpu().numpy(),
                    }
                    engine_outputs = current_head_engine.infer(engine_inputs)
                    
                    # 找到pred_anchor输出
                    engine_pred_anchor = None
                    for name in ["pred_anchor", "anchor", "output_1"]:
                        if name in engine_outputs:
                            engine_pred_anchor = engine_outputs[name]
                            break
                    
                    if engine_pred_anchor is not None:
                        error_patterns = analyze_error_patterns(pytorch_pred_anchor, engine_pred_anchor, logger)
                        head_metrics["error_patterns"] = error_patterns
            
            # 如果启用关键点捕获，分析关键点
            if args.capture_keypoints and hasattr(head_wrapper, 'captured_keypoints') and head_wrapper.captured_keypoints:
                keypoints_analysis = compare_keypoints(head_wrapper.captured_keypoints, logger)
                head_metrics["keypoints_analysis"] = keypoints_analysis
        else:
            # 第二帧验证（需要额外的输入：temp_instance_feature, temp_anchor, mask, track_id）
            # 注意：当前脚本的head_wrapper只支持第一帧，第二帧需要额外的包装类
            # 这里暂时跳过第二帧验证，或者需要实现Sparse4DHead2ndWrapper
            logger.warning(f"样本 {sample_idx}: 第二帧验证功能尚未完全实现，跳过Head验证")
            head_metrics = {"note": "第二帧验证未实现"}
        
        # 如果是第一帧，缓存结果供下一帧使用（如果启用连续帧验证）
        if is_first_frame and validate_mode == "continuous" and sample_idx < start_idx + num_samples_to_process - 1:
            # 执行head推理以获取输出并缓存
            with torch.no_grad():
                outputs = head_wrapper(
                    feature,
                    spatial_shapes,
                    level_start_index,
                    instance_feature,
                    anchor,
                    time_interval,
                    image_wh,
                    lidar2img,
                )
                pred_instance_feature, pred_anchor, pred_class_score, pred_quality_score = outputs
            
            # 缓存结果到instance_bank（模拟实际推理流程）
            # 注意：这里需要调用instance_bank.cache，但需要确保metas正确
            try:
                # 确保metas["img_metas"]是列表格式（instance_bank.get需要）
                cache_metas = metas.copy() if metas is not None else {}
                # 使用统一的格式修复函数
                ensure_img_metas_list_format(cache_metas, "cache_metas", logger)
                model.head.instance_bank.cache(pred_instance_feature, pred_anchor, pred_class_score, cache_metas)
                logger.info(f"样本 {sample_idx}: 已缓存结果供下一帧使用")
            except Exception as e:
                logger.warning(f"样本 {sample_idx}: 缓存结果失败: {e}")
        
        # 计算AP值（如果存在ground truth）
        ap_results = {}
        gt_boxes, gt_labels = get_ground_truth(processed_data, device, logger)
        
        if gt_boxes is not None and gt_labels is not None and len(gt_boxes) > 0:
            logger.info("=" * 80)
            logger.info("计算AP值")
            logger.info("=" * 80)
            
            # 获取类别名称
            class_names = cfg.get("class_names", None)
            
            try:
                # PyTorch模型输出转换为检测结果
                logger.info("处理PyTorch模型输出...")
                # 使用之前head验证时已经得到的model_outs
                if head_model_outs is not None:
                    # 使用head的post_process方法获取检测结果
                    with torch.no_grad():
                        pytorch_results = model.head.post_process(head_model_outs, output_idx=-1)
                    
                    if pytorch_results and len(pytorch_results) > 0:
                        pytorch_result = pytorch_results[0]
                        if "boxes_3d" in pytorch_result:
                            pred_boxes_pytorch = pytorch_result["boxes_3d"].detach().cpu().numpy()
                            pred_scores_pytorch = pytorch_result["scores_3d"].detach().cpu().numpy()
                            pred_labels_pytorch = pytorch_result["labels_3d"].detach().cpu().numpy()
                            
                            # 计算PyTorch的AP
                            pytorch_ap = compute_ap(
                                pred_boxes_pytorch,
                                pred_scores_pytorch,
                                pred_labels_pytorch,
                                gt_boxes,
                                gt_labels,
                                iou_threshold=0.5,
                                class_names=class_names
                            )
                            ap_results["pytorch"] = pytorch_ap
                            logger.info(f"PyTorch mAP: {pytorch_ap['mAP']:.4f}")
                            if "per_class_AP_with_names" in pytorch_ap:
                                for cls_name, ap_val in pytorch_ap["per_class_AP_with_names"].items():
                                    logger.info(f"  {cls_name}: {ap_val:.4f}")
                        else:
                            logger.warning("PyTorch结果中未找到boxes_3d")
                            ap_results["pytorch"] = {"error": "未找到检测结果"}
                else:
                    logger.warning("未找到head_model_outs，无法计算PyTorch AP")
                    ap_results["pytorch"] = {"error": "未找到head_model_outs"}
            except Exception as e:
                logger.warning(f"计算PyTorch AP时出错: {e}")
                import traceback
                logger.warning(traceback.format_exc())
                ap_results["pytorch"] = {"error": str(e)}
            
            # TensorRT引擎输出转换为检测结果
            try:
                logger.info("处理TensorRT引擎输出...")
                # 重新执行head引擎推理以获取完整输出
                engine_inputs = {
                    "feature": feature.detach().cpu().numpy(),
                    "spatial_shapes": spatial_shapes.detach().cpu().numpy().astype(np.int32),
                    "level_start_index": level_start_index.detach().cpu().numpy().astype(np.int32),
                    "instance_feature": instance_feature.detach().cpu().numpy(),
                    "anchor": anchor.detach().cpu().numpy(),
                    "time_interval": time_interval.detach().cpu().numpy(),
                    "image_wh": image_wh.detach().cpu().numpy(),
                    "lidar2img": lidar2img.detach().cpu().numpy(),
                }
                engine_outputs = current_head_engine.infer(engine_inputs)
                
                # 尝试从引擎输出中提取所需的数据
                pred_anchor_engine = None
                pred_class_score_engine = None
                pred_quality_score_engine = None
                
                # 查找输出名称
                for name in engine_outputs.keys():
                    if "anchor" in name.lower() or "pred_anchor" in name.lower():
                        pred_anchor_engine = torch.from_numpy(engine_outputs[name]).to(device)
                    elif "class" in name.lower() or "cls" in name.lower():
                        pred_class_score_engine = torch.from_numpy(engine_outputs[name]).to(device)
                    elif "quality" in name.lower():
                        pred_quality_score_engine = torch.from_numpy(engine_outputs[name]).to(device)
                
                # 如果找不到，尝试使用默认名称
                if pred_anchor_engine is None and len(engine_outputs) > 0:
                    # 尝试按顺序获取
                    output_names = list(engine_outputs.keys())
                    if len(output_names) >= 2:
                        pred_anchor_engine = torch.from_numpy(engine_outputs[output_names[1]]).to(device)
                    if len(output_names) >= 3:
                        pred_class_score_engine = torch.from_numpy(engine_outputs[output_names[2]]).to(device)
                    if len(output_names) >= 4:
                        pred_quality_score_engine = torch.from_numpy(engine_outputs[output_names[3]]).to(device)
                
                if pred_anchor_engine is not None and pred_class_score_engine is not None:
                    # 确保tensor的形状正确
                    # pred_class_score_engine应该是[1, 900, num_classes]
                    # pred_anchor_engine应该是[1, 900, 11]
                    logger.debug(f"引擎输出形状 - pred_class_score: {pred_class_score_engine.shape}, pred_anchor: {pred_anchor_engine.shape}")
                    
                    # 检查并修复pred_class_score_engine的形状
                    logger.info(f"引擎输出原始形状 - pred_class_score: {pred_class_score_engine.shape}, pred_anchor: {pred_anchor_engine.shape}")
                    
                    if len(pred_class_score_engine.shape) == 2:
                        # 如果是[900, num_classes]，添加batch维度
                        pred_class_score_engine = pred_class_score_engine.unsqueeze(0)
                        logger.info(f"添加batch维度后 - pred_class_score: {pred_class_score_engine.shape}")
                    elif len(pred_class_score_engine.shape) == 1:
                        # 如果是[900*num_classes]，需要reshape
                        # 需要知道num_classes，通常从配置或模型中获取
                        num_classes = len(cfg.get("class_names", [])) if cfg.get("class_names") else 10
                        num_queries = 900
                        if pred_class_score_engine.shape[0] == num_queries * num_classes:
                            pred_class_score_engine = pred_class_score_engine.reshape(1, num_queries, num_classes)
                            logger.info(f"reshape后 - pred_class_score: {pred_class_score_engine.shape}")
                        else:
                            logger.warning(f"无法reshape pred_class_score: shape={pred_class_score_engine.shape}, 期望={num_queries * num_classes}")
                            raise ValueError(f"pred_class_score_engine形状不支持: {pred_class_score_engine.shape}")
                    elif len(pred_class_score_engine.shape) == 3:
                        # 已经是[1, 900, num_classes]或[num_decoder, 1, 900, num_classes]格式
                        if pred_class_score_engine.shape[0] == 1:
                            # [1, 900, num_classes]，不需要修改
                            logger.info(f"pred_class_score已经是3维: {pred_class_score_engine.shape}")
                        else:
                            # 可能是[num_decoder, 1, 900, num_classes]，取最后一个
                            logger.info(f"pred_class_score是4维，取最后一个decoder: {pred_class_score_engine.shape}")
                            pred_class_score_engine = pred_class_score_engine[-1]
                    elif len(pred_class_score_engine.shape) == 4:
                        # [num_decoder, 1, 900, num_classes]，取最后一个decoder
                        logger.info(f"pred_class_score是4维，取最后一个decoder: {pred_class_score_engine.shape}")
                        pred_class_score_engine = pred_class_score_engine[-1]
                    else:
                        logger.warning(f"pred_class_score_engine形状不支持: {pred_class_score_engine.shape}")
                        raise ValueError(f"pred_class_score_engine形状不支持: {pred_class_score_engine.shape}")
                    
                    if len(pred_anchor_engine.shape) == 2:
                        # 如果是[900, 11]，添加batch维度
                        pred_anchor_engine = pred_anchor_engine.unsqueeze(0)
                        logger.debug(f"添加batch维度后 - pred_anchor: {pred_anchor_engine.shape}")
                    elif len(pred_anchor_engine.shape) == 1:
                        # 如果是[900*11]，需要reshape
                        pred_anchor_engine = pred_anchor_engine.reshape(1, -1, 11)
                        logger.debug(f"reshape后 - pred_anchor: {pred_anchor_engine.shape}")
                    
                    if pred_quality_score_engine is not None:
                        if len(pred_quality_score_engine.shape) == 2:
                            pred_quality_score_engine = pred_quality_score_engine.unsqueeze(0)
                        elif len(pred_quality_score_engine.shape) == 1:
                            pred_quality_score_engine = pred_quality_score_engine.reshape(1, -1, 2)
                    
                    # 构建model_outs字典
                    # decoder.decode期望的输入格式：
                    # - cls_scores: 列表，每个元素是[1, 900, num_classes] tensor，或单个[1, 900, num_classes] tensor
                    # - box_preds: 列表，每个元素是[1, 900, 11] tensor，或单个[1, 900, 11] tensor
                    # 如果输入是列表，decoder会使用output_idx=-1来索引最后一个元素
                    # 如果输入是单个tensor，decoder会使用output_idx=-1来索引（但tensor需要是3维的）
                    
                    # 确保pred_class_score_engine和pred_anchor_engine是3维的[1, 900, ...]格式
                    logger.info(f"最终decoder输入形状 - classification: {pred_class_score_engine.shape}, prediction: {pred_anchor_engine.shape}")
                    
                    engine_model_outs = {
                        "classification": [pred_class_score_engine],  # 列表格式，decoder会取最后一个
                        "prediction": [pred_anchor_engine],  # 列表格式，decoder会取最后一个
                    }
                    if pred_quality_score_engine is not None:
                        engine_model_outs["quality"] = [pred_quality_score_engine]
                    
                    # 使用decoder进行后处理
                    try:
                        pred_boxes_engine, pred_scores_engine, pred_labels_engine = convert_model_outputs_to_detections(
                            engine_model_outs, model.head.decoder, device
                        )
                        
                        # 计算TensorRT的AP
                        if pred_boxes_engine is not None and len(pred_boxes_engine) > 0:
                            engine_ap = compute_ap(
                                pred_boxes_engine,
                                pred_scores_engine,
                                pred_labels_engine,
                                gt_boxes,
                                gt_labels,
                                iou_threshold=0.5,
                                class_names=class_names
                            )
                        else:
                            logger.warning("TensorRT检测结果为空")
                            engine_ap = {"mAP": 0.0, "per_class_AP": {}}
                    except Exception as e:
                        logger.warning(f"转换引擎输出为检测结果时出错: {e}")
                        import traceback
                        logger.warning(traceback.format_exc())
                        # 不抛出异常，而是记录错误并继续
                        ap_results["engine"] = {"error": f"转换失败: {str(e)}"}
                        engine_ap = None
                    
                    if engine_ap is not None:
                        ap_results["engine"] = engine_ap
                        logger.info(f"TensorRT mAP: {engine_ap['mAP']:.4f}")
                        if "per_class_AP_with_names" in engine_ap:
                            for cls_name, ap_val in engine_ap["per_class_AP_with_names"].items():
                                logger.info(f"  {cls_name}: {ap_val:.4f}")
                        
                        # 计算AP差异
                        if "pytorch" in ap_results and "mAP" in ap_results["pytorch"]:
                            pytorch_map = ap_results["pytorch"]["mAP"]
                            engine_map = engine_ap["mAP"]
                            ap_diff = pytorch_map - engine_map
                            ap_results["ap_diff"] = {
                                "pytorch_mAP": pytorch_map,
                                "engine_mAP": engine_map,
                                "difference": ap_diff,
                                "relative_diff": ap_diff / (pytorch_map + 1e-8) * 100.0
                            }
                            logger.info(f"AP差异: PyTorch {pytorch_map:.4f} vs TensorRT {engine_map:.4f}, 差异: {ap_diff:.4f} ({ap_results['ap_diff']['relative_diff']:.2f}%)")
                else:
                    logger.warning(f"无法从引擎输出中提取所需数据。可用输出: {list(engine_outputs.keys())}")
                    ap_results["engine"] = {"error": "无法提取引擎输出数据"}
            except Exception as e:
                logger.warning(f"计算TensorRT AP时出错: {e}")
                import traceback
                logger.warning(traceback.format_exc())
                ap_results["engine"] = {"error": str(e)}
        else:
            logger.info("未找到ground truth数据，跳过AP计算")
            ap_results["note"] = "未找到ground truth数据"
        
        # 保存结果
        sample_result = {
            "sample_idx": sample_idx,
            "backbone": backbone_metrics,
            "head": head_metrics,
            "ap": ap_results,
        }
        all_results.append(sample_result)
    
    # 保存验证结果
    import json
    results_file = os.path.join(args.output_dir, "validation_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    logger.info("=" * 80)
    logger.info("验证完成!")
    logger.info(f"结果已保存到: {results_file}")
    logger.info("=" * 80)
    
    # 打印汇总信息
    logger.info("\n汇总信息:")
    for result in all_results:
        logger.info(f"\n样本 {result['sample_idx']}:")
        logger.info(f"  Backbone - MSE: {result['backbone']['mse']:.6e}, MAE: {result['backbone']['mae']:.6e}")
        for name, metrics in result['head'].items():
            if isinstance(metrics, dict) and 'mse' in metrics:
                logger.info(f"  {name} - MSE: {metrics['mse']:.6e}, MAE: {metrics['mae']:.6e}")
        
        # 打印AP结果
        if 'ap' in result:
            ap_info = result['ap']
            if 'pytorch' in ap_info and 'mAP' in ap_info['pytorch']:
                logger.info(f"  PyTorch mAP: {ap_info['pytorch']['mAP']:.4f}")
            if 'engine' in ap_info and 'mAP' in ap_info['engine']:
                logger.info(f"  TensorRT mAP: {ap_info['engine']['mAP']:.4f}")
            if 'ap_diff' in ap_info:
                diff_info = ap_info['ap_diff']
                logger.info(f"  AP差异: {diff_info.get('difference', 0):.4f} ({diff_info.get('relative_diff', 0):.2f}%)")


if __name__ == "__main__":
    main()

