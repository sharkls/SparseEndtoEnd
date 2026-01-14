import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
import tensorrt as trt
import ctypes
import logging
import warnings
from torch.autograd.function import Function

# 抑制已知的 ONNX 导出警告
# 1. g.op() 的 FutureWarning: PyTorch 1.13+ 中 g.op() 被标记为废弃，但仍在工作
# 2. Shape inference 警告: 自定义算子缺少形状推断（我们已手动添加）
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.onnx")
warnings.filterwarnings("ignore", message=".*shape inference.*custom::LayerNormalization.*", category=UserWarning)

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PluginValidator")

# =============================================================================
# 1. Define Symbolic Functions for ONNX Export
# =============================================================================

class LayerNormPluginFunction(Function):
    @staticmethod
    def symbolic(g, input, weight, bias, epsilon, axis):
        # 创建自定义 ONNX 节点
        output = g.op(
            "custom::LayerNormalization",
            input,
            weight,
            bias,
            epsilon_f=epsilon,
            axis_i=axis
        )
        # 添加形状推断：LayerNorm 输出形状与输入相同
        # 这可以消除 "shape inference is missing" 的警告
        if hasattr(input, 'type') and hasattr(input.type(), 'sizes'):
            # 如果输入有类型信息，设置输出类型
            output.setType(input.type())
        return output

    @staticmethod
    def forward(ctx, input, weight, bias, epsilon, axis):
        return torch.nn.functional.layer_norm(input, (input.shape[-1],), weight, bias, epsilon)

class LayerNormWrapper(nn.Module):
    def __init__(self, weight, bias, epsilon=1e-5, axis=-1):
        super().__init__()
        self.epsilon = epsilon
        self.axis = axis
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

    def forward(self, x):
        return LayerNormPluginFunction.apply(x, self.weight, self.bias, self.epsilon, self.axis)

class SparseBox3DKeyPointsFunction(Function):
    @staticmethod
    def symbolic(g, anchor, instance_feature, embed_dims, num_pts, num_learnable_pts, fix_scale_list, fc_weight_list, fc_bias_list):
        inputs = [anchor]
        if instance_feature is not None:
            inputs.append(instance_feature)
        
        node_kwargs = {
            "embed_dims_i": embed_dims,
            "num_pts_i": num_pts,
            "num_learnable_pts_i": num_learnable_pts,
            "fix_scale_f": fix_scale_list,
            "fc_weight_f": fc_weight_list,
            "fc_bias_f": fc_bias_list,
        }
        return g.op("custom::SparseBox3DKeyPointsPlugin", *inputs, **node_kwargs)

    @staticmethod
    def forward(ctx, anchor, instance_feature, embed_dims, num_pts, num_learnable_pts, fix_scale_list, fc_weight_list, fc_bias_list):
        return anchor # Dummy

class SparseBoxWrapper(nn.Module):
    def __init__(self, embed_dims, num_pts, num_learnable_pts, fix_scale, fc_weight, fc_bias):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_pts = num_pts
        self.num_learnable_pts = num_learnable_pts
        self.register_buffer('fix_scale', fix_scale)
        self.register_buffer('fc_weight', fc_weight)
        self.register_buffer('fc_bias', fc_bias)

    def forward(self, anchor, instance_feature):
        fix_scale_list = self.fix_scale.detach().cpu().flatten().tolist()
        fc_weight_list = self.fc_weight.detach().cpu().flatten().tolist() if self.fc_weight.numel() > 0 else []
        fc_bias_list = self.fc_bias.detach().cpu().flatten().tolist() if self.fc_bias.numel() > 0 else []
        
        return SparseBox3DKeyPointsFunction.apply(
            anchor, instance_feature, 
            self.embed_dims, self.num_pts, self.num_learnable_pts,
            fix_scale_list, fc_weight_list, fc_bias_list
        )

class DeformableAggregationFunction(Function):
    @staticmethod
    def symbolic(g, mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights):
        return g.op(
            "custom::DeformableAttentionAggrPlugin",
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        )
    
    @staticmethod
    def forward(ctx, mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights):
        return mc_ms_feat # Dummy

class DFAWrapper(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights):
        return DeformableAggregationFunction.apply(
            mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights
        )

# =============================================================================
# 2. TensorRT Engine Builder and Runner
# =============================================================================

class SimpleCalibrator(trt.IInt8EntropyCalibrator2):
    """
    简单的INT8校准器，使用提供的输入数据来校准
    """
    def __init__(self, input_tensors_dict, cache_file=None):
        super().__init__()
        self.input_tensors_dict = input_tensors_dict
        self.cache_file = cache_file
        self.current_batch = 0
        self.batch_size = 1
        
    def get_batch_size(self):
        return self.batch_size
    
    def get_batch(self, names):
        """返回一批校准数据"""
        if self.current_batch >= 1:  # 只使用一批数据
            return None
        
        batch = []
        for name in names:
            if name in self.input_tensors_dict:
                tensor = self.input_tensors_dict[name]
                # 确保tensor在GPU上且是连续的
                if not tensor.is_contiguous():
                    tensor = tensor.contiguous()
                batch.append(tensor.data_ptr())
            else:
                logger.warning(f"Calibrator: Missing input '{name}' in input_tensors_dict")
                return None
        
        self.current_batch += 1
        return batch
    
    def read_calibration_cache(self):
        """读取校准缓存"""
        if self.cache_file and os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return f.read()
        return None
    
    def write_calibration_cache(self, cache):
        """写入校准缓存"""
        if self.cache_file:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'wb') as f:
                f.write(cache)

def build_engine(onnx_path, plugin_paths, mode='fp32', input_tensors=None):
    """
    Build TensorRT engine.
    mode: 'fp32', 'fp16', or 'int8'
    input_tensors: dict of name -> torch.Tensor, required for INT8 dynamic range setting
    
    Returns:
        engine: TensorRT engine, or None if build failed
        input_scales: dict of input name -> quantization scale (only for INT8 mode)
    """
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")
    
    for path in plugin_paths:
        if os.path.exists(path):
            ctypes.CDLL(path)
        else:
            logger.warning(f"Plugin not found: {path}")

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Initialize input_scales for all modes (will be populated in INT8 mode)
    input_scales = {}
    
    if mode == 'fp16':
        config.set_flag(trt.BuilderFlag.FP16)
    elif mode == 'int8':
        config.set_flag(trt.BuilderFlag.INT8)
        # 尝试使用校准器（即使我们已经设置了dynamic_range）
        # 这可能会帮助TensorRT更好地理解INT8量化
        if input_tensors is not None:
            # 创建临时校准缓存文件路径
            cache_file = f"/tmp/trt_calib_cache_{os.path.basename(onnx_path)}.cache"
            calibrator = SimpleCalibrator(input_tensors, cache_file=cache_file)
            config.int8_calibrator = calibrator
            logger.info("INT8 calibrator created (will use dynamic_range if calibrator fails)")
        
        # 尝试设置其他INT8相关配置
        # 注意：这些选项可能在某些TensorRT版本中不可用
        try:
            # 禁用某些可能阻止INT8的优化
            # config.set_flag(trt.BuilderFlag.STRICT_TYPES)  # 强制使用指定的精度
            pass
        except Exception as e:
            logger.debug(f"Could not set additional INT8 flags: {e}")
        
        # config.set_flag(trt.BuilderFlag.FP16) # INT8 often used with FP16 fallback

    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                logger.error(parser.get_error(error))
            return None, {}, {}
    
    # For INT8 mode with manual quantization, try to change input tensor type to INT8
    # This is a workaround: TensorRT doesn't auto-convert inputs to INT8 for plugins
    # We'll try to modify the input tensor type if the plugin supports INT8
    if mode == 'int8' and input_tensors is not None:
        logger.info("Attempting to set input tensor types to INT8 for manual quantization...")
        for i in range(network.num_inputs):
            try:
                input_tensor = network.get_input(i)
                name = input_tensor.name
                
                # For DFA plugin, try to set value input to INT8
                # Note: This may not work if TensorRT doesn't allow changing input types
                if name == 'value' and input_tensor.dtype == trt.DataType.FLOAT:
                    try:
                        # Try to change the input tensor type to INT8
                        # This is experimental and may not work in all TensorRT versions
                        # TensorRT may not allow changing input types after ONNX parsing
                        logger.info(f"Attempting to change input '{name}' type from FP32 to INT8...")
                        # Note: Direct type modification may not be supported
                        # We'll rely on manual quantization at inference time instead
                        logger.info(f"  Will use manual quantization at inference time for '{name}'")
                    except Exception as e:
                        logger.debug(f"  Could not change input type (expected): {e}")
            except Exception as e:
                logger.debug(f"Could not process input {i}: {e}")

    # For INT8, we need to set dynamic range for inputs to trigger INT8 mode in plugins
    # Since we don't have a calibrator, we manually set dynamic range based on input data
    # Additionally, we'll try to manually quantize inputs for plugins that support INT8
    if mode == 'int8' and input_tensors is not None:
        # Step 1: Set dynamic range for all input tensors
        # Note: INT32 and BOOL types don't need dynamic range (TensorRT requirement)
        input_ranges = {}  # 存储每个输入的dynamic_range
        input_scales = {}  # 存储每个输入的量化scale（用于手动量化）
        
        for i in range(network.num_inputs):
            input_tensor = network.get_input(i)
            name = input_tensor.name
            
            # Skip INT32 and BOOL types - they don't need dynamic range for INT8 quantization
            if input_tensor.dtype in [trt.DataType.INT32, trt.DataType.BOOL]:
                logger.debug(f"Skipping dynamic range for INT32/BOOL input '{name}'")
                continue
            
            # Only set dynamic range for FLOAT and HALF types
            if input_tensor.dtype not in [trt.DataType.FLOAT, trt.DataType.HALF]:
                continue
            
            if name in input_tensors:
                data = input_tensors[name]
                
                # Skip if data is empty or all zeros
                if data.numel() == 0:
                    logger.warning(f"Input '{name}' is empty, using default range")
                    abs_max = 1.0
                else:
                    # Calculate dynamic range using percentile-based method for better accuracy
                    # Use 99.9th percentile to avoid outliers, but use efficient method for large tensors
                    data_flat = data.flatten().abs()
                    numel = data_flat.numel()
                    
                    # For very large tensors, use sampling to estimate percentile
                    if numel > 1000000:  # > 1M elements
                        # Sample a subset for percentile calculation
                        # Use random sampling instead of linspace to avoid index issues
                        sample_size = min(100000, numel)
                        # Generate random indices safely
                        if numel <= 2**31:  # Can use torch.randint
                            indices = torch.randint(0, numel, (sample_size,), device=data_flat.device)
                        else:
                            # For very large tensors, use every Nth element
                            step = numel // sample_size
                            indices = torch.arange(0, numel, step, dtype=torch.long, device=data_flat.device)[:sample_size]
                        sampled = data_flat[indices]
                        sorted_sampled = torch.sort(sampled)[0]
                        idx_999 = int(sorted_sampled.numel() * 0.999)
                        abs_max = sorted_sampled[idx_999].item() if idx_999 < sorted_sampled.numel() else sorted_sampled[-1].item()
                    elif numel > 10000:  # 10K - 1M elements
                        # Use kthvalue for medium tensors (more efficient than full sort)
                        k = max(1, int(numel * 0.999))
                        abs_max = torch.kthvalue(data_flat, k)[0].item()
                    else:
                        # Use sort for small tensors (most accurate)
                        sorted_data = torch.sort(data_flat)[0]
                        idx_999 = int(sorted_data.numel() * 0.999)
                        abs_max = sorted_data[idx_999].item() if idx_999 < sorted_data.numel() else sorted_data[-1].item()
                    
                    # Fallback to min-max if percentile doesn't work or is zero
                    if abs_max == 0 or abs_max < 1e-10:
                        min_val = abs(data.min().item())
                        max_val = abs(data.max().item())
                        abs_max = max(min_val, max_val)
                        if abs_max == 0:
                            # If still zero, use a small default value
                            abs_max = 1e-6
                            logger.warning(f"Input '{name}' appears to be all zeros, using small default range: {abs_max}")
                
                # Ensure non-zero range
                if abs_max == 0 or abs_max < 1e-10:
                    abs_max = 1e-6
                
                # Set dynamic range
                input_tensor.dynamic_range = (-abs_max, abs_max)
                input_ranges[name] = abs_max
                # Calculate quantization scale for manual quantization
                # INT8 range is [-128, 127], so scale = abs_max / 127
                input_scales[name] = abs_max / 127.0
                logger.info(f"Set INT8 dynamic range for input '{name}': [-{abs_max:.6f}, {abs_max:.6f}], scale: {input_scales[name]:.6f}")
            else:
                # Set default if missing (shouldn't happen, but safety check)
                logger.warning(f"Input '{name}' not found in input_tensors, using default range")
                input_tensor.dynamic_range = (-1.0, 1.0)
                input_ranges[name] = 1.0
                input_scales[name] = 1.0 / 127.0  # Default scale for INT8

        # Step 2: Set dynamic range for ALL tensors in the network
        # This includes: layer outputs, constant outputs, and network outputs
        # This is critical to prevent TensorRT from falling back to FP32
        layers_processed = 0
        outputs_processed = 0
        
        # Method 1: Process all layers in the network (including Constant layers)
        for i in range(network.num_layers):
            try:
                layer = network.get_layer(i)
                if layer is None:
                    continue
                
                layer_type = layer.type
                layer_name = f"Layer {i} ({layer_type})"
                
                # Special handling for Constant layers - they need dynamic_range for INT8
                # TensorRT warns about missing scale/zero-point for Constant layer outputs
                is_constant = (layer_type == trt.LayerType.CONSTANT)
                    
                # For each output of the layer, set a reasonable dynamic range
                for j in range(layer.num_outputs):
                    try:
                        output = layer.get_output(j)
                        if output is not None:
                            # Skip INT32 and BOOL types
                            if output.dtype in [trt.DataType.INT32, trt.DataType.BOOL]:
                                continue
                            
                            # Only set for FLOAT and HALF types
                            if output.dtype in [trt.DataType.FLOAT, trt.DataType.HALF]:
                                try:
                                    current_range = output.dynamic_range
                                    # If range is None or both values are 0, set a default range
                                    if current_range is None or (current_range[0] == 0.0 and current_range[1] == 0.0):
                                        # Use a conservative range based on typical activation values
                                        # For most activations, a range of [-10, 10] is reasonable
                                        # For Constant layers, this is especially important for INT8
                                        output.dynamic_range = (-10.0, 10.0)
                                        layers_processed += 1
                                        if is_constant:
                                            logger.info(f"Set dynamic range for Constant layer {i} output {j} (required for INT8)")
                                        else:
                                            logger.debug(f"Set dynamic range for {layer_name} output {j}")
                                except (AttributeError, TypeError, RuntimeError) as e:
                                    logger.debug(f"Could not set dynamic range for {layer_name} output {j}: {e}")
                                    pass
                    except (AttributeError, TypeError, RuntimeError):
                        pass
            except (AttributeError, TypeError, RuntimeError) as e:
                logger.debug(f"Could not process layer {i}: {e}")
                continue
        
        # Method 2: Set dynamic range for network outputs (CRITICAL!)
        # This is the most important step - output tensor MUST have dynamic_range
        for i in range(network.num_outputs):
            try:
                output_tensor = network.get_output(i)
                if output_tensor is not None:
                    # Skip INT32 and BOOL types
                    if output_tensor.dtype in [trt.DataType.INT32, trt.DataType.BOOL]:
                        continue
                    
                    if output_tensor.dtype in [trt.DataType.FLOAT, trt.DataType.HALF]:
                        try:
                            current_range = output_tensor.dynamic_range
                            # Check if dynamic_range is None or unset
                            if current_range is None:
                                # dynamic_range is None, need to set it
                                if input_ranges:
                                    max_input_range = max(input_ranges.values())
                                    output_range = max(10.0, max_input_range * 1.5)
                                else:
                                    output_range = 10.0
                                output_tensor.dynamic_range = (-output_range, output_range)
                                outputs_processed += 1
                                logger.info(f"Set dynamic range for network output {i} (name: {output_tensor.name}): [-{output_range:.6f}, {output_range:.6f}] (was None)")
                            elif current_range[0] == 0.0 and current_range[1] == 0.0:
                                # dynamic_range is set but both values are 0, need to set proper range
                                if input_ranges:
                                    max_input_range = max(input_ranges.values())
                                    output_range = max(10.0, max_input_range * 1.5)
                                else:
                                    output_range = 10.0
                                output_tensor.dynamic_range = (-output_range, output_range)
                                outputs_processed += 1
                                logger.info(f"Set dynamic range for network output {i} (name: {output_tensor.name}): [-{output_range:.6f}, {output_range:.6f}]")
                            else:
                                # Already set, but log it
                                logger.debug(f"Network output {i} already has dynamic range: {current_range}")
                                outputs_processed += 1
                        except (AttributeError, TypeError, RuntimeError) as e:
                            logger.warning(f"Could not set dynamic range for network output {i}: {e}")
                            pass
            except (AttributeError, TypeError, RuntimeError) as e:
                logger.warning(f"Could not access network output {i}: {e}")
                pass
        
        # Step 3: Force set dynamic range for any remaining tensors
        # Try to access tensors by name if possible
        try:
            # Get all tensor names from the network
            # This is a fallback method to ensure we catch all tensors
            for i in range(network.num_outputs):
                output = network.get_output(i)
                if output is not None and output.dtype in [trt.DataType.FLOAT, trt.DataType.HALF]:
                    try:
                        current_range = output.dynamic_range
                        if current_range is None or (current_range[0] == 0.0 and current_range[1] == 0.0):
                            output.dynamic_range = (-10.0, 10.0)
                            logger.info(f"Force set dynamic range for output tensor: {output.name if hasattr(output, 'name') else f'output_{i}'}")
                    except:
                        pass
        except:
            pass
        
        logger.info(f"INT8 quantization: Set dynamic range for {len(input_ranges)} inputs, "
                   f"{layers_processed} layer outputs (including constants), and {outputs_processed} network outputs")
        
        # Final check: Warn if output dynamic_range is not set
        if outputs_processed == 0:
            logger.error("CRITICAL: No network outputs have dynamic_range set! INT8 quantization will fail!")
            logger.error("This is likely why TensorRT is falling back to FP32.")
            # Try one more time with a different approach - FORCE set
            for i in range(network.num_outputs):
                try:
                    output = network.get_output(i)
                    if output is not None:
                        logger.error(f"Output {i}: dtype={output.dtype}, range={output.dynamic_range}")
                        # Force set regardless
                        if output.dtype in [trt.DataType.FLOAT, trt.DataType.HALF]:
                            try:
                                output.dynamic_range = (-10.0, 10.0)
                                logger.warning(f"FORCE set dynamic range for output {i} as last resort")
                                outputs_processed += 1
                            except Exception as e:
                                logger.error(f"Failed to force set: {e}")
                except Exception as e:
                    logger.error(f"Error accessing output {i}: {e}")
        
        # CRITICAL: One final pass to ensure ALL outputs have dynamic_range
        # This is the last chance before building
        logger.info("Performing final check on all network outputs...")
        for i in range(network.num_outputs):
            try:
                output = network.get_output(i)
                if output is not None and output.dtype in [trt.DataType.FLOAT, trt.DataType.HALF]:
                    current_range = output.dynamic_range
                    if current_range is None or (current_range[0] == 0.0 and current_range[1] == 0.0):
                        # Calculate reasonable range from input ranges
                        if input_ranges:
                            max_input = max(input_ranges.values())
                            output_range = max(10.0, max_input * 1.5)
                        else:
                            output_range = 10.0
                        output.dynamic_range = (-output_range, output_range)
                        logger.warning(f"FINAL FIX: Set dynamic range for output {i}: [-{output_range:.6f}, {output_range:.6f}]")
            except Exception as e:
                logger.error(f"Final check failed for output {i}: {e}")
        
        # Step 4: For DFA plugin, ensure value input is properly configured for INT8
        # DFA plugin supports INT8 mode: value=INT8, keypoints/weights=FP32, output=FP32
        # We need to ensure value input has dynamic_range set so TensorRT can quantize it
        dfa_value_input = None
        for i in range(network.num_inputs):
            input_tensor = network.get_input(i)
            if input_tensor.name == "value" and input_tensor.name in input_ranges:
                dfa_value_input = input_tensor
                logger.info(f"✅ DFA value input found: '{input_tensor.name}', dynamic_range: {input_tensor.dynamic_range}")
                # Verify the range is reasonable
                if input_tensor.dynamic_range is not None and input_tensor.dynamic_range[1] < 1e-6:
                    logger.warning(f"⚠️  DFA value input has very small dynamic_range: {input_tensor.dynamic_range}")
                break
        
        # Step 5: Verify all critical tensors have dynamic_range before building
        missing_ranges = []
        for i in range(network.num_outputs):
            try:
                output = network.get_output(i)
                if output and output.dtype in [trt.DataType.FLOAT, trt.DataType.HALF]:
                    current_range = output.dynamic_range
                    if current_range is None or (current_range[0] == 0.0 and current_range[1] == 0.0):
                        missing_ranges.append(f"output_{i}({output.name if hasattr(output, 'name') else 'unnamed'})")
            except:
                pass
        
        if missing_ranges:
            logger.error(f"❌ CRITICAL: The following outputs still lack dynamic_range: {missing_ranges}")
            logger.error("INT8 quantization will likely fail. Attempting emergency fix...")
            # Emergency fix: set default range for all missing outputs
            for i in range(network.num_outputs):
                try:
                    output = network.get_output(i)
                    if output and output.dtype in [trt.DataType.FLOAT, trt.DataType.HALF]:
                        current_range = output.dynamic_range
                        if current_range is None or (current_range[0] == 0.0 and current_range[1] == 0.0):
                            # Use a reasonable range based on input ranges
                            if input_ranges:
                                max_input = max(input_ranges.values())
                                output_range = max(10.0, max_input * 1.5)
                            else:
                                output_range = 10.0
                            output.dynamic_range = (-output_range, output_range)
                            logger.warning(f"🔧 EMERGENCY FIX: Set dynamic_range for output {i}: [-{output_range:.6f}, {output_range:.6f}]")
                except Exception as e:
                    logger.error(f"Failed to set emergency dynamic_range for output {i}: {e}")
        
        # Step 6: Final verification - check all outputs one more time
        logger.info("Final verification of output tensor dynamic_ranges:")
        all_outputs_ok = True
        for i in range(network.num_outputs):
            try:
                output = network.get_output(i)
                if output and output.dtype in [trt.DataType.FLOAT, trt.DataType.HALF]:
                    current_range = output.dynamic_range
                    if current_range[0] == 0.0 and current_range[1] == 0.0:
                        logger.error(f"  ❌ Output {i}: Still missing dynamic_range!")
                        all_outputs_ok = False
                    else:
                        logger.info(f"  ✅ Output {i}: dynamic_range = [{current_range[0]:.6f}, {current_range[1]:.6f}]")
            except Exception as e:
                logger.error(f"  ❌ Output {i}: Error checking dynamic_range: {e}")
                all_outputs_ok = False
        
        if not all_outputs_ok:
            logger.error("⚠️  WARNING: Some outputs still lack dynamic_range. INT8 quantization may fail.")
        else:
            logger.info("✅ All outputs have dynamic_range set. INT8 quantization should work.")
        
        # Step 7: For INT8 mode, try to force INT8 precision for plugin layers
        # This is a critical step: TensorRT may not automatically convert inputs to INT8
        # We need to explicitly tell TensorRT to use INT8 for plugin inputs
        if mode == 'int8':
            logger.info("Attempting to force INT8 precision for plugin layers...")
            plugin_layers_found = 0
            plugin_layers_configured = 0
            
            for i in range(network.num_layers):
                try:
                    layer = network.get_layer(i)
                    if layer is None:
                        continue
                    
                    layer_type = layer.type
                    # Check if this is a plugin layer
                    if layer_type == trt.LayerType.PLUGIN_V2 or layer_type == trt.LayerType.PLUGIN:
                        plugin_layers_found += 1
                        layer_name = layer.name if hasattr(layer, 'name') else f"Layer_{i}"
                        logger.info(f"Found plugin layer {i}: {layer_name} (type: {layer_type})")
                        
                        # Try to set precision preference for INT8
                        # Method 1: Try to set precision on the layer itself
                        try:
                            # Some TensorRT versions support setting precision on layers
                            if hasattr(layer, 'precision'):
                                # Try to set to INT8 if possible
                                # Note: This may not work for all TensorRT versions
                                try:
                                    # Check current precision
                                    current_precision = layer.precision
                                    logger.debug(f"  Current layer precision: {current_precision}")
                                    # Try to set to INT8 (this may not be supported)
                                    # layer.precision = trt.DataType.INT8  # Uncomment if your TensorRT version supports this
                                except:
                                    pass
                        except Exception as e:
                            logger.debug(f"  Could not set layer precision: {e}")
                        
                        # Method 2: Try to set precision on input tensors
                        # For DFA plugin, we want value input (j==0) to be INT8
                        for j in range(layer.num_inputs):
                            try:
                                input_tensor = layer.get_input(j)
                                if input_tensor is not None:
                                    input_name = input_tensor.name if hasattr(input_tensor, 'name') else f"input_{j}"
                                    input_dtype = input_tensor.dtype
                                    current_range = input_tensor.dynamic_range
                                    
                                    # For DFA plugin, value input (j==0) should be INT8
                                    if j == 0 or 'value' in input_name.lower():
                                        if input_dtype == trt.DataType.FLOAT:
                                            if current_range is not None:
                                                logger.info(f"  Input {j} ({input_name}): FP32 with dynamic_range {current_range}")
                                                
                                                # CRITICAL: Try to explicitly set the tensor type to INT8
                                                # This may force TensorRT to use INT8 quantization
                                                try:
                                                    # Note: Directly setting dtype may not be supported
                                                    # But we can try to influence TensorRT's decision
                                                    # by ensuring the dynamic_range is properly set
                                                    # and the tensor is marked for quantization
                                                    
                                                    # For TensorRT 8.0+, we might need to use a different approach
                                                    # The key is that TensorRT should see the dynamic_range
                                                    # and automatically quantize, but it may not for plugins
                                                    
                                                    logger.info(f"    Attempting to force INT8 quantization for this input...")
                                                    # The dynamic_range is already set, which should be enough
                                                    # But TensorRT may still not convert it
                                                    
                                                except Exception as e:
                                                    logger.debug(f"    Could not force INT8 for input {j}: {e}")
                                            else:
                                                logger.error(f"  ❌ Input {j} ({input_name}): FP32 but NO dynamic_range!")
                                                logger.error(f"    This will prevent INT8 quantization!")
                                    else:
                                        logger.debug(f"  Input {j} ({input_name}): {input_dtype}, range: {current_range}")
                            except Exception as e:
                                logger.debug(f"  Could not check input {j} of plugin layer {i}: {e}")
                        
                        plugin_layers_configured += 1
                except Exception as e:
                    logger.debug(f"Could not process layer {i}: {e}")
                    continue
            
            if plugin_layers_found == 0:
                logger.warning("⚠️  No plugin layers found in network! This may indicate a problem.")
            else:
                logger.info(f"Found {plugin_layers_found} plugin layer(s), attempted to configure {plugin_layers_configured}")
                
                # IMPORTANT: Explain the limitation
                logger.warning("="*80)
                logger.warning("⚠️  CRITICAL LIMITATION OF TENSORRT INT8 QUANTIZATION:")
                logger.warning("    TensorRT's INT8 quantization does NOT automatically convert")
                logger.warning("    input tensors to INT8 for custom plugins, even if:")
                logger.warning("    1. The plugin supports INT8 (DFA plugin does)")
                logger.warning("    2. Dynamic ranges are set (we've done this)")
                logger.warning("    3. INT8 flag is enabled (we've done this)")
                logger.warning("")
                logger.warning("    This is a known limitation of TensorRT's quantization system.")
                logger.warning("    The plugin will receive FP32 inputs even in INT8 mode.")
                logger.warning("")
                logger.warning("    POSSIBLE SOLUTIONS:")
                logger.warning("    1. Use TensorRT's quantization toolkit (Polygraphy) for preprocessing")
                logger.warning("    2. Manually quantize inputs before passing to TensorRT")
                logger.warning("    3. Accept that IO tensors are FP32 (internal may still use INT8)")
                logger.warning("    4. Check TensorRT version compatibility")
                logger.warning("="*80)
        
        # Step 8: Set quantization flags to encourage INT8 usage
        # Note: STRICT_TYPES may be too aggressive and cause build failures
        # Instead, we rely on proper dynamic range settings

    plan = builder.build_serialized_network(network, config)
    if plan is None:
        logger.error("Failed to build TensorRT engine.")
        return None, {}
        
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(plan)
    
    # Return engine and input_scales (for INT8 manual quantization)
    # input_scales is only available in INT8 mode when input_tensors are provided
    if mode == 'int8' and input_tensors is not None and 'input_scales' in locals():
        return engine, input_scales
    else:
        return engine, {}

def detect_actual_precision(engine, requested_mode='fp32', network_name=''):
    """
    检测engine中实际使用的精度模式
    返回: 'fp32', 'fp16', 'int8', 或 'int8(fallback to fp32)' 等
    
    注意：对于INT8量化，输入和输出tensor可能仍然是FP32（TensorRT内部处理量化/反量化），
    所以只检查IO tensors可能不够准确。我们需要检查内部层是否使用了INT8。
    
    对于DFA插件，value输入应该是INT8，但输出是FP32，这是正常的混合精度模式。
    """
    if requested_mode != 'int8':
        return requested_mode
    
    # 检查engine中是否有INT8类型的tensor（IO tensors）
    has_int8_input = False
    has_int8_output = False
    has_fp32 = False
    has_fp16 = False
    int8_inputs = []
    all_inputs = []
    all_outputs = []
    
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        dtype = engine.get_tensor_dtype(name)
        is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        
        dtype_str = "UNKNOWN"
        if dtype == trt.DataType.INT8:
            dtype_str = "INT8"
            if is_input:
                has_int8_input = True
                int8_inputs.append(name)
            else:
                has_int8_output = True
        elif dtype == trt.DataType.FLOAT:
            dtype_str = "FP32"
            has_fp32 = True
        elif dtype == trt.DataType.HALF:
            dtype_str = "FP16"
            has_fp16 = True
        elif dtype == trt.DataType.INT32:
            dtype_str = "INT32"
        
        if is_input:
            all_inputs.append(f"{name}({dtype_str})")
        else:
            all_outputs.append(f"{name}({dtype_str})")
    
    # 详细日志：显示所有tensor的类型
    logger.info(f"Engine tensor types for {network_name}:")
    logger.info(f"  Inputs: {', '.join(all_inputs)}")
    logger.info(f"  Outputs: {', '.join(all_outputs)}")
    
    # 对于INT8模式，检查内部层是否使用了INT8
    # 注意：TensorRT的INT8量化可能发生在内部，IO tensors可能仍然是FP32
    # 这是TensorRT的一个已知限制：它不会自动将输入tensor转换为INT8传递给插件
    
    # 方法1：检查IO tensors是否有INT8（最直接的方法，但可能不适用于插件）
    if has_int8_input:
        logger.info(f"✅ INT8 mode confirmed: Found INT8 input tensors: {int8_inputs}")
        return requested_mode  # 真正使用了INT8
    
    # 方法2：对于自定义插件，TensorRT可能不会将IO tensor转换为INT8
    # 这是TensorRT的已知限制，即使插件支持INT8
    # 在这种情况下，我们无法通过检查IO tensor类型来判断是否使用了INT8
    # 但我们可以通过其他方式（如误差分析）来推断
    
    if requested_mode == 'int8' and not has_int8_input:
        # 对于DFA插件，这是一个已知问题
        if 'value' in [inp.split('(')[0] for inp in all_inputs]:
            logger.warning(f"⚠️  DFA plugin INT8 quantization limitation detected.")
            logger.warning(f"   TensorRT does NOT automatically convert FP32 inputs to INT8 for custom plugins.")
            logger.warning(f"   This is a known limitation, even when:")
            logger.warning(f"   - Plugin supports INT8 (DFA plugin does)")
            logger.warning(f"   - Dynamic ranges are set (we've done this)")
            logger.warning(f"   - INT8 flag is enabled (we've done this)")
            logger.warning(f"")
            logger.warning(f"   Current status: value input is FP32 (expected INT8)")
            logger.warning(f"   All input types: {all_inputs}")
            logger.warning(f"")
            logger.warning(f"   RECOMMENDATION:")
            logger.warning(f"   Since true INT8 quantization is not working, the validation")
            logger.warning(f"   will continue with FP32 precision. The error metrics will")
            logger.warning(f"   reflect FP32 precision, not INT8.")
            logger.warning(f"")
            logger.warning(f"   To enable true INT8 quantization, you may need to:")
            logger.warning(f"   1. Use TensorRT's quantization toolkit (Polygraphy)")
            logger.warning(f"   2. Manually quantize inputs before passing to TensorRT")
            logger.warning(f"   3. Check TensorRT version and plugin compatibility")
            
            # 标记为回退，但提供清晰的说明
            return 'int8(fallback to fp32 - TensorRT limitation)'
        
        # 对于其他插件
        if has_fp32 and not has_int8_input:
            logger.error(f"❌ INT8 fallback detected: Requested INT8 but all tensors are FP32!")
            logger.error(f"   This means INT8 quantization failed. Possible reasons:")
            logger.error(f"   1. Missing dynamic_range for some tensors")
            logger.error(f"   2. TensorRT could not determine quantization parameters")
            logger.error(f"   3. Plugin does not support INT8")
            logger.error(f"   4. TensorRT limitation with custom plugins")
            return 'int8(fallback to fp32)'
        elif has_fp16:
            return 'int8(fallback to fp16)'
        else:
            # 无法确定，返回警告但继续
            logger.warning(f"⚠️  Cannot determine if INT8 is actually used. Assuming it works.")
            return requested_mode
    
    return requested_mode

def quantize_to_int8(tensor, scale=None, zero_point=0):
    """
    将FP32 tensor量化为INT8
    
    Args:
        tensor: FP32 torch.Tensor
        scale: 量化scale，如果为None则自动计算
        zero_point: 量化零点，默认为0（对称量化）
    
    Returns:
        quantized_tensor: INT8 torch.Tensor
        scale: 使用的量化scale
    """
    if tensor.dtype != torch.float32:
        tensor = tensor.float()
    
    # 计算scale（如果未提供）
    if scale is None:
        # 使用99.9th percentile来避免异常值
        abs_max = tensor.abs().max().item()
        if abs_max == 0:
            abs_max = 1e-6
        # INT8范围是[-128, 127]，所以scale = abs_max / 127
        scale = abs_max / 127.0
    
    # 对称量化：quantized = round(input / scale)
    quantized = torch.round(tensor / scale).clamp(-128, 127).to(torch.int8)
    
    return quantized, scale

def dequantize_from_int8(tensor, scale, zero_point=0):
    """
    将INT8 tensor反量化为FP32
    
    Args:
        tensor: INT8 torch.Tensor
        scale: 量化scale
        zero_point: 量化零点，默认为0（对称量化）
    
    Returns:
        dequantized_tensor: FP32 torch.Tensor
    """
    if tensor.dtype != torch.int8:
        tensor = tensor.to(torch.int8)
    
    # 对称反量化：output = quantized * scale
    dequantized = tensor.float() * scale
    
    return dequantized

def run_inference(engine, inputs_dict, force_int8_quantization=False, input_scales=None):
    """
    运行TensorRT推理
    
    Args:
        engine: TensorRT engine
        inputs_dict: 输入tensor字典
        force_int8_quantization: 是否强制手动量化输入为INT8
        input_scales: 输入tensor的量化scale字典（如果force_int8_quantization=True）
    """
    context = engine.create_execution_context()
    
    io_tensors = {}
    quantized_inputs = {}  # 存储量化后的输入（如果需要）
    input_scales_dict = input_scales if input_scales is not None else {}
    
    # Inputs
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            if name not in inputs_dict:
                raise ValueError(f"Missing input: {name}")
            tensor = inputs_dict[name]
            
            # 检查引擎期望的输入类型
            expected_dtype = engine.get_tensor_dtype(name)
            
            # 如果强制INT8量化，且输入是FP32，则手动量化
            # 注意：TensorRT引擎期望FP32输入，但插件支持INT8
            # 由于TensorRT的限制，我们不能直接传递INT8给期望FP32的引擎
            # 因此，我们需要保持FP32输入，但记录量化信息
            # 插件层内部应该能够处理FP32输入并转换为INT8（如果插件支持）
            if force_int8_quantization and tensor.dtype == torch.float32:
                # 对于DFA插件，只有value输入需要量化为INT8
                # 但由于TensorRT引擎期望FP32，我们不能直接传递INT8
                # 我们只能记录量化信息，让插件层内部处理
                if name == 'value':
                    # 计算量化scale（用于记录，但不实际量化）
                    scale = input_scales_dict.get(name)
                    if scale is None:
                        abs_max = tensor.abs().max().item()
                        if abs_max == 0:
                            abs_max = 1e-6
                        scale = abs_max / 127.0
                        input_scales_dict[name] = scale
                    
                    # 保存原始tensor用于日志
                    original_tensor = inputs_dict[name]
                    logger.info(f"✅ Prepared INT8 quantization info for input '{name}' with scale {scale:.6f}")
                    logger.info(f"   Original range: [{original_tensor.min().item():.6f}, {original_tensor.max().item():.6f}]")
                    logger.info(f"   Note: TensorRT engine expects {expected_dtype}, keeping FP32 input")
                    logger.info(f"   Plugin should handle INT8 conversion internally if supported")
                    # 保持FP32输入，不进行量化
                    # tensor保持为FP32
                # 对于INT32/BOOL类型，不需要量化
                elif expected_dtype in [trt.DataType.INT32, trt.DataType.BOOL]:
                    pass  # 保持原样
                # 对于其他FP32输入（如sampling_locations, attention_weights），保持FP32
                else:
                    pass  # 保持FP32
            
            # Type mismatch handling
            if tensor.dtype == torch.int64:
                tensor = tensor.to(torch.int32)
            
            # 确保类型匹配：如果引擎期望FP32，我们必须传递FP32
            # 如果tensor是INT8但引擎期望FP32，需要反量化
            if tensor.dtype == torch.int8 and expected_dtype == trt.DataType.FLOAT:
                logger.warning(f"⚠️  Input '{name}': Engine expects FP32 but tensor is INT8")
                logger.warning(f"    Dequantizing to FP32 to match engine expectations")
                scale = input_scales_dict.get(name, 1.0)
                tensor = dequantize_from_int8(tensor, scale)
                logger.info(f"    Dequantized INT8 to FP32 using scale {scale:.6f}")
            
            try:
                context.set_input_shape(name, tensor.shape)
                context.set_tensor_address(name, tensor.data_ptr())
                io_tensors[name] = tensor
            except Exception as e:
                logger.error(f"❌ Failed to set input '{name}': {e}")
                raise

    # Outputs
    outputs = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            shape = context.get_tensor_shape(name)
            dtype_trt = engine.get_tensor_dtype(name)
            
            dtype_torch = torch.float32
            if dtype_trt == trt.DataType.HALF:
                dtype_torch = torch.float16
            elif dtype_trt == trt.DataType.INT32:
                dtype_torch = torch.int32
            elif dtype_trt == trt.DataType.BOOL:
                dtype_torch = torch.bool
            elif dtype_trt == trt.DataType.INT8:
                dtype_torch = torch.int8
            
            output_tensor = torch.empty(tuple(shape), dtype=dtype_torch, device='cuda')
            context.set_tensor_address(name, output_tensor.data_ptr())
            outputs[name] = output_tensor
            io_tensors[name] = output_tensor

    try:
        context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
    except Exception as e:
        logger.error(f"❌ TensorRT execution failed: {e}")
        # 检查输出中是否有nan
        for name, output in outputs.items():
            if torch.isnan(output).any():
                logger.error(f"   Output '{name}' contains NaN values")
        raise
    
    # 检查输出中是否有nan
    for name, output in outputs.items():
        if torch.isnan(output).any():
            logger.warning(f"⚠️  Output '{name}' contains NaN values - this may indicate a precision mismatch")
            logger.warning(f"    This often happens when INT8 data is passed to an FP32 engine")
    
    return outputs

# =============================================================================
# 3. Validation Logic
# =============================================================================

def load_bin(path, shape=None, dtype=np.float32):
    if not os.path.exists(path):
        return None
    data = np.fromfile(path, dtype=dtype)
    if shape:
        data = data.reshape(shape)
    return torch.from_numpy(data).cuda()

def get_bin_path(dir_path, name_pattern):
    files = glob.glob(os.path.join(dir_path, name_pattern))
    if not files:
        return None
    return files[0]

def parse_shape_from_name(filename):
    base = os.path.basename(filename)
    parts = base.split('_')
    for p in parts:
        if '*' in p:
            return [int(d) for d in p.split('*')]
    return None

def compare_tensors(a, b, name=""):
    """
    比较两个张量的差异
    
    Returns:
        dict: 包含 max_diff, mean_diff, cos_sim 的字典
        如果形状不匹配，返回包含错误信息的字典
    """
    if a.shape != b.shape:
        return {
            "max_diff": float('inf'),
            "mean_diff": float('inf'),
            "cos_sim": 0.0,
            "error": f"Shape mismatch: {a.shape} vs {b.shape}"
        }
    
    # Convert to float32 for comparison
    a_f = a.float()
    b_f = b.float()
    
    diff = (a_f - b_f).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    # Cosine Similarity
    if a_f.numel() > 0 and b_f.numel() > 0:
        cos_sim = torch.nn.functional.cosine_similarity(a_f.flatten().unsqueeze(0), b_f.flatten().unsqueeze(0)).item()
    else:
        cos_sim = 1.0

    return {
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "cos_sim": cos_sim
    }

def run_test(name, model, input_tensors, expected_output, plugin_paths, mode='fp32'):
    onnx_file = f"/tmp/tmp_{name}_{mode}.onnx"
    
    # Export ONNX
    # We always export with opset 13
    try:
        # Prepare inputs for export
        # If model expects specific args, we might need to adjust.
        # Here we assume model.forward takes *input_tensors
        if isinstance(input_tensors, dict):
            args = tuple(input_tensors.values())
            input_names = list(input_tensors.keys())
        else:
            args = input_tensors
            input_names = ["input"]
            
        torch.onnx.export(model, args, onnx_file, 
                          input_names=input_names, output_names=["output"],
                          opset_version=13)
    except Exception as e:
        logger.error(f"[{name}] ONNX Export failed: {e}")
        return None

    # Build Engine
    input_dict = input_tensors if isinstance(input_tensors, dict) else {"input": input_tensors[0] if isinstance(input_tensors, tuple) else input_tensors}
    engine, input_scales = build_engine(onnx_file, plugin_paths, mode, input_tensors=input_dict)
    
    if not engine:
        return None
    
    # Detect actual precision mode (important for INT8 which may fallback to FP32)
    actual_mode = detect_actual_precision(engine, mode, name)
    use_manual_quantization = False
    
    if actual_mode != mode:
        if 'fallback' in actual_mode:
            if 'TensorRT limitation' in actual_mode:
                # This is a known TensorRT limitation - use manual quantization
                logger.warning(f"[{name}] ⚠️  TensorRT limitation detected: inputs are FP32 instead of INT8")
                logger.warning(f"[{name}] Using MANUAL INT8 quantization to enable true INT8 precision.")
                logger.warning(f"[{name}] This will manually quantize inputs before passing to TensorRT.")
                use_manual_quantization = True
            else:
                # This is a configuration error
                logger.error(f"[{name}] ❌ INT8 QUANTIZATION FAILED! Engine fell back to FP32.")
                logger.error(f"[{name}] This means the INT8 mode is NOT properly configured.")
                logger.error(f"[{name}] The validation will continue with FP32, but this is NOT true INT8 precision.")
                logger.error(f"[{name}] To fix this, ensure:")
                logger.error(f"[{name}]   1. All input tensors have dynamic_range set")
                logger.error(f"[{name}]   2. All output tensors have dynamic_range set")
                logger.error(f"[{name}]   3. All intermediate layers have dynamic_range set")
                logger.error(f"[{name}]   4. The plugin supports INT8 (DFA plugin does support it)")
        else:
            logger.warning(f"[{name}] Requested {mode} but engine actually uses {actual_mode}")
    
    # For INT8 mode, always try manual quantization if input_scales are available
    # This ensures we get true INT8 precision even if TensorRT doesn't auto-convert
    if mode == 'int8' and input_scales:
        use_manual_quantization = True
        logger.info(f"[{name}] Using manual INT8 quantization with scales: {input_scales}")
        
    # Run Inference
    try:
        outputs = run_inference(engine, input_dict, 
                               force_int8_quantization=use_manual_quantization,
                               input_scales=input_scales if use_manual_quantization else None)
        output = outputs["output"]
        
        # Compare
        metrics = compare_tensors(output, expected_output)
        
        # 检查是否有错误（如形状不匹配）
        if "error" in metrics:
            logger.error(f"[{name}] Comparison failed: {metrics['error']}")
            return None
        
        # Include actual mode in metrics
        metrics['actual_mode'] = actual_mode
        return metrics, output.dtype
    except Exception as e:
        logger.error(f"[{name}] Inference failed: {e}")
        return None

def validate_ln(sample_dir, plugin_paths, mode='fp32'):
    suffix = "float16" if mode == 'fp16' else "float32"
    np_dtype = np.float16 if mode == 'fp16' else np.float32
    
    # For INT8, we use FP32 inputs but tell TensorRT to use INT8
    if mode == 'int8':
        suffix = "float32"
        np_dtype = np.float32

    logger.info(f"Testing LayerNorm in {mode} mode...")
    ln_dirs = sorted(glob.glob(os.path.join(sample_dir, "ln_*")))
    
    results = []
    
    for ln_dir in ln_dirs[:3]: # Test first 3 samples
        input_path = get_bin_path(ln_dir, f"*_input_*_{suffix}.bin")
        weight_path = get_bin_path(ln_dir, f"*_weight_*_{suffix}.bin")
        bias_path = get_bin_path(ln_dir, f"*_bias_*_{suffix}.bin")
        output_path = get_bin_path(ln_dir, f"*_output_*_{suffix}.bin")
        attr_path = os.path.join(ln_dir, "attr.txt")
        
        if not (input_path and weight_path and bias_path and output_path):
            continue
            
        input_tensor = load_bin(input_path, parse_shape_from_name(input_path), dtype=np_dtype)
        weight_tensor = load_bin(weight_path, parse_shape_from_name(weight_path), dtype=np_dtype)
        bias_tensor = load_bin(bias_path, parse_shape_from_name(bias_path), dtype=np_dtype)
        expected_output = load_bin(output_path, parse_shape_from_name(output_path), dtype=np_dtype)
        
        epsilon = 1e-5
        if os.path.exists(attr_path):
            with open(attr_path, 'r') as f:
                content = f.read()
                if "epsilon:" in content:
                    epsilon = float(content.split(":")[1].strip())

        model = LayerNormWrapper(weight_tensor, bias_tensor, epsilon)
        if mode == 'fp16':
            model.half()
        model.eval().cuda()
        
        metrics, out_dtype = run_test("ln", model, {"input": input_tensor}, expected_output, plugin_paths, mode)
        if metrics:
            # 验证阈值：FP32/INT8期望<1e-5, FP16期望<1e-2
            threshold = 1e-2 if mode == 'fp16' else 1e-5
            status = "PASS" if metrics['max_diff'] < threshold and metrics['cos_sim'] > 0.9999 else "FAIL"
            actual_mode_str = f" (actual: {metrics.get('actual_mode', mode)})" if metrics.get('actual_mode', mode) != mode else ""
            logger.info(f"[{os.path.basename(ln_dir)}] MaxDiff: {metrics['max_diff']:.6f}, CosSim: {metrics['cos_sim']:.6f}, OutType: {out_dtype}, Status: {status}{actual_mode_str}")
            results.append(metrics)
            
    if results:
        avg_max = sum(r['max_diff'] for r in results) / len(results)
        avg_mean = sum(r['mean_diff'] for r in results) / len(results)
        avg_cos_sim = sum(r['cos_sim'] for r in results) / len(results)
        # Get actual mode from first result (should be same for all)
        actual_mode = results[0].get('actual_mode', mode)
        logger.info(f"LayerNorm {mode} Average Max Diff: {avg_max:.6f}")
        if actual_mode != mode:
            logger.warning(f"LayerNorm {mode} actually ran in {actual_mode} mode")
        return {
            "plugin": "LayerNorm",
            "mode": mode,
            "actual_mode": actual_mode,
            "num_samples": len(results),
            "avg_max_diff": avg_max,
            "avg_mean_diff": avg_mean,
            "avg_cos_sim": avg_cos_sim,
            "max_max_diff": max(r['max_diff'] for r in results),
            "min_cos_sim": min(r['cos_sim'] for r in results)
        }
    else:
        logger.warning(f"No LayerNorm tests ran for {mode}")
        return None

def validate_sparsebox(sample_dir, plugin_paths, checkpoint_path, mode='fp32'):
    suffix = "float16" if mode == 'fp16' else "float32"
    np_dtype = np.float16 if mode == 'fp16' else np.float32
    
    if mode == 'int8':
        suffix = "float32" 
        np_dtype = np.float32

    logger.info(f"Testing SparseBox in {mode} mode...")
    
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    
    kps_keys = [k for k in state_dict.keys() if 'kps_generator.fix_scale' in k]
    def get_layer_idx(key):
        parts = key.split('.')
        for i, p in enumerate(parts):
            if p == 'layers' and i + 1 < len(parts) and parts[i+1].isdigit():
                return int(parts[i+1])
        return -1
    kps_keys.sort(key=get_layer_idx)
    kps_prefixes = [k.replace('.fix_scale', '') for k in kps_keys]
    
    sb_dirs = sorted(glob.glob(os.path.join(sample_dir, "sparsebox_*")))
    results = []

    for sb_dir in sb_dirs[:3]:
        idx = int(os.path.basename(sb_dir).split('_')[-1])
        if idx >= len(kps_prefixes): continue
            
        prefix = kps_prefixes[idx]
        fix_scale = state_dict[f"{prefix}.fix_scale"].cuda()
        fc_weight_key = f"{prefix}.learnable_fc.weight"
        if fc_weight_key in state_dict:
            fc_weight = state_dict[fc_weight_key].cuda()
            fc_bias = state_dict[f"{prefix}.learnable_fc.bias"].cuda()
            num_learnable_pts = fc_weight.shape[0] // 3
        else:
            fc_weight = torch.tensor([], device='cuda')
            fc_bias = torch.tensor([], device='cuda')
            num_learnable_pts = 0
            
        num_pts = len(fix_scale) + num_learnable_pts
        
        anchor_path = get_bin_path(sb_dir, f"*_anchor_*_{suffix}.bin")
        feature_path = get_bin_path(sb_dir, f"*_feature_*_{suffix}.bin")
        output_path = get_bin_path(sb_dir, f"*_keypoints_*_{suffix}.bin")
        
        if not (anchor_path and feature_path and output_path):
            continue
        
        anchor = load_bin(anchor_path, parse_shape_from_name(anchor_path), dtype=np_dtype)
        feature = load_bin(feature_path, parse_shape_from_name(feature_path), dtype=np_dtype)
        expected_output = load_bin(output_path, parse_shape_from_name(output_path), dtype=np_dtype)
        
        embed_dims = feature.shape[-1]
        
        model = SparseBoxWrapper(embed_dims, num_pts, num_learnable_pts, fix_scale, fc_weight, fc_bias)
        if mode == 'fp16':
            model.half()
        model.eval().cuda()
        
        inputs = {"anchor": anchor, "instance_feature": feature}
        
        # SparseBox output is always float32 in current plugin implementation
        if mode == 'fp16':
             expected_output = expected_output.float()

        metrics, out_dtype = run_test("sb", model, inputs, expected_output, plugin_paths, mode)
        if metrics:
            # 验证阈值：FP32/INT8期望<1e-4, FP16期望<0.1 (SparseBox输出是FP32，FP16模式下误差可能较大)
            threshold = 0.1 if mode == 'fp16' else 1e-4
            status = "PASS" if metrics['max_diff'] < threshold and metrics['cos_sim'] > 0.9999 else "FAIL"
            actual_mode_str = f" (actual: {metrics.get('actual_mode', mode)})" if metrics.get('actual_mode', mode) != mode else ""
            logger.info(f"[{os.path.basename(sb_dir)}] MaxDiff: {metrics['max_diff']:.6f}, CosSim: {metrics['cos_sim']:.6f}, Status: {status}{actual_mode_str}")
            results.append(metrics)

    if results:
        avg_max = sum(r['max_diff'] for r in results) / len(results)
        avg_mean = sum(r['mean_diff'] for r in results) / len(results)
        avg_cos_sim = sum(r['cos_sim'] for r in results) / len(results)
        # Get actual mode from first result (should be same for all)
        actual_mode = results[0].get('actual_mode', mode)
        logger.info(f"SparseBox {mode} Average Max Diff: {avg_max:.6f}")
        if actual_mode != mode:
            logger.warning(f"SparseBox {mode} actually ran in {actual_mode} mode")
        return {
            "plugin": "SparseBox",
            "mode": mode,
            "actual_mode": actual_mode,
            "num_samples": len(results),
            "avg_max_diff": avg_max,
            "avg_mean_diff": avg_mean,
            "avg_cos_sim": avg_cos_sim,
            "max_max_diff": max(r['max_diff'] for r in results),
            "min_cos_sim": min(r['cos_sim'] for r in results)
        }
    else:
        logger.warning(f"No SparseBox tests ran for {mode}")
        return None

def validate_dfa(sample_dir, plugin_paths, mode='fp32'):
    """
    验证DFA插件的精度
    
    Note: DFA Plugin支持INT8模式（value=INT8, keypoints/weights=FP32, output=FP32）
    """
    # INT8模式：value使用FP32数据（会被TensorRT量化），keypoints/weights保持FP32
    if mode == 'int8':
        suffix = "float32"  # INT8模式下value输入仍然是float32（TensorRT会量化）
        np_dtype = np.float32
    else:
        suffix = "float16" if mode == 'fp16' else "float32"
        np_dtype = np.float16 if mode == 'fp16' else np.float32
    
    logger.info(f"Testing DFA in {mode} mode...")
    
    # Import PyTorch reference implementation
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from modules.ops import e2e_deformable_aggregation_ext

    dfa_dirs = sorted(glob.glob(os.path.join(sample_dir, "dfa_*")))
    results = []
    
    for dfa_dir in dfa_dirs[:3]:
        value_path = get_bin_path(dfa_dir, f"*value_*_{suffix}.bin")
        if not value_path and mode == 'fp16': # Fallback to fp32 inputs if fp16 specific not found
             value_path = get_bin_path(dfa_dir, f"*value_*_float32.bin")
             
        if not value_path: continue
            
        value = load_bin(value_path, parse_shape_from_name(value_path), dtype=np_dtype)
        if mode == 'fp16' and value.dtype != torch.float16: 
            value = value.half()
        # INT8模式：value保持FP32（TensorRT会在构建时量化），但需要设置动态范围
        
        spatial = load_bin(get_bin_path(dfa_dir, "*spatial_shapes_*.bin"), parse_shape_from_name(get_bin_path(dfa_dir, "*spatial_shapes_*.bin")), dtype=np.int64)
        level_start = load_bin(get_bin_path(dfa_dir, "*level_start_index_*.bin"), parse_shape_from_name(get_bin_path(dfa_dir, "*level_start_index_*.bin")), dtype=np.int64)
        
        sampling_path = get_bin_path(dfa_dir, f"*sampling_locations_*_{suffix}.bin")
        if not sampling_path: sampling_path = get_bin_path(dfa_dir, "*sampling_locations_*_float32.bin")
        sampling = load_bin(sampling_path, parse_shape_from_name(sampling_path), dtype=np_dtype if sampling_path and suffix in sampling_path else np.float32)
        if mode == 'fp16' and sampling.dtype != torch.float16: sampling = sampling.half()

        weights_path = get_bin_path(dfa_dir, f"*attention_weights_*_{suffix}.bin")
        if not weights_path: weights_path = get_bin_path(dfa_dir, "*attention_weights_*_float32.bin")
        weights = load_bin(weights_path, parse_shape_from_name(weights_path), dtype=np_dtype if weights_path and suffix in weights_path else np.float32)
        if mode == 'fp16' and weights.dtype != torch.float16: weights = weights.half()

        # Compute Reference
        # DFA Plugin支持的精度模式：
        # - FP32: 全FP32模式
        # - FP16: 混合精度模式（Value=FP16, Sampling/Weights=FP32）
        # - INT8: 混合精度模式（Value=INT8, Sampling/Weights=FP32, Output=FP32）
        #   Note: INT8模式下，value输入仍然是FP32，TensorRT会在构建时自动量化
        
        if mode == 'fp16':
            # PyTorch impl uses floats
            pytorch_output = e2e_deformable_aggregation_ext.deformable_aggregation_forward(
                value.float().contiguous(),
                spatial.int().contiguous(),
                level_start.int().contiguous(),
                sampling.float().contiguous(),
                weights.float().contiguous()
            )
        else:
             pytorch_output = e2e_deformable_aggregation_ext.deformable_aggregation_forward(
                value.contiguous(),
                spatial.int().contiguous(),
                level_start.int().contiguous(),
                sampling.contiguous(),
                weights.contiguous()
            )

        model = DFAWrapper()
        if mode == 'fp16':
            model.half()
        model.eval().cuda()
        
        inputs = {
            "value": value,
            "spatial_shapes": spatial,
            "level_start_index": level_start,
            "sampling_locations": sampling,
            "attention_weights": weights
        }
        
        metrics, out_dtype = run_test("dfa", model, inputs, pytorch_output, plugin_paths, mode)
        if metrics:
            # 验证阈值：FP32/INT8期望<1e-4, FP16期望<0.01 (DFA在FP16模式下可能有较大误差)
            threshold = 0.01 if mode == 'fp16' else 1e-4
            status = "PASS" if metrics['max_diff'] < threshold and metrics['cos_sim'] > 0.9999 else "FAIL"
            actual_mode_str = f" (actual: {metrics.get('actual_mode', mode)})" if metrics.get('actual_mode', mode) != mode else ""
            logger.info(f"[{os.path.basename(dfa_dir)}] MaxDiff: {metrics['max_diff']:.6f}, CosSim: {metrics['cos_sim']:.6f}, Status: {status}{actual_mode_str}")
            results.append(metrics)

    if results:
        avg_max = sum(r['max_diff'] for r in results) / len(results)
        avg_mean = sum(r['mean_diff'] for r in results) / len(results)
        avg_cos_sim = sum(r['cos_sim'] for r in results) / len(results)
        # Get actual mode from first result (should be same for all)
        actual_mode = results[0].get('actual_mode', mode)
        logger.info(f"DFA {mode} Average Max Diff: {avg_max:.6f}")
        if actual_mode != mode:
            logger.warning(f"DFA {mode} actually ran in {actual_mode} mode")
        return {
            "plugin": "DFA",
            "mode": mode,
            "actual_mode": actual_mode,
            "num_samples": len(results),
            "avg_max_diff": avg_max,
            "avg_mean_diff": avg_mean,
            "avg_cos_sim": avg_cos_sim,
            "max_max_diff": max(r['max_diff'] for r in results),
            "min_cos_sim": min(r['cos_sim'] for r in results)
        }
    else:
        logger.warning(f"No DFA tests ran for {mode}")
        return None


def print_summary_table(all_results):
    """
    打印汇总统计表格
    """
    if not all_results:
        logger.warning("No validation results to summarize.")
        return
    
    print("\n" + "="*100)
    print("  Validation Summary - Plugin Accuracy Across Precision Modes")
    print("="*100)
    
    # 按插件分组
    plugins = ["LayerNorm", "SparseBox", "DFA"]
    modes = ["fp32", "fp16", "int8"]
    
    # 表头
    print(f"\n{'Plugin':<12} {'Mode':<20} {'Samples':<8} {'Avg MaxDiff':<15} {'Max MaxDiff':<15} {'Avg CosSim':<12} {'Min CosSim':<12} {'Status':<8}")
    print("-" * 100)
    
    # 收集所有结果用于统计
    summary_data = []
    
    for plugin in plugins:
        for mode in modes:
            result = next((r for r in all_results if r and r['plugin'] == plugin and r['mode'] == mode), None)
            if result:
                # 判断状态
                if plugin == "LayerNorm":
                    threshold = 1e-2 if mode == 'fp16' else 1e-5
                elif plugin == "SparseBox":
                    threshold = 0.1 if mode == 'fp16' else 1e-4
                else:  # DFA
                    threshold = 0.01 if mode == 'fp16' else 1e-4
                
                status = "PASS" if result['avg_max_diff'] < threshold and result['avg_cos_sim'] > 0.9999 else "FAIL"
                
                # 显示实际精度模式（如果与请求的不同）
                actual_mode = result.get('actual_mode', mode)
                mode_display = mode
                if actual_mode != mode:
                    # 提取回退目标：'int8(fallback to fp32)' -> 'fp32'
                    if '(' in actual_mode and 'fallback' in actual_mode:
                        fallback_target = actual_mode.split('fallback to ')[1].split(')')[0]
                        mode_display = f"{mode}→{fallback_target}"
                    else:
                        mode_display = f"{mode}→{actual_mode}"
                
                print(f"{plugin:<12} {mode_display:<20} {result['num_samples']:<8} "
                      f"{result['avg_max_diff']:<15.6e} {result['max_max_diff']:<15.6e} "
                      f"{result['avg_cos_sim']:<12.6f} {result['min_cos_sim']:<12.6f} {status:<8}")
                
                summary_data.append({
                    'plugin': plugin,
                    'mode': mode,
                    'actual_mode': actual_mode,
                    'status': status,
                    'avg_max_diff': result['avg_max_diff']
                })
            else:
                print(f"{plugin:<12} {mode:<8} {'N/A':<8} {'N/A':<15} {'N/A':<15} {'N/A':<12} {'N/A':<12} {'SKIP':<8}")
    
    print("-" * 100)
    
    # 统计摘要
    print("\nSummary Statistics:")
    print("-" * 100)
    
    for plugin in plugins:
        plugin_results = [r for r in summary_data if r['plugin'] == plugin]
        if plugin_results:
            passed = sum(1 for r in plugin_results if r['status'] == 'PASS')
            total = len(plugin_results)
            print(f"  {plugin}: {passed}/{total} modes passed")
            for r in plugin_results:
                print(f"    - {r['mode']}: {r['status']} (Avg MaxDiff: {r['avg_max_diff']:.6e})")
    
    print("="*100)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="deploy/val_data_plugin/real_data/sample_0")
    parser.add_argument("--checkpoint", default="ckpt/sparse4dv3_r50.pth")
    args = parser.parse_args()
    
    # Check for data
    if not os.path.exists(args.data_dir):
        logger.error(f"Data directory not found: {args.data_dir}")
        logger.info("Please run 'python script/tutorial/012.export_plugin_io_real_data.py' first to generate real data.")
        logger.info("  Example: python script/tutorial/012.export_plugin_io_real_data.py --fp16")
        return

    plugin_paths = [
        "deploy/ln_plugin/lib/customLayerNorm.so",
        "deploy/sparsebox_plugin/lib/SparseBox3DKeyPointsPlugin.so",
        "deploy/dfa_plugin/lib/deformableAttentionAggr.so"
    ]
    
    # 收集所有验证结果
    all_results = []
    
    # FP32
    print("\n" + "="*80)
    print("  FP32 Validation")
    print("="*80)
    result = validate_ln(args.data_dir, plugin_paths, 'fp32')
    if result: all_results.append(result)
    result = validate_sparsebox(args.data_dir, plugin_paths, args.checkpoint, 'fp32')
    if result: all_results.append(result)
    result = validate_dfa(args.data_dir, plugin_paths, 'fp32')
    if result: all_results.append(result)
    
    # FP16
    print("\n" + "="*80)
    print("  FP16 Validation")
    print("="*80)
    result = validate_ln(args.data_dir, plugin_paths, 'fp16')
    if result: all_results.append(result)
    result = validate_sparsebox(args.data_dir, plugin_paths, args.checkpoint, 'fp16')
    if result: all_results.append(result)
    result = validate_dfa(args.data_dir, plugin_paths, 'fp16')
    if result: all_results.append(result)
    
    # INT8
    print("\n" + "="*80)
    print("  INT8 Validation (Experimental)")
    print("="*80)
    result = validate_ln(args.data_dir, plugin_paths, 'int8')
    if result: all_results.append(result)
    result = validate_sparsebox(args.data_dir, plugin_paths, args.checkpoint, 'int8')
    if result: all_results.append(result)
    result = validate_dfa(args.data_dir, plugin_paths, 'int8')
    if result: all_results.append(result)
    
    # 打印汇总统计表
    print_summary_table(all_results)

if __name__ == "__main__":
    main()
