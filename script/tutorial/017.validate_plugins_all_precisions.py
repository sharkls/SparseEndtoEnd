import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
import tensorrt as trt
import ctypes
import logging
from torch.autograd.function import Function

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PluginValidator")

# =============================================================================
# 1. Define Symbolic Functions for ONNX Export
# =============================================================================

class LayerNormPluginFunction(Function):
    @staticmethod
    def symbolic(g, input, weight, bias, epsilon, axis):
        return g.op(
            "custom::CustomLayerNormalization",
            input,
            weight,
            bias,
            epsilon_f=epsilon,
            axis_i=axis
        )

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

def build_engine(onnx_path, plugin_paths, mode='fp32', input_tensors=None, scale_info=None):
    """
    Build TensorRT engine.
    mode: 'fp32', 'fp16', or 'int8'
    input_tensors: dict of name -> torch.Tensor, required for INT8 dynamic range setting
    scale_info: dict of tensor_name -> scale, used for INT8 mode to set correct dynamic range
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
    
    if mode == 'fp16':
        config.set_flag(trt.BuilderFlag.FP16)
    elif mode == 'int8':
        config.set_flag(trt.BuilderFlag.INT8)
        # config.set_flag(trt.BuilderFlag.FP16) # INT8 often used with FP16 fallback

    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                logger.error(parser.get_error(error))
            return None

    # For INT8, we need to set dynamic range for inputs to trigger INT8 mode in plugins
    # Since we don't have a calibrator, we manually set dynamic range based on input data
    if mode == 'int8' and input_tensors is not None:
        for i in range(network.num_inputs):
            input_tensor = network.get_input(i)
            name = input_tensor.name
            if name in input_tensors:
                data = input_tensors[name]
                
                # Handle INT8 inputs: need to dequantize to get actual range
                if data.dtype == torch.int8:
                    # If we have scale_info, use it to calculate the original range
                    if scale_info and name in scale_info:
                        scale = scale_info[name]
                        # From scale = abs_max / 127, we get abs_max = scale * 127
                        abs_max = scale * 127.0
                    else:
                        # Fallback: estimate from INT8 data
                        # INT8 range is [-128, 127], scale = abs_max / 127
                        int8_min = data.min().item()
                        int8_max = data.max().item()
                        abs_max_int8 = max(abs(int8_min), abs(int8_max))
                        if abs_max_int8 == 0:
                            abs_max_int8 = 127.0
                        # Estimate: assume the INT8 data uses most of the range
                        # If int8_max = 127, then original_max ≈ scale * 127
                        # We use a conservative estimate
                        estimated_scale = 1.0 / 127.0  # Conservative estimate
                        abs_max = abs_max_int8 * estimated_scale
                    if abs_max == 0: abs_max = 1.0
                else:
                    # For FP32/FP16 inputs that should be quantized
                    min_val = data.min().item()
                    max_val = data.max().item()
                    abs_max = max(abs(min_val), abs(max_val))
                    if abs_max == 0: abs_max = 1.0
                
                input_tensor.dynamic_range = (-abs_max, abs_max)
                # logger.info(f"Set dynamic range for {name}: {-abs_max} to {abs_max} (dtype: {data.dtype})")
            else:
                # Set default if missing (e.g. spatial_shapes which are int)
                # Int inputs don't need dynamic range usually, but safety check
                if input_tensor.dtype == trt.DataType.FLOAT or input_tensor.dtype == trt.DataType.HALF:
                     input_tensor.dynamic_range = (-1.0, 1.0)

        # Also need to set dynamic range for all layers to avoid errors if strict types are used
        # But without calibration, this is hard. 
        # Hopefully plugins handle 'missing' ranges by falling back or we just need input ranges.
        # Note: Plugins receiving INT8 *must* have input dynamic range set.

    plan = builder.build_serialized_network(network, config)
    if plan is None:
        logger.error("Failed to build TensorRT engine.")
        return None
        
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(plan)
    return engine

def run_inference(engine, inputs_dict):
    context = engine.create_execution_context()
    
    io_tensors = {}
    
    # Inputs
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            if name not in inputs_dict:
                raise ValueError(f"Missing input: {name}")
            tensor = inputs_dict[name]
            
            # Type mismatch handling
            if tensor.dtype == torch.int64:
                 tensor = tensor.int()
            
            context.set_input_shape(name, tensor.shape)
            context.set_tensor_address(name, tensor.data_ptr())
            io_tensors[name] = tensor

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

    context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    
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
    if a.shape != b.shape:
        return f"Shape mismatch: {a.shape} vs {b.shape}"
    
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

def run_test(name, model, input_tensors, expected_output, plugin_paths, mode='fp32', scale_info=None):
    """
    scale_info: dict of tensor_name -> scale, used for INT8 mode to set correct dynamic range
    """
    onnx_file = f"/tmp/tmp_{name}_{mode}.onnx"
    
    # Export ONNX
    # For INT8 mode, we need to export with FP32 inputs (ONNX export may not handle INT8 well)
    # Then TensorRT will quantize based on dynamic range
    try:
        # Prepare inputs for export
        if isinstance(input_tensors, dict):
            # For INT8 mode, convert INT8 inputs to FP32 for ONNX export
            if mode == 'int8':
                export_inputs = {}
                for k, v in input_tensors.items():
                    if v.dtype == torch.int8:
                        # Dequantize for export using the scale from scale_info
                        if scale_info and k in scale_info:
                            scale = scale_info[k]
                            export_inputs[k] = (v.float() * scale)
                        else:
                            # Fallback: use 1.0 as scale
                            export_inputs[k] = v.float()
                    else:
                        export_inputs[k] = v
                args = tuple(export_inputs.values())
                input_names = list(input_tensors.keys())
            else:
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
    engine = build_engine(onnx_file, plugin_paths, mode, input_tensors=input_dict, scale_info=scale_info)
    
    if not engine:
        return None
        
    # Run Inference
    try:
        # For INT8 mode, ensure inputs are in correct format
        inference_inputs = {}
        for k, v in input_dict.items():
            if mode == 'int8' and v.dtype == torch.int8:
                # Keep INT8 for inference
                inference_inputs[k] = v
            else:
                inference_inputs[k] = v
        
        outputs = run_inference(engine, inference_inputs)
        output = outputs["output"]
        
        # Compare
        metrics = compare_tensors(output, expected_output)
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
            logger.info(f"[{os.path.basename(ln_dir)}] MaxDiff: {metrics['max_diff']:.6f}, CosSim: {metrics['cos_sim']:.6f}, OutType: {out_dtype}")
            results.append(metrics)
            
    if results:
        avg_max = sum(r['max_diff'] for r in results) / len(results)
        logger.info(f"LayerNorm {mode} Average Max Diff: {avg_max:.6f}")
    else:
        logger.warning(f"No LayerNorm tests ran for {mode}")

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
            logger.info(f"[{os.path.basename(sb_dir)}] MaxDiff: {metrics['max_diff']:.6f}, CosSim: {metrics['cos_sim']:.6f}")
            results.append(metrics)

    if results:
        avg_max = sum(r['max_diff'] for r in results) / len(results)
        logger.info(f"SparseBox {mode} Average Max Diff: {avg_max:.6f}")
    else:
        logger.warning(f"No SparseBox tests ran for {mode}")

def validate_dfa(sample_dir, plugin_paths, mode='fp32'):
    suffix = "float16" if mode == 'fp16' else "float32"
    np_dtype = np.float16 if mode == 'fp16' else np.float32
    
    # For INT8, we use FP32 inputs but will quantize value to INT8
    if mode == 'int8':
        suffix = "float32"
        np_dtype = np.float32
    
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
        
        spatial = load_bin(get_bin_path(dfa_dir, "*spatial_shapes_*.bin"), parse_shape_from_name(get_bin_path(dfa_dir, "*spatial_shapes_*.bin")), dtype=np.int64)
        level_start = load_bin(get_bin_path(dfa_dir, "*level_start_index_*.bin"), parse_shape_from_name(get_bin_path(dfa_dir, "*level_start_index_*.bin")), dtype=np.int64)
        
        # For INT8 mode: sampling_locations and attention_weights MUST be FP32 (as per plugin design)
        # For FP16 mode: can use FP32 or FP16 (plugin supports mixed precision)
        # For FP32 mode: use FP32
        if mode == 'int8':
            # INT8 mode: sampling and weights must be FP32
            sampling_path = get_bin_path(dfa_dir, "*sampling_locations_*_float32.bin")
            weights_path = get_bin_path(dfa_dir, "*attention_weights_*_float32.bin")
            sampling = load_bin(sampling_path, parse_shape_from_name(sampling_path), dtype=np.float32) if sampling_path else None
            weights = load_bin(weights_path, parse_shape_from_name(weights_path), dtype=np.float32) if weights_path else None
        else:
            sampling_path = get_bin_path(dfa_dir, f"*sampling_locations_*_{suffix}.bin")
            if not sampling_path: sampling_path = get_bin_path(dfa_dir, "*sampling_locations_*_float32.bin")
            sampling = load_bin(sampling_path, parse_shape_from_name(sampling_path), dtype=np_dtype if sampling_path and suffix in sampling_path else np.float32)
            
            weights_path = get_bin_path(dfa_dir, f"*attention_weights_*_{suffix}.bin")
            if not weights_path: weights_path = get_bin_path(dfa_dir, "*attention_weights_*_float32.bin")
            weights = load_bin(weights_path, parse_shape_from_name(weights_path), dtype=np_dtype if weights_path and suffix in weights_path else np.float32)
        
        if sampling is None or weights is None:
            logger.warning(f"[{os.path.basename(dfa_dir)}] Missing sampling_locations or attention_weights, skipping")
            continue
        
        # Type conversions for FP16 mode
        if mode == 'fp16':
            if value.dtype != torch.float16: value = value.half()
            # Plugin uses mixed precision: FP16 value + FP32 sampling/weights
            # So we keep sampling and weights as FP32
        
        # Prepare inputs for TensorRT
        # For INT8: quantize value to INT8, keep sampling/weights as FP32
        if mode == 'int8':
            # Quantize value to INT8
            # Calculate scale: abs_max / 127
            value_fp32 = value.float()
            abs_max = value_fp32.abs().max().item()
            if abs_max == 0:
                abs_max = 1.0
            value_scale = abs_max / 127.0
            
            # Quantize: value_int8 = clamp(round(value_fp32 / scale), -128, 127)
            value_int8 = (value_fp32 / value_scale).round().clamp(-128, 127).to(torch.int8)
            
            # For reference: dequantize back to FP32 for PyTorch implementation
            value_for_ref = (value_int8.float() * value_scale)
            
            inputs = {
                "value": value_int8,
                "spatial_shapes": spatial,
                "level_start_index": level_start,
                "sampling_locations": sampling.float(),
                "attention_weights": weights.float()
            }
            
            # Store scale info for build_engine
            scale_info = {"value": value_scale}
            
            # Reference output using dequantized value
            pytorch_output = e2e_deformable_aggregation_ext.deformable_aggregation_forward(
                value_for_ref.contiguous(),
                spatial.int().contiguous(),
                level_start.int().contiguous(),
                sampling.float().contiguous(),
                weights.float().contiguous()
            )
        elif mode == 'fp16':
            # FP16 mode: value is FP16, sampling/weights are FP32 (mixed precision)
            inputs = {
                "value": value,
                "spatial_shapes": spatial,
                "level_start_index": level_start,
                "sampling_locations": sampling.float(),  # Keep FP32
                "attention_weights": weights.float()     # Keep FP32
            }
            
            scale_info = None  # No scale needed for FP16
            
            # PyTorch impl uses floats
            pytorch_output = e2e_deformable_aggregation_ext.deformable_aggregation_forward(
                value.float().contiguous(),
                spatial.int().contiguous(),
                level_start.int().contiguous(),
                sampling.float().contiguous(),
                weights.float().contiguous()
            )
        else:
            # FP32 mode
            inputs = {
                "value": value,
                "spatial_shapes": spatial,
                "level_start_index": level_start,
                "sampling_locations": sampling,
                "attention_weights": weights
            }
            
            scale_info = None  # No scale needed for FP32
            
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
        
        # Pass scale_info for INT8 mode
        metrics, out_dtype = run_test("dfa", model, inputs, pytorch_output, plugin_paths, mode, scale_info=scale_info)
        if metrics:
            logger.info(f"[{os.path.basename(dfa_dir)}] MaxDiff: {metrics['max_diff']:.6f}, CosSim: {metrics['cos_sim']:.6f}")
            results.append(metrics)

    if results:
        avg_max = sum(r['max_diff'] for r in results) / len(results)
        logger.info(f"DFA {mode} Average Max Diff: {avg_max:.6f}")
    else:
        logger.warning(f"No DFA tests ran for {mode}")


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
    
    # FP32
    print("\n" + "="*80)
    print("  FP32 Validation")
    print("="*80)
    validate_ln(args.data_dir, plugin_paths, 'fp32')
    validate_sparsebox(args.data_dir, plugin_paths, args.checkpoint, 'fp32')
    validate_dfa(args.data_dir, plugin_paths, 'fp32')
    
    # FP16
    print("\n" + "="*80)
    print("  FP16 Validation")
    print("="*80)
    validate_ln(args.data_dir, plugin_paths, 'fp16')
    validate_sparsebox(args.data_dir, plugin_paths, args.checkpoint, 'fp16')
    validate_dfa(args.data_dir, plugin_paths, 'fp16')
    
    # INT8
    print("\n" + "="*80)
    print("  INT8 Validation (Experimental)")
    print("="*80)
    validate_ln(args.data_dir, plugin_paths, 'int8')
    validate_sparsebox(args.data_dir, plugin_paths, args.checkpoint, 'int8')
    validate_dfa(args.data_dir, plugin_paths, 'int8')

if __name__ == "__main__":
    main()
