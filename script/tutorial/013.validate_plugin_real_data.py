import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
import tensorrt as trt
import ctypes
from torch.autograd.function import Function

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
        # Prepare inputs
        inputs = [anchor]
        if instance_feature is not None:
            inputs.append(instance_feature)
        
        # Attributes are passed as lists directly
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
        # Convert tensors to lists for symbolic function
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

def build_engine(onnx_path, plugin_paths, logger, fp16=False):
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
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
    
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                logger.error(parser.get_error(error))
            return None

    # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) # 1GB
    
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        return None
        
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(plan)
    return engine

def run_inference(engine, inputs_dict):
    context = engine.create_execution_context()
    
    io_tensors = {}
    allocations = []
    
    # Inputs
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            if name not in inputs_dict:
                raise ValueError(f"Missing input: {name}")
            tensor = inputs_dict[name]
            
            # Handle type mismatch if necessary (e.g. Int64 -> Int32)
            # TensorRT often uses Int32 for indices
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
    # format: name_d1*d2*d3_type.bin
    base = os.path.basename(filename)
    parts = base.split('_')
    # find part with '*'
    for p in parts:
        if '*' in p:
            return [int(d) for d in p.split('*')]
    return None

def validate_ln(sample_dir, plugin_paths, logger, fp16=False):
    logger.info(f">>> Validating LayerNorm Plugin (FP16={fp16})...")
    ln_dirs = sorted(glob.glob(os.path.join(sample_dir, "ln_*")))
    
    suffix = "float16" if fp16 else "float32"
    np_dtype = np.float16 if fp16 else np.float32
    
    errs = []
    
    # Test a few samples
    for ln_dir in ln_dirs[:5]:
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
        if fp16:
            model.half()
        model.eval().cuda()
        
        onnx_file = "tmp_ln.onnx"
        torch.onnx.export(model, (input_tensor,), onnx_file, 
                          input_names=["input"], output_names=["output"],
                          opset_version=13)
                          
        engine = build_engine(onnx_file, plugin_paths, logger, fp16=fp16)
        if engine:
            outputs = run_inference(engine, {"input": input_tensor})
            trt_output = outputs["output"]
            
            diff = (trt_output - expected_output).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            logger.info(f"[{os.path.basename(ln_dir)}] Max Diff: {max_diff:.6f}, Mean Diff: {mean_diff:.6f}")
            errs.append(max_diff)
            
    if errs:
        logger.info(f"LayerNorm Average Max Diff: {sum(errs)/len(errs):.6f}")

def validate_sparsebox(sample_dir, plugin_paths, checkpoint_path, logger, fp16=False):
    logger.info(f">>> Validating SparseBox Plugin (FP16={fp16})...")
    
    suffix = "float16" if fp16 else "float32"
    np_dtype = np.float16 if fp16 else np.float32
    
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
    
    for sb_dir in sb_dirs:
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
        if fp16:
            model.half()
        model.eval().cuda()
        
        onnx_file = "tmp_sb.onnx"
        torch.onnx.export(model, (anchor, feature), onnx_file, 
                          input_names=["anchor", "instance_feature"], output_names=["keypoints"],
                          opset_version=13)
        
        engine = build_engine(onnx_file, plugin_paths, logger, fp16=fp16)
        if engine:
            outputs = run_inference(engine, {"anchor": anchor, "instance_feature": feature})
            trt_output = outputs["keypoints"]
            
            if fp16:
                expected_output = expected_output.float() # SparseBox output is float32
            
            # TRT output is likely float32 (plugin forces float32)
            if trt_output.dtype != expected_output.dtype:
                 trt_output = trt_output.to(expected_output.dtype)

            diff = (trt_output - expected_output).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            
            if np.isnan(max_diff):
                logger.error(f"[{os.path.basename(sb_dir)}] NaN detected in output diff!")
                logger.error(f"  TRT Output NaNs: {torch.isnan(trt_output).sum().item()}")
                logger.error(f"  Expected Output NaNs: {torch.isnan(expected_output).sum().item()}")
                logger.error(f"  TRT Output range: [{trt_output.min().item()}, {trt_output.max().item()}]")
                logger.error(f"  Expected Output range: [{expected_output.min().item()}, {expected_output.max().item()}]")
            
            logger.info(f"[{os.path.basename(sb_dir)}] Max Diff: {max_diff:.6f}, Mean Diff: {mean_diff:.6f}")

def validate_dfa(sample_dir, plugin_paths, logger, fp16=False):
    logger.info(f">>> Validating DFA Plugin (FP16={fp16})...")
    
    # 导入 PyTorch DFA CUDA 扩展用于实时计算参考输出
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from modules.ops import e2e_deformable_aggregation_ext
    
    suffix = "float16" if fp16 else "float32"
    np_dtype = np.float16 if fp16 else np.float32
    
    dfa_dirs = sorted(glob.glob(os.path.join(sample_dir, "dfa_*")))
    
    errs = []
    for dfa_dir in dfa_dirs:
        value_path = get_bin_path(dfa_dir, f"*value_*_{suffix}.bin")
        if not value_path: continue
            
        value = load_bin(value_path, parse_shape_from_name(value_path), dtype=np_dtype)
        spatial = load_bin(get_bin_path(dfa_dir, "*spatial_shapes_*.bin"), parse_shape_from_name(get_bin_path(dfa_dir, "*spatial_shapes_*.bin")), dtype=np.int64)
        level_start = load_bin(get_bin_path(dfa_dir, "*level_start_index_*.bin"), parse_shape_from_name(get_bin_path(dfa_dir, "*level_start_index_*.bin")), dtype=np.int64)
        
        sampling_path = get_bin_path(dfa_dir, f"*sampling_locations_*_{suffix}.bin")
        if not sampling_path and fp16:
             sampling_path = get_bin_path(dfa_dir, "*sampling_locations_*_float32.bin")
        sampling = load_bin(sampling_path, parse_shape_from_name(sampling_path), dtype=np_dtype if sampling_path and suffix in sampling_path else np.float32)
        
        weights_path = get_bin_path(dfa_dir, f"*attention_weights_*_{suffix}.bin")
        if not weights_path and fp16:
             weights_path = get_bin_path(dfa_dir, "*attention_weights_*_float32.bin")
        weights = load_bin(weights_path, parse_shape_from_name(weights_path), dtype=np_dtype if weights_path and suffix in weights_path else np.float32)

        # 使用实时 PyTorch CUDA 输出作为参考（而不是预保存的文件）
        # 这确保了对比的公平性，避免预保存数据与当前环境不一致的问题
        pytorch_output = e2e_deformable_aggregation_ext.deformable_aggregation_forward(
            value.contiguous().float(),
            spatial.int().contiguous(),
            level_start.int().contiguous(),
            sampling.contiguous().float(),
            weights.contiguous().float()
        )

        model = DFAWrapper()
        if fp16:
            model.half()
        model.eval().cuda()
        
        # Ensure inputs match model dtype (except ints)
        if fp16:
             if value.dtype != torch.float16: value = value.half()
             if sampling.dtype != torch.float16: sampling = sampling.half()
             if weights.dtype != torch.float16: weights = weights.half()
        
        onnx_file = "tmp_dfa.onnx"
        torch.onnx.export(model, (value, spatial, level_start, sampling, weights), onnx_file,
                          input_names=["value", "spatial_shapes", "level_start_index", "sampling_locations", "attention_weights"],
                          output_names=["output"],
                          opset_version=13)
                          
        engine = build_engine(onnx_file, plugin_paths, logger, fp16=fp16)
        if engine:
            outputs = run_inference(engine, {
                "value": value,
                "spatial_shapes": spatial,
                "level_start_index": level_start,
                "sampling_locations": sampling,
                "attention_weights": weights
            })
            trt_output = outputs["output"]
            
            # 对比 TRT 输出和实时 PyTorch 输出
            if trt_output.dtype != pytorch_output.dtype:
                 trt_output = trt_output.to(pytorch_output.dtype)

            diff = (trt_output - pytorch_output).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            errs.append(max_diff)
            logger.info(f"[{os.path.basename(dfa_dir)}] Max Diff: {max_diff:.6f}, Mean Diff: {mean_diff:.6f}")
    
    if errs:
        logger.info(f"DFA Average Max Diff: {sum(errs)/len(errs):.6f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="deploy/val_data_plugin/real_data/sample_0")
    parser.add_argument("--checkpoint", default="ckpt/sparse4dv3_r50.pth")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 validation")
    args = parser.parse_args()
    
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("PluginValidator")
    
    plugin_paths = [
        "deploy/ln_plugin/lib/customLayerNorm.so",
        "deploy/sparsebox_plugin/lib/SparseBox3DKeyPointsPlugin.so",
        "deploy/dfa_plugin/lib/deformableAttentionAggr.so"
    ]
    
    validate_ln(args.data_dir, plugin_paths, logger, fp16=args.fp16)
    validate_sparsebox(args.data_dir, plugin_paths, args.checkpoint, logger, fp16=args.fp16)
    validate_dfa(args.data_dir, plugin_paths, logger, fp16=args.fp16)

if __name__ == "__main__":
    main()
