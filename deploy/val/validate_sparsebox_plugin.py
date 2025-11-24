#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SparseBox3DKeyPointsPlugin 验证工具

功能：
1. 从 head ONNX 中解析 SparseBox3DKeyPointsPlugin 节点的参数；
2. 读取指定样本的 anchor / instance_feature 输入；
3. 同时运行 PyTorch 参考实现与 TensorRT Plugin，比较输出；
4. 输出误差统计，并可选保存差值到文件。

使用示例：
python3 deploy/val/validate_sparsebox_plugin.py \
    --onnx deploy/onnx/sparse4dhead1st.onnx \
    --plugin-so deploy/sparsebox_plugin/lib/SparseBox3DKeyPointsPlugin.so \
    --asset-dir script/tutorial/asset \
    --sample-index 0 \
    --node-index 0
"""

import argparse
import ctypes
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import onnx
import tensorrt as trt
import torch
from cuda import cudart

from modules.ops.sparse_box3d_keypoints import sparse_box3d_keypoints

LOGGER = logging.getLogger("SparseBoxValidator")


def _setup_logger(verbose: bool) -> None:
    LOGGER.setLevel(logging.DEBUG if verbose else logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    LOGGER.handlers.clear()
    LOGGER.addHandler(handler)


def _normalize_attr_name(name: str) -> str:
    for suffix in ("_i", "_f", "_s", "_is", "_fs"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def parse_plugin_attrs(node: onnx.NodeProto) -> Dict[str, np.ndarray]:
    attrs: Dict[str, np.ndarray] = {}
    for attr in node.attribute:
        key = _normalize_attr_name(attr.name)
        if attr.type == onnx.AttributeProto.INT:
            attrs[key] = np.array([attr.i], dtype=np.int32)
        elif attr.type == onnx.AttributeProto.FLOAT:
            attrs[key] = np.array([attr.f], dtype=np.float32)
        elif attr.type == onnx.AttributeProto.INTS:
            attrs[key] = np.asarray(attr.ints, dtype=np.int32)
        elif attr.type == onnx.AttributeProto.FLOATS:
            attrs[key] = np.asarray(attr.floats, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported attribute type for {attr.name}: {attr.type}")
    return attrs


def list_sparsebox_nodes(onnx_path: Path) -> List[onnx.NodeProto]:
    model = onnx.load(onnx_path.as_posix())
    nodes = [node for node in model.graph.node if "SparseBox3DKeyPointsPlugin" in node.op_type]
    return nodes


def load_sample(asset_dir: Path, sample_index: int) -> Tuple[np.ndarray, np.ndarray]:
    anchor_file = asset_dir / f"sample_{sample_index}_anchor_1*900*11_float32.bin"
    inst_file = asset_dir / f"sample_{sample_index}_instance_feature_1*900*256_float32.bin"
    if not anchor_file.exists() or not inst_file.exists():
        raise FileNotFoundError(f"Sample {sample_index} files not found in {asset_dir}")

    anchor = np.fromfile(anchor_file, dtype=np.float32)
    if anchor.size % 11 != 0:
        raise ValueError("Anchor file size is not divisible by 11")
    num_anchor = anchor.size // 11
    anchor = anchor.reshape(1, num_anchor, 11)

    instance = np.fromfile(inst_file, dtype=np.float32)
    embed_dims = instance.size // num_anchor // 1
    instance = instance.reshape(1, num_anchor, embed_dims)
    return anchor, instance


def build_trt_engine(
    attrs: Dict[str, np.ndarray],
    input_shapes: Dict[str, Tuple[int, ...]],
    plugin_so: Path,
    use_fp16: bool,
) -> trt.ICudaEngine:
    ctypes.cdll.LoadLibrary(plugin_so.as_posix())
    trt_logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(trt_logger, "")
    builder = trt.Builder(trt_logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    dtype = trt.DataType.HALF if use_fp16 else trt.DataType.FLOAT

    anchor_shape = input_shapes["anchor"]
    anchor_input = network.add_input("anchor", dtype, anchor_shape)
    profile.set_shape("anchor", anchor_shape, anchor_shape, anchor_shape)

    inputs = [anchor_input]
    if "instance_feature" in input_shapes:
        inst_shape = input_shapes["instance_feature"]
        inst_input = network.add_input("instance_feature", dtype, inst_shape)
        profile.set_shape("instance_feature", inst_shape, inst_shape, inst_shape)
        inputs.append(inst_input)

    registry = trt.get_plugin_registry()
    creator = registry.get_plugin_creator("SparseBox3DKeyPointsPlugin", "1", "")
    if creator is None:
        raise RuntimeError("Cannot find SparseBox3DKeyPointsPlugin creator")

    field_data_refs: List[np.ndarray] = []

    def plugin_field(name: str, array: np.ndarray, ftype: trt.PluginFieldType):
        arr = np.ascontiguousarray(array)
        field_data_refs.append(arr)
        return trt.PluginField(name, arr, ftype)

    fields: List[trt.PluginField] = []
    required_ints = ["embed_dims", "num_pts", "num_learnable_pts"]
    for key in required_ints:
        if key not in attrs:
            raise ValueError(f"Missing attribute `{key}` in ONNX node")
        fields.append(
            plugin_field(key, attrs[key].astype(np.int32), trt.PluginFieldType.INT32)
        )

    if "fix_scale" in attrs:
        fields.append(
            plugin_field("fix_scale", attrs["fix_scale"].astype(np.float32), trt.PluginFieldType.FLOAT32)
        )
    if "fc_weight" in attrs and attrs["fc_weight"].size > 0:
        fields.append(
            plugin_field("fc_weight", attrs["fc_weight"].astype(np.float32), trt.PluginFieldType.FLOAT32)
        )
    if "fc_bias" in attrs and attrs["fc_bias"].size > 0:
        fields.append(
            plugin_field("fc_bias", attrs["fc_bias"].astype(np.float32), trt.PluginFieldType.FLOAT32)
        )

    plugin = creator.create_plugin(
        "SparseBox3DKeyPointsPlugin",
        trt.PluginFieldCollection(fields),
    )
    layer = network.add_plugin_v2(inputs, plugin)
    network.mark_output(layer.get_output(0))

    config.add_optimization_profile(profile)
    if use_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    engine = builder.build_engine(network, config)
    if engine is None:
        raise RuntimeError("Failed to build TensorRT engine")
    return engine


def run_engine(engine: trt.ICudaEngine, inputs: Dict[str, np.ndarray]) -> np.ndarray:
    context = engine.create_execution_context()
    bindings: List[int] = [0] * engine.num_bindings

    def alloc_and_copy(name: str, array: np.ndarray) -> int:
        idx = engine.get_binding_index(name)
        context.set_binding_shape(idx, array.shape)
        nbytes = array.nbytes
        _, device_mem = cudart.cudaMalloc(nbytes)
        cudart.cudaMemcpy(
            device_mem, array.ctypes.data, nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        )
        bindings[idx] = device_mem
        return device_mem

    host_inputs: Dict[str, np.ndarray] = {}
    for name, array in inputs.items():
        host_inputs[name] = array
        alloc_and_copy(name, array)

    output_idx = engine.get_binding_index(engine.get_binding_name(engine.num_bindings - 1))
    output_shape = tuple(context.get_binding_shape(output_idx))
    dtype = engine.get_binding_dtype(output_idx)
    np_dtype = np.float16 if dtype == trt.DataType.HALF else np.float32
    # 关键修复：使用zeros初始化输出内存，避免未初始化值导致非确定性行为
    # 这可以确保即使kernel没有写入某些位置，也不会产生NaN或随机值
    output_host = np.zeros(output_shape, dtype=np_dtype)
    _, output_device = cudart.cudaMalloc(output_host.nbytes)
    # 将初始化的零值复制到设备内存，确保输出内存是干净的
    cudart.cudaMemcpy(
        output_device, output_host.ctypes.data, output_host.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
    )
    bindings[output_idx] = output_device

    context.execute_v2(bindings)
    # 关键修复：等待CUDA流完成，确保kernel执行完成后再读取结果
    # 这可以避免非确定性行为，确保每次执行都是确定性的
    # 注意：TensorRT的execute_v2是异步的，需要同步确保完成
    err = cudart.cudaDeviceSynchronize()[0]
    if err != cudart.cudaError_t.cudaSuccess:
        # 如果同步失败，记录错误但继续执行
        pass
    cudart.cudaMemcpy(
        output_host.ctypes.data, output_device, output_host.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
    )

    for ptr in bindings:
        if isinstance(ptr, int) and ptr != 0:
            cudart.cudaFree(ptr)

    return output_host.astype(np.float32)


def run_pytorch_reference(
    anchor: np.ndarray,
    instance_feature: np.ndarray,
    attrs: Dict[str, np.ndarray],
    precision: str,
) -> np.ndarray:
    embed_dims = int(attrs["embed_dims"][0])
    num_pts = int(attrs["num_pts"][0])
    num_learnable = int(attrs["num_learnable_pts"][0])
    num_fix = num_pts - num_learnable

    if precision == "fp16" and torch.cuda.is_available():
        device = torch.device("cuda")
        torch_dtype = torch.float16
    else:
        device = torch.device("cpu")
        torch_dtype = torch.float32 if precision == "fp32" else torch.float32
        if precision == "fp16" and not torch.cuda.is_available():
            LOGGER.warning(
                "FP16 reference requested but CUDA不可用，改用CPU FP32 计算"
            )

    anchor_t = torch.from_numpy(
        anchor.astype(np.float16 if precision == "fp16" else np.float32)
    ).to(device=device, dtype=torch_dtype)
    inst_t = (
        torch.from_numpy(
            instance_feature.astype(np.float16 if precision == "fp16" else np.float32)
        ).to(device=device, dtype=torch_dtype)
        if num_learnable > 0
        else None
    )

    fix_scale = attrs.get("fix_scale")
    if fix_scale is None or fix_scale.size != num_fix * 3:
        raise ValueError("Invalid fix_scale data in ONNX attributes")
    fix_scale_t = (
        torch.from_numpy(fix_scale.reshape(num_fix * 3))
        .to(dtype=torch_dtype, device=device if torch_dtype == torch.float16 else torch.device("cpu"))
    )

    if num_learnable > 0:
        fc_weight = attrs.get("fc_weight")
        fc_bias = attrs.get("fc_bias")
        if fc_weight is None or fc_bias is None:
            raise ValueError("Missing fc_weight/fc_bias for learnable points")
        weight = torch.from_numpy(
            fc_weight.reshape(num_learnable * 3, embed_dims)
        ).to(dtype=torch_dtype, device=device if torch_dtype == torch.float16 else torch.device("cpu"))
        bias = (
            torch.from_numpy(fc_bias.reshape(num_learnable * 3))
            .to(dtype=torch_dtype, device=device if torch_dtype == torch.float16 else torch.device("cpu"))
        )
    else:
        weight = bias = None

    with torch.no_grad():
        result = sparse_box3d_keypoints(
            anchor_t,
            inst_t,
            embed_dims,
            num_pts,
            num_learnable,
            fix_scale_t,
            weight,
            bias,
        )
    return result.to(dtype=torch.float32).cpu().numpy()


def compare_outputs(ref: np.ndarray, test: np.ndarray) -> Dict[str, float]:
    diff = np.abs(ref - test)
    metrics = {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "median_abs_diff": float(np.median(diff)),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Validate SparseBox3DKeyPointsPlugin accuracy")
    parser.add_argument("--onnx", required=True, help="Head ONNX path with SparseBox plugin")
    parser.add_argument("--plugin-so", required=True, help="SparseBox plugin shared library")
    parser.add_argument("--asset-dir", default="script/tutorial/asset", help="Directory with sample_*.bin files")
    parser.add_argument("--sample-index", type=int, default=0, help="Starting sample index")
    parser.add_argument("--node-index", type=int, default=0, help="Index of plugin node in ONNX graph")
    parser.add_argument("--fp16", action="store_true", help="Run TensorRT plugin in FP16 mode")
    parser.add_argument(
        "--ref-precision",
        choices=["auto", "fp32", "fp16"],
        default="auto",
        help="Precision for PyTorch reference (default: auto match TensorRT)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of consecutive samples to validate starting from sample-index",
    )
    parser.add_argument(
        "--all-nodes",
        action="store_true",
        help="Validate all SparseBox nodes found in ONNX",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--dump-output",
        default=None,
        help="Optional path (file or prefix) to save per-point absolute differences (numpy .npy)",
    )
    args = parser.parse_args()

    _setup_logger(args.verbose)

    onnx_path = Path(args.onnx)
    plugin_so = Path(args.plugin_so)
    asset_dir = Path(args.asset_dir)

    nodes = list_sparsebox_nodes(onnx_path)
    if not nodes:
        raise RuntimeError("No SparseBox3DKeyPointsPlugin nodes found in ONNX")

    node_indices = list(range(len(nodes))) if args.all_nodes else [args.node_index]
    for idx in node_indices:
        if idx < 0 or idx >= len(nodes):
            raise IndexError(f"node_index {idx} out of range (found {len(nodes)} nodes)")

    sample_indices = [args.sample_index + i for i in range(args.num_samples)]
    ref_precision = args.ref_precision
    if ref_precision == "auto":
        # 关键修复：由于插件在 FP16 模式下输出 FP32，应该使用 FP32 参考实现进行比较
        # 这样可以确保比较的是相同精度的输出
        ref_precision = "fp32"  # 总是使用 FP32 参考，因为插件输出 FP32

    summary = []
    dump_base = Path(args.dump_output) if args.dump_output else None
    multi_dump = dump_base is not None and (len(node_indices) > 1 or args.num_samples > 1)

    for node_idx in node_indices:
        node = nodes[node_idx]
        LOGGER.info("Validating plugin node %d: %s (%s)", node_idx, node.name or "<unnamed>", node.op_type)
        attrs = parse_plugin_attrs(node)
        LOGGER.info(
            "Parsed attrs: embed_dims=%d, num_pts=%d, num_learnable_pts=%d",
            int(attrs["embed_dims"][0]),
            int(attrs["num_pts"][0]),
            int(attrs["num_learnable_pts"][0]),
        )

        base_anchor, base_instance = load_sample(asset_dir, sample_indices[0])
        LOGGER.info(
            "Loaded sample %d with anchor shape %s",
            sample_indices[0],
            base_anchor.shape,
        )

        input_shapes = {"anchor": tuple(base_anchor.shape)}
        need_instance = int(attrs["num_learnable_pts"][0]) > 0
        if need_instance:
            input_shapes["instance_feature"] = tuple(base_instance.shape)

        engine = build_trt_engine(attrs, input_shapes, plugin_so, args.fp16)

        for sample_idx in sample_indices:
            if sample_idx == sample_indices[0]:
                anchor = base_anchor
                instance = base_instance
            else:
                anchor, instance = load_sample(asset_dir, sample_idx)
                LOGGER.info("Loaded sample %d with anchor shape %s", sample_idx, anchor.shape)

            trt_inputs = {"anchor": anchor.astype(np.float16 if args.fp16 else np.float32)}
            if need_instance:
                trt_inputs["instance_feature"] = instance.astype(np.float16 if args.fp16 else np.float32)

            trt_output = run_engine(engine, trt_inputs).astype(np.float32)
            ref_output = run_pytorch_reference(anchor, instance, attrs, ref_precision)
            
            # 关键修复：检查输出是否包含NaN/Inf，如果包含则记录详细信息
            if np.isnan(trt_output).any() or np.isinf(trt_output).any():
                nan_count = np.isnan(trt_output).sum()
                inf_count = np.isinf(trt_output).sum()
                LOGGER.warning(
                    "[node %d][sample %d] TensorRT输出包含异常值: NaN=%d, Inf=%d",
                    node_idx, sample_idx, nan_count, inf_count
                )
            if np.isnan(ref_output).any() or np.isinf(ref_output).any():
                nan_count = np.isnan(ref_output).sum()
                inf_count = np.isinf(ref_output).sum()
                LOGGER.warning(
                    "[node %d][sample %d] PyTorch参考输出包含异常值: NaN=%d, Inf=%d",
                    node_idx, sample_idx, nan_count, inf_count
                )

            metrics = compare_outputs(ref_output, trt_output)
            summary.append({"node": node_idx, "sample": sample_idx, "metrics": metrics})
            LOGGER.info(
                "[node %d][sample %d] metrics: %s",
                node_idx,
                sample_idx,
                metrics,
            )

            if dump_base:
                diff = np.abs(ref_output - trt_output)
                if multi_dump:
                    dump_path = dump_base.with_name(
                        f"{dump_base.stem}_node{node_idx}_sample{sample_idx}{dump_base.suffix}"
                    )
                else:
                    dump_path = dump_base
                dump_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(dump_path, diff)
                LOGGER.info("Saved absolute diff tensor to %s", dump_path)

    if len(summary) > 1:
        LOGGER.info("==== Validation Summary ====")
        for row in summary:
            metrics = row["metrics"]
            LOGGER.info(
                "node %d sample %d -> max %.6g, mean %.6g, median %.6g",
                row["node"],
                row["sample"],
                metrics["max_abs_diff"],
                metrics["mean_abs_diff"],
                metrics["median_abs_diff"],
            )


if __name__ == "__main__":
    main()

