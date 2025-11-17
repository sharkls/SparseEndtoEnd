#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import onnx
from onnxsim import simplify


def parse_shapes(shapes_str: str):
    shapes = {}
    for kv in shapes_str.split(","):
        name, shape = kv.split(":")
        dims = [int(x) for x in shape.split("x")]
        shapes[name] = dims
    return shapes


def get_onnx_input_shapes(model):
    """从 ONNX 模型中提取实际的输入形状"""
    input_shapes = {}
    for inp in model.graph.input:
        shape = []
        for dim in inp.type.tensor_type.shape.dim:
            if dim.dim_value > 0:
                shape.append(dim.dim_value)
            elif dim.dim_param:
                # 动态维度，保持 None 或使用默认值
                shape.append(None)
            else:
                shape.append(1)  # 未知维度，使用默认值 1
        input_shapes[inp.name] = shape
    return input_shapes


def merge_shapes(onnx_shapes, user_shapes):
    """合并 ONNX 原始形状和用户指定的形状"""
    merged = {}
    # 先使用 ONNX 原始形状
    for name, shape in onnx_shapes.items():
        merged[name] = shape.copy()
    
    # 用用户指定的形状覆盖（如果维数匹配）
    for name, user_shape in user_shapes.items():
        if name in merged:
            onnx_shape = merged[name]
            # 检查维度数量是否匹配
            if len(user_shape) == len(onnx_shape):
                # 检查是否有动态维度（None）
                if any(d is None for d in onnx_shape):
                    # 有动态维度，直接使用用户指定的形状
                    merged[name] = user_shape
                else:
                    # 没有动态维度，检查用户指定的形状是否合理（允许 -1 表示保持原值）
                    new_shape = []
                    for i, (u_dim, o_dim) in enumerate(zip(user_shape, onnx_shape)):
                        if u_dim == -1:
                            new_shape.append(o_dim)
                        else:
                            new_shape.append(u_dim)
                    merged[name] = new_shape
            else:
                print(f"[WARN] Shape dimension mismatch for '{name}': ONNX has {len(onnx_shape)} dims, user specified {len(user_shape)} dims. Using ONNX shape.")
        else:
            print(f"[WARN] Input '{name}' not found in ONNX model. Ignoring.")
    
    return merged


def main():
    parser = argparse.ArgumentParser(description="Simplify ONNX with fixed shapes to reduce Transpose/Gather chains")
    parser.add_argument("--inp", required=True, help="input onnx path")
    parser.add_argument("--out", required=True, help="output onnx path")
    parser.add_argument(
        "--shapes",
        type=str,
        default="",
        help=(
            "input shapes, e.g. feature:1x89760x256,instance_feature:1x900x256,anchor:1x900x11,"
            "temp_instance_feature:1x600x256,temp_anchor:1x600x11,track_id:1x900. "
            "Use -1 to keep original dimension. If not specified, will use ONNX original shapes."
        ),
    )
    parser.add_argument("--save-shapes-json", type=str, default="", help="optional: dump shapes json for debug")
    args = parser.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading ONNX: {inp}")
    model = onnx.load(str(inp))

    # 获取 ONNX 模型的原始输入形状
    onnx_shapes = get_onnx_input_shapes(model)
    print(f"[INFO] ONNX model input shapes: {onnx_shapes}")

    # 解析用户指定的形状
    user_shapes = {}
    if args.shapes:
        user_shapes = parse_shapes(args.shapes)
        print(f"[INFO] User specified shapes: {user_shapes}")

    # 合并形状
    if user_shapes:
        final_shapes = merge_shapes(onnx_shapes, user_shapes)
        print(f"[INFO] Using merged shapes for simplification: {final_shapes}")
    else:
        final_shapes = onnx_shapes
        print(f"[INFO] Using ONNX original shapes for simplification: {final_shapes}")

    # 保存形状到 JSON（用于调试）
    if args.save_shapes_json:
        with open(args.save_shapes_json, "w") as f:
            json.dump(final_shapes, f, indent=2)
        print(f"[INFO] Saved shapes to: {args.save_shapes_json}")

    print("[INFO] Simplifying ONNX (const folding, pattern fusion)...")
    try:
        # 使用 merged shapes 进行简化
        # 过滤掉 None 值（动态维度），因为 onnxsim 不接受 None
        clean_shapes = {}
        for name, shape in final_shapes.items():
            if any(d is None for d in shape):
                print(f"[WARN] Skipping '{name}' with dynamic dimensions: {shape}")
                continue
            clean_shapes[name] = shape
        
        if clean_shapes:
            model_simp, check = simplify(model, overwrite_input_shapes=clean_shapes, dynamic_input_shape=False)
        else:
            # 如果没有有效的形状，使用默认简化（不固定形状）
            print("[INFO] No valid shapes to fix, using default simplification...")
            model_simp, check = simplify(model)
        
        if not check:
            raise RuntimeError("Simplified ONNX model could not be validated")
        
        # 应用额外的ONNX优化pass（减少Transpose/Gather节点）
        print("[INFO] Applying additional ONNX optimizations...")
        try:
            from onnx import optimizer
            
            optimization_passes = [
                'eliminate_nop_transpose',      # 消除无用的Transpose
                'fuse_matmul_add_bias_into_gemm',  # 融合MatMul+Add为GEMM
                'fuse_transpose_into_gemm',     # 将Transpose融合到GEMM
                'eliminate_nop_monotone_argmax',  # 消除无用的ArgMax
                'eliminate_unused_initializer',  # 消除未使用的初始化器
                'eliminate_identity',           # 消除Identity节点
                'fuse_bn_into_conv',           # 融合BN到Conv
                'fuse_consecutive_concats',     # 融合连续的Concat
                'fuse_consecutive_reduce_unsqueeze',  # 融合连续的Reduce+Unsqueeze
                'fuse_consecutive_squeezes',    # 融合连续的Squeeze
            ]
            
            # 过滤掉不支持的pass（根据ONNX版本）
            available_passes = []
            for pass_name in optimization_passes:
                try:
                    # 检查pass是否可用
                    if hasattr(optimizer, 'get_available_passes'):
                        available = optimizer.get_available_passes()
                        if pass_name in available:
                            available_passes.append(pass_name)
                    else:
                        # 旧版本ONNX，尝试直接使用
                        available_passes.append(pass_name)
                except:
                    pass
            
            if available_passes:
                print(f"[INFO] Applying {len(available_passes)} optimization passes: {', '.join(available_passes)}")
                optimized_model = optimizer.optimize(model_simp, available_passes)
                onnx.save(optimized_model, str(out))
                print(f"[INFO] Applied additional optimizations, saved to: {out}")
            else:
                print("[WARN] No available optimization passes, using simplified model")
                onnx.save(model_simp, str(out))
                print(f"[INFO] Saved simplified ONNX: {out}")
        except ImportError:
            print("[WARN] ONNX optimizer not available, skipping additional optimizations")
            onnx.save(model_simp, str(out))
            print(f"[INFO] Saved simplified ONNX: {out}")
        except Exception as e:
            print(f"[WARN] Additional optimizations failed: {e}, using simplified model")
            onnx.save(model_simp, str(out))
            print(f"[INFO] Saved simplified ONNX: {out}")
    except Exception as e:
        print(f"[ERROR] Failed to simplify ONNX: {e}")
        print("[INFO] Falling back to original ONNX...")
        # 如果简化失败，尝试不指定形状进行简化
        try:
            model_simp, check = simplify(model)
            if check:
                onnx.save(model_simp, str(out))
                print(f"[INFO] Saved simplified ONNX (without shape constraints): {out}")
            else:
                raise RuntimeError("Simplified ONNX model could not be validated")
        except Exception as e2:
            print(f"[ERROR] Fallback simplification also failed: {e2}")
            raise


if __name__ == "__main__":
    main()


