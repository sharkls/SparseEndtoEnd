# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import torch
import argparse
import numpy as np
import tensorrt as trt
import copy
import sys
import ctypes

from typing import Optional, Dict, Any
from tool.trainer.utils import set_random_seed
from tool.utils.config import read_cfg
from tool.utils.dist_utils import init_dist, get_dist_info
from tool.utils.distributed import E2EDistributedDataParallel
from tool.utils.data_parallel import E2EDataParallel
from tool.runner.fp16_utils import wrap_fp16_model
from tool.runner.checkpoint import load_checkpoint
from dataset.dataloader_wrapper import *
from tool.tester.test_sdk import *
from dataset import *
from modules.sparse4d_detector import *
from modules.ops import feature_maps_format

# Try to load custom plugins
# (Moved inside main or global scope based on args)

# TRT Logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def parse_args():
    parser = argparse.ArgumentParser(description="Test TensorRT Engine INT8 Head")
    parser.add_argument("--config", default="dataset/config/sparse4d_temporal_r50_1x4_bs22_256x704.py", help="train config file path")
    parser.add_argument("--checkpoint", default="ckpt/sparse4dv3_r50.pth", help="checkpoint file for backbone")
    parser.add_argument("--engine_head1", default="deploy/engine/sparse4dhead1st_int8.engine", help="path to head1 engine")
    parser.add_argument("--engine_head2", default="deploy/engine/sparse4dhead2nd_int8.engine", help="path to head2 engine")
    parser.add_argument("--launcher", choices=["none", "pytorch"], default="none")
    parser.add_argument("--eval", type=str, nargs="+", default="bbox", help='evaluation metrics')
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--plugins", type=str, nargs="+", default=[], help="Path to TensorRT plugin .so files")
    return parser.parse_args()

class TRTEngine(object):
    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        
        # Buffer allocation
        self.inputs = []
        self.outputs = []
        self.bindings = [None] * self.engine.num_bindings
        
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            shape = self.engine.get_binding_shape(i)
            
            # Use torch dtype
            torch_dtype = torch.from_numpy(np.array([], dtype=dtype)).dtype
            
            # Input or Output
            if self.engine.binding_is_input(i):
                self.inputs.append({'name': name, 'index': i, 'dtype': torch_dtype, 'shape': shape})
            else:
                # Allocate output buffer on GPU using PyTorch
                # Note: Shape might contain -1 for dynamic dimensions, assuming fixed for now based on export
                # If dynamic, we need to resize based on input
                # For Sparse4D export, shapes are usually fixed.
                size = list(shape)
                if size[0] < 0: size[0] = 1 # Force batch size 1 if dynamic
                
                device_mem = torch.empty(tuple(size), dtype=torch_dtype, device='cuda')
                self.bindings[i] = int(device_mem.data_ptr())
                self.outputs.append({'name': name, 'index': i, 'mem': device_mem})

    def infer(self, input_dict):
        # 1. Bind Inputs
        for inp in self.inputs:
            name = inp['name']
            idx = inp['index']
            
            if name not in input_dict:
                print(f"[Warning] Input {name} not provided!")
                continue
            
            data = input_dict[name]
            
            # Ensure data is Tensor on GPU
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data).cuda()
            elif isinstance(data, torch.Tensor):
                if not data.is_cuda:
                    data = data.cuda()
            
            # Ensure dtype matches
            if data.dtype != inp['dtype']:
                data = data.to(dtype=inp['dtype'])
                
            # Flatten if needed? TRT expects contiguous memory. 
            # If tensor is contiguous, data_ptr is valid.
            if not data.is_contiguous():
                data = data.contiguous()
                
            self.bindings[idx] = int(data.data_ptr())

        # 2. Execute
        # Use PyTorch stream to synchronize
        stream = torch.cuda.current_stream().cuda_stream
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=stream)
        
        # 3. Return Outputs (Already in GPU tensors)
        # We need to clone or make sure we don't overwrite if reused?
        # Since we allocate outputs in init, they are reused. 
        # For safety in loop, we might want to return views or copies if needed.
        # But usually downstream consumes them immediately.
        
        # Return dict mapped by name
        results = {}
        for out in self.outputs:
            results[out['name']] = out['mem']
            
        return results

class Sparse4DHeadEngineWrapper(torch.nn.Module):
    def __init__(self, pytorch_head, engine_head1_path, engine_head2_path):
        super().__init__()
        self.pytorch_head = pytorch_head
        print(f"Loading Engine Head1: {engine_head1_path}")
        self.engine1 = TRTEngine(engine_head1_path)
        print(f"Loading Engine Head2: {engine_head2_path}")
        self.engine2 = TRTEngine(engine_head2_path)

    def to_tensor(self, data):
        if data is None:
            return None
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).cuda()
        if isinstance(data, torch.Tensor):
            return data.cuda()
        return data

    def forward(self, feature_maps, metas):
        # 1. Prepare Feature Maps
        # Check if feature_maps is already formatted (list of 3 tensors)
        # Sparse4D.extract_feat calls feature_maps_format if use_deformable_func is True
        if isinstance(feature_maps, (list, tuple)) and len(feature_maps) == 3 and \
           all(isinstance(x, torch.Tensor) for x in feature_maps):
            fmt_feats = feature_maps
        else:
            fmt_feats = feature_maps_format(feature_maps)
        
        # 2. Instance Bank
        batch_size = feature_maps[0].shape[0]
        (
            instance_feature,
            anchor,
            temp_instance_feature, 
            temp_anchor,
            time_interval,
        ) = self.pytorch_head.instance_bank.get(batch_size, metas)

        # 3. Construct Inputs
        inputs = {}
        inputs['feature'] = self.to_tensor(fmt_feats[0])
        inputs['spatial_shapes'] = self.to_tensor(fmt_feats[1]).int() # TRT usually expects int32 for shapes
        inputs['level_start_index'] = self.to_tensor(fmt_feats[2]).int()
        inputs['instance_feature'] = self.to_tensor(instance_feature)
        inputs['anchor'] = self.to_tensor(anchor)
        inputs['time_interval'] = self.to_tensor(time_interval)
        
        if 'image_wh' in metas:
            inputs['image_wh'] = self.to_tensor(metas['image_wh'])
        elif 'img_metas' in metas:
             inputs['image_wh'] = torch.stack([torch.tensor(m['image_wh']).cuda() for m in metas['img_metas']], dim=0)
             
        if 'lidar2img' in metas:
            inputs['lidar2img'] = self.to_tensor(metas['lidar2img'])
        elif 'img_metas' in metas:
            inputs['lidar2img'] = torch.stack([torch.tensor(m['lidar2img']).cuda() for m in metas['img_metas']], dim=0)

        # 4. Inference
        is_first_frame = temp_instance_feature is None
        
        if is_first_frame:
            # Head 1
            out_map = self.engine1.infer(inputs)
            
            num_anchor = 900 
            pred_inst_feat = out_map['pred_instance_feature'].reshape(1, num_anchor, 256)
            pred_anchor = out_map['pred_anchor'].reshape(1, num_anchor, 11)
            pred_cls = out_map['pred_class_score'].reshape(1, num_anchor, 10)
            pred_qt = out_map['pred_quality_score'].reshape(1, num_anchor, 2)
            
            self.pytorch_head.instance_bank.update(pred_inst_feat, pred_anchor, pred_cls)

        else:
            # Head 2
            inputs['temp_instance_feature'] = self.to_tensor(temp_instance_feature)
            inputs['temp_anchor'] = self.to_tensor(temp_anchor)
            inputs['mask'] = self.to_tensor(self.pytorch_head.instance_bank.mask).int()
            inputs['track_id'] = self.to_tensor(self.pytorch_head.instance_bank.track_id).int()
            
            out_map = self.engine2.infer(inputs)
            
            num_anchor = 900
            pred_inst_feat = out_map['pred_instance_feature'].reshape(1, num_anchor, 256)
            pred_anchor = out_map['pred_anchor'].reshape(1, num_anchor, 11)
            pred_cls = out_map['pred_class_score'].reshape(1, num_anchor, 10)
            pred_qt = out_map['pred_quality_score'].reshape(1, num_anchor, 2)
            pred_track_id = out_map['pred_track_id'].reshape(1, num_anchor) # Optional usage

        # 5. Cache
        self.pytorch_head.instance_bank.cache(
            pred_inst_feat, 
            pred_anchor, 
            pred_cls, 
            metas, 
            feature_maps 
        )
        
        # 6. Return
        return {
            "classification": [pred_cls],
            "prediction": [pred_anchor],
            "quality": [pred_qt],
            "track_id": self.pytorch_head.instance_bank.get_track_id(pred_cls, self.pytorch_head.decoder.score_threshold) if not self.training else None
        }

    def post_process(self, model_outs, output_idx=-1):
        results = self.pytorch_head.decoder.decode(
            model_outs["classification"],
            model_outs["prediction"],
            model_outs.get("track_id"),
            model_outs.get("quality"),
            output_idx=output_idx,
        )
        return results

    def loss(self, *args, **kwargs):
        return {} 

def build_module(cfg, default_args: Optional[Dict] = None) -> Any:
    cfg2 = cfg.copy()
    if default_args is not None:
        for name, value in default_args.items():
            cfg2.setdefault(name, value)
    type = cfg2.pop("type")
    return eval(type)(**cfg2)

def main():
    args = parse_args()
    
    # Load Plugins
    if args.plugins:
        for plugin in args.plugins:
            try:
                print(f"Loading plugin: {plugin}")
                ctypes.CDLL(plugin)
            except Exception as e:
                print(f"[Error] Failed to load plugin {plugin}: {e}")
                sys.exit(1)
    
    cfg = read_cfg(args.config)
    cfg["model"]["img_backbone"]["init_cfg"] = {}
    
    if args.launcher != "none":
        init_dist(args.launcher, **cfg.dist_params)
    
    set_random_seed(cfg.get("seed", 0), deterministic=args.deterministic)
    
    samples_per_gpu = cfg["data"]["test"].pop("samples_per_gpu", 1)
    dataset = build_module(cfg["data"]["test"])
    data_loader = dataloader_wrapper_without_dist(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg["data"]["workers_per_gpu"],
        dist=False,
        shuffle=False,
    )
    
    model = build_module(cfg["model"])
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.cuda().eval()
    
    print("Replacing Head with Engine Wrapper...")
    engine_head = Sparse4DHeadEngineWrapper(model.head, args.engine_head1, args.engine_head2)
    model.head = engine_head
    
    # Wrap model with DataParallel to handle DataContainer unpacking
    model = E2EDataParallel(model, device_ids=[0])

    outputs = single_gpu_test(model, data_loader)
    
    rank, _ = get_dist_info()
    if rank == 0:
        eval_kwargs = cfg.get("evaluation", {}).copy()
        for key in ["interval"]:
            eval_kwargs.pop(key, None)
        eval_kwargs.update(dict(metric=args.eval))
        print("\nEvaluation Config:", eval_kwargs)
        results_dict = dataset.evaluate(outputs, **eval_kwargs)
        print(results_dict)

if __name__ == "__main__":
    main()
