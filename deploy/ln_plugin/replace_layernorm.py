import torch
import torch.nn as nn
from torch.autograd.function import Function

class LayerNormPluginFunction(Function):
    @staticmethod
    def symbolic(g, input, weight, bias, epsilon, axis):
        # Register the custom op for ONNX
        # The plugin expects inputs: input, weight, bias
        # And attributes: epsilon, axis
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
        # Fallback implementation for PyTorch inference
        # Input shape: [..., C]
        # Normalized shape is (C,) derived from input.shape[-1]
        # We assume standard LayerNorm behavior over the last dimension
        normalized_shape = (input.shape[-1],)
        return torch.nn.functional.layer_norm(input, normalized_shape, weight, bias, epsilon)

class LayerNormPluginWrapper(nn.Module):
    def __init__(self, layer_norm: nn.LayerNorm, axis=-1):
        super().__init__()
        self.epsilon = layer_norm.eps
        self.axis = axis 
        
        # Clone weight and bias
        self.register_parameter("weight", nn.Parameter(layer_norm.weight.data.clone()))
        self.register_parameter("bias", nn.Parameter(layer_norm.bias.data.clone()))

    def forward(self, x):
        return LayerNormPluginFunction.apply(x, self.weight, self.bias, self.epsilon, self.axis)

def replace_layernorm_with_plugin(model, verbose=False):
    """
    Recursively replace nn.LayerNorm with LayerNormPluginWrapper
    Only replaces LayerNorm if it normalizes over the last dimension (1D normalized_shape).
    """
    replaced_count = 0
    for name, module in model.named_children():
        if isinstance(module, nn.LayerNorm):
            # Check if it matches our assumption (normalization over last dim)
            if len(module.normalized_shape) == 1:
                # Create wrapper
                new_module = LayerNormPluginWrapper(module)
                # Replace
                setattr(model, name, new_module)
                replaced_count += 1
                if verbose:
                    pass # Reduce verbosity for individual layers
            else:
                if verbose:
                    print(f"Skipping {name}: normalized_shape={module.normalized_shape} (only 1D supported)")
        else:
            # Recurse
            replaced_count += replace_layernorm_with_plugin(module, verbose)
            
    return replaced_count

