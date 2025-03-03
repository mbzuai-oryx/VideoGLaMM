import torch
import numpy as np

def get_dimensions(obj):
    if isinstance(obj, (torch.Tensor, np.ndarray)):
        return list(obj.shape)
    elif isinstance(obj, list) or isinstance(obj, tuple):
        dimensions = [len(obj)]
        if len(obj) > 0:
            # Assuming uniform dimensions across all items in the list
            dimensions += get_dimensions(obj[0])
        return dimensions
    else:
        raise TypeError("Unsupported object type. Must be a PyTorch tensor, NumPy array, or a nested list of tensors/arrays.")

def print_dimensions(name, obj):
    dimensions = get_dimensions(obj)
    print('>>',name, ':', dimensions)