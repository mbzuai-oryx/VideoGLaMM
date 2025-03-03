import torch
import torch.nn.functional as F
import numpy as np
from model.segment_anything.utils.transforms import ResizeLongestSide

####################
# SAM1 Transforms

# transform = ResizeLongestSide(image_size)

# def preprocess_for_sam(self, x: torch.Tensor) -> torch.Tensor:
#     """Normalize pixel values and pad to a square input."""
#     # Normalize colors
#     x = (x - self.pixel_mean) / self.pixel_std

#     # Pad
#     h, w = x.shape[-2:]
#     padh = self.img_size - h
#     padw = self.img_size - w
#     x = F.pad(x, (0, padw, 0, padh))
#     return x

####################
# SAM2 Transforms

def sam_preprocess(
    x: np.ndarray, # np array in RGB format # (H, W, 3)
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
    model_type="ori" # "ori" for SAM, "effi" for Effi-SAM, "sam2" for SAM2
    ) -> torch.Tensor:
    '''
    
    Preprocessing function of Segment Anything Model, including scaling, normalization and padding.  
    Preprocess differs between SAM and Effi-SAM, where Effi-SAM use no padding.
    
    - input: ndarray
    - output: torch.Tensor
    
    Usage:
        image_np = cv2.imread(image_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        original_size_list = [image_np.shape[:2]]

        image_sam, resize_shape = __sam_preprocess(image_np, model_type=args.model_type)
        
    '''
    assert img_size==1024, \
        "both SAM and Effi-SAM receive images of size 1024^2, don't change this setting unless you're sure that your employed model works well with another size."
    x = ResizeLongestSide(img_size).apply_image(x)
    resize_shape = x.shape[:2]
    x = torch.from_numpy(x).permute(2,0,1).contiguous()

    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    if model_type=="effi" or model_type=="sam2":
        x = F.interpolate(x.unsqueeze(0), (img_size, img_size), mode="bilinear").squeeze(0)
    else:
        # Pad
        h, w = x.shape[-2:]
        padh = img_size - h
        padw = img_size - w
        x = F.pad(x, (0, padw, 0, padh))
    return x, resize_shape


class SAM_v1_Preprocess:
    def preprocess(self, x: np.ndarray) -> torch.Tensor:
        return sam_preprocess(x, model_type="ori")
    
class SAM_v2_Preprocess:    
    def preprocess(self, x: np.ndarray) -> torch.Tensor:
        return sam_preprocess(x, model_type="sam2")
