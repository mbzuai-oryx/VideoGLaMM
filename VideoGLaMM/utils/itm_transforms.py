import random

import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms.functional as TF

# Define a class for the transformation
class RandomTransforms:
    def __init__(self, size=(1024, 1024)):
        self.size = size

    def __call__(self, img, mask=None):
        # # Random horizontal flip with a probability of 0.5
        # if random.random() > 0.5:
        #     img = TF.hflip(img)
        #     if mask is not None:
        #         mask = TF.hflip(mask)

        # Random resize
        resize_scale_min = 1.0
        resize_scale_max = 1.2
        scale = random.uniform(resize_scale_min, resize_scale_max)
        new_size = int(self.size[0] * scale)
        img = TF.resize(img, [new_size, new_size])
        if mask is not None:
            mask = TF.resize(mask, [new_size, new_size])

        # Random crop
        i, j, h, w = torchvision.transforms.RandomCrop.get_params(img, output_size=self.size)
        img = TF.crop(img, i, j, h, w)
        if mask is not None:
            mask = TF.crop(mask, i, j, h, w)

        # Photometric distortion (here using a simple color jitter as an example)
        color_jitter = torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)
        img = color_jitter(img)

        return img, mask
    
__transform = RandomTransforms()
def __transform_fn(preprocessed_for_sam, masks):
    ''' 
        preprocessed_for_sample: [T_sam, 3, 1024, 1024]
        masks: [num_masks, T_sam, h', w']
    '''
    h,w = masks.shape[-2:]
    masks_resized = F.interpolate(masks.float(), size=(1024, 1024), mode='nearest') # [num_masks, T_sam, h', w']-> [num_masks, T_sam, 1024, 1024]
    # Apply the transforms to an image and its corresponding mask
    image_transformed, mask_transformed = __apply_transforms(preprocessed_for_sam, masks_resized, __transform)
    
    masks_transformed_original_shape = F.interpolate(mask_transformed, size=(h, w), mode='nearest') # [num_masks, T_sam, h', w']
    
    return image_transformed, masks_transformed_original_shape

def __adjust_temporal_dimension(images, masks, T_train=5):
    """
    Adjust the time dimension of the images and masks tensors to match T_train.

    Parameters:
    - images (torch.Tensor): The images tensor with shape [T_sam, 3, 1024, 1024]
    - masks (torch.Tensor): The masks tensor with shape [num_seg_tokens_per_sample, T_sam, H, W]
    - T_train (int): The desired time dimension size.

    Returns:
    - tuple of torch.Tensor: The new_images and new_masks tensors with adjusted time dimensions.
    """
    # Get current T_sam from images or masks
    T_sam = images.shape[0]

    if T_sam < T_train:
        # Calculate number of repetitions needed
        repeat_times = T_train // T_sam
        extra_repeat = T_train % T_sam

        # Repeat the whole tensor the necessary number of whole times
        new_images = images.repeat(repeat_times, 1, 1, 1)
        new_masks = masks.repeat(1, repeat_times, 1, 1)

        # If there's a remainder, repeat enough samples to fill up to T_train
        if extra_repeat > 0:
            new_images = torch.cat((new_images, images[:extra_repeat]), dim=0)
            new_masks = torch.cat((new_masks, masks[:, :extra_repeat]), dim=1)
    else:
        # If T_sam is already greater than or equal to T_train, optionally slice the tensors
        new_images = images[:T_train]
        new_masks = masks[:, :T_train]

    return new_images, new_masks

def apply_augmentations_and_transforms(preprocessed_for_sam, masks, T_train=5):
    # make sure the required number of image, mask pairs are there
    preprocessed_for_sam, masks = __adjust_temporal_dimension(preprocessed_for_sam, masks, T_train)
    # augment frames
    preprocessed_for_sam, masks = __transform_fn(preprocessed_for_sam, masks) 
    
    return preprocessed_for_sam, masks

def __apply_transforms(images, masks, transform):
    '''
    Assume we have a batch of images and masks
        Images: [T, 3, 1024, 1024] 
        Masks: [num_classes, T, 1024, 1024]
    '''
    transformed_images = []
    transformed_masks = []

    T = images.shape[0]
    num_classes = masks.shape[0]

    for t in range(T):
        img = images[t]  # Current frame
        mask_frames = masks[:, t]  # Masks corresponding to the current frame

        if num_classes>0:
            # Apply the same transform to image and each class mask
            img_transformed, mask_transformed = transform(img, mask_frames)
            transformed_images.append(img_transformed)
            transformed_masks.append(mask_transformed)
        else:
            img_transformed, _ = transform(img)
            transformed_images.append(img_transformed)
            transformed_masks.append(mask_frames)

    return torch.stack(transformed_images), torch.stack(transformed_masks, dim=1)
