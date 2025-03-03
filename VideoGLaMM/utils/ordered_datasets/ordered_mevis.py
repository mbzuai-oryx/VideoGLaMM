##################################################
import json
import os
import random

import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor

# from model.llava import conversation as conversation_lib
from model.chatunivi import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide
from PIL import Image
import math
import numpy as np
import os
import torch

from model.chatunivi.constants import *

from ..refer_datasets.mevis import MeVISBaseDataset

##################################################

def subsample_images(images, t):
    if isinstance(images, list):
        num_images = len(images)
        if t < num_images:
            indices = np.linspace(0, num_images - 1, num=t, dtype=int)
            return [images[i] for i in indices]
        else:
            return images
    elif isinstance(images, np.ndarray):
        T = images.shape[0]
        if t < T:
            indices = np.linspace(0, T - 1, num=t, dtype=int)
            return images[indices]
        else:
            return images
    else:
        raise ValueError("Input images must be either a list of PIL images or a numpy array.")


##################################################
def preprocess_multimodal(source):
    ''' 
    If the <video> token is at the end of the sentence, this function will move it to the beginning
    '''
    for sentence in source:
        if DEFAULT_VIDEO_TOKEN in sentence["value"]:
            sentence["value"] = (
                sentence["value"].replace(DEFAULT_VIDEO_TOKEN, "").strip()
            )
            sentence["value"] = DEFAULT_VIDEO_TOKEN + "\n" + sentence["value"]
            sentence["value"] = sentence["value"].strip()
    return source
    

QUESTION_LIST_FOR_DECLARATIVE = [
    DEFAULT_VIDEO_TOKEN + "\n" + "Can you segment {phrase} in this video?",
    DEFAULT_VIDEO_TOKEN + "\n" + "Please locate {phrase} in this video.",
    DEFAULT_VIDEO_TOKEN + "\n" + "What is {phrase} in this video? Please respond with segmentation masks.",
    DEFAULT_VIDEO_TOKEN + "\n" + "Perform spatial segmentation of {phrase}",
]

ANSWER_LIST = [
    "It is [SEG].",
    "Sure, [SEG].",
    "Sure, it is [SEG].",
    "Sure, the segmentation result is [SEG].",
    "[SEG].",
]

class OrderedMEVISDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_video_dataset_dir,
        tokenizer,
        vision_tower,
        precision: str = "fp32",
        image_size: int = 224,
        image_set="train", 
        num_frames_for_clip = -1,
        num_frames_for_sam = -1,
        debug_mode=False
    ):

        self.base_video_dataset_dir = base_video_dataset_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        
        self.image_set = image_set
        assert self.image_set in ["train", "valid", "valid_u"], f"invalid image_set:{self.image_set}"
        
        self.dataset = MeVISBaseDataset(base_video_dataset_dir, image_set=self.image_set, num_frames=num_frames_for_clip)
        print("Done loading {} samples.".format(len(self.dataset)))
        
        self.num_frames_for_clip = num_frames_for_clip
        self.num_frames_for_sam = num_frames_for_sam
        
        self.debug_mode = debug_mode
        

    def __len__(self):
        return len(self.dataset)

    def preprocess_for_sam(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
    
    def generate_converation_from_template(self, caption, image_set='train'):
                    
        conversations = []

        if image_set=='train':
            conversations.append({'from': 'human', 
                                'value': random.choice(QUESTION_LIST_FOR_DECLARATIVE).format(phrase=caption.lower())})

            conversations.append({'from': 'gpt', 
                                    'value': random.choice(ANSWER_LIST)})
        else:
            conversations.append({'from': 'human', 
                                'value': DEFAULT_VIDEO_TOKEN + "\n" + "What is {phrase} in this video? Please respond with segmentation masks.".format(phrase=caption.lower())
                                    })

            conversations.append({'from': 'gpt', 
                                    'value': "[SEG]."})

        return conversations

    def __getitem__(self, idx):
        
        # data_i = self.dataset[idx]
        # pil_images, gt_masks = get_imgs_and_masks_from_video(data_i)
        # if not self.num_frames_for_clip==-1:
        #     pil_images, gt_masks = subsample_images(pil_images, self.num_frames_for_clip), subsample_images(gt_masks, self.num_frames_for_clip) 
        
        np_images, target = self.dataset[idx]
        pil_images = target["pil_images"]
        caption = target['caption']
        video_path = '' #TODO
        
        ###
        preprocessed_for_clip = [self.clip_image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0] for img in pil_images]
        
        ###
        conv = conversation_lib.default_conversation.copy()
        # source = item["conversations"]
        source = self.generate_converation_from_template(caption, image_set=self.image_set)
        source = preprocess_multimodal(source)
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        conversations = []
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"role:{role} is not the same as {conv.roles[j % 2]}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
        ### 

        questions = None # FIXME: Is this used anywhere?
        sampled_sents = None # FIXME: Is this used anywhere?
        
        ###
        if not self.num_frames_for_sam==-1:
            np_images = subsample_images(np_images, self.num_frames_for_sam)
            gt_masks  = subsample_images(gt_masks, self.num_frames_for_sam)
            
        preprocessed_for_sam = [self.transform.apply_image(image) for image in np_images]
        resize = preprocessed_for_sam[0].shape[:2]
        preprocessed_for_sam = [self.preprocess_for_sam(torch.from_numpy(image).permute(2, 0, 1).contiguous()) for image in preprocessed_for_sam]
        
        # Datatype of gt_masks: uint8
        # Value range of gt_masks: [0, 1]
        mask = torch.tensor(gt_masks)
        masks = [mask]
        masks = torch.stack(masks) # [N, T,H,W]
        
        label = torch.ones(masks.shape[1], masks.shape[2],  masks.shape[3]) * self.ignore_label # [T,H,W]
        ###

        if self.debug_mode:
            return (
                pil_images,
                video_path, 
                preprocessed_for_sam,
                preprocessed_for_clip,
                conversations,
                masks,
                label,
                resize,
                questions,
                sampled_sents,
            )
        if self.image_set=='valid' or self.image_set=='valid_u':
            inference = True
            return (
                video_path, 
                preprocessed_for_sam,
                preprocessed_for_clip,
                conversations,
                masks,
                label,
                resize,
                questions,
                sampled_sents,
                inference
            )
        return (
            video_path, 
            preprocessed_for_sam,
            preprocessed_for_clip,
            conversations,
            masks,
            label,
            resize,
            questions,
            sampled_sents,
        )
