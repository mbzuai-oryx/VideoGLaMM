##################################################
import json
import os
import random
import math

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image

from .refer_datasets.mevis import MeVISBaseDataset

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

class MEVISDataset(torch.utils.data.Dataset):
    ignore_label = 255

    def __init__(
        self,
        base_video_dataset_dir,
        enc_preprocessor,
        sam_preprocessor,
        conversation_generator,
        image_set="train", 
        num_frames_for_sam = -1,
        debug_mode=False
    ):
        self.sam_preprocessor = sam_preprocessor
        self.enc_preprocessor = enc_preprocessor
        self.conversation_generator = conversation_generator
        
        self.default_video_token = DEFAULT_VIDEO_TOKEN = self.conversation_generator.DEFAULT_VIDEO_TOKEN
        self.question_list_for_declarative = [
                DEFAULT_VIDEO_TOKEN + "\n" + "Can you segment {phrase} in this video?",
                DEFAULT_VIDEO_TOKEN + "\n" + "Please locate {phrase} in this video.",
                DEFAULT_VIDEO_TOKEN + "\n" + "What is {phrase} in this video? Please respond with segmentation masks.",
                DEFAULT_VIDEO_TOKEN + "\n" + "Perform spatial segmentation of {phrase}",
            ]
        self.answer_list = [
                "It is [SEG].",
                "Sure, [SEG].",
                "Sure, it is [SEG].",
                "Sure, the segmentation result is [SEG].",
                "[SEG].",
            ]
        
        self.base_video_dataset_dir = base_video_dataset_dir
        
        self.image_set = image_set
        assert self.image_set in ["train", "valid", "valid_u"], f"invalid image_set:{self.image_set}"
        
        if self.image_set=='train':
            self.num_frames_for_clip = self.enc_preprocessor.num_frames
        else:
            self.num_frames_for_clip = -1
        self.num_frames_for_sam = num_frames_for_sam
        
        self.dataset = MeVISBaseDataset(base_video_dataset_dir, image_set=self.image_set, num_frames=self.num_frames_for_clip)
        print("Done loading {} samples.".format(len(self.dataset)))
        
        self.debug_mode = debug_mode
        

    def __len__(self):
        return len(self.dataset)
    
    def generate_converation_from_template(self, caption, image_set='train'):
                    
        conversations = []

        if image_set=='train':
            conversations.append({'from': 'human', 
                                'value': random.choice(self.question_list_for_declarative).format(phrase=caption.lower())})

            conversations.append({'from': 'gpt', 
                                    'value': random.choice(self.answer_list)})
        else:
            conversations.append({'from': 'human', 
                                'value': self.default_video_token + "\n" + "What is {phrase} in this video? Please respond with segmentation masks.".format(phrase=caption.lower())
                                    })

            conversations.append({'from': 'gpt', 
                                    'value': "[SEG]."})

        return conversations

    def __getitem__(self, idx):
        
        np_images, target = self.dataset[idx]
        pil_images = target["pil_images"]
        caption = target['caption']
        gt_masks = target['masks']
        video_path = target['video_path']
        
        ###
        enc_img = self.enc_preprocessor.preprocess(pil_images)
        
        ###
        source = self.generate_converation_from_template(caption, image_set=self.image_set)
        conversations = self.conversation_generator.apply(source)
        
        ###
        if not self.num_frames_for_sam==-1:
            np_images = subsample_images(np_images, self.num_frames_for_sam) # [T,H,W,C]
            gt_masks  = subsample_images(gt_masks, self.num_frames_for_sam) # [T,H,W]
            
        #
        preprocessed_for_sam_and_resize_shapes = [self.sam_preprocessor.preprocess(image) for image in np_images]
        preprocessed_for_sam = [x[0] for x in preprocessed_for_sam_and_resize_shapes]
        resize = preprocessed_for_sam_and_resize_shapes[0][1]
        
        # Datatype of gt_masks: uint8
        # Value range of gt_masks: [0, 1]
        mask = torch.tensor(gt_masks)
        masks = [mask]
        masks = torch.stack(masks) # [num_objects, T,H,W]
        
        label = torch.ones(masks.shape[2],  masks.shape[3]) * self.ignore_label # [H,W]
        ###
        
        assert masks.shape[1] == self.num_frames_for_sam
        assert len(preprocessed_for_sam) == self.num_frames_for_sam

        data_dict = {
            'file_path': video_path,
            'preprocessed_for_sam': preprocessed_for_sam,
            'images': enc_img['images'],
            'context_images': enc_img['context_images'],
            'conversations': conversations,
            'masks': masks,
            'label': label,
            'resize': resize,
            'questions': None,
            'sampled_classes': None,
        }
        if self.image_set=='valid' or self.image_set=='valid_u':
            data_dict['inference'] = True
        if self.debug_mode:
            data_dict['pil_images'] = pil_images

        return data_dict