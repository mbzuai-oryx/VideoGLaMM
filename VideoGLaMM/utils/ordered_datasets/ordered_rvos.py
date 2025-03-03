
##################################################
import json
import os
import random

import cv2
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor

# from model.llava import conversation as conversation_lib
from model.chatunivi import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide
from PIL import Image
import math
from decord import VideoReader, cpu
import numpy as np
import os
import torch
from pathlib import Path

from model.chatunivi.constants import *

from ..refer_datasets.a2d import A2DSentencesDataset
from ..refer_datasets.davis import DAVIS17Dataset
from ..refer_datasets.jhmdb import JHMDBSentencesDataset
from ..refer_datasets.ytvos import YTVOSDataset
from ..refer_datasets.transforms import make_coco_transforms

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

class OrderedReferVOSDataset(torch.utils.data.Dataset):
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
        num_classes_per_sample: int = 3,
        exclude_val=False,
        refer_vos_data="ytvos||davis17||a2d||jhmdb", 
        image_set="train", 
        num_frames = 5 #TODO: match this with maximum video size in LISA.py
    ):
        self.exclude_val = exclude_val
        self.num_classes_per_sample = num_classes_per_sample

        self.base_video_dataset_dir = base_video_dataset_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        
        self.image_set = image_set
        assert self.image_set in ["train", "val", "test"], f"invalid image_set:{self.image_set}"
        
        # 
        self.refer_vos_ds_list = refer_vos_data.split("||")  # ['ytvos', 'davis17', 'a2d', 'jhmdb']
        self.all_refer_vos_datasets = []
        for dataset_name in self.refer_vos_ds_list:
            if dataset_name=="ytvos":
                # Refer-YTVOS
                ytvos_root = Path(os.path.join(base_video_dataset_dir, "refer_youtube_vos"))
                assert ytvos_root.exists(), f'provided YTVOS path {ytvos_root} does not exist'
                PATHS = {
                    "train": (ytvos_root / "train", ytvos_root / "meta_expressions" / "train" / "meta_expressions.json"),
                    "val": (ytvos_root / "valid", ytvos_root / "meta_expressions" / "val" / "meta_expressions.json"),    # not used actually
                }
                img_folder, ann_file = PATHS[self.image_set]
                dataset = YTVOSDataset(img_folder, ann_file, 
                        transforms=make_coco_transforms(self.image_set, do_not_normalize=True), 
                        return_masks=True, num_frames=num_frames)
                
            elif dataset_name=="davis17":
                # Refer-DAVIS-17
                davis_root = Path(os.path.join(base_video_dataset_dir, "processed/refer_davis/2017"))
                assert davis_root.exists(), f'provided DAVIS path {davis_root} does not exist'
                PATHS = {
                    "train": (davis_root / "train", davis_root / "meta_expressions" / "train" / "meta_expressions.json"),
                    "val": (davis_root / "valid", davis_root / "meta_expressions" / "val" / "meta_expressions.json"),    # not used actually
                }
                img_folder, ann_file = PATHS[self.image_set]
                dataset = DAVIS17Dataset(img_folder, ann_file, 
                                        transforms=make_coco_transforms(self.image_set, do_not_normalize=True), 
                                        return_masks=True, num_frames=num_frames)
                
            elif dataset_name=="a2d":
                # A2D-sentence dataset
                root = Path(os.path.join(base_video_dataset_dir, "a2d_sentences"))
                assert root.exists(), f'provided A2D-Sentences path {root} does not exist'
                PATHS = {
                    "train": (root, root / "a2d_sentences_single_frame_train_annotations.json"),
                    "val": (root, root / "a2d_sentences_single_frame_test_annotations.json"),   
                }
                img_folder, ann_file = PATHS[self.image_set]
                dataset = A2DSentencesDataset(img_folder, ann_file, 
                                        transforms=make_coco_transforms(self.image_set, do_not_normalize=True), 
                                        return_masks=True, num_frames=num_frames, subset=self.image_set)
                
            elif dataset_name=="jhmdb":
                # JHMDB-sentence dataset
                root = Path(os.path.join(base_video_dataset_dir, "jhmdb_sentences"))
                assert root.exists(), f'provided JHMDB-Sentences path {root} does not exist'
                PATHS = {
                    "train": (root, root / "jhmdb_sentences_samples_metadata.json"), # not used
                    "val": (root, root / "jhmdb_sentences_samples_metadata.json"),   
                }
                img_folder, ann_file = PATHS[self.image_set]
                dataset = JHMDBSentencesDataset(base_video_dataset_dir, ann_file, 
                                        transforms=make_coco_transforms(self.image_set, do_not_normalize=True), 
                                        return_masks=True, num_frames=num_frames)
            else:
                raise Exception(f'Unsupported dataset type : {dataset_name}')
            
            # self.refer_vos_data[dataset_name] = dataset
            self.all_refer_vos_datasets.append(dataset)
        
        self.concatenated_dataset = torch.utils.data.ConcatDataset(self.all_refer_vos_datasets)
        

    def __len__(self):
        return len(self.concatenated_dataset)

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
    
    def generate_converation_from_template(self, caption):
                    
        conversations = []

        conversations.append({'from': 'human', 
                              'value': random.choice(QUESTION_LIST_FOR_DECLARATIVE).format(phrase=caption.lower())})

        conversations.append({'from': 'gpt', 
                                'value': random.choice(ANSWER_LIST)})

        return conversations

    def __getitem__(self, idx):
        
        np_images, target = self.concatenated_dataset[idx] # THWC
        pil_images = target['pil_images']
        caption = target['caption']
        video_path = None #FIXME

        ori_size = len(np_images), *np_images[0].shape[:2] # [T,H,W] #np_images[0].shape[:2]
        ###
        
        preprocessed_for_clip = [self.clip_image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0] for img in pil_images]
        
        ###
        conv = conversation_lib.default_conversation.copy()
        # source = item["conversations"]
        source = self.generate_converation_from_template(caption)
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
        # preprocessed_for_sam = [self.transform.apply_image(torch.permute(image, (2,0,1))) for image in imgs]  # preprocess image for sam
        preprocessed_for_sam = [self.transform.apply_image(image) for image in np_images]
        # print('preprocessed_for_sam', len(preprocessed_for_sam), preprocessed_for_sam[0].shape)
        resize = preprocessed_for_sam[0].shape[:2]
        # preprocessed_for_sam = [self.preprocess_for_sam(torch.from_numpy(image).permute(2, 0, 1).contiguous()) for image in preprocessed_for_sam]
        preprocessed_for_sam = [self.preprocess_for_sam(torch.from_numpy(image).permute(2, 0, 1).contiguous()) for image in preprocessed_for_sam]
        # masks = torch.rand(0, *ori_size)
        # label = torch.ones(ori_size) * self.ignore_label
        mask = target['masks']
        masks = [mask]
        masks = torch.stack(masks)
        
        # masks: [N, T,H,W] 
        # label: [T,H,W]
        label = torch.ones(masks.shape[1], masks.shape[2],  masks.shape[3]) * self.ignore_label
        ###

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
