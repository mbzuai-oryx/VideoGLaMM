
##################################################
import json
import os
import random
import math
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn.functional as F

from PIL import Image
from decord import VideoReader, cpu

from .refer_datasets.a2d import A2DSentencesDataset
from .refer_datasets.davis import DAVIS17Dataset
from .refer_datasets.jhmdb import JHMDBSentencesDataset
from .refer_datasets.ytvos import YTVOSDataset
from .refer_datasets.transforms import make_coco_transforms

##################################################

class ReferVOSDataset(torch.utils.data.Dataset):
    ignore_label = 255

    def __init__(
        self,
        base_video_dataset_dir,
        enc_preprocessor,
        sam_preprocessor,
        conversation_generator,
        refer_vos_data="ytvos||davis17||a2d||jhmdb", 
        image_set="train", 
        num_frames_for_sam = 1
    ):
        
        self.sam_preprocessor = sam_preprocessor
        self.enc_preprocessor = enc_preprocessor
        self.conversation_generator = conversation_generator
        
        self.DEFAULT_VIDEO_TOKEN = self.conversation_generator.DEFAULT_VIDEO_TOKEN
        self.QUESTION_LIST_FOR_DECLARATIVE = [
            self.DEFAULT_VIDEO_TOKEN + "\n" + "Can you segment {phrase} in this video?",
            self.DEFAULT_VIDEO_TOKEN + "\n" + "Please locate {phrase} in this video.",
            self.DEFAULT_VIDEO_TOKEN + "\n" + "What is {phrase} in this video? Please respond with segmentation masks.",
            self.DEFAULT_VIDEO_TOKEN + "\n" + "Perform spatial segmentation of {phrase}",
        ]

        self.ANSWER_LIST = [
            "It is [SEG].",
            "Sure, [SEG].",
            "Sure, it is [SEG].",
            "Sure, the segmentation result is [SEG].",
            "[SEG].",
        ]
        
        self.num_frames_for_clip = 5
        self.num_frames_for_sam = num_frames_for_sam
        
        self.base_video_dataset_dir = base_video_dataset_dir
        
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
                        transforms=make_coco_transforms(image_set="val", do_not_normalize=True), # val transform to avoid random transforms
                        return_masks=True, num_frames=self.num_frames_for_clip)
                
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
                                        transforms=make_coco_transforms(image_set="val", do_not_normalize=True), # val transform to avoid random transforms
                                        return_masks=True, num_frames=self.num_frames_for_clip)
                
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
                                        transforms=make_coco_transforms(image_set="val", do_not_normalize=True), # val transform to avoid random transforms
                                        return_masks=True, num_frames=self.num_frames_for_clip, subset=self.image_set)
                
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
                                        transforms=make_coco_transforms(image_set="val", do_not_normalize=True), # val transform to avoid random transforms
                                        return_masks=True, num_frames=self.num_frames_for_clip)
            else:
                raise Exception(f'Unsupported dataset type : {dataset_name}')
            
            # self.refer_vos_data[dataset_name] = dataset
            self.all_refer_vos_datasets.append(dataset)
        
        self.concatenated_dataset = torch.utils.data.ConcatDataset(self.all_refer_vos_datasets)
        

    def __len__(self):
        return len(self.concatenated_dataset)
    
    def generate_converation_from_template(self, caption):
        '''
        Returns a conversation template for the given caption
        e.g. 
            [   
                {'from': 'human', 'value': 'Can you segment a person in this video?'},
                {'from': 'gpt', 'value': 'Sure, [SEG].'}
            ]
        '''
                    
        conversations = []

        conversations.append({'from': 'human', 
                              'value': random.choice(self.QUESTION_LIST_FOR_DECLARATIVE).format(phrase=caption.lower())})

        conversations.append({'from': 'gpt', 
                                'value': random.choice(self.ANSWER_LIST)})

        return conversations

    def __getitem__(self, idx):
        
        np_images, target = self.concatenated_dataset[idx] # THWC
        pil_images = target['pil_images']
        caption = target['caption']
        mask = target['masks'] # [T,H,W]

        ori_size = len(np_images), *np_images[0].shape[:2] # [T,H,W] #np_images[0].shape[:2]
        ###
        enc_out = self.enc_preprocessor.preprocess(pil_images)
        
        ###
        source = self.generate_converation_from_template(caption)
        conversations = self.conversation_generator.apply(source)
        
        ###        
        if not self.num_frames_for_sam == -1:
            # only take self.num_frames_for_sam number of frames and masks by uniformly sampling
            frame_indices = np.linspace(0, len(np_images)-1, self.num_frames_for_sam, dtype=int)
            np_images_for_sam = [np_images[i] for i in frame_indices] # [T,H,W,C]
            mask = mask[frame_indices] # [T,H,W]
        
        ## preprocess for sam
        preprocessed_for_sam_and_resize_shapes = [self.sam_preprocessor.preprocess(image) for image in np_images_for_sam]
        preprocessed_for_sam = [x[0] for x in preprocessed_for_sam_and_resize_shapes]
        resize = preprocessed_for_sam_and_resize_shapes[0][1]
        
        
        masks = [mask]
        masks = torch.stack(masks) # [num_objects, T,H,W]
        # masks: [num_objects, T,H,W] # num_objects=1 in this case
        
        # label = torch.ones(masks.shape[1], masks.shape[2],  masks.shape[3]) * self.ignore_label # label: [T,H,W]
        label = torch.ones(masks.shape[2],  masks.shape[3]) * self.ignore_label # label: [H,W]
        ###
        
        assert masks.shape[1] == self.num_frames_for_sam
        assert len(preprocessed_for_sam) == self.num_frames_for_sam
        
        data_dict = {
            'file_path': None,
            'preprocessed_for_sam': preprocessed_for_sam,
            'images': enc_out['images'],
            'context_images': enc_out['context_images'],
            'conversations': conversations,
            'masks': masks,
            'label': label,
            'resize': resize,
            'questions': None,
            'sampled_classes': None,
        }

        return data_dict
