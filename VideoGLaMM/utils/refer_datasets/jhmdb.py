"""
JHMDB-Sentences data loader
modified from https://github.com/mttr2021/MTTR/blob/main/datasets/jhmdb_sentences/jhmdb_sentences_dataset.py
"""
from pathlib import Path

import torch
from torchvision.io import read_video
import torchvision.transforms.functional as F

from torch.utils.data import Dataset

import os
from PIL import Image
import json
import numpy as np
import random

import scipy.io

def get_image_id(video_id, frame_idx):
    image_id = f'v_{video_id}_f_{frame_idx}'
    return image_id

class JHMDBSentencesDataset(Dataset):
    """
    A Torch dataset for JHMDB-Sentences.
    For more information check out: https://kgavrilyuk.github.io/publication/actor_action/ or the original paper at:
    https://arxiv.org/abs/1803.07485
    """
    def __init__(self, dataset_path: str, ann_file: Path, transforms, return_masks: bool, 
                 num_frames: int):
        super(JHMDBSentencesDataset, self).__init__()
        self.dataset_path = dataset_path #'video_dataset'
        self.ann_file = ann_file
        self.samples_metadata = self.get_samples_metadata()

        self._transforms = transforms    
        self.return_masks = return_masks # not used
        self.num_frames = num_frames     

        print('\n')  

    def get_samples_metadata(self):
        with open(str(self.ann_file), 'r') as f:
            samples_metadata = [tuple(a) for a in json.load(f)]
            return samples_metadata

    @staticmethod
    def bounding_box(img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax # y1, y2, x1, x2 

    def __len__(self):
        return len(self.samples_metadata)
    
    def __getitem__(self, idx):
        # only support for evaluation
        video_id, chosen_frame_path, video_masks_path, video_total_frames, text_query = self.samples_metadata[idx]
        text_query = " ".join(text_query.lower().split())  # clean up the text query

        # read the source window frames:
        chosen_frame_idx = int(chosen_frame_path.split('/')[-1].split('.')[0])
        # get a window of window_size frames with frame chosen_frame_idx in the middle.
        start_idx, end_idx = chosen_frame_idx - self.num_frames // 2, chosen_frame_idx + (self.num_frames + 1) // 2
        frame_indices = list(range(start_idx, end_idx))  # note that jhmdb-sentences frames are 1-indexed
        # extract the window source frames:
        sample_indx = []
        for i in frame_indices:
            i = min(max(i, 1), video_total_frames)  # pad out of range indices with edge frames
            sample_indx.append(i)
        sample_indx.sort()
        # find the valid frame index in sampled frame list, there is only one valid frame
        valid_indices = sample_indx.index(chosen_frame_idx)

        # read frames
        imgs, boxes, masks, valid = [], [], [], []
        for i in sample_indx:
            p = '/'.join(chosen_frame_path.split('/')[:-1]) + f'/{i:05d}.png'
            frame_path = os.path.join(self.dataset_path, p[2:])
            imgs.append(Image.open(frame_path).convert('RGB'))

        # read the instance masks:
        video_masks_path = os.path.join(self.dataset_path, video_masks_path)
        all_video_masks = scipy.io.loadmat(video_masks_path)['part_mask'].transpose(2, 0, 1) # [T, H, W]
        # note that to take the center-frame corresponding mask we switch to 0-indexing:
        instance_mask = torch.tensor(all_video_masks[chosen_frame_idx - 1]) # [H, W]
        mask = instance_mask.numpy()
        if (mask > 0).any():
            y1, y2, x1, x2 = self.bounding_box(mask)
            box = torch.tensor([x1, y1, x2, y2]).to(torch.float)
            valid.append(1)
        else: # some frame didn't contain the instance
            box = torch.tensor([0, 0, 0, 0]).to(torch.float) 
            valid.append(0)
        mask = torch.from_numpy(mask)
        boxes.append(box)
        masks.append(mask)

        # transform
        h, w = instance_mask.shape[-2:]
        boxes = torch.stack(boxes, dim=0) 
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        masks = torch.stack(masks, dim=0) 
        # there is only one valid frame
        target = {
            'frames_idx': torch.tensor(sample_indx), # [T,]
            'valid_indices': torch.tensor([valid_indices]),
            'boxes': boxes,                          # [1, 4], xyxy
            'masks': masks,                          # [1, H, W]
            'valid': torch.tensor(valid),            # [1,]
            'caption': text_query,
            'orig_size': torch.as_tensor([int(h), int(w)]), 
            'size': torch.as_tensor([int(h), int(w)]),
            'image_id': get_image_id(video_id, chosen_frame_idx)
        }

        # "boxes" normalize to [0, 1] and transform from xyxy to cxcywh in self._transform
        imgs, target = self._transforms(imgs, target) 
        imgs = torch.stack(imgs, dim=0) # [T, 3, H, W]
        
        # # in 'val', valid always satisfies
        # return imgs, target
        imgs = torch.permute(imgs, (0,2,3,1)) # TCHW->THWC
        np_images = imgs.numpy()
        np_images = (np_images*255).astype(np.uint8)

        return np_images, target
