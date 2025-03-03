# -------------------------------------------------------------------------------------------------------------------
# 1. Ref-Youtube-VOS
ytvos_category_dict = {
    'airplane': 0, 'ape': 1, 'bear': 2, 'bike': 3, 'bird': 4, 'boat': 5, 'bucket': 6, 'bus': 7, 'camel': 8, 'cat': 9, 
    'cow': 10, 'crocodile': 11, 'deer': 12, 'dog': 13, 'dolphin': 14, 'duck': 15, 'eagle': 16, 'earless_seal': 17, 
    'elephant': 18, 'fish': 19, 'fox': 20, 'frisbee': 21, 'frog': 22, 'giant_panda': 23, 'giraffe': 24, 'hand': 25, 
    'hat': 26, 'hedgehog': 27, 'horse': 28, 'knife': 29, 'leopard': 30, 'lion': 31, 'lizard': 32, 'monkey': 33, 
    'motorbike': 34, 'mouse': 35, 'others': 36, 'owl': 37, 'paddle': 38, 'parachute': 39, 'parrot': 40, 'penguin': 41, 
    'person': 42, 'plant': 43, 'rabbit': 44, 'raccoon': 45, 'sedan': 46, 'shark': 47, 'sheep': 48, 'sign': 49, 
    'skateboard': 50, 'snail': 51, 'snake': 52, 'snowboard': 53, 'squirrel': 54, 'surfboard': 55, 'tennis_racket': 56, 
    'tiger': 57, 'toilet': 58, 'train': 59, 'truck': 60, 'turtle': 61, 'umbrella': 62, 'whale': 63, 'zebra': 64
}

ytvos_category_list = [
    'airplane', 'ape', 'bear', 'bike', 'bird', 'boat', 'bucket', 'bus', 'camel', 'cat', 'cow', 'crocodile', 
    'deer', 'dog', 'dolphin', 'duck', 'eagle', 'earless_seal', 'elephant', 'fish', 'fox', 'frisbee', 'frog', 
    'giant_panda', 'giraffe', 'hand', 'hat', 'hedgehog', 'horse', 'knife', 'leopard', 'lion', 'lizard', 
    'monkey', 'motorbike', 'mouse', 'others', 'owl', 'paddle', 'parachute', 'parrot', 'penguin', 'person', 
    'plant', 'rabbit', 'raccoon', 'sedan', 'shark', 'sheep', 'sign', 'skateboard', 'snail', 'snake', 'snowboard', 
    'squirrel', 'surfboard', 'tennis_racket', 'tiger', 'toilet', 'train', 'truck', 'turtle', 'umbrella', 'whale', 'zebra'
]

"""
Ref-YoutubeVOS data loader
"""
from pathlib import Path

import torch
from torch.utils.data import Dataset

import os
from PIL import Image
import json
import numpy as np
import random


class YTVOSDataset(Dataset):
    """
    A dataset class for the Refer-Youtube-VOS dataset which was first introduced in the paper:
    "URVOS: Unified Referring Video Object Segmentation Network with a Large-Scale Benchmark"
    (see https://link.springer.com/content/pdf/10.1007/978-3-030-58555-6_13.pdf).
    The original release of the dataset contained both 'first-frame' and 'full-video' expressions. However, the first
    dataset is not publicly available anymore as now only the harder 'full-video' subset is available to download
    through the Youtube-VOS referring video object segmentation competition page at:
    https://competitions.codalab.org/competitions/29139
    Furthermore, for the competition the subset's original validation set, which consists of 507 videos, was split into
    two competition 'validation' & 'test' subsets, consisting of 202 and 305 videos respectively. Evaluation can
    currently only be done on the competition 'validation' subset using the competition's server, as
    annotations were publicly released only for the 'train' subset of the competition.

    """
    def __init__(self, img_folder: Path, ann_file: Path, transforms, return_masks: bool, 
                 num_frames: int ):
        self.img_folder = img_folder     
        self.ann_file = ann_file         
        self._transforms = transforms    
        self.return_masks = return_masks # not used
        self.num_frames = num_frames     
        # create video meta data
        self.prepare_metas()       

        print('\n video num: ', len(self.videos), ' clip num: ', len(self.metas))  
        print('\n')    

    def prepare_metas(self):
        # read object information
        with open(os.path.join(str(self.img_folder), 'meta.json'), 'r') as f:
            subset_metas_by_video = json.load(f)['videos']
        
        # read expression data
        with open(str(self.ann_file), 'r') as f:
            subset_expressions_by_video = json.load(f)['videos']
        self.videos = list(subset_expressions_by_video.keys())

        self.metas = []
        for vid in self.videos:
            vid_meta = subset_metas_by_video[vid]
            vid_data = subset_expressions_by_video[vid]
            vid_frames = sorted(vid_data['frames'])
            vid_len = len(vid_frames)
            for exp_id, exp_dict in vid_data['expressions'].items():
                for frame_id in range(0, vid_len, self.num_frames):
                    meta = {}
                    meta['video'] = vid
                    meta['exp'] = exp_dict['exp']
                    meta['obj_id'] = int(exp_dict['obj_id'])
                    meta['frames'] = vid_frames
                    meta['frame_id'] = frame_id
                    # get object category
                    obj_id = exp_dict['obj_id']
                    meta['category'] = vid_meta['objects'][obj_id]['category']
                    self.metas.append(meta)

    @staticmethod
    def bounding_box(img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax # y1, y2, x1, x2 
        
    def __len__(self):
        return len(self.metas)
        
    def __getitem__(self, idx):
        instance_check = False
        while not instance_check:
            meta = self.metas[idx]  # dict

            video, exp, obj_id, category, frames, frame_id = \
                        meta['video'], meta['exp'], meta['obj_id'], meta['category'], meta['frames'], meta['frame_id']
            # clean up the caption
            exp = " ".join(exp.lower().split())
            category_id = ytvos_category_dict[category]
            vid_len = len(frames)

            num_frames = self.num_frames
            # random sparse sample
            sample_indx = [frame_id]
            if self.num_frames != 1:
                # local sample
                sample_id_before = random.randint(1, 3)
                sample_id_after = random.randint(1, 3)
                local_indx = [max(0, frame_id - sample_id_before), min(vid_len - 1, frame_id + sample_id_after)]
                sample_indx.extend(local_indx)
    
                # global sampling
                if num_frames > 3:
                    all_inds = list(range(vid_len))
                    global_inds = all_inds[:min(sample_indx)] + all_inds[max(sample_indx):]
                    global_n = num_frames - len(sample_indx)
                    if len(global_inds) > global_n:
                        select_id = random.sample(range(len(global_inds)), global_n)
                        for s_id in select_id:
                            sample_indx.append(global_inds[s_id])
                    elif vid_len >=global_n:  # sample long range global frames
                        select_id = random.sample(range(vid_len), global_n)
                        for s_id in select_id:
                            sample_indx.append(all_inds[s_id])
                    else:
                        select_id = random.sample(range(vid_len), global_n - vid_len) + list(range(vid_len))           
                        for s_id in select_id:                                                                   
                            sample_indx.append(all_inds[s_id])
            sample_indx.sort()

            # read frames and masks
            imgs, labels, boxes, masks, valid = [], [], [], [], []
            for j in range(self.num_frames):
                frame_indx = sample_indx[j]
                frame_name = frames[frame_indx]
                img_path = os.path.join(str(self.img_folder), 'JPEGImages', video, frame_name + '.jpg')
                mask_path = os.path.join(str(self.img_folder), 'Annotations', video, frame_name + '.png')
                img = Image.open(img_path).convert('RGB')
                mask = Image.open(mask_path).convert('P')

                # create the target
                label =  torch.tensor(category_id) 
                mask = np.array(mask)
                mask = (mask==obj_id).astype(np.float32) # 0,1 binary
                if (mask > 0).any():
                    y1, y2, x1, x2 = self.bounding_box(mask)
                    box = torch.tensor([x1, y1, x2, y2]).to(torch.float)
                    valid.append(1)
                else: # some frame didn't contain the instance
                    box = torch.tensor([0, 0, 0, 0]).to(torch.float) 
                    valid.append(0)
                mask = torch.from_numpy(mask)

                # append
                imgs.append(img)
                labels.append(label)
                masks.append(mask)
                boxes.append(box)

            # transform
            w, h = img.size
            labels = torch.stack(labels, dim=0) 
            boxes = torch.stack(boxes, dim=0) 
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)
            masks = torch.stack(masks, dim=0) 
            target = {
                'frames_idx': torch.tensor(sample_indx), # [T,]
                'labels': labels,                        # [T,]
                'boxes': boxes,                          # [T, 4], xyxy
                'masks': masks,                          # [T, H, W]
                'valid': torch.tensor(valid),            # [T,]
                'caption': exp,
                'orig_size': torch.as_tensor([int(h), int(w)]), 
                'size': torch.as_tensor([int(h), int(w)])
            }

            # "boxes" normalize to [0, 1] and transform from xyxy to cxcywh in self._transform
            imgs, target = self._transforms(imgs, target) 
            imgs = torch.stack(imgs, dim=0) # [T, 3, H, W]
            
            # FIXME: handle "valid", since some box may be removed due to random crop
            if torch.any(target['valid'] == 1):  # at leatst one instance
                instance_check = True
            else:
                idx = random.randint(0, self.__len__() - 1)

            imgs = torch.permute(imgs, (0,2,3,1)) # TCHW->THWC
            np_images = imgs.numpy()
            np_images = (np_images*255).astype(np.uint8)
            
            pil_images = [Image.fromarray(f) for f in np_images]
            target['pil_images'] = pil_images

        return np_images, target
