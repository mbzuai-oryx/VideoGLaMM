# 2. Ref-DAVIS17
davis_category_dict = {
    'airplane': 0, 'backpack': 1, 'ball': 2, 'bear': 3, 'bicycle': 4, 'bird': 5, 'boat': 6, 'bottle': 7, 'box': 8, 'bus': 9, 
    'camel': 10, 'car': 11, 'carriage': 12, 'cat': 13, 'cellphone': 14, 'chamaleon': 15, 'cow': 16, 'deer': 17, 'dog': 18, 
    'dolphin': 19, 'drone': 20, 'elephant': 21, 'excavator': 22, 'fish': 23, 'goat': 24, 'golf cart': 25, 'golf club': 26, 
    'grass': 27, 'guitar': 28, 'gun': 29, 'helicopter': 30, 'horse': 31, 'hoverboard': 32, 'kart': 33, 'key': 34, 'kite': 35, 
    'koala': 36, 'leash': 37, 'lion': 38, 'lock': 39, 'mask': 40, 'microphone': 41, 'monkey': 42, 'motorcycle': 43, 'oar': 44, 
    'paper': 45, 'paraglide': 46, 'person': 47, 'pig': 48, 'pole': 49, 'potted plant': 50, 'puck': 51, 'rack': 52, 'rhino': 53, 
    'rope': 54, 'sail': 55, 'scale': 56, 'scooter': 57, 'selfie stick': 58, 'sheep': 59, 'skateboard': 60, 'ski': 61, 'ski poles': 62, 
    'snake': 63, 'snowboard': 64, 'stick': 65, 'stroller': 66, 'surfboard': 67, 'swing': 68, 'tennis racket': 69, 'tractor': 70, 
    'trailer': 71, 'train': 72, 'truck': 73, 'turtle': 74, 'varanus': 75, 'violin': 76, 'wheelchair': 77
}

davis_category_list = [
    'airplane', 'backpack', 'ball', 'bear', 'bicycle', 'bird', 'boat', 'bottle', 'box', 'bus', 'camel', 'car', 'carriage', 
    'cat', 'cellphone', 'chamaleon', 'cow', 'deer', 'dog', 'dolphin', 'drone', 'elephant', 'excavator', 'fish', 'goat', 
    'golf cart', 'golf club', 'grass', 'guitar', 'gun', 'helicopter', 'horse', 'hoverboard', 'kart', 'key', 'kite', 'koala', 
    'leash', 'lion', 'lock', 'mask', 'microphone', 'monkey', 'motorcycle', 'oar', 'paper', 'paraglide', 'person', 'pig', 
    'pole', 'potted plant', 'puck', 'rack', 'rhino', 'rope', 'sail', 'scale', 'scooter', 'selfie stick', 'sheep', 'skateboard', 
    'ski', 'ski poles', 'snake', 'snowboard', 'stick', 'stroller', 'surfboard', 'swing', 'tennis racket', 'tractor', 'trailer', 
    'train', 'truck', 'turtle', 'varanus', 'violin', 'wheelchair'
]

"""
Ref-Davis17 data loader
"""
from pathlib import Path

import torch
# from torch.autograd.grad_mode import F
from torch.utils.data import Dataset
# import datasets.transforms_video as T
import os
from PIL import Image

import json
import numpy as np
import random

class DAVIS17Dataset(Dataset):
    """
    A dataset class for the Refer-DAVIS17 dataset which was first introduced in the paper:
    "Video Object Segmentation with Language Referring Expressions"
    (see https://arxiv.org/pdf/1803.08006.pdf).
    There are 60/30 videos in train/validation set, respectively.
    """
    def __init__(self, img_folder: Path, ann_file: Path, transforms, return_masks: bool, 
                 num_frames: int, 
                 ):
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
            category_id = davis_category_dict[category]
            vid_len = len(frames)

            num_frames = self.num_frames
            # random sparse sample
            sample_indx = [frame_id]
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
            # imgs = imgs.numpy()
            np_images = imgs.numpy()
            np_images = (np_images*255).astype(np.uint8)
            
            pil_images = [Image.fromarray(f) for f in np_images]
            target['pil_images'] = pil_images

        return np_images, target
