import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class ReferDAVISDataset(Dataset):
    def __init__(self, video_dataset_dir, split, transform=None):
        """
        Initialize the Refer-DAVIS dataset.

        Args:
            video_dataset_dir (str): Path to the root video dataset folder
            split (str): Dataset split, either 'train' or 'valid'.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.davis_path = os.path.join(video_dataset_dir, "processed/refer_davis/2017")
        self.split = split
        self.transform = transform

        # Load meta expressions JSON
        meta_file = os.path.join(self.davis_path, "meta_expressions", split, "meta_expressions.json")
        with open(meta_file, "r") as f:
            self.data = json.load(f)["videos"]

        # Store list of videos
        self.video_list = list(self.data.keys())
        self.img_folder = os.path.join(self.davis_path, split, "JPEGImages")
        
        # metadata
        self.metas = []
        for video in self.video_list:
            video_data = self.data[video]
            frames = video_data["frames"]
            expressions = video_data["expressions"]
            
            for expression_id, expression in expressions.items():
                self.metas.append({
                    'video': video,
                    'expression_id': expression_id,
                    'expression': expression['exp'],
                    'frames': frames
                })

    def __len__(self):
        """
        Returns the number of videos in the dataset.
        """
        return len(self.metas)

    def __getitem__(self, idx):
        """
        Get data for a single video and expression.

        Args:
            idx (int): Index of the video to retrieve.

        Returns:
            dict: A dictionary containing video frames, expression, and metadata.
        """
        meta = self.metas[idx]
        video = meta['video']
        expression_id = meta['expression_id']
        expression_text = meta['expression']
        frames = meta['frames']

        # Extract frame paths and apply transformations
        img_paths = [os.path.join(self.img_folder, video, frame + ".jpg") for frame in frames]
        imgs = [Image.open(img_path).convert('RGB') for img_path in img_paths]

        if self.transform:
            imgs = [self.transform(img) for img in imgs]
            imgs = torch.stack(imgs, dim=0)  # [video_len, 3, H, W]

        
        np_images = [np.array(image) for image in imgs]
        np_images = np.stack(np_images, axis=0)
        
        target = {
            'caption': expression_text,
            'pil_images': imgs,
            'video_path': (video, expression_id),
            'frame_ids': frames
        }
        
        return np_images, target

