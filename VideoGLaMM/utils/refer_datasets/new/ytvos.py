import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class ReferYouTubeVOSDataset(Dataset):
    def __init__(self, video_dataset_dir, split, transform=None):
        """
        Initialize the Refer-YouTube-VOS dataset.
        
        Args:
            video_dataset_dir (str): Path to the root video dataset folder
            split (str): Dataset split, either 'train' or 'val'.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        assert split in ['train', 'valid', 'test'], "Invalid split. Must be one of ['train', 'valid', 'test']"
        
        self.ytvos_path = os.path.join(video_dataset_dir, "refer_youtube_vos")
        self.split = split
        self.transform = transform

        # Load meta expressions JSON
        if split == 'train':
            meta_file = os.path.join(self.ytvos_path, "meta_expressions", 'train', "meta_expressions.json")
            with open(meta_file, "r") as f:
                data = json.load(f)["videos"]
            self.data = data
            self.videos = sorted(data.keys())
            assert len(self.videos) == 3471, "Expected 3471 training videos, got {}".format(len(self.videos))
        
        elif split == 'valid':
            valid_meta_file = os.path.join(self.ytvos_path, "meta_expressions", 'valid', "meta_expressions.json")
            with open(valid_meta_file, "r") as f:
                valid_data = json.load(f)["videos"]
            self.data = valid_data
                
            # for some reasons the competition's validation expressions dict contains both the validation (202) & test videos (305). 
            # so we simply load the test expressions dict and use it to filter out the test videos from the validation expressions dict:
            test_meta_file = os.path.join(self.ytvos_path, "meta_expressions", "test", "meta_expressions.json")
            with open(test_meta_file, 'r') as f:
                test_data = json.load(f)['videos']
            test_videos = set(test_data.keys())
            self.videos = sorted([video for video in valid_data.keys() if video not in test_videos])
            assert len(self.videos) == 202, "Expected 202 validation videos, got {}".format(len(self.videos))
            
        elif split == 'test':
            meta_file = os.path.join(self.ytvos_path, "meta_expressions", 'test', "meta_expressions.json")
            with open(meta_file, "r") as f:
                data = json.load(f)["videos"]
            self.data = data
            self.videos = sorted(data.keys())
            assert len(self.videos) == 305, "Expected 305 test videos, got {}".format(len(self.videos))
        
        self.img_folder = os.path.join(self.ytvos_path, split, "JPEGImages")
        
        # metadata
        self.metas = []
        for video in self.videos:
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
        Returns the number of valid videos in the dataset.
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

