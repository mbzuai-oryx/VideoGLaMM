import json
import os
import random
import math
import re
import numpy as np
import cv2
import torch
import torch.nn.functional as F

from PIL import Image
from decord import VideoReader, cpu

def load_video_instruct_100k_json(input_json_file_path):
    input_json_contents = json.load(open(input_json_file_path, 'r'))
    output_json_contents = []
    for i, content in enumerate(input_json_contents):
        output_content = {'id': content['video_id'], 'video': f"{content['video_id']}.mp4", 'conversations': []}

        if i % 2 == 0:
            output_content['conversations'].append({'from': 'human', 'value': f"{content['q']}\n<video>"})
        else:
            output_content['conversations'].append({'from': 'human', 'value': f"<video>\n{content['q']}"})

        output_content['conversations'].append({'from': 'gpt', 'value': content['a']})
        output_json_contents.append(output_content)

    print(f"Total annotations retained: {len(output_json_contents)}")    
    return output_json_contents

def filter_missing_videos(vqa_video_root, vqa_data):
    extensions = ['.mp4', '.mkv', '.webm']
    vqa_data_new = []
    for item in vqa_data:
        exists=False
        for ext in extensions:
            filename = item['id']+ext
            if os.path.exists(os.path.join(vqa_video_root, filename)):
                exists = True
                break
        if exists:
            # print("found:", filename)
            vqa_data_new.append({'id':item['id'], 'video':filename, 'conversations':item['conversations']})
    return vqa_data_new

def _get_rawvideo_dec_2(video_path, max_frames=64, clip_input_image_resolution=224, video_framerate=1, s=None, e=None):
    # speed up video decode via decord.
    max_video_length = 0

    if s is None:
        start_time, end_time = None, None
    else:
        start_time = int(s)
        end_time = int(e)
        start_time = start_time if start_time >= 0. else 0.
        end_time = end_time if end_time >= 0. else 0.
        if start_time > end_time:
            start_time, end_time = end_time, start_time
        elif start_time == end_time:
            end_time = start_time + 1

    if os.path.exists(video_path):
        vreader = VideoReader(video_path, ctx=cpu(0))
    else:
        print(video_path)
        raise FileNotFoundError

    fps = vreader.get_avg_fps()
    f_start = 0 if start_time is None else int(start_time * fps)
    f_end = int(min(1000000000 if end_time is None else end_time * fps, len(vreader) - 1))
    num_frames = f_end - f_start + 1
    if num_frames > 0:
        # T x 3 x H x W
        sample_fps = int(video_framerate)
        t_stride = int(round(float(fps) / sample_fps))

        all_pos = list(range(f_start, f_end + 1, t_stride))
        if len(all_pos) > max_frames:
            sample_pos = [all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=max_frames, dtype=int)]
        else:
            sample_pos = all_pos

        np_images = [f for f in vreader.get_batch(sample_pos).asnumpy()]
        
        pil_images = [Image.fromarray(f) for f in np_images]

    else:
        print("video path: {} error.".format(video_path))

    return np_images


class VideoInstruct100kDataset(torch.utils.data.Dataset):
    ignore_label = 255

    def __init__(
        self,
        base_video_dataset_dir,
        enc_preprocessor,
        sam_preprocessor,
        conversation_generator,
        image_set="train",
        video_vqa_data = "video_instruct_100k",
    ):
        self.sam_preprocessor = sam_preprocessor
        self.enc_preprocessor = enc_preprocessor
        self.conversation_generator = conversation_generator
        
        self.num_max_frames = self.enc_preprocessor.num_frames

        self.base_video_dataset_dir = base_video_dataset_dir
        
        json_dir = os.path.join(base_video_dataset_dir, video_vqa_data, 'VideoInstruct100K.json') 
        self.vqa_video_root = os.path.join(base_video_dataset_dir, f"activitynet/videos/{image_set}")
        self.vqa_data = filter_missing_videos(self.vqa_video_root, load_video_instruct_100k_json(json_dir))
        self.vqa_data = sorted(self.vqa_data, key=lambda x: x['id'])
        
        # Remove ignore indices
        ignore_filenames_file = os.path.join(self.base_video_dataset_dir, f"processed/video_instruct_100k/ignore_filenames.txt")
        self.ignore_filenames = []
        with open(ignore_filenames_file, 'r') as file:
            for line in file:
                self.ignore_filenames.append((line.strip()))  
                
        self.vqa_data = [item for item in self.vqa_data if item['video'] not in self.ignore_filenames]

        print("video_vqa_data: ", len(self.vqa_data))

    def __len__(self):
        return len(self.vqa_data)

    def __getitem__(self, idx):
        # idx = random.randint(0, len(self.vqa_data) - 1)
        item = self.vqa_data[idx]
        video_path = os.path.join(self.vqa_video_root, item["video"]) 
        
        ###
        # np_images, preprocessed_for_clip, _ = _get_rawvideo_dec(video_path, self.clip_image_processor, max_frames=self.num_max_frames, clip_input_image_resolution=224, video_framerate=1, s=None, e=None)
        np_images =  _get_rawvideo_dec_2(video_path, max_frames=self.num_max_frames, clip_input_image_resolution=224, video_framerate=1, s=None, e=None)
        
        preprocessed_for_clip = self.enc_preprocessor.preprocess(np_images)
        
        # np_images_for_sam = np_images[time_s:time_e+1]
        # np_images_for_sam = np_images[0].unsqueeze(0)
        np_images_for_sam = np.expand_dims(np_images[0], axis=0)
        ori_size = len(np_images_for_sam), *np_images_for_sam[0].shape[:2] # np_images_for_sam[0].shape[:2]
        
        ###
        
        source = item["conversations"]
        # Replace repeated <image> tokens with a single <video> token
        for sentence in source:
            sentence["value"] = re.sub(r'(<image>)+', '<video>', sentence["value"])
            
        conversations = self.conversation_generator.apply(source)
        ### 

        # preprocess for sam
        
        np_images_for_sam = [np_images[0]] # Only take one frame
        
        preprocessed_for_sam_and_resize_shapes = [self.sam_preprocessor.preprocess(image) for image in np_images_for_sam]
        preprocessed_for_sam = [x[0] for x in preprocessed_for_sam_and_resize_shapes]
        resize = preprocessed_for_sam_and_resize_shapes[0][1]
        
        
        masks = torch.rand(0, *ori_size[-2:]) # [0, H,W]
        label = torch.ones(ori_size[-2:]) * self.ignore_label # [H,W]
        ###

        data_dict = {
            'file_path': video_path,
            'preprocessed_for_sam': preprocessed_for_sam,
            'images': preprocessed_for_clip['images'],
            'context_images': preprocessed_for_clip['context_images'],
            'conversations': conversations,
            'masks': masks,
            'label': label,
            'resize': resize,
            'questions': None,
            'sampled_classes': None,
        }
        return data_dict
