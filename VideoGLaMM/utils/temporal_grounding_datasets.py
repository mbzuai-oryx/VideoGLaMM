from torch.utils.data import Dataset
import os
import os
import torch
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
import math
import json
import jsonlines


def _get_rawvideo_dec(video_path, video_framerate=1, s=None, e=None):
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
        sample_pos = list(range(f_start, f_end + 1, t_stride))
        np_images = [f for f in vreader.get_batch(sample_pos).asnumpy()]
        # pil_images = [Image.fromarray(f) for f in np_images]
    else:
        print("video path: {} error.".format(video_path))

    return np_images


class CharadesSTA(Dataset):
    def __init__(self, base_video_dataset_dir, split='train', video_framerate=1, max_num_frames=100):
        self.base_video_dataset_dir = base_video_dataset_dir
        self.video_framerate = video_framerate
        self.max_num_frames = max_num_frames
        
        assert split in ['train', 'test'], "Dataset split must be either 'train' or 'test'"
        self.split = split
        annotations_file = os.path.join(base_video_dataset_dir, f'charades_sta/charades_sta_{self.split}.txt')
        with open(annotations_file, 'r') as file:
            annotations = file.read()
        annotations_list = annotations.split('\n')
        annotations_list = [ann for ann in annotations_list if ann != '']
        
        self.parsed_annotations = []
        for ann in annotations_list:
            ann_parts = ann.split('##')
            video_id, t_start, t_end = ann_parts[0].split(' ')
            text_query = ann_parts[1]
            # print(video_id, t_start, t_end, text_query)
            self.parsed_annotations.append({
                'video_id': video_id,
                't_start': t_start,
                't_end': t_end,
                'text_query': text_query
            })
        
        
    def __len__(self):
        return len(self.parsed_annotations)
    
    def __getitem__(self, idx):
        video_id = self.parsed_annotations[idx]['video_id']
        t_start, t_end = self.parsed_annotations[idx]['t_start'], self.parsed_annotations[idx]['t_end']
        text_query = self.parsed_annotations[idx]['text_query']
        video_path = os.path.join(self.base_video_dataset_dir, 'charades_sta/Charades_v1_480', f'{video_id}.mp4')
        # print('video_path', video_path)
        np_images = _get_rawvideo_dec(
            video_path, 
            video_framerate=self.video_framerate, 
            s=None, e=None)

        f_start = math.floor(float(t_start) * self.video_framerate)
        f_end = math.ceil(float(t_end) * self.video_framerate)
        
        # Subsample np_images if it has more than max_num_frames
        if len(np_images) > self.max_num_frames:
            new_np_images_idxs = np.linspace(0, len(np_images)-1, self.max_num_frames, dtype=int)
            new_np_images = [np_images[i] for i in new_np_images_idxs]
            new_f_start = int(f_start * (self.max_num_frames / len(np_images)))
            new_f_end = int(f_end * (self.max_num_frames / len(np_images)))
            np_images = new_np_images
            f_start, f_end = new_f_start, new_f_end
            
        return np_images, text_query, (f_start, f_end)
    
class ActivityNetCaptions(Dataset):
    def __init__(self, base_video_dataset_dir, split='train', video_framerate=1, max_num_frames=100):
        self.base_video_dataset_dir = base_video_dataset_dir
        self.video_framerate = video_framerate
        self.max_num_frames = max_num_frames
        
        assert split in ['train', 'val_1', 'val_2', 'test'], "Dataset split must be ['train', 'val_1', 'val_2', 'test']"
        self.split = split
        annotations_file = os.path.join(base_video_dataset_dir, f'activitynet_captions/{self.split}.json')
        with open(annotations_file, 'r') as file:
            annotations = json.load(file)
            
        video_ids = list(annotations.keys())
        self.parsed_annotations = []
        for video_id in video_ids:
            
            # Check if video exists
            extensions = ['.mp4', '.mkv', '.webm']
            video_exists = False
            for ext in extensions:
                video_path = os.path.join(self.base_video_dataset_dir, 'activitynet','videos',self.split, f'{video_id}{ext}')
                if os.path.exists(video_path):
                    video_exists = True
                    break
            if not video_exists:
                continue
                
            # print(video_id)
            # print(annotations[video_id])
            duration = annotations[video_id]['duration']
            timestamps = annotations[video_id]['timestamps']
            sentences = annotations[video_id]['sentences']
            
            for timestamp, sentence in zip(timestamps, sentences):
                # print(timestamp, sentence)
                t_start, t_end = timestamp
                self.parsed_annotations.append({
                    'video_id': video_id,
                    'video_path':video_path,
                    't_start': t_start,
                    't_end': t_end,
                    'text_query': sentence
                })
        
    def __len__(self):  
        return len(self.parsed_annotations)
    
    def __getitem__(self, idx):
        # video_id = self.parsed_annotations[idx]['video_id']
        video_path = self.parsed_annotations[idx]['video_path']

        # extensions = ['.mp4', '.mkv', '.webm']
        # for ext in extensions:
        #     video_path = os.path.join(self.base_video_dataset_dir, 'activitynet','videos',self.split, f'{video_id}{ext}')
        #     if os.path.exists(video_path):
        #         break

        t_start, t_end = self.parsed_annotations[idx]['t_start'], self.parsed_annotations[idx]['t_end']
        text_query = self.parsed_annotations[idx]['text_query'] 
        np_images = _get_rawvideo_dec(
            video_path, 
            video_framerate=self.video_framerate, 
            s=None, e=None)

        f_start = math.floor(float(t_start) * self.video_framerate)
        f_end = math.ceil(float(t_end) * self.video_framerate)
        
        # Subsample np_images if it has more than max_num_frames
        if len(np_images) > self.max_num_frames:
            new_np_images_idxs = np.linspace(0, len(np_images)-1, self.max_num_frames, dtype=int)
            new_np_images = [np_images[i] for i in new_np_images_idxs]
            new_f_start = int(f_start * (self.max_num_frames / len(np_images)))
            new_f_end = int(f_end * (self.max_num_frames / len(np_images)))
            np_images = new_np_images
            f_start, f_end = new_f_start, new_f_end
        
        return np_images, text_query, (f_start, f_end)
    
    
    
class QVHighlights(Dataset):
    def __init__(self, base_video_dataset_dir, split='train', video_framerate=1, max_num_frames=100):
        self.base_video_dataset_dir = base_video_dataset_dir
        self.video_framerate = video_framerate
        self.max_num_frames = max_num_frames
        assert split in ['train', 'test'], "Dataset split must be either 'train' or 'test'"
        self.split = split
        annotations_file = os.path.join(base_video_dataset_dir, f'qv_highlights/highlight_{self.split}_release.jsonl')
        self.parsed_annotations = []
        with jsonlines.open(annotations_file, 'r') as file:
            for line in file:
                self.parsed_annotations.append(line)
            
    def __len__(self):
        return len(self.parsed_annotations)
    
    def __getitem__(self, idx):
        video_path = os.path.join(self.base_video_dataset_dir, 'qv_highlights/videos', f"{self.parsed_annotations[idx]['vid']}.mp4")

        np_images = _get_rawvideo_dec(
            video_path, 
            video_framerate=self.video_framerate, 
            s=None, e=None)
        
        text_query = self.parsed_annotations[idx]['query']
        relevant_windows = self.parsed_annotations[idx]['relevant_windows']
        relevant_clip_ids = self.parsed_annotations[idx]['relevant_clip_ids']
        
        # Subsample np_images to create new_np_images
        new_np_images = np.linspace(0, len(np_images)-1,self.max_num_frames, dtype=int)
        new_np_images = [np_images[i] for i in new_np_images]
        new_relevant_windows = [[int(s * (self.max_num_frames / len(np_images))), int(e * (self.max_num_frames / len(np_images)))] for s,e in relevant_windows]


        chosen_relevant_window = max(new_relevant_windows, key=lambda x: x[1] - x[0])
        ignored_relevant_windows = [window for window in new_relevant_windows if window != chosen_relevant_window]
        
        final_np_images = []
        chosen_indices = []

        kk=0
        for t in range(len(new_np_images)):
            if  chosen_relevant_window[0]<=t and t <= chosen_relevant_window[1]:
                # print('Chosen:', kk)
                final_np_images.append(new_np_images[t])
                chosen_indices.append(kk)
                kk+=1
            elif any(f_start <= t <= f_end for f_start, f_end in ignored_relevant_windows):
                pass
            else:
                # print('Kept:', kk)
                final_np_images.append(new_np_images[t])
                kk+=1
        
        
        final_f_start, final_f_end = chosen_indices[0], chosen_indices[-1]
        
        return final_np_images, text_query, (final_f_start, final_f_end)
    
############################################################################################################

import json
import os
import random

import torch
import torch.nn.functional as F


from PIL import Image
import math
from decord import VideoReader, cpu
import numpy as np
import os
import torch

##################################################

class TemporalGroundingDataset(torch.utils.data.Dataset):
    ignore_label = 255

    def __init__(
        self,
        base_video_dataset_dir,
        enc_preprocessor,
        sam_preprocessor,
        conversation_generator,
        tg_data="charades||anetcaps||qvh", 
        image_set="train", 
        num_frames = 5 #TODO: match this with maximum video size in LISA.py
    ):
        self.sam_preprocessor = sam_preprocessor
        self.enc_preprocessor = enc_preprocessor
        self.conversation_generator = conversation_generator
        
        DEFAULT_VIDEO_TOKEN = self.conversation_generator.DEFAULT_VIDEO_TOKEN
        self.QUESTION_LIST_FOR_DECLARATIVE = [
            DEFAULT_VIDEO_TOKEN + "\n" + "Can you temporally locate {phrase} in this video?",
            DEFAULT_VIDEO_TOKEN + "\n" + "Please temporally locate {phrase} in this video.",
            DEFAULT_VIDEO_TOKEN + "\n" + "Perform temporal segmentation of {phrase}",
            DEFAULT_VIDEO_TOKEN + "\n" + "Can you indentify the range of frames containing {phrase}?",
        ]
        self.ANSWER_LIST = [
            "It is in frames:({t_start},{t_end}).",
            "Sure, frames:({t_start},{t_end}).",
            "Sure, it is within frames:({t_start},{t_end}).",
            "Sure, the localization result is in frames:({t_start},{t_end}).",
            "Frames:({t_start},{t_end}).",
        ]
        
        self.max_num_frames = self.enc_preprocessor.num_frames

        self.base_video_dataset_dir = base_video_dataset_dir
        
        self.image_set = image_set
        assert self.image_set in ["train", "val", "test"], f"invalid image_set:{self.image_set}"
        
        # 
        self.ds_list = tg_data.split("||")  # "charades||anetcaps||qvh"
        # self.dataset_dict = {}
        self.dataset_list = []
        for dataset_name in self.ds_list:
            if dataset_name=="charades":
                # Charades-STA
                dataset = CharadesSTA(base_video_dataset_dir, split=self.image_set, video_framerate=1, max_num_frames=self.max_num_frames)
                
            elif dataset_name=="anetcaps":
                dataset = ActivityNetCaptions(base_video_dataset_dir, split=self.image_set, video_framerate=1, max_num_frames=self.max_num_frames)
                
            elif dataset_name=="qvh":
                # QVHighlights
                dataset = QVHighlights(base_video_dataset_dir, split=self.image_set, video_framerate=1, max_num_frames=self.max_num_frames)
            else:
                raise Exception(f'Unsupported dataset type : {dataset_name}')
            
            # self.dataset_dict[dataset_name] = dataset
            self.dataset_list.append(dataset)
        
        self.concatenated_dataset = torch.utils.data.ConcatDataset(self.dataset_list)
        

    def __len__(self):
        return len(self.concatenated_dataset)
        # TODO : change this if val
    
    def generate_converation_from_template(self, caption, time_interval):
        
        (f_start, f_end) = time_interval
                    
        conversations = []

        conversations.append({'from': 'human', 
                              'value': random.choice(self.QUESTION_LIST_FOR_DECLARATIVE).format(phrase=caption.lower())})

        conversations.append({'from': 'gpt', 
                                'value': random.choice(self.ANSWER_LIST).format(t_start=f_start, t_end=f_end)})

        return conversations

    def __getitem__(self, idx):
        
        # ds_idx = random.randint(0, len(self.ds_list) - 1)
        # ds_name = self.ds_list[ds_idx]
        # # print(">> sampling from ", ds_name)
        # dataset = self.dataset_dict[ds_name]
        # idx = random.randint(0, len(dataset) - 1)
        
        # np_images, caption, (f_start, f_end) = dataset[idx]
        np_images, caption, (f_start, f_end) = self.concatenated_dataset[idx]
        
        ###
        pil_images = [Image.fromarray(f) for f in np_images]
        preprocessed_for_clip = self.enc_preprocessor.preprocess(pil_images)
        
        ###
        source = self.generate_converation_from_template(caption, (f_start, f_end))
        conversations = self.conversation_generator.apply(source)

        ###
        # np_images_for_sam = np_images[time_s:time_e+1]
        # np_images_for_sam = np_images[0].unsqueeze(0)
        np_images_for_sam = np.expand_dims(np_images[0], axis=0)
        ori_size = len(np_images_for_sam), *np_images_for_sam[0].shape[:2] 
        
        # preprocess for sam
        preprocessed_for_sam_and_resize_shapes = [self.sam_preprocessor.preprocess(image) for image in np_images_for_sam]
        preprocessed_for_sam = [x[0] for x in preprocessed_for_sam_and_resize_shapes]
        resize = preprocessed_for_sam_and_resize_shapes[0][1]
        
        
        masks = torch.rand(0, *ori_size[-2:])  # [0, H,W]
        label = torch.ones(ori_size[-2:]) * self.ignore_label # [H,W]
        
        ###        
        data_dict = {
            'file_path': '',
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
