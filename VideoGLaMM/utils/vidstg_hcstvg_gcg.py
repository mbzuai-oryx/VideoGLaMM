from torch.utils.data import Dataset
import os
from PIL import Image
import re
import numpy as np
import torch
import json


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



def get_phrase_and_obj_ids_from_caption(caption):
    # Pattern to match phrases in square brackets and object IDs in parentheses
    pattern = r"\[([^\]]+)\]\(([^)]+)\)"
    matches = re.findall(pattern, caption)

    # Prepare the results in a dictionary format
    results = [{"phrase": phrase, "object_ids": ids.split(", ")} for phrase, ids in matches]

    list_of_obj_ids = []
    phrases = []
    # Print the results
    for result in results:
        # print(f"Phrase: '{result['phrase']}', Object IDs: {result['object_ids']}")
        list_of_obj_ids.append(result['object_ids'])
        phrases.append(result['phrase'])
    return list_of_obj_ids, phrases

def add_seg_tokens(text):
    # Pattern to match phrases in square brackets and object IDs in parentheses
    pattern = r"\[([^\]]+)\]\(([^)]+)\)"
    
    # Substitute matched phrases with the desired format
    formatted_text = re.sub(pattern, r"<p> \1 </p> [SEG]", text)
    
    return formatted_text



class VidSTG_HCSTVG_GCGBaseDataset(Dataset):
    def __init__(self, base_video_dataset_dir = './video_dataset', image_set="train", source_dataset="vidstg"):
        # self.captions_dir = './video_dataset/vidstg_gcg/train_captions'
        # self.videos_dir = './video_dataset/vidstg_gcg/train'
        
        assert source_dataset in ["vidstg", "hcstvg"], f"Invalid type:{source_dataset}. Only vidstg and hcstvg are supported."
        
        if source_dataset=="vidstg":
            self.captions_dir = os.path.join(base_video_dataset_dir, 'vidstg_gcg', f'{image_set}_captions')
            self.videos_dir = os.path.join(base_video_dataset_dir, 'vidstg_gcg', image_set)
        elif source_dataset=="hcstvg":
            self.captions_dir = os.path.join(base_video_dataset_dir, 'hcstvg_gcg', f'{image_set}_captions')
            self.videos_dir = os.path.join(base_video_dataset_dir, 'hcstvg_gcg', image_set)
        
        self.json_files = [f for f in os.listdir(self.captions_dir) if f.endswith('.json')]
        
    def __len__(self):
        return len(self.json_files)
    
    def __getitem__(self, idx):
        ''' 
        Returns:
        - pil_images: list of PIL images              # shape: Tx()
        - all_gt_masks: list of numpy arrays    # shape: num_objsx(TxHxW)
        '''
        json_file = self.json_files[idx]
        video_id = json_file.split('.')[0]
        
        with open(os.path.join(self.captions_dir, json_file), 'r') as f:
            data_ = json.load(f)
            caption = data_['caption']
            
        list_of_obj_ids, phrases = get_phrase_and_obj_ids_from_caption(caption)
        
        # print('list_of_obj_ids:', list_of_obj_ids)
        new_caption = add_seg_tokens(caption)
        
        # list frames from videos_dir/{video_id}/frames
        frames = os.listdir(os.path.join(self.videos_dir, video_id, 'frames'))
        frames.sort()
        pil_images = []
        for frame in frames:
            pil_images.append(Image.open(os.path.join(self.videos_dir, video_id, 'frames', frame)))
            
            
        # masks are in videos_dir/{video_id}/masks/{object_id}/{frame_id}.png
        all_gt_masks = [] # 2
        for obj_ids in list_of_obj_ids: # [['1'], ['2']]
            # print('obj_ids:', obj_ids) # ['1']
            obj_id = obj_ids[0]
            # gt_masks = [] # 1
            # for obj_id in obj_ids:
            # print('obj_id:', obj_id)
            obj_masks = [] # 15
            for frame in frames:
                obj_masks.append(Image.open(os.path.join(self.videos_dir, video_id, 'masks', str(obj_id).zfill(3), frame)))
            # gt_masks.append(obj_masks)
            # all_gt_masks.append(gt_masks)
            all_gt_masks.append(obj_masks)
            
        # all_gt_masks : num_objsxTx{PIL Image}
        all_gt_masks = np.array(all_gt_masks)
        # all_gt_masks in 0-1 range
        all_gt_masks = (all_gt_masks/255).astype(np.uint8)
        
        assert len(all_gt_masks)==len(list_of_obj_ids), f"len(all_gt_masks): {len(all_gt_masks)}     len(list_of_obj_ids):{len(list_of_obj_ids)}"
        assert len(all_gt_masks)== new_caption.count('[SEG]'), f"len(all_gt_masks): {len(all_gt_masks)}     new_caption.count('[SEG]'):{new_caption.count('[SEG]')}"
                
        # # Datatype of all_gt_masks: uint8        # Value_range: [0, 1]
        
        all_masks = {}
        # for i in range(num_objs_per_video): # for each obj in video
        for i in range(len(list_of_obj_ids)):
            all_masks[i] = all_gt_masks[i].astype(bool)
                
        return video_id, json_file, pil_images, all_masks, new_caption, phrases


import os
import json
import re
import random

from PIL import Image
import numpy as np
import torch

from torch.utils.data import Dataset
    
GCG_QUESTIONS = [
    'Could you please give me a detailed description of the video? Please respond with interleaved segmentation masks for the corresponding parts of the answer.',
    'Can you provide a thorough description of the this video? Please output with interleaved segmentation masks for the corresponding phrases.',
    'Please describe in detail the contents of the video. Please respond with interleaved segmentation masks for the corresponding parts of the answer.',
    'Could you give a comprehensive explanation of what can be found within this video? Please output with interleaved segmentation masks for the corresponding phrases.',
    'Could you give me an elaborate explanation of this video? Please respond with interleaved segmentation masks for the corresponding phrases.',
    'Could you provide me with a detailed analysis of this video? Please output with interleaved segmentation masks for the corresponding parts of the answer.',
] 


class VidSTG_HCSTVG_GCGDataset(Dataset):
    ignore_label = 255

    def __init__(
        self,
        base_video_dataset_dir,
        enc_preprocessor,
        sam_preprocessor,
        conversation_generator,
        image_set="train", 
        num_frames_for_sam = 1,
        source_dataset = "vidstg"
    ):

        self.sam_preprocessor = sam_preprocessor
        self.enc_preprocessor = enc_preprocessor
        self.conversation_generator = conversation_generator
        
        self.DEFAULT_VIDEO_TOKEN = self.conversation_generator.DEFAULT_VIDEO_TOKEN
        
        self.image_set = image_set
        assert self.image_set in ["train" ], f"invalid image_set:{self.image_set}"
        
        self.dataset = VidSTG_HCSTVG_GCGBaseDataset(base_video_dataset_dir=base_video_dataset_dir, image_set=image_set, source_dataset=source_dataset)
        print("Done loading {} samples.".format(len(self.dataset)))
        
        self.num_frames_for_clip = self.enc_preprocessor.num_frames
        self.num_frames_for_sam = num_frames_for_sam
                

    def __len__(self):
        return len(self.dataset)
    
    def generate_converation_from_template(self, answer):
                    
        conversations = []

        conversations.append({'from': 'human', 
                            'value': self.DEFAULT_VIDEO_TOKEN + "\n" + random.choice(GCG_QUESTIONS)})

        conversations.append({'from': 'gpt', 
                                'value': answer})
    

        return conversations

    def __getitem__(self, idx):
        
        # pil_images_for_clip, pil_images_for_sam, gt_masks, refined_caption = self.dataset[idx]
        _, _, pil_images_for_clip, all_gt_masks, caption, _ = self.dataset[idx]
        # all_gt_masks as list of numpy arrays
        all_gt_masks = [np.array(mask) for mask in all_gt_masks.values()] # num_objsx(TxHxW)
        
        pil_images_for_sam = pil_images_for_clip.copy() # T_samx()
        pil_images_for_clip = subsample_images(pil_images_for_clip, self.num_frames_for_clip) # T_clipx()
        

        # We have
        #   ## all_gt_masks:  # shape: num_objsx(TxHxW)
        #    pil_images_for_sam : Tx()
        if self.num_frames_for_sam==1:
            pil_images_for_sam = [pil_images_for_sam[0]] # 1x()
            gt_masks = [mask[0] for mask in all_gt_masks] # num_objsx(HxW) #NOTE: taking the first frame only
            gt_masks = [np.expand_dims(mask, axis=0) for mask in gt_masks] # num_objsx(1xHxW)
        else:
            frame_indices = np.linspace(0, len(pil_images_for_sam)-1, self.num_frames_for_sam, dtype=int)
            pil_images_for_sam = [pil_images_for_sam[i] for i in frame_indices] # T_samx()
            gt_masks = [mask[frame_indices] for mask in all_gt_masks] # num_objsx(T_samxHxW)

        ###
        preprocessed_for_clip = self.enc_preprocessor.preprocess(pil_images_for_clip)
        
        ###
        source = self.generate_converation_from_template(caption)
        conversations = self.conversation_generator.apply(source)
        ### 
        
        ###
        np_images_for_sam = [np.array(image) for image in pil_images_for_sam]
        preprocessed_for_sam_and_resize_shapes = [self.sam_preprocessor.preprocess(image) for image in np_images_for_sam]
        preprocessed_for_sam = [x[0] for x in preprocessed_for_sam_and_resize_shapes]
        resize = preprocessed_for_sam_and_resize_shapes[0][1]
        

        ###
        # Datatype of gt_masks: uint8
        # Value range of gt_masks: [0, 1]
        gt_masks_ = [torch.from_numpy(image) for image in gt_masks] # num_objsx(1xHxW) or num_objsx(T_samxHxW)
        gt_masks_ = torch.stack(gt_masks_) # (num_objs,1,H,W)
        mask = (gt_masks_).to(torch.bool)
        masks = mask # masks  # [num_objects, T,H,W] # torch.bool
        
        assert masks.shape[1] == self.num_frames_for_sam
        assert len(preprocessed_for_sam) == self.num_frames_for_sam, f"len(preprocessed_for_sam):{len(preprocessed_for_sam)} != self.num_frames_for_sam:{self.num_frames_for_sam}"
        
        label = torch.ones(masks.shape[2],  masks.shape[3]) * self.ignore_label # [H,W]
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
