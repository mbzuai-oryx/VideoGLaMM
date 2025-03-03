from torch.utils.data import Dataset
import json
from PIL import Image
import os
import re
import numpy as np
import random

import torch
from torchvision.transforms.functional import to_tensor
import torch.nn.functional as F

class GroundedVideoQABaseDataset(Dataset):
    def __init__(self, base_video_dataset_dir='./video_dataset', image_set='train'):
        
        # QA annotations file
        ann_file_path = os.path.join(base_video_dataset_dir, 'grounded_video_qa/grounded_video_qa_trainval.json') #TODO add test set
        with open(ann_file_path, 'r') as file:
            self.qa_pair_anns = json.load(file)
        
        # Video directory
        self.dataset_base_dir = os.path.join(base_video_dataset_dir, 'processed', 'activitynet_entities')
        
    
    def __len__(self):
        return len(self.qa_pair_anns)
    
    def __getitem__(self, idx):
        item = self.qa_pair_anns[idx]
        video_id = item['video_id']
        seg_idx = item['seg_idx']
        qa_idx = item['qa_idx']
        
        question = item['question']
        answer = item['answer']
        
        seg_tokens_x = re.findall(r'\[SEG:(\d+)\]', answer) 
        seg_tokens = [f'[SEG:{seg_token_x}]' for seg_token_x in seg_tokens_x]
        seg_tokens_set = set(seg_tokens)
            
        
        split_dir = os.path.join(self.dataset_base_dir, f'splits/{video_id}/{seg_idx}')
        
        # List all files in the current directory
        files = sorted(os.listdir(split_dir))
        filenames = [os.path.join(split_dir, file) for file in files]
        pil_images_for_clip = [Image.open(filename) for filename in filenames]
        
        pil_images_for_sam = []
        gt_masks = []
        
        # Fetch HQ-SAM generated masks
        for seg_token in seg_tokens_set:
            frame_id = item['seg_token_to_obj'][seg_token]['frame_id']
            bbox    = item['seg_token_to_obj'][seg_token]['bbox']

            img_file_path = os.path.join(self.dataset_base_dir, f'splits/{video_id}/{seg_idx}/{str(frame_id+1).zfill(2)}.jpg')
            # print(img_file_path)
            
            # Load the image
            pil_image = Image.open(img_file_path)
            pil_images_for_sam.append(pil_image)
            
            mask_file_path = os.path.join(self.dataset_base_dir, f'masks/{video_id}/{seg_idx}/{str(qa_idx).zfill(6)}/mask.png')
            mask_image = Image.open(mask_file_path)
            gt_masks.append(mask_image)
        
        return {
            'video_id': video_id,
            'seg_idx': seg_idx,
            'filenames': filenames,
            'question': question,
            'answer': answer,
            'seg_tokens_set': seg_tokens_set,
            'pil_images_for_clip': pil_images_for_clip, # T x PIL Image
            'pil_images_for_sam': pil_images_for_sam, # N x PIL Image
            'gt_masks': gt_masks # N x PIL Image
        }
        

####################################################################################################
####################################################################################################

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


class GroundedVideoQADataset(torch.utils.data.Dataset):
    ignore_label = 255

    def __init__(
        self,
        base_video_dataset_dir,
        enc_preprocessor,
        sam_preprocessor,
        conversation_generator,
        image_set="train", 
        num_frames_for_sam = 1,
    ):

        self.enc_preprocessor=enc_preprocessor
        self.sam_preprocessor = sam_preprocessor
        self.conversation_generator = conversation_generator
        
        self.base_video_dataset_dir = base_video_dataset_dir
        
        self.image_set = image_set
        assert self.image_set in ["train", "test", ], f"invalid image_set:{self.image_set}"
        
        self.dataset = GroundedVideoQABaseDataset(base_video_dataset_dir, image_set=self.image_set)
        print("Done loading {} samples.".format(len(self.dataset)))
        
        self.num_frames_for_clip = self.enc_preprocessor.num_frames
        
        if num_frames_for_sam != 1:
            print('Warning! GroundedVideoQADataset is designed for num_frames_for_sam=1 only')
        self.num_frames_for_sam = 1
        
        self.DEFAULT_VIDEO_TOKEN = self.conversation_generator.DEFAULT_VIDEO_TOKEN
        self.QUESTION_TEMPLATES = [
            self.DEFAULT_VIDEO_TOKEN + "\n" + '{question} Please respond with interleaved segmentation masks for the corresponding parts of the answer.',
            self.DEFAULT_VIDEO_TOKEN + "\n" + '{question} Please output with interleaved segmentation masks for the corresponding phrases.',
            self.DEFAULT_VIDEO_TOKEN + "\n" + '{question} Please respond with interleaved segmentation masks for the corresponding phrases.',
        ] 

                

    def __len__(self):
        return len(self.dataset)
    
    def generate_converation_from_template(self, question, answer, image_set='train'):
                    
        conversations = []

        if image_set=='train':
            conversations.append({'from': 'human', 
                                'value': random.choice(self.QUESTION_TEMPLATES).format(question=question.lower())})

            conversations.append({'from': 'gpt', 
                                    'value': answer})
        else:
            conversations.append({'from': 'human', 
                                'value': self.DEFAULT_VIDEO_TOKEN + "\n" + '{question} Please respond with interleaved segmentation masks for the corresponding parts of the answer.'.format(question=question.lower())
                                    })

            conversations.append({'from': 'gpt', 
                                    'value': answer})

        return conversations

    def __getitem__(self, idx):
        
        ### 
        item = self.dataset[idx]
        question = item['question']
        answer = item['answer']
        answer = re.sub(r'\[SEG:\d+\]', '[SEG]', answer)
        pil_images_for_clip = item['pil_images_for_clip']
        pil_images_for_sam = item['pil_images_for_sam']
        gt_masks = item['gt_masks']
                
        ###
        if not self.num_frames_for_clip==-1:
            pil_images_for_clip = subsample_images(pil_images_for_clip, self.num_frames_for_clip)
        # preprocessed_for_clip = [self.clip_image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0] for img in pil_images_for_clip]
        enc_img = self.enc_preprocessor.preprocess(pil_images_for_clip)
        
        ###
        source = self.generate_converation_from_template(question, answer, image_set=self.image_set)
        conversations = self.conversation_generator.apply(source)
        
        ###
        np_images_for_sam = [np.array(image) for image in pil_images_for_sam]
        
        preprocessed_for_sam_and_resize_shapes = [self.sam_preprocessor.preprocess(image) for image in np_images_for_sam]
        preprocessed_for_sam = [x[0] for x in preprocessed_for_sam_and_resize_shapes]
        resize = preprocessed_for_sam_and_resize_shapes[0][1]
        
        ###
        # Datatype of gt_masks: uint8
        # Value range of gt_masks: [0, 1]
        gt_masks_ = [to_tensor(image) for image in gt_masks]
        gt_masks_ = torch.stack(gt_masks_)
        mask = (gt_masks_).to(torch.bool)
        masks = mask # masks  # [N, T,H,W] # torch.bool
        
        assert masks.shape[1] == len(preprocessed_for_sam), f"masks.shape[1]:{masks.shape[1]} != len(preprocessed_for_sam):{len(preprocessed_for_sam)}"
        
        label = torch.ones(masks.shape[2],  masks.shape[3]) * self.ignore_label # [H,W]
        ###

        data_dict = {
            'file_path': '',
            'preprocessed_for_sam': preprocessed_for_sam,
            'images': enc_img['images'],
            'context_images': enc_img['context_images'],
            'conversations': conversations,
            'masks': masks,
            'label': label,
            'resize': resize,
            'questions': None,
            'sampled_classes': None,
        }
        return data_dict
