import os
import json
import re
import random

from PIL import Image
import numpy as np
import torch

from torch.utils.data import Dataset


class ANetEntitiesGCGBaseDataset(Dataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        
        self.ann_dir = os.path.join(dataset_dir, 'anns')
        self.mask_dir = os.path.join(dataset_dir, 'masks')
        self.video_frames_dir = os.path.join(dataset_dir, 'video_frames')
        
        # list all json files in the ann_dir
        self.ann_filenames = [f for f in os.listdir(self.ann_dir) if f.endswith('.json')]
        
    def __len__(self):
        return len(self.ann_filenames)
    
    def __getitem__(self, idx):
        ann_filename = self.ann_filenames[idx]
        
        # 'v_fzp5ooc727c____6.json'
        vid, seg = ann_filename.split('.')[0].split('.')[0].split('____')
        
        # load ann object
        # json load
        obj = json.load(open(os.path.join(self.ann_dir, ann_filename)))
        
        refined_caption = obj['refined_caption']
        
        seg_tokens_x = re.findall(r'\[SEG:(\d+)\]', refined_caption) 
        seg_tokens = [f'[SEG:{seg_token_x}]' for seg_token_x in seg_tokens_x]
        seg_tokens_set = set(seg_tokens)
        
        frame_dir = os.path.join(self.video_frames_dir, vid, seg)
        
        # List all files in the current directory
        # files = sorted(os.listdir(frame_dir))
        # list only jpg files
        img_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
        filenames = [os.path.join(frame_dir, file) for file in img_files]
        pil_images_for_clip = [Image.open(filename) for filename in filenames]
        
        
        # Fetch HQ-SAM generated masks
        img_mask_store = {}
        for seg_token in seg_tokens_set:
            seg_token_x = seg_token.split(':')[1].split(']')[0]

            frame_id = obj['seg_token_to_obj'][seg_token]['frame_id']
            bbox    = obj['seg_token_to_obj'][seg_token]['bbox']

            pil_image = Image.open(os.path.join(frame_dir, f'{str(frame_id).zfill(2)}.jpg'))
            
            
            mask_file_path = os.path.join(self.mask_dir, f'{vid}____{seg}', f'{str(seg_token_x).zfill(2)}', f'mask.png')
            mask_image = Image.open(mask_file_path).convert("L")
            mask_image = np.array(mask_image, dtype=np.uint8)
            mask_image = (mask_image / 255).astype(np.uint8) # [H,W]
            img_mask_store[seg_token] = {
                'image': pil_image,
                'mask': mask_image # [H,W]
            }
        
        # add the images and masks in the order of the seg tokens
        pil_images_for_sam , gt_masks = [], []
        seg_tokens_in_order = re.findall(r'\[SEG:\d\]', refined_caption)
        for seg_token in seg_tokens_in_order:
            pil_images_for_sam.append(img_mask_store[seg_token]['image'])
            gt_masks.append(img_mask_store[seg_token]['mask'])
            
            
        gt_masks = [np.expand_dims(mask, 0) for mask in gt_masks] # num_objectsx[H,W] -> num_objectsx[1,H,W]
        # gt_masks = np.stack(gt_masks) # [num_objects, 1,H,W]
        
        
        return pil_images_for_clip, pil_images_for_sam, gt_masks, refined_caption
    
GCG_QUESTIONS = [
    'Could you please give me a detailed description of the video? Please respond with interleaved segmentation masks for the corresponding parts of the answer.',
    'Can you provide a thorough description of the this video? Please output with interleaved segmentation masks for the corresponding phrases.',
    'Please describe in detail the contents of the video. Please respond with interleaved segmentation masks for the corresponding parts of the answer.',
    'Could you give a comprehensive explanation of what can be found within this video? Please output with interleaved segmentation masks for the corresponding phrases.',
    'Could you give me an elaborate explanation of this video? Please respond with interleaved segmentation masks for the corresponding phrases.',
    'Could you provide me with a detailed analysis of this video? Please output with interleaved segmentation masks for the corresponding parts of the answer.',
] 

class ANetEntitiesGCGDataset(Dataset):
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

        self.sam_preprocessor = sam_preprocessor
        self.enc_preprocessor = enc_preprocessor
        self.conversation_generator = conversation_generator
        
        self.DEFAULT_VIDEO_TOKEN = self.conversation_generator.DEFAULT_VIDEO_TOKEN
        
        self.image_set = image_set
        assert self.image_set in ["train" ], f"invalid image_set:{self.image_set}"
        
        self.dataset = ANetEntitiesGCGBaseDataset(os.path.join(base_video_dataset_dir, 'activitynet_entities_gcg'))
        print("Done loading {} samples.".format(len(self.dataset)))
        
        if num_frames_for_sam != 1:
            print('Warning! ANetEntitiesGCGDataset is designed for num_frames_for_sam=1 only')
        self.num_frames_for_sam = 1
                

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
        
        pil_images_for_clip, pil_images_for_sam, gt_masks, refined_caption = self.dataset[idx]
                
        refined_caption = re.sub(r'\[SEG:\d\]', '[SEG]', refined_caption) # remove the index from the seg tokens

        ###
        preprocessed_for_clip = self.enc_preprocessor.preprocess(pil_images_for_clip)
        
        ###
        source = self.generate_converation_from_template(refined_caption)
        conversations = self.conversation_generator.apply(source)
        ### 
        
        # uniformly sample num_frames_for_sam frames using linspace
        pil_images_for_sam = [pil_images_for_sam[i] for i in np.linspace(0, len(pil_images_for_sam)-1, self.num_frames_for_sam, dtype=int)] # num_frames_for_sam is 1
        
        ###
        np_images_for_sam = [np.array(image) for image in pil_images_for_sam]
        
        preprocessed_for_sam_and_resize_shapes = [self.sam_preprocessor.preprocess(image) for image in np_images_for_sam]
        preprocessed_for_sam = [x[0] for x in preprocessed_for_sam_and_resize_shapes]
        resize = preprocessed_for_sam_and_resize_shapes[0][1]
        
        # gt_masks : num_objects x [1,H,W]
        # assert gt_masks[0].dtype == np.uint8, f"gt_masks.dtype:{gt_masks[0].dtype}"
        # assert np.max(gt_masks[0]) <= 1, f"np.max(gt_masks):{np.max(gt_masks[0])}"
        
        ###
        # Datatype of gt_masks: uint8
        # Value range of gt_masks: [0, 1]
        gt_masks_ = [torch.from_numpy(np.array(image)) for image in gt_masks]
        gt_masks_ = torch.stack(gt_masks_)
        mask = (gt_masks_).to(torch.bool)
        masks = mask # masks  # [num_objects, T,H,W] # torch.bool
                
        assert masks.shape[1] == len(preprocessed_for_sam), f"masks.shape[1]:{masks.shape[1]} != len(preprocessed_for_sam):{len(preprocessed_for_sam)}"
        
        label = torch.ones(masks.shape[2],  masks.shape[3]) * self.ignore_label # [H,W]
        ###
        
        data_dict = {
            'file_path': '',
            'preprocessed_for_sam': preprocessed_for_sam, # 1x[H,W,C]
            'images': preprocessed_for_clip['images'],
            'context_images': preprocessed_for_clip['context_images'],
            'conversations': conversations,
            'masks': masks, # [num_objects, 1,H,W]
            'label': label,
            'resize': resize,
            'questions': None,
            'sampled_classes': None,
        }
        return data_dict
