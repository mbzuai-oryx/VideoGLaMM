import json
import torch
import os
import random

import torch
import torch.nn.functional as F

from PIL import Image
import math
import numpy as np
import pycocotools.mask as cocomask
import pycocotools

def get_masks_from_annotation(annotation, w, h, l):
    
    np_masks = []
    obj_present = np.zeros(l, dtype=bool)
    # print('l', l, '     annotation', len(annotation['segmentations']))
    for t in range(l):
        segmentation_for_frame = annotation['segmentations'][t]
        if segmentation_for_frame is not None:
            try:
                mask_t = cocomask.decode(segmentation_for_frame).astype(bool)
            except TypeError:
                rle = segmentation_for_frame
                segmentation_for_frame_ = pycocotools.mask.frPyObjects(rle, rle['size'][0], rle['size'][1])
                mask_t = cocomask.decode(segmentation_for_frame_).astype(bool)
                # mask_t = np.zeros((h,w), dtype=bool)
            obj_present[t] = True
        else:
            mask_t = np.zeros((h,w), dtype=bool)
            obj_present[t] = False
        np_masks.append(mask_t)
    np_masks = np.stack(np_masks)
    return np_masks, obj_present # (l,h,w)   # (l)

class BURST_YTVIS_GCGBaseDataset(torch.utils.data.Dataset):
    def __init__(self, base_video_dataset_dir, image_set="train", max_num_frames=5):
        
        self.max_num_frames = max_num_frames
        self.image_set = image_set
        
        # base_video_dataset_dir = './video_dataset'
        self.base_video_dataset_dir = base_video_dataset_dir
        self.image_set = image_set
        if self.image_set == 'train':
            self.is_train = True
            annotation_json_file = os.path.join(base_video_dataset_dir, 'video_gcg', 'instruction_data', 'train.json')
        elif self.image_set == 'val':
            self.is_train = False
            annotation_json_file = os.path.join(base_video_dataset_dir, 'video_gcg', 'instruction_data', 'val.json')
        elif self.image_set == 'test':
            self.is_train = False
            annotation_json_file = os.path.join(base_video_dataset_dir, 'video_gcg', 'instruction_data', 'test.json')
        else:
            raise ValueError('Invalid image_set {} provided for BURST_YTVIS_GCGBaseDataset'.format(self.image_set))
        with open(annotation_json_file, 'r') as file:
            data = json.load(file)
        
        self.videos = data['videos']
        self.annotations = data['annotations']
        
        self.mask_id_to_ann_id = {}
        for j, ann in enumerate(self.annotations):
            self.mask_id_to_ann_id[ann['id']] = j
        
        
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx):
        filenames = self.videos[idx]['file_names']
        w,h,l = self.videos[idx]['width'], self.videos[idx]['height'], self.videos[idx]['length']
        v_id2o_id = self.videos[idx]['dense_cap']['v_id2o_id']
        word_indices = self.videos[idx]['dense_cap']['token_pos']
        mask_ids = self.videos[idx]['dense_cap']['mask_id']
        caption = self.videos[idx]['dense_cap']['caption']
        dataset_split = self.videos[idx]['metadata']['dataset'] if 'metadata' in self.videos[idx] else self.videos[idx]['dataset_split']

        # filepaths = [os.path.join(self.burst_dataset_root, 'frames', self.image_set, filename)  for filename in filenames]
        
        if dataset_split == 'yt19':
            # filepaths = [os.path.join(self.base_video_dataset_dir, 'ytvis', 'vos', (self.image_set if self.image_set=='train' else 'valid'
            #                                                      ), 'JPEGImages', filename)  for filename in filenames]
            filepaths = [os.path.join(self.base_video_dataset_dir, 'ytvis', 'vos', 'train', 'JPEGImages', filename)  for filename in filenames]
        else:
            filepaths = [os.path.join(self.base_video_dataset_dir, 'burst', 'frames', ('train' if self.image_set=='train' else 'val'), filename)  for filename in filenames]
        
        ## create caption with [SEG] tokens
        caption_split = caption.split(' ')
        new_caption = []
        for i in range(len(caption_split)):
            if i in word_indices:
                new_caption.append(f'<p> {caption_split[i]} </p> [SEG]')
                # new_caption.append(f'{caption_split[i]}[SEG]')
            else:
                new_caption.append(caption_split[i])
        new_caption = ' '.join(new_caption)

        ## obtain mapping from word index to mask ids
        word_index_to_mask_ids = {}
        for word_index, mask_id in zip(word_indices, mask_ids):
            if word_index in word_index_to_mask_ids:
                word_index_to_mask_ids[word_index].append(mask_id)
            else:
                word_index_to_mask_ids[word_index] = [mask_id]
        ## obtain mapping to mask ids, in the order of them present in the caption
        new_mask_ids = {}
        for i, key in enumerate(sorted(word_index_to_mask_ids.keys())):
            new_mask_ids[i] = word_index_to_mask_ids[key]
            
        ## number of objects present in the video
        num_objs_per_video = len(new_mask_ids)
        
        # print('dataset_split', dataset_split)
        
        ## concatenate masks for each object
        all_masks = {}
        obj_presence = {}
        for i in range(num_objs_per_video): # for each obj in video
            mask_ids_per_obj = new_mask_ids[i]
            mask_per_obj = np.zeros((l,h,w), dtype=bool)
            frames_obj_present = np.zeros(l, dtype=bool)
            for mask_id in mask_ids_per_obj:
                # annotation = self.annotations[mask_id]
                annotation = self.annotations[self.mask_id_to_ann_id[mask_id]]
                np_masks, obj_present = get_masks_from_annotation(annotation, w, h, l)  # (l,h,w)
                mask_per_obj = mask_per_obj + np_masks
                frames_obj_present = frames_obj_present + obj_present
            all_masks[i] = mask_per_obj
            obj_presence[i] = frames_obj_present
        
        
        if self.is_train:
            
            ## select frames such that each object is covered
            select_frames = set()
            for i in range(num_objs_per_video): # for each obj in video
                true_indices = np.where(obj_presence[i])[0]
                random_index = np.random.choice(true_indices)            
                select_frames.add(random_index)

            ## padding with more frames
            
            iter_count = 0  # Counter to track the number of iterations
            max_iter = l  # Setting a max limit to the iterations to avoid infinite loop
            # Check if len(select_frames) is less than max_num_frames
            while len(select_frames) < self.max_num_frames and iter_count < max_iter:
                # Generate a random value between 0 and L-1
                random_value = np.random.randint(0, l)
                # Add the random value to select_frames if it's not already in the list
                if random_value not in select_frames:
                    select_frames.add(random_value)
                iter_count += 1  # Increment the iteration counter
            select_frames = sorted(select_frames)
        
            ## select number of frames
            all_masks_filtered = {}
            for key, obj_mask in enumerate(all_masks.values()):
                all_masks_filtered[key] = obj_mask[select_frames]
            filtered_filepaths = [filepaths[i] for i in select_frames]
            
        else:
            all_masks_filtered = all_masks
            filtered_filepaths = filepaths
        
        ## load images
        pil_images = []
        for i in range(len(filtered_filepaths)):
            img_path = filtered_filepaths[i]
            img = Image.open(img_path).convert('RGB')
            w, h = img.size
            pil_images.append(img)
        
        #
        video_name = ''
        json_file = ''
        phrases = [] #TODO
        return video_name, json_file, pil_images, all_masks_filtered, new_caption, phrases
            
        
GCG_QUESTIONS = [
    'Could you please give me a detailed description of the video? Please respond with interleaved segmentation masks for the corresponding parts of the answer.',
    'Can you provide a thorough description of the this video? Please output with interleaved segmentation masks for the corresponding phrases.',
    'Please describe in detail the contents of the video. Please respond with interleaved segmentation masks for the corresponding parts of the answer.',
    'Could you give a comprehensive explanation of what can be found within this video? Please output with interleaved segmentation masks for the corresponding phrases.',
    'Could you give me an elaborate explanation of this video? Please respond with interleaved segmentation masks for the corresponding phrases.',
    'Could you provide me with a detailed analysis of this video? Please output with interleaved segmentation masks for the corresponding parts of the answer.',
] 

        
class BURST_YTVIS_GCGDataset(torch.utils.data.Dataset):
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

        self.num_frames_for_sam = num_frames_for_sam
        
        self.enc_preprocessor = enc_preprocessor
        self.sam_preprocessor = sam_preprocessor
        self.conversation_generator = conversation_generator
        
        self.base_video_dataset_dir = base_video_dataset_dir
        
        self.image_set = image_set
        assert self.image_set in ["train", "val", "test"], f"invalid image_set:{self.image_set}"
        
        self.dataset = BURST_YTVIS_GCGBaseDataset(base_video_dataset_dir, max_num_frames=5)
        print("Done loading {} samples.".format(len(self.dataset)))
        
        self.DEFAULT_VIDEO_TOKEN = self.conversation_generator.DEFAULT_VIDEO_TOKEN
                

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
        
        _, _, pil_images, all_masks, new_caption, _ = self.dataset[idx]
                
        np_images = [np.array(image) for image in pil_images]
        np_images = np.stack(np_images, axis=0)
        
        ###
        preprocessed_for_clip = self.enc_preprocessor.preprocess(pil_images)
        
        ###
        source = self.generate_converation_from_template(new_caption)
        conversations = self.conversation_generator.apply(source)
        
        ###
        # preprocess image for sam
        preprocessed_for_sam_and_resize_shapes = [self.sam_preprocessor.preprocess(image) for image in np_images]
        preprocessed_for_sam = [x[0] for x in preprocessed_for_sam_and_resize_shapes]
        resize = preprocessed_for_sam_and_resize_shapes[0][1]
        
        # Datatype of gt_masks: uint8
        # Value range of gt_masks: [0, 1]
        masks = []
        for key, gt_mask in all_masks.items():
            mask = torch.tensor(gt_mask)
            masks.append(mask)
        masks = torch.stack(masks) # [N, T,H,W]
        
        # subsample preprocessed_for_sam and masks
        if not self.num_frames_for_sam==-1:
            # uniform sampling
            frame_indices = np.linspace(0, len(np_images)-1, self.num_frames_for_sam, dtype=int)
            preprocessed_for_sam = [preprocessed_for_sam[i] for i in frame_indices] # [T_sam,3,H,W]
            masks = masks[:, frame_indices] # [N, T_sam,H,W]
                
        label = torch.ones(masks.shape[2],  masks.shape[3]) * self.ignore_label # [H,W]
        ###

        assert masks.shape[1] == self.num_frames_for_sam
        assert len(preprocessed_for_sam) == self.num_frames_for_sam
        
        data_dict = {
            'file_path': '',
            'preprocessed_for_sam': preprocessed_for_sam, # [T,3,H,W]
            'images': preprocessed_for_clip['images'],
            'context_images': preprocessed_for_clip['context_images'],
            'conversations': conversations,
            'masks': masks, # [N, T,H,W]
            'label': label,
            'resize': resize,
            'questions': None,
            'sampled_classes': None,
        }
        return data_dict
