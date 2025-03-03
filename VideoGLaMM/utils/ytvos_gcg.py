from pathlib import Path
import re

def get_phrase_and_obj_ids_from_caption(caption):
    # Pattern to match phrases in square brackets and object IDs in parentheses
    pattern = r"\[([^\]]+)\]\(([^)]+)\)"
    matches = re.findall(pattern, caption)

    # Prepare the results in a dictionary format
    results = [{"phrase": phrase, "object_ids": ids.split(", ")} for phrase, ids in matches]

    list_of_obj_ids = []
    # Print the results
    for result in results:
        # print(f"Phrase: '{result['phrase']}', Object IDs: {result['object_ids']}")
        list_of_obj_ids.append(result['object_ids'])
    return list_of_obj_ids

def add_seg_tokens(text):
    # Pattern to match phrases in square brackets and object IDs in parentheses
    pattern = r"\[([^\]]+)\]\(([^)]+)\)"
    
    # Substitute matched phrases with the desired format
    formatted_text = re.sub(pattern, r"<p> \1 </p> [SEG]", text)
    
    return formatted_text

def prepare_metas_2(img_folder, ann_file, num_frames):
    # read object information
    with open(os.path.join(str(img_folder), 'meta.json'), 'r') as f:
        subset_metas_by_video = json.load(f)['videos']
    
    # read expression data
    with open(str(ann_file), 'r') as f:
        subset_expressions_by_video = json.load(f)['videos']
    videos = list(subset_expressions_by_video.keys())

    # metas = []
    metas_videowise = []
    for vid in videos:
        vid_meta = subset_metas_by_video[vid]
        vid_data = subset_expressions_by_video[vid]
        vid_frames = sorted(vid_data['frames'])
        vid_len = len(vid_frames)
        
        meta_vid = {}
        meta_vid['video'] = vid
        meta_vid['frames'] = vid_frames
        meta_vid['objs'] = []
        
        for exp_id, exp_dict in vid_data['expressions'].items():
            # for frame_id in range(0, vid_len, num_frames):
            meta = {}
            # meta['video'] = vid
            meta['exp'] = exp_dict['exp']
            meta['obj_id'] = int(exp_dict['obj_id'])
            # meta['frames'] = vid_frames
            # meta['frame_id'] = frame_id
            # get object category
            obj_id = exp_dict['obj_id']
            meta['category'] = vid_meta['objects'][obj_id]['category']
            meta_vid['objs'].append(meta)
        metas_videowise.append(meta_vid)
    
    ###
    # dataset_videowise = []
    dataset_videowise = {}
    for meta_vid in metas_videowise:
        record = {}
        record["video_name"] = meta_vid['video']
        record["frames"] = meta_vid['frames']
        record["objs"] = []
        
        for meta in meta_vid['objs']:
            exp = meta['exp']
            obj_ids = meta['obj_id']
            category = meta['category']
            
            record["objs"].append({'sentence':exp, 'obj_ids':obj_ids,
                                #    'annotations':video_objs 
                                })
        # dataset_videowise.append(record)
        dataset_videowise[meta_vid['video']] = record
        
    
    
    print('\n video num: ', len(videos), ' clip num: ', len(metas_videowise))  
    print('video wise dataset: ', len(dataset_videowise))
    return dataset_videowise

import random
import numpy as np
from PIL import Image
import torch

def get_imgs_and_masks_from_video(dataset, video_name, img_folder, num_frames, list_of_obj_ids):

    meta = dataset[video_name]  # dict

    video, frames = video_name, meta['frames']
    
    vid_len = len(frames)

    # uniform sampling
    sample_indx = random.sample(range(vid_len), num_frames)
    
    # load images
    imgs = []
    all_masks = []
    for j in range(num_frames):
        frame_indx = sample_indx[j]
        frame_name = frames[frame_indx]
        img_path = os.path.join(str(img_folder), 'JPEGImages', video, frame_name + '.jpg')
        img = Image.open(img_path).convert('RGB')
        imgs.append(img)
        
        mask_path = os.path.join(str(img_folder), 'Annotations', video, frame_name + '.png')
        mask = Image.open(mask_path).convert('P')
        mask = np.array(mask) # [H, W]
        
        # load masks
        frame_masks = []
        # for obj in meta['objs']:
        for obj_ids in list_of_obj_ids:
            # obj_id = obj['obj_ids']
            # print('>>obj_id: ', obj_ids) # ['0', '1']
            # obj_id = int(obj_id)
            mask_ = np.zeros_like(mask).astype(np.uint8)
            for obj_id in obj_ids:
                obj_id = int(obj_id)
                mask_ += (mask==obj_id).astype(np.uint8)
            # mask_ = (mask==obj_id).astype(np.uint8) # 0,1 binary
            # mask_ = torch.from_numpy(mask_) # [H, W]
            frame_masks.append(mask_)
        
        # frame_masks : num_objsx[ H, W]
        all_masks.append(frame_masks)
    
    # all_masks : num_frames x num_objsx[ H, W]
    # imgs: num_frames x ()  # pil images
    
    # Transform all_masks: 
    # num_frames x num_objsx  H x W] -> num_objsx num_frames x H x W
    all_masks = np.array(all_masks)
    all_masks = np.transpose(all_masks, (1, 0, 2, 3)) # num_objsx num_frames x H x W

    return imgs, all_masks


import os
import json
        
from torch.utils.data import Dataset

class YTVOSGCGBaseDataset(Dataset):
    def __init__(self, base_video_dataset_dir = './video_dataset', image_set = 'train', num_frames=5):
        self.ytvos_gcg_captions_dir = os.path.join(base_video_dataset_dir, 'ytvos_gcg', image_set)
        self.json_files = [f for f in os.listdir(self.ytvos_gcg_captions_dir) if f.endswith('.json')]
        
        self.num_frames = num_frames
        
        ytvos_root = Path(os.path.join(base_video_dataset_dir, "refer_youtube_vos"))
        assert ytvos_root.exists(), f'provided YTVOS path {ytvos_root} does not exist'
        PATHS = {
            "train": (ytvos_root / "train", ytvos_root / "meta_expressions" / "train" / "meta_expressions.json"),
            "val": (ytvos_root / "valid", ytvos_root / "meta_expressions" / "val" / "meta_expressions.json"),    # not used actually
        }
        self.img_folder, ann_file = PATHS[image_set]
        
        self.dataset = prepare_metas_2(self.img_folder, ann_file, self.num_frames)


    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        
        json_file = self.json_files[idx]
        video_name = json_file.split('.')[0]
        
        with open(os.path.join(self.ytvos_gcg_captions_dir, json_file)) as f:
            data_ = json.load(f)
            caption = data_['caption']
        
        list_of_obj_ids = get_phrase_and_obj_ids_from_caption(caption)
        new_caption = add_seg_tokens(caption)
        
        pil_images, all_masks = get_imgs_and_masks_from_video(self.dataset, video_name, self.img_folder, num_frames=self.num_frames, list_of_obj_ids=list_of_obj_ids)    
        
        return pil_images, all_masks, new_caption


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

class YTVOSGCGDataset(Dataset):
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
        
        self.dataset = YTVOSGCGBaseDataset(num_frames=5)
        print("Done loading {} samples.".format(len(self.dataset)))
        
        self.num_frames_for_sam = num_frames_for_sam
        self.num_frames_for_clip = self.enc_preprocessor.num_frames #TODO
                

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
        pil_images_for_clip, all_gt_masks, caption = self.dataset[idx]
        pil_images_for_sam = pil_images_for_clip.copy() # Tx()
        
        # We have
        #   ## all_gt_masks:  # shape: num_objsx(TxHxW)
        #    pil_images_for_sam : Tx()
        if self.num_frames_for_sam == 1:
            pil_images_for_sam = [pil_images_for_sam[0]] # 1x()
            gt_masks = [mask[0] for mask in all_gt_masks] # num_objsx(HxW)
            gt_masks = [np.expand_dims(mask, axis=0) for mask in gt_masks] # num_objsx(1xHxW)
        else:
            # sample frames
            frame_indices = np.linspace(0, len(pil_images_for_sam)-1, self.num_frames_for_sam, dtype=int)
            pil_images_for_sam = [pil_images_for_sam[i] for i in frame_indices] # T_sam x ()
            gt_masks = [mask[frame_indices] for mask in all_gt_masks]  # num_objsx(T_samxHxW)

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
        gt_masks_ = torch.stack(gt_masks_) # (num_objs,T_sam,H,W)
        mask = (gt_masks_).to(torch.bool)
        masks = mask # masks  # [num_objects, T_sam,H,W] # torch.bool
        
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
