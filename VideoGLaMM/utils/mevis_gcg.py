import os
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

def load_mevis_json_2(mevis_root, image_set):
        
    image_root = os.path.join(mevis_root, image_set) # "./video_dataset/mevis/train"
    json_file = os.path.join(mevis_root, image_set, "meta_expressions.json") # "./video_dataset/mevis/train/meta_expressions.json"

    num_instances_without_valid_segmentation = 0
    num_instances_valid_segmentation = 0


    ann_file = json_file
    with open(str(ann_file), 'r') as f:
        subset_expressions_by_video = json.load(f)['videos']
    videos = list(subset_expressions_by_video.keys())
    print('number of video in the datasets:{}'.format(len(videos)))
    
    # 
    # metas = []
    metas_videowise = []
    if image_set=='train' or image_set=='valid_u': # only train and valid_u sets have masks
        mask_json = os.path.join(image_root, 'mask_dict.json')
        print(f'Loading masks form {mask_json} ...')
        with open(mask_json) as fp:
            mask_dict = json.load(fp)

        for vid in videos:
            vid_data = subset_expressions_by_video[vid]
            vid_frames = sorted(vid_data['frames'])
            vid_len = len(vid_frames)
            if vid_len < 2:
                continue
            
            meta_vid = {}
            meta_vid['video'] = vid
            meta_vid['frames'] = vid_frames
            meta_vid['length'] = vid_len
            meta_vid['objs'] = []
            
            for exp_id, exp_dict in vid_data['expressions'].items():
                meta = {}
                # meta['video'] = vid
                meta['exp'] = exp_dict['exp']
                meta['obj_id'] = [int(x) for x in exp_dict['obj_id']]
                meta['anno_id'] = [str(x) for x in exp_dict['anno_id']]
                # meta['frames'] = vid_frames
                meta['exp_id'] = exp_id
                meta['category'] = 0
                # meta['length'] = vid_len
                meta_vid['objs'].append(meta)
            metas_videowise.append(meta_vid)
            
    else: # valid set does not have masks
        for vid in videos:
            vid_data = subset_expressions_by_video[vid]
            vid_frames = sorted(vid_data['frames'])
            vid_len = len(vid_frames)
            
            
            meta_vid = {}
            meta_vid['video'] = vid
            meta_vid['frames'] = vid_frames
            meta_vid['length'] = vid_len
            meta_vid['objs'] = []
            
            for exp_id, exp_dict in vid_data['expressions'].items():
                meta = {}
                # meta['video'] = vid
                meta['exp'] = exp_dict['exp']
                meta['obj_id'] = -1
                meta['anno_id'] = -1
                # meta['frames'] = vid_frames
                meta['exp_id'] = exp_id
                meta['category'] = 0
                # meta['length'] = vid_len
                meta_vid['objs'].append(meta)
            metas_videowise.append(meta_vid)
            
    ###
    dataset_videowise = {}
    for meta_vid in metas_videowise:
        record = {}
        video_name = meta_vid['video']
        # file_names = [os.path.join(image_root, 'JPEGImages', meta_vid['video'], meta_vid['frames'][i] + '.jpg') for i in range(meta_vid["length"])]
        record["file_names"] = [os.path.join(image_root, 'JPEGImages', meta_vid['video'], meta_vid["frames"][i]+ '.jpg') for i in range(meta_vid["length"])]
        record["length"] = meta_vid["length"]
        record["video_name"] = meta_vid['video']
        record["objs"] = []
        
        record["objs_anns"] = {}
        
        for meta in meta_vid['objs']:
            exp = meta['exp']
            obj_ids = meta['obj_id']
            anno_ids = meta['anno_id']
            category = meta['category']
            exp_id = meta['exp_id']
            
            exp = " ".join(exp.lower().split())
            if "eval_idx" in meta:
                eval_idx = meta["eval_idx"]
            
            video_objs = []
            if image_set=='train' or image_set=='valid_u':
                for frame_idx in range(meta_vid["length"]): # for time
                    frame_objs = []
                    for x, obj_id in zip(anno_ids, obj_ids): # for objects
                        obj = {}
                        segm = mask_dict[x][frame_idx]
                        if not segm:
                            num_instances_without_valid_segmentation += 1
                            continue
                        num_instances_valid_segmentation += 1
                        bbox = [0, 0, 0, 0]
                        obj["id"] = obj_id
                        obj["segmentation"] = segm
                        obj["category_id"] = category
                        obj["bbox"] = bbox
                        frame_objs.append(obj)
                    video_objs.append(frame_objs)            
            record["objs"].append({'sentence':exp, 'obj_ids':obj_ids, 'exp_id':exp_id, 'annotations':video_objs})
            record["objs_anns"][tuple([int(i) for i in obj_ids])] = {'annotations':video_objs}
        
        # dataset_videowise.append(record)
        dataset_videowise[video_name] = record

    if num_instances_without_valid_segmentation > 0:
        print(
            "Total {} instance and Filtered out {} instances without valid segmentation. ".format(
                num_instances_valid_segmentation, num_instances_without_valid_segmentation
            )
        )
    return dataset_videowise

from PIL import Image
import numpy as np
from pycocotools import mask as cocomask



def get_imgs_and_masks_from_video(data_i, list_of_obj_ids):
    
    # video_annotations = data_i['annotations']
    
    # assert len(video_annotations)==data_i['length'], f"len(video_annotations): {len(video_annotations)}     data_i['length']:{data_i['length']}"
    vid_len = data_i['length']
    
    # h,w = data_i['annotations'][0][0]['segmentation']['size']
    imgs = []
    for frame_idx in range(vid_len):
        img_path = data_i['file_names'][frame_idx]
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        imgs.append(img)
    
    all_gt_masks = []
    for obj_ids in list_of_obj_ids:
        
        gt_masks = np.zeros((vid_len, h, w), dtype=np.uint8)
        # pred_masks = np.zeros((vid_len, h, w), dtype=np.uint8)
        
        data_i_annotations = data_i['objs_anns'][tuple([int(i) for i in obj_ids])]['annotations']

        if len(data_i_annotations)==vid_len:
            for frame_idx in range(vid_len):
                num_annotations = len(data_i_annotations[frame_idx])
                for anno_id in range(num_annotations):
                    if data_i_annotations[frame_idx]:
                        try:
                            mask_rle = data_i_annotations[frame_idx][anno_id]['segmentation']
                            if mask_rle:
                                gt_masks[frame_idx] += cocomask.decode(mask_rle)
                        except Exception as e:
                            print(e)
                            print('frame_idx:', frame_idx, '    anno_id:', anno_id)
                            
        all_gt_masks.append(gt_masks)
            
    return imgs, all_gt_masks

import re

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


from torch.utils.data import Dataset

class MevisGCGBaseDataset(Dataset):
    def __init__(self, base_video_dataset_dir = './video_dataset', image_set="train"):
        
        self.captions_dir = os.path.join(base_video_dataset_dir, "mevis_gcg", image_set) # ./video_dataset/mevis_gcg/train
        self.json_files = [f for f in os.listdir(self.captions_dir) if f.endswith('.json')]
        self.dataset_videowise = load_mevis_json_2(os.path.join(base_video_dataset_dir, "mevis"), image_set)

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        ''' 
        Returns:
        - pil_images: list of PIL images              # shape: Tx()
        - all_gt_masks: list of numpy arrays    # shape: num_objsx(TxHxW)
        '''
        json_file = self.json_files[idx]
        video_name = json_file.split('.')[0]
        data_i = self.dataset_videowise[video_name]
        
        with open(os.path.join(self.captions_dir, json_file)) as f:
            data_ = json.load(f)
            caption = data_['caption']
        
        list_of_obj_ids, phrases = get_phrase_and_obj_ids_from_caption(caption)
        
        new_caption = add_seg_tokens(caption)
        
        pil_images, all_gt_masks = get_imgs_and_masks_from_video(data_i, list_of_obj_ids)
        
        assert len(all_gt_masks)==len(list_of_obj_ids), f"len(all_gt_masks): {len(all_gt_masks)}     len(list_of_obj_ids):{len(list_of_obj_ids)}"
        assert len(all_gt_masks)== new_caption.count('[SEG]'), f"len(all_gt_masks): {len(all_gt_masks)}     new_caption.count('[SEG]'):{new_caption.count('[SEG]')}"
        
        # np_images = [np.array(image) for image in pil_images]
        # np_images = np.stack(np_images, axis=0) # TxHxWxC
        
        # for SAM
        # pil_images_for_sam = pil_images.copy() # Tx()
        # all_gt_masks # num_objsx(TxHxW)
        
        # # Datatype of all_gt_masks: uint8        # Value_range: [0, 1]
        
        # return video_name, json_file, pil_images, all_gt_masks, new_caption, phrases
        
        all_masks = {}
        # for i in range(num_objs_per_video): # for each obj in video
        for i in range(len(list_of_obj_ids)):
            all_masks[i] = all_gt_masks[i].astype(bool)
                
        return video_name, json_file, pil_images, all_masks, new_caption, phrases
    
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

class MevisGCGDataset(Dataset):
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
        assert self.image_set in ["train" ], f"invalid image_set:{self.image_set}" #TODO
        
        self.dataset = MevisGCGBaseDataset(base_video_dataset_dir, image_set)
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
        _,_, pil_images_for_clip, all_gt_masks, caption, _ = self.dataset[idx]
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
