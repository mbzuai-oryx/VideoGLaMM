import json
res_data = json.load(open('res_data_new.json'))

import json
import os

import torch
import numpy as np

from decord import VideoReader, cpu
from PIL import Image
import numpy as np

import ffmpeg
from PIL import Image
import io

def load_frames(filepath, s_time, e_time, num_frames):
    # Calculate the intervals at which frames should be sampled
    intervals = np.linspace(s_time, e_time, num_frames + 1) + (e_time - s_time) / num_frames / 2.
    frames = []

    for i, t in enumerate(intervals[:-1]):  # Ignore the last interval
        out, _ = (
            ffmpeg
            .input(filepath, ss=t)
            .filter('scale', 720, -1)  # Scale the width to 720px, height is auto
            .output('pipe:', vframes=1, format='image2', vcodec='mjpeg')
            .run(capture_stdout=True, quiet=True)
        )
        
        image = Image.open(io.BytesIO(out))
        frames.append(image)
    
    return frames

class ANetEntitiesDataset(object):

    def __init__(self, 
                 base_video_dataset_dir,
                 data_split=None # ['hidden_test', 'training', 'validation', 'testing']
                 ):
        
        self.data_split = data_split
        
        self.base_video_dataset_dir = base_video_dataset_dir # './video_dataset'
        activitynet_entities_json_dir = os.path.join(self.base_video_dataset_dir, 'activitynet_entities', 'data')
        
        if self.data_split == 'training' or self.data_split == 'validation':
            reference_file= os.path.join(activitynet_entities_json_dir, 'anet_entities_cleaned_class_thresh50_trainval.json')
        elif self.data_split == 'testing':
            reference_file= os.path.join(activitynet_entities_json_dir, 'anet_entities_cleaned_class_thresh50_test_skeleton.json')
        elif self.data_split == 'hidden_test':
            reference_file= os.path.join(activitynet_entities_json_dir, 'anet_entities_cleaned_class_thresh50_hidden_test_skeleton.json')
        
        split_file=os.path.join(self.base_video_dataset_dir, 'activitynet_entities', 'data', 'split_ids_anet_entities.json')
        # './video_dataset/activitynet_entities/data/split_ids_anet_entities.json'
        
        anet_split = 'train' if self.data_split == 'training' or self.data_split == 'validation' else 'test'
        self.video_root = os.path.join(self.base_video_dataset_dir, f"activitynet/videos/{anet_split}")

        if not reference_file:
            raise IOError('Please input a valid reference file!')
        
        self.import_ref(reference_file, split_file)
        
        # load annotations into a list
        ls = []
        extensions = ['.mp4', '.mkv', '.webm']
        for vid, anns in self.ref.items():
            for seg, ann in anns['segments'].items():
                # check if video file exists
                for ext in extensions:
                    filename = vid+ext
                    filepath = os.path.join(self.video_root, filename)
                    if os.path.exists(filepath):
                        exists = True
                        ls.append((vid, seg, filepath))
                        break
                if not exists:
                    print(f"Video file not found for {vid}")
                    continue                
        self.ls = ls
        
    def __len__(self):
        return len(self.ls)
        
    def import_ref(self, reference_file=None, split_file=None):

        with open(split_file) as f:
            split_dict = json.load(f)
        split = {}
        # for s in self.data_split:
        s = self.data_split
        split.update({i:i for i in split_dict[s]})

        with open(reference_file) as f:
            ref = json.load(f)['annotations']
        ref = {k:v for k,v in ref.items() if k in split}
        self.ref = ref
        

    def __getitem__(self, idx):
        video_id, segment_id, filepath = self.ls[idx]
            
        ann = self.ref[video_id]['segments'][segment_id]
        s_t, e_t = ann['timestamps']
        
        frames = load_frames(filepath, s_t, e_t, num_frames=10)
        
        return video_id, segment_id, frames, ann
    
def _has_repeated_sublists(process_idx):
    # Convert each sublist to a tuple to make it hashable (needed for set operations)
    seen = set()
    for sublist in process_idx:
        # Tuples allow sublists with same elements in the same order to be recognized as duplicates
        tuple_sublist = tuple(sublist)
        if tuple_sublist in seen:
            return True  # Found a duplicate sublist
        seen.add(tuple_sublist)
    return False

def render_caption(ann):
    
    tokens = ann['tokens']
    
    process_idx = ann['process_idx']
    process_bnd_box = ann['process_bnd_box']
    frame_ind = ann['frame_ind']
    crowds = ann['crowds']
    
    num_boxes_in_ann = len(process_bnd_box)
    
    # If no bounding boxes, skip
    if num_boxes_in_ann == 0:
        raise ValueError('No bounding boxes found in the annotation')

    # Ignore multi-object grounding
    if _has_repeated_sublists(process_idx):
        raise ValueError('Having multiple bounding boxes per token is not supported')
    
    # Ignore plural-objects grounding
    if any(crowds):
        raise ValueError('Plurals/Crowds not supported')

    # Add segmentation tokens to the caption
    seg_token_to_obj = {f"[SEG:{i}]": {
                            'bbox': process_bnd_box[i], 
                            'frame_id': frame_ind[i],
                            'process_idx': process_idx[i],
                            'crowds': crowds[i],
                        } for i in range(num_boxes_in_ann)}
        
    process_idx_to_seg = {i: f"[SEG:{i}]" for i in range(num_boxes_in_ann)}
    
    indices_to_add_seg_token = {}
    for k, process_idx_ in seg_token_to_obj.items():
        for p in process_idx_['process_idx']:
            indices_to_add_seg_token[p] = k
            
    new_tokens = []
    for tok_idx, tok in enumerate(tokens):
        if tok_idx in indices_to_add_seg_token.keys():
            new_tokens.append(f'<p> {tok} </p> {indices_to_add_seg_token[tok_idx]}' )
        else:
            new_tokens.append(tok)
            
    new_caption = (' '.join(new_tokens))
    
    # If not all bounding box annotations are not available in one frame, skip
    frames_ids_of_bboxes = [seg_token_to_obj[f"[SEG:{i}]"]['frame_id'] for i in range(num_boxes_in_ann)]
    if len(set(frames_ids_of_bboxes)) > 1:
        raise ValueError('All bounding boxes must be available in one frame')
    
                    
    return new_caption, seg_token_to_obj

    

dataset = ANetEntitiesDataset(base_video_dataset_dir = '/home/shehan/workspace_grounding_lmm/LISA2/video_dataset', data_split='training')

from tools.sam_hq import SAMHQ
samhq_model = SAMHQ(sam_type="vit_h")


def get_segmentation_mask(video_frames, seg_token_to_obj, save_dir):
    
    # Apply HQ-SAM
    # for seg_token in seg_tokens_set:
    for seg_token, v in seg_token_to_obj.items():
        frame_id = v['frame_id']
        bbox     = v['bbox']
        
        seg_token_x = seg_token.split(':')[1].split(']')[0]
        
        # Load the image
        pil_image = video_frames[frame_id]
        
        pred_mask = samhq_model.predict_sam(pil_image, torch.tensor(bbox))
        pred_mask = pred_mask[0, 0].cpu().numpy()
        
        
        save_mask_path = os.path.join(save_dir, 
                                      f'{str(seg_token_x).zfill(2)}/mask.png')
        save_mask_directory = os.path.dirname(save_mask_path)
        if not os.path.exists(save_mask_directory):
            os.makedirs(save_mask_directory)
        mask_image = Image.fromarray(pred_mask.astype('uint8') * 255)
        mask_image.save(save_mask_path)
        print(f"Saved mask to {save_mask_path}")
        print('---')
        

########

from tqdm import tqdm

extensions = ['.mp4', '.mkv', '.webm']

dir_save_masks = './tmp_masks'
dir_save_objs = './tmp_anns'

if not os.path.exists(dir_save_masks):
    os.makedirs(dir_save_masks)
if not os.path.exists(dir_save_objs):
    os.makedirs(dir_save_objs)
    

##
keys = list(dataset.ref.keys())
total_keys = len(keys)
chunk_size = total_keys // 4
##

# for vid in tqdm(keys[:chunk_size]):
# for vid in tqdm(keys[chunk_size:2*chunk_size]):
# for vid in tqdm(keys[2*chunk_size:3*chunk_size]):
for vid in tqdm(keys[3*chunk_size:4*chunk_size]):

# for vid in tqdm(dataset.ref.keys()):
    for seg, ann in dataset.ref[vid]['segments'].items():
        
        
        # print(vid, seg, ann)
        
        # if this sample doesn't exist in res_data, skip
        if vid not in res_data or seg not in res_data[vid]:
            continue
        
        refined_caption = res_data[vid][seg]['refined_caption']
        
        save_dir_for_masks = os.path.join(dir_save_masks, f'{vid}____{seg}')
        if os.path.exists(save_dir_for_masks):
            print(f"Already processed {vid}____{seg}")
            continue
        
        # check if video file exists
        for ext in extensions:
            filename = vid+ext
            filepath = os.path.join(dataset.video_root, filename)
            if os.path.exists(filepath):
                exists = True
                break
        if not exists:
            print(f"Video file not found for {vid}")
            continue  
        
        ann = dataset.ref[vid]['segments'][seg]
        s_t, e_t = ann['timestamps']
        
        video_frames = load_frames(filepath, s_t, e_t, num_frames=10)
        
        new_caption, seg_token_to_obj = render_caption(ann)
        
        
        get_segmentation_mask(video_frames, seg_token_to_obj, save_dir= save_dir_for_masks)
        
        obj = {
            'video_id': vid,
            'segment_id': seg,
            'new_caption': new_caption,
            'refined_caption': refined_caption,
            'ann': ann,
            'seg_token_to_obj': seg_token_to_obj
        }
        
        # save obj to file
        save_path = os.path.join(dir_save_objs, f'{vid}____{seg}.json')
        with open(save_path, 'w') as f:
            json.dump(obj, f)
    

    