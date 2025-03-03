import json
import os

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
                 data_split=None # ['hidden_test', 'training', 'validation', 'testing']
                 ):
        
        self.data_split = data_split
        
        self.base_video_dataset_dir = './video_dataset'
        activitynet_entities_json_dir = os.path.join(self.base_video_dataset_dir, 'activitynet_entities', 'data')
        
        if self.data_split == 'training' or self.data_split == 'validation':
            reference_file= os.path.join(activitynet_entities_json_dir, 'anet_entities_cleaned_class_thresh50_trainval.json')
        elif self.data_split == 'testing':
            reference_file= os.path.join(activitynet_entities_json_dir, 'anet_entities_cleaned_class_thresh50_test_skeleton.json')
        elif self.data_split == 'hidden_test':
            reference_file= os.path.join(activitynet_entities_json_dir, 'anet_entities_cleaned_class_thresh50_hidden_test_skeleton.json')
        
        split_file='./video_dataset/activitynet_entities/data/split_ids_anet_entities.json'
        
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
    

dataset = ANetEntitiesDataset(data_split='training')

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


from tqdm import tqdm

output_save_dir = './tmp_anet'

# for idx in tqdm(range(0, len(dataset))):

# for idx in tqdm(range(0, len(dataset)//4)):
# for idx in tqdm(range( len(dataset)//4, 2*len(dataset)//4)):
# for idx in tqdm(range( 2*len(dataset)//4, 3*len(dataset)//4)):
for idx in tqdm(range( 3*len(dataset)//4, len(dataset))):
    
    try:
        video_id, segment_id, video_frames, ann = dataset[idx]
        print(f"Processing {video_id} - {segment_id}")
        
        new_caption, seg_token_to_obj = render_caption(ann)
        
        print('---')
        
        obj = {
            'video_id': video_id,
            'segment_id': segment_id,
            'new_caption': new_caption,
            'ann': ann,
            'seg_token_to_obj': seg_token_to_obj
        }
        
        # Create directory if it doesn't exist
        directory = os.path.join(output_save_dir, f"{video_id}/{segment_id}")
        if not os.path.exists(directory):
            os.makedirs(directory)
        # Save the frames
        for frame_idx, frame in enumerate(video_frames):
            frame.save(os.path.join(directory, f"{str(frame_idx).zfill(2)}.jpg") )
        
        # Save obj as JSON file
        json_file = os.path.join(directory, 'obj.json')
        with open(json_file, 'w') as f:
            json.dump(obj, f)
        
    except Exception as e:
        print(f"Error processing {video_id} - {segment_id} ::: \033[91m{e}\033[0m")
        continue


