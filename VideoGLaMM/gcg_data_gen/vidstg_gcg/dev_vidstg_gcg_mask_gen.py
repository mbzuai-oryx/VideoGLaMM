import json
import os
import ffmpeg
import numpy as np
from tqdm import tqdm

import torch
from gcg_data_gen.vidstg_gcg.sam21.sam2.build_sam import build_sam2, build_sam2_video_predictor
# from sam21.sam2.build_sam import build_sam2, build_sam2_video_predictor
sam2_ckpt = '/home/shehan/workspace_grounding_lmm/LISA2/gcg_data_gen/vidstg_gcg/sam21/checkpoints/sam2.1_hiera_large.pt'

device = 'cuda'
precision = 'bf16'
predictor = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_l.yaml", sam2_ckpt, device=device)

from utils.sam_transforms import SAM_v2_Preprocess
sam_preprocessor = SAM_v2_Preprocess()

import cv2
def write_masks(base_save_dir, video_id, video_segments, video_frames_np, used_frame_ids):
    ''' Write masks to disk 
    Args:
    - video_segments: dictionary with keys being frame indices, and values being dictionaries with keys being segment indices
    - video_frames_np: numpy array of video frames # [T, H, W, C]  # numpy array
    '''
    
    save_dir = os.path.join(base_save_dir, video_id)
    os.makedirs(save_dir, exist_ok=True)
    
    # video_segments is a dictionary with keys being frame indices
    # video_segments[t] is a dictionary with keys being segment indices
    
    # save directory format
    # - base_dir
    #   - video_id (save_dir)
    #     - frames
    #         - frame_id
    #     - masks
    #         - target_id
    #             - frame_id
    
    os.makedirs(os.path.join(save_dir, 'frames'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'masks'), exist_ok=True)
    
    for t, pred_mask in video_segments.items():
        
        frame_id = used_frame_ids[t]
        frame_id = str(frame_id).zfill(6)
        
        # save frame
        save_path = "{}/frames/{}.png".format(save_dir, frame_id)
        cv2.imwrite(save_path, cv2.cvtColor(video_frames_np[t], cv2.COLOR_RGB2BGR))
        print("{} has been saved.".format(save_path))
        
        for obj_id, pred_mask_i in pred_mask.items():
            pred_mask_i = pred_mask_i > 0

            # save_path = "{}/mask_{}_{}.jpg".format(save_dir, t, obj_id)
            # cv2.imwrite(save_path, pred_mask_i * 100)
            # print("{} has been saved.".format(save_path))
            
            obj_id_str = str(obj_id).zfill(3)
            
            # save binary masks as png
            save_path = "{}/masks/{}/{}.png".format(save_dir, obj_id_str, frame_id)
            os.makedirs(os.path.join(save_dir, 'masks', obj_id_str), exist_ok=True)
            cv2.imwrite(save_path, pred_mask_i.astype(np.uint8) * 255)
            print("{} has been saved.".format(save_path))

            # save masked images
            save_path = "{}/masked_img/{}/{}.png".format(save_dir, obj_id_str, frame_id)
            os.makedirs(os.path.join(save_dir, 'masked_img', obj_id_str), exist_ok=True)
            save_img = video_frames_np[t].copy()
            save_img[pred_mask_i] = (video_frames_np[t] * 0.5 + pred_mask_i[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5)[pred_mask_i]
            (video_frames_np[t] * 0.5 + pred_mask_i[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5)[pred_mask_i]
            save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, save_img)
            print("{} has been saved.".format(save_path))


max_num_frames_to_sample = 20

# Function to load JSON data from a file
def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Function to load annotations from the dataset directory
def load_annotations(vidstg_ann_dir, split='train'):
    # Define file paths for the current split
    file_ids_path = os.path.join(vidstg_ann_dir, f"{split}_files.json")
    annotations_path = os.path.join(vidstg_ann_dir, f"{split}_annotations.json")
    
    # Load the file ids and annotations
    file_ids = load_json(file_ids_path)
    annotations = load_json(annotations_path)
    
    # Create a dictionary with video ID as keys for easier access
    annotations_dict = {annotation['vid']: annotation for annotation in annotations}
    
    return file_ids, annotations_dict

# Main function to load all splits
def load_vidstg_data(vidstg_ann_dir):
    splits = ['train', 'val', 'test']
    data = {}
    
    for split in splits:
        file_ids, annotations = load_annotations(vidstg_ann_dir, split)
        data[split] = {
            'file_ids': file_ids,
            'annotations': annotations
        }
    
    return data

vidor_ann_dir = './video_dataset/vidstg/vidor_annotations'
vidor_video_dir = os.path.join('./video_dataset/', 'vidstg', 'video')
vidstg_ann_dir = './video_dataset/vidstg/vidstg_annotations'
vidstg_data = load_vidstg_data(vidstg_ann_dir)

image_set = 'test' # 'train', 'val', 'test' #TODO
if image_set == 'train':
    vidor = json.load(open(os.path.join(vidstg_ann_dir, "vidor_training.json"), "r")) # train
    chosen_file_ids = vidstg_data['train']['file_ids']
    chosen_annotations = vidstg_data['train']['annotations']
elif image_set == 'val':
    vidor = json.load(open(os.path.join(vidstg_ann_dir, "vidor_training.json"), "r"))
    chosen_file_ids = vidstg_data['val']['file_ids']
    chosen_annotations = vidstg_data['val']['annotations']
elif image_set == 'test':
    vidor = json.load(open(os.path.join(vidstg_ann_dir, "vidor_validation.json"), "r"))
    chosen_file_ids = vidstg_data['test']['file_ids']
    chosen_annotations = vidstg_data['test']['annotations']
    
# VidOR contains 7,000, 835 and 2,165 videos for training, validation and testing, respectively. 
# Since box annota-tions of testing videos are unavailable yet, VidSTG omit testing videos, split 10% training videos as our validation data and regard original validation videos as the testing data


# divide into 4
# chosen_file_ids = chosen_file_ids[:len(chosen_file_ids)//4] 
# chosen_file_ids = chosen_file_ids[len(chosen_file_ids)//4:2*len(chosen_file_ids)//4] 
# chosen_file_ids = chosen_file_ids[2*len(chosen_file_ids)//4:3*len(chosen_file_ids)//4]
# chosen_file_ids = chosen_file_ids[3*len(chosen_file_ids)//4:] 

# divide into 2 #TODO
# chosen_file_ids = chosen_file_ids[:len(chosen_file_ids)//2]
chosen_file_ids = chosen_file_ids[len(chosen_file_ids)//2:]

for video_id in tqdm(chosen_file_ids):
    try:
        ann = chosen_annotations[video_id]
    except:
        print('video_id:', video_id, 'not found in chosen_annotations')
        continue
    
    used_target_ids = (ann["used_relation"]["subject_tid"], ann["used_relation"]["object_tid"])
    
    annot_vidor = vidor[ann["vid"]]
    
    
    video_path = annot_vidor['video_path']
    frame_count = annot_vidor['frame_count']
    width = annot_vidor['width']
    height = annot_vidor['height']

    start_frame = ann['temporal_gt']['begin_fid']
    end_frame = ann['temporal_gt']['end_fid']

    duration = (end_frame - start_frame + 1) / annot_vidor['fps']
    ss = start_frame / annot_vidor['fps']

    # print('video_path:', video_path) # video_path: 0054/2463204137.mp4
    video_path = os.path.join(vidor_video_dir, video_path)

    print('video_path:', video_path)

    # Extract frames using ffmpeg
    out, _ = (
        ffmpeg
        .input(video_path, ss=ss, t=duration)
        .filter('fps', fps=(end_frame - start_frame + 1) / duration)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run(capture_stdout=True, quiet=True)
    )
    frames = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])

    # Extract bounding boxes for each frame
    trajectories = annot_vidor['trajectories']

    boxes_dict = {}
    used_frame_ids = list(range(start_frame, end_frame + 1))
    for target_id in used_target_ids:
        target_id_ = str(target_id)
        if target_id_ in trajectories:
            object_trajectory = trajectories[target_id_]
            
            boxes = []
            for frame_id in range(start_frame, end_frame + 1):
                frame_boxes = object_trajectory.get(str(frame_id), None)
                if frame_boxes:
                    bbox = frame_boxes['bbox']
                    boxes.append(bbox)
                else:
                    boxes.append(None)
        else:
            boxes = [None] * (end_frame - start_frame + 1)
            
        # assert len(boxes) == len(used_frame_ids), f"len(boxes): {len(boxes)}, len(used_frame_ids): {len(used_frame_ids)}"
        # assert len(boxes) == len(frames), f"len(boxes): {len(boxes)}, len(frames): {len(frames)}"
            
        boxes_dict[target_id] = boxes
        
    frames = frames[:len(used_frame_ids)] # remove the last frame to fix bug
    
    if len(frames)>max_num_frames_to_sample: #TODO
        indices = np.linspace(0, len(frames)-1, max_num_frames_to_sample).astype(int)
        frames = [frames[i] for i in indices]
        boxes_dict = {k: [v[i] for i in indices] for k, v in boxes_dict.items()}
        used_frame_ids_ = [used_frame_ids[i] for i in indices]
        used_frame_ids = used_frame_ids_
        
        
    ### Preprocess image for SAM
    np_images_ = frames # T x (H x W x C)

    original_size_list = np_images_[0].shape[:2] # (H, W)
    preprocessed_for_sam_and_resize_shapes = [sam_preprocessor.preprocess(image) for image in np_images_]
    image_sam = [x[0] for x in preprocessed_for_sam_and_resize_shapes]
    resize_shape = preprocessed_for_sam_and_resize_shapes[0][1]
    resize_list = [resize_shape]
    image_sam = torch.stack(image_sam, dim=0).cuda() # (T x 3 x 1024 x 1024)

    image_sam = image_sam.bfloat16() if precision == "bf16" else (image_sam.half() if precision == "fp16" else image_sam.float())
    image_sam = image_sam.cuda(non_blocking=True) # (T x 3 x 1024 x 1024)

    video_height, video_width = original_size_list
    
    #
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        inference_state = predictor.init_state_from_tensor(image_sam, video_height, video_width)
        predictor.reset_state(inference_state)
        
        # print('boxes_dict:', boxes_dict)
        # print('boxes_dict.keys():', boxes_dict.keys())
        # print('boxes_dict[0]', len(boxes_dict[0]))
        # print('boxes_dict[1]', len(boxes_dict[1]))
        
        # print('len(image_sam):', len(image_sam))
        # print('len(frames):', len(frames))
        # print('len(used_frame_ids):', len(used_frame_ids))
                
        for ann_frame_idx in range(0, len(image_sam)): # for each frame in the video
            for ann_obj_id in used_target_ids: # for each object in the video
                if boxes_dict[ann_obj_id][ann_frame_idx] is None:
                    continue
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=ann_obj_id,
                    box=boxes_dict[ann_obj_id][ann_frame_idx] # (x_min, y_min, x_max, y_max)
                )
            
        # run propagation throughout the video and collect the results in a dict
        video_segments = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                # out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()[0] # select only one mask per object
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        
    video_frames_np = frames # [T, H, W, C]  # numpy array
    # base_save_dir = f'./video_dataset/vidstg_gcg_generated/{image_set}' #TODO
    base_save_dir = f'/media/shehan/Extreme SSD/SALMANPC/video_dataset/vidstg_gcg_generated/{image_set}' #TODO

    write_masks(base_save_dir, video_id, video_segments, video_frames_np, used_frame_ids)
    
    