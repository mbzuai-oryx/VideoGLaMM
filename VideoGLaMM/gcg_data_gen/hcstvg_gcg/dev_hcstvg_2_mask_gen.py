# Adapted from https://github.com/hassony2/torch_videovision
from PIL import Image
import numbers
import torch
import cv2
import numpy as np
import PIL
from PIL import Image

# Adapted from
# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# from datasets.torch_videovision import ClipToTensor, normalize, resize_clip, crop_clip
# from util.box_ops import box_xyxy_to_cxcywh

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

import torch
import random
import numpy as np
import copy
import PIL
# from util.misc import interpolate

def interpolate(
    input, size=None, scale_factor=None, mode="nearest", align_corners=None
):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    Equivalent to nn.functional.interpolate, but with support for empty channel sizes.
    """
    if input.numel() > 0:
        return torch.nn.functional.interpolate(
            input, size, scale_factor, mode, align_corners
        )

    assert (
        input.shape[0] != 0 or input.shape[1] != 0
    ), "At least one of the two first dimensions must be non zero"

    if input.shape[1] == 0:
        # Pytorch doesn't support null dimension on the channel dimension, so we transpose to fake a null batch dim
        return torch.nn.functional.interpolate(
            input.transpose(0, 1), size, scale_factor, mode, align_corners
        ).transpose(0, 1)

    # empty batch dimension is now supported in pytorch
    return torch.nn.functional.interpolate(
        input, size, scale_factor, mode, align_corners
    )


def prepare(w, h, anno):
    """
    :param w: pixel width of the frame
    :param h: pixel height of the frame
    :param anno: dictionary with key bbox
    :return: dictionary with preprocessed keys tensors boxes and orig_size
    """
    boxes = [obj["bbox"] for obj in anno]
    # guard against no boxes via resizing
    boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    boxes[:, 2:] += boxes[:, :2]
    boxes[:, 0::2].clamp_(min=0, max=w)
    boxes[:, 1::2].clamp_(min=0, max=h)

    keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
    boxes = boxes[keep]

    target = {}
    target["boxes"] = boxes
    target["orig_size"] = torch.as_tensor([int(h), int(w)])

    return target

####################################################################################################
import cv2
def write_masks(base_save_dir, video_id, video_segments, video_frames_np):
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
        
        frame_id = t
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

####################################################################################################

import os
import json
from torch.utils.data import Dataset
from pathlib import Path
import time
import ffmpeg
import numpy as np
import random
from PIL import Image
import copy

class HCSTVG_Dataset(Dataset):
    def __init__(
        self,
        vid_folder,
        ann_file,
        image_set = 'test',
        take_only_temp_loc_frames = False,
        video_max_len=100,
        required_fps=5,
    ):
        """
        :param vid_folder: path to the folder containing a folder "video"
        :param ann_file: path to the json annotation file
        :param transforms: video data transforms to be applied on the videos and boxes
        :param is_train: whether training or not
        :param video_max_len: maximum number of frames to be extracted from a video
        :param video_max_len_train: maximum number of frames to be extracted from a video at training time
        :param fps: number of frames per second
        :param tmp_loc: whether to use temporal localization annotations
        """
        self.vid_folder = vid_folder
        print("loading annotations into memory...")
        tic = time.time()
        self.annotations = json.load(open(ann_file, "r"))
        print("Done (t={:0.2f}s)".format(time.time() - tic))
        self.video_max_len = video_max_len
        self.tmp_loc = not take_only_temp_loc_frames
        
        self.vid2imgids = {}
        for i_vid, video in enumerate(self.annotations):
            video_num_images = video["frame_count"]
            video_fps = video_num_images / 20  # duration of videos in HC-STVG is 20s
            sampling_rate = required_fps / video_fps
            assert sampling_rate <= 1  # downsampling at fps
            start_frame = 0 if self.tmp_loc else video["tube_start_frame"]
            end_frame = (
                video_num_images - 1 if self.tmp_loc else video["tube_end_frame"]
            )
            frame_ids = [start_frame]
            for frame_id in range(start_frame, end_frame):
                if int(frame_ids[-1] * sampling_rate) < int(frame_id * sampling_rate):
                    frame_ids.append(frame_id)

            if len(frame_ids) > video_max_len:  # subsample at video_max_len
                frame_ids = [
                    frame_ids[(j * len(frame_ids)) // video_max_len]
                    for j in range(video_max_len)
                ]


            inter_frames = []
            for frame_id in frame_ids:
                if video["tube_start_frame"] <= frame_id < video["tube_end_frame"]:
                    inter_frames.append(frame_id)
            
            self.vid2imgids[video["video_id"]] = [frame_ids, inter_frames]

    def __len__(self) -> int:
        return len(self.annotations)
    
    def get_caption(self, idx):
        video = self.annotations[idx]
        video_caption = video["caption"]
        return video_caption

    def __getitem__(self, idx):
        """
        :param idx: int
        :return:
        images: a CTHW video tensor
        targets: list of frame-level target, one per frame, dictionary with keys image_id, boxes, orig_sizes
        tmp_target: video-level target, dictionary with keys video_id, inter_idx, frames_id, caption
        """
        video = self.annotations[idx]
        video_caption = video["caption"]
        video_id = video["video_id"]
        video_original_id = video["original_video_id"]
        trajectory = video["trajectory"]
        frame_ids, inter_frames = self.vid2imgids[video_id]

        video_num_images = video["frame_count"]
        video_fps = video_num_images / 20
        
        start_frame = 0 if self.tmp_loc else video["tube_start_frame"]
        end_frame   = video_num_images - 1 if self.tmp_loc else video["tube_end_frame"]
            
        # ffmpeg decoding
        # vid_path = os.path.join(self.vid_folder, "video", video["video_path"])
        vid_path = os.path.join(self.vid_folder, video["video_path"])
        # ss = 0
        # t = 20
        ss = start_frame / video_fps
        t = (end_frame - start_frame) / video_fps
        # print('vid_path: ', vid_path)
        cmd = ffmpeg.input(vid_path, ss=ss, t=t).filter("fps", fps=len(frame_ids) / t)
        out, _ = cmd.output("pipe:", format="rawvideo", pix_fmt="rgb24").run(
            capture_stdout=True, quiet=True
        )
        w = video["width"]
        h = video["height"]
        images_list = np.frombuffer(out, np.uint8).reshape([-1, h, w, 3])
        assert len(images_list) == len(frame_ids)

        # prepare frame-level targets
        targets_list = []
        targets_list_2 = []
        inter_idx = []  # list of indexes of the frames in the annotated moment
        for i_img, img_id in enumerate(frame_ids):
            if img_id in inter_frames:
                bbox = trajectory[img_id - video["tube_start_frame"]]  # dictionary with bbox [left, top, width, height] key
                # bbox = trajectory[i_img] # dictionary with bbox [left, top, width, height] key #NOTE
                anns = {"bbox": bbox}
                anns = [anns]

                anns2 = {"bbox": [bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]]}
                anns2 = [anns2]
                inter_idx.append(i_img)
            else:
                anns = []
                anns2 = []
            target = prepare(w, h, anns)
            target["image_id"] = f"{video_id}_{img_id}"
            targets_list.append(target)
            targets_list_2.append(anns2)

        images, targets = images_list, targets_list
        
        #######
        img2box = {} 
        for frame_id in frame_ids:
            if video["tube_start_frame"] <= frame_id < video["tube_end_frame"]:
                x1, y1, w, h = video["trajectory"][frame_id - video["tube_start_frame"]] 
                x2 = x1 + w
                y2 = y1 + h
                img2box[frame_id] = [[x1, y1, x2, y2]]
        #######


        qtype = 'declarative'
        
        # video level annotations
        video_level_ann = {
            "video_id": video_id,
            "qtype": qtype,
            "frame_ids": frame_ids,
            "inter_frames" : inter_frames,
            "inter_idx": inter_idx,
            "caption": video_caption,
            "img2box":img2box,
        }
        inter_idx_to_inter_frames_map = {}
        for idx, orig_frame_id in zip(video_level_ann['inter_idx'], sorted(list(video_level_ann['inter_frames']))):
            inter_idx_to_inter_frames_map[idx]=orig_frame_id
        video_level_ann["inter_idx_to_inter_frames_map"]=inter_idx_to_inter_frames_map

        return images, targets, qtype, video_caption, video_level_ann, targets_list_2


video_dir = "/home/shehan/workspace_grounding_lmm/LISA2/video_dataset/hcstvg/Video"
ann_dir = "/home/shehan/workspace_grounding_lmm/LISA2/video_dataset/hcstvg/anno_v2"

dataset = HCSTVG_Dataset(
    vid_folder=video_dir,
    ann_file=os.path.join(ann_dir, "train_v2_proc.json"),
    image_set='train',
    take_only_temp_loc_frames = True,
    video_max_len=40,
    required_fps=5,
)

####################################################################################################
import torch
# from model.segment_anything_2.sam2.build_sam import build_sam2, build_sam2_video_predictor
from gcg_data_gen.vidstg_gcg.sam21.sam2.build_sam import build_sam2, build_sam2_video_predictor
# sam2_ckpt = "/home/shehan/workspace_grounding_lmm/segment-anything-2/checkpoints/sam2_hiera_large.pt"
sam2_ckpt = '/home/shehan/workspace_grounding_lmm/LISA2/gcg_data_gen/vidstg_gcg/sam21/checkpoints/sam2.1_hiera_large.pt'

device = 'cuda'
precision = 'bf16'

predictor = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_l.yaml", sam2_ckpt, device=device)

from utils.sam_transforms import SAM_v2_Preprocess
sam_preprocessor = SAM_v2_Preprocess()


base_save_dir = './video_dataset/hcstvg_gcg/train' #TODO


####################################################################################################
idxs = range(len(dataset))

# divide into 4
# idxs = idxs[:len(idxs)//4]
# idxs = idxs[len(idxs)//4:2*len(idxs)//4]
# idxs = idxs[2*len(idxs)//4:3*len(idxs)//4]
idxs = idxs[3*len(idxs)//4:]

from tqdm import tqdm

# for idx in range(len(dataset)):
for idx in tqdm(idxs):
    
    np_images, targets, qtype, video_caption, video_level_ann, targets_list_2 = dataset[idx]
    boxes_dict = {}
    boxes_dict[0] = [targets_list_2[i][0]['bbox'] if targets_list_2[i] != [] else None for i in range(len(targets_list_2)) ]
    
    
    #
    # './video_dataset/hcstvg_gcg/train/009320'
    video_id = str(video_level_ann['video_id']) 
    video_id = video_id.zfill(6)
    if os.path.exists(os.path.join(base_save_dir, video_id)):
        print(f"Skipping {video_id}. Already exists.")
        continue
    
    ### Preprocess image for SAM
    np_images_ = np_images # T x (H x W x C)

    original_size_list = np_images_[0].shape[:2] # (H, W)
    preprocessed_for_sam_and_resize_shapes = [sam_preprocessor.preprocess(image) for image in np_images_]
    image_sam = [x[0] for x in preprocessed_for_sam_and_resize_shapes]
    resize_shape = preprocessed_for_sam_and_resize_shapes[0][1]
    resize_list = [resize_shape]
    image_sam = torch.stack(image_sam, dim=0).cuda() # (T x 3 x 1024 x 1024)

    image_sam = image_sam.bfloat16() if precision == "bf16" else (image_sam.half() if precision == "fp16" else image_sam.float())
    image_sam = image_sam.cuda(non_blocking=True) # (T x 3 x 1024 x 1024)


    video_height, video_width = original_size_list
    
    
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        inference_state = predictor.init_state_from_tensor(image_sam, video_height, video_width)
        predictor.reset_state(inference_state)
        
        
        for ann_frame_idx in range(0, len(image_sam)): # for each frame in the video
            # for ann_obj_id in used_target_ids: # for each object in the video
            ann_obj_id = 0
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
            
        for t in range(len(np_images)):
            if not t in video_segments:
                # add empty masks for the frames where no object is present
                video_segments[t] = {}
                ann_obj_id = 0
                video_segments[t][ann_obj_id] = np.zeros((np_images[t].shape[0], np_images[t].shape[1])).astype(np.bool)
    
        


        video_id = str(video_level_ann['video_id']) 
        video_id = video_id.zfill(6)

        write_masks(base_save_dir, video_id, video_segments, np_images)