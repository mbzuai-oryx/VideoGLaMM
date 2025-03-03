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


#########################################################################################################

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


#########################################################################################################

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


#########################################################################################################

from openai import OpenAI
import ast

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


_EXAMPLES = """\
Example 1: 

Input :
The woman in the blue turban let go of the stool in her hand and turns to look around.
Subject ID:0
  
Output:
{'caption': '[The woman in the blue turban](0) let go of the stool in her hand and turns to look around.'}

Example 2:

Input:
The man in the jacket walks to the woman in red and stops, takes a golf club, gives it to the woman in red, and pushes her.
Subject ID:0

Output:
{'caption': '[The man in the jacket](0) walks to the woman in red and stops, takes a golf club, gives it to the woman in red, and pushes her.'}

"""

_PROMPT = """\
Your task is to identify the subject noun phrase in given original unannotated video descriptions.

You are requested to generate a new caption annotating the subject in the caption with the corresponding target subject ID.

You may look at the following examples:
{examples}

Now please process the following.

{original_caption}
Subject ID:{subject_id}


In the new caption, the noun phrases should be included within square brackets and subject ID should be included within paranthesis. E.g. [noun phrase](subject ID) .

Please provide the generated caption in JSON format, with a key "caption".
"""

def get_annotated_caption(caption):
    """
    """
    formatted_prompt = _PROMPT.format(
        examples = _EXAMPLES,
        original_caption = caption,
        subject_id = 0,
    )
    
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": formatted_prompt}
        ],
        response_format={ "type": "json_object" }
    )

    response_message = completion.choices[0].message.content
    response_dict = ast.literal_eval(response_message)
    return response_dict



output_dir = 'hcstvg_gcg_captions'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
def process_idx(idx):
    
    np_images, targets, qtype, video_caption, video_level_ann, targets_list_2 = dataset[idx]
    
    video_id = str(video_level_ann['video_id']) 
    video_id = video_id.zfill(6)
    output_path = os.path.join(output_dir, f"{video_id}.json")
    if os.path.exists(output_path):
        print(f"Processed video ID: {video_id}", " Caption already exists at:", output_path)
        return

    print('original: ', video_caption)
    
    caption = get_annotated_caption(video_caption)
    
    print('annotated: ', caption['caption'], '\n')
    
    
    with open(output_path, 'w') as f:
        json.dump(caption, f)

    print(f"Processed video ID: {video_id}", " Caption saved at:", output_path)
    print('-----------------------------------\n')
    return


from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(process_idx, idx) for idx in range(len(dataset))]
    
    for future in tqdm(as_completed(futures), total=len(dataset), desc="Processing ..."):
        future.result()