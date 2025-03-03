import cv2
import numpy as np
import torch
import skimage
import json
import re
from tqdm import tqdm
import argparse
from collections import defaultdict
import os
from PIL import Image

from utils.refer_datasets.mevis import MeVISBaseDataset


from utils.grounding_utils.box_ops import masks_to_boxes
from utils.grounding_utils.box_ops import np_box_iou

from chat import initialize_model_videogptplus, initialize_model_chatunivi, preprocess_vision, IMAGE_TOKEN_INDEX

# args = argparse.Namespace(
#     llava_version_or_path="./checkpoints_hf/VideoGPTPlus-Phi3-SAM2-8frame-tunevlproj-epoch29",
#     vis_save_path="./vis_output/eval_davis17",
#     precision="fp16",
#     model_max_length=2048,
#     vision_tower="openai/clip-vit-large-patch14",
#     local_rank=0,
#     load_in_8bit=False,
#     load_in_4bit=False,
#     use_mm_start_end=True,
#     conv_type="llava_v1",
#     use_sam2_video_branch=True,
#     base_model_type="vgpt|phi3",
#     video_dataset_dir='./video_dataset',
#     dataset_name="ReferDAVIS|valid"
# )


def parse_args():
    parser = argparse.ArgumentParser(description="Eval GCG")
    
    # Model parameters
    parser.add_argument("--llava_version_or_path", type=str, default="/home/shehan/workspace_grounding_lmm/LISA2/checkpoints_hf/ChatUniVi-SAM2-test")
    parser.add_argument("--vis_save_path", type=str, default="./vis_output/eval_davis17")
    parser.add_argument("--precision", type=str, default="fp16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--model_max_length", type=int, default=2048)
    parser.add_argument("--vision_tower", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--conv_type", type=str, default="llava_v1")
    parser.add_argument("--use_sam2_video_branch", action="store_true")
    parser.add_argument("--base_model_type", type=str, default="vgpt|phi3", choices=["vgpt|phi3","vgpt|llama3_1", "chatunivi"])
    
    # Dataset parameters
    parser.add_argument("--video_dataset_dir", default='./video_dataset', type=str)
    parser.add_argument("--dataset_name", default="ReferDAVIS|valid", type=str, choices=["ReferDAVIS|valid"])
    
    return parser.parse_args()

args = parse_args()

# Load model, tokenizer, and image processor, conv_generator
if args.base_model_type.split('|')[0] == "vgpt":
    model, tokenizer, enc_preprocessor, conv_generator, sam_preprocessor = initialize_model_videogptplus(
        model_base=args.llava_version_or_path,
        precision=args.precision,
        local_rank=args.local_rank,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        use_sam2_video_branch=args.use_sam2_video_branch,
        base_llm_type=args.base_model_type.split('|')[1]
    )
elif args.base_model_type.split('|')[0] == "chatunivi":
    model, tokenizer, enc_preprocessor, conv_generator, sam_preprocessor = initialize_model_chatunivi(
        llava_version_or_path=args.llava_version_or_path,
        precision=args.precision,
        model_max_length=args.model_max_length,
        vision_tower=args.vision_tower,
        local_rank=args.local_rank,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        use_mm_start_end=args.use_mm_start_end,
        use_sam2_video_branch=args.use_sam2_video_branch
    )
else:
    raise ValueError(f"Invalid base model type: {args.base_model_type}")


import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image, ImageDraw
import math
import torch.nn.functional as F
import json


from tqdm import tqdm


split = 'valid'
# davis_path= './video_dataset/processed/refer_davis/2017' #'data/ref-davis'
davis_path = os.path.join(args.video_dataset_dir, "processed/refer_davis/2017")
output_dir = args.vis_save_path


# Get palette
palette_img = os.path.join(davis_path, "valid/Annotations/blackswan/00000.png")
palette = Image.open(palette_img).getpalette()

# def main():

# Load data
root = Path(davis_path)  # data/ref-davis
img_folder = os.path.join(root, split, "JPEGImages")
meta_file = os.path.join(root, "meta_expressions", split, "meta_expressions.json")
with open(meta_file, "r") as f:
    data = json.load(f)["videos"]
video_list = list(data.keys())


# Save path
save_path_prefix = os.path.join(output_dir, split)
if not os.path.exists(save_path_prefix):
    os.makedirs(save_path_prefix)
save_visualize_path_prefix = os.path.join(output_dir, split + '_images')



# Process each video sequentially
for video in tqdm(video_list, desc="Processing videos"):
    # Run model on each video
    # sub_processor( data, save_path_prefix, save_visualize_path_prefix, img_folder, video)
    
    # Processing each expression in the video
    expressions = data[video]["expressions"]
    expression_list = list(expressions.keys())
    num_expressions = len(expression_list)
    video_len = len(data[video]["frames"])

    # Read all the annotation metadata
    metas = []
    for i in range(num_expressions):
        meta = {}
        meta["video"] = video
        meta["exp"] = expressions[expression_list[i]]["exp"]
        meta["exp_id"] = expression_list[i]  # start from 0
        meta["frames"] = data[video]["frames"]
        metas.append(meta)
    
    # Process annotations for each annotator
    num_obj = num_expressions // 4
    for anno_id in range(4):  # 4 annotators
        save_name_expression = {0: 'Davis17_annot1', 1:'Davis17_annot1_full_video', 2:'Davis17_annot2', 3:'Davis17_annot2_full_video'}
        anno_logits = []
        anno_masks = []

        # For each object in the video
        for obj_id in range(num_obj):
            i = obj_id * 4 + anno_id
            video_name = metas[i]["video"]
            exp = metas[i]["exp"]
            exp_id = metas[i]["exp_id"]
            frames = metas[i]["frames"]
            video_len = len(frames)
            
            all_pred_logits = []
            all_pred_masks = []


            # For each clip in the video
            for clip_id in range(0, video_len, 64):
                clip_frames_ids = range(clip_id, min(clip_id + 64, video_len))
                
                imgs = [(Image.open(os.path.join(img_folder, video_name, frames[t] + ".jpg")).convert('RGB')) for t in clip_frames_ids]

                
                np_images = [np.array(img) for img in imgs]
                np_images = [np_images] # Add batch dimension   # B x T x (H x W x C)
                
                gt_caption = exp
                
                # Prepare inputs
                prompt_text = "What is {phrase} in this video? Please respond with segmentation masks.".format(phrase=gt_caption.lower())
                enc_image, enc_context_image, image_sam, original_size_list, resize_list = preprocess_vision(np_images, type="video", 
                                                                                                    enc_preprocessor=enc_preprocessor, 
                                                                                                    sam_preprocessor=sam_preprocessor, 
                                                                                                    conv_generator=conv_generator,
                                                                                                    precision=args.precision)
                input_ids = conv_generator.apply_for_chat(prompt_text, type='video', tokenizer=tokenizer)
                
                with torch.cuda.amp.autocast():
                    output_ids_batch, video_segments_batch = model.inference(
                        images=enc_image,
                        context_images=enc_context_image,
                        images_for_sam=image_sam,
                        input_ids=input_ids,
                        resize_list=resize_list,
                        original_size_list=original_size_list,
                        max_new_tokens=512,
                        use_sam2_video_branch=args.use_sam2_video_branch,
                    )
                    
                assert len(output_ids_batch) == 1 and len(video_segments_batch) == 1, "Batch size must be 1"
        
                output_ids = output_ids_batch[0]
                
                output_ids = output_ids[output_ids != IMAGE_TOKEN_INDEX]

                text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
                print("text_output : ", text_output)
                text_output = text_output.replace("\n", "").replace("  ", " ")
                text_output = text_output.split("ASSISTANT: ")[-1]
                
                
                
                video_segments = video_segments_batch[0]
                # video_segments is a dict of dict
                # keys of outer dict: t, 
                # keys of inner dict: 0 always
                # Values of inner dict: mask
                
                # pred_masks into [T, H, W]
                h,w = video_segments[0][0].shape[-2:]
                pred_masks = torch.zeros(len(video_segments), h, w)
                for t in range(len(video_segments)):
                    pred_masks[t] = torch.tensor(video_segments[t][0])
                    
                                
                # store the clip results
                all_pred_masks.append(pred_masks)
                
            all_pred_masks = torch.cat(all_pred_masks, dim=0)   # (video_len, h, w) 
            anno_masks.append(all_pred_masks)
            
        # handle a complete image (all objects of a annotator)
        anno_masks = torch.stack(anno_masks)   # [num_obj, video_len, h, w]
        t, h, w = anno_masks.shape[-3:]
        anno_masks[anno_masks < 0.5] = 0.0
        background = 0.1 * torch.ones(1, t, h, w).to(anno_masks.device)
        anno_masks = torch.cat([background, anno_masks], dim=0) # [num_obj+1, video_len, h, w]
        out_masks = torch.argmax(anno_masks, dim=0) # int, the value indicate which object, [video_len, h, w]

        out_masks = out_masks.detach().cpu().numpy().astype(np.uint8) # [video_len, h, w]

        # save results
        anno_save_path = os.path.join(save_path_prefix, f"{save_name_expression[anno_id]}", video)
        if not os.path.exists(anno_save_path):
            os.makedirs(anno_save_path)
        for f in range(out_masks.shape[0]):
            img_E = Image.fromarray(out_masks[f])
            img_E.putpalette(palette)
            img_E.save(os.path.join(anno_save_path, '{:05d}.png'.format(f)))
            # print(f"Save {os.path.join(anno_save_path, '{:05d}.png'.format(f))}")
            

