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
from utils.refer_datasets.new.ytvos import ReferYouTubeVOSDataset
from utils.refer_datasets.new.davis17 import ReferDAVISDataset


from utils.grounding_utils.box_ops import masks_to_boxes
from utils.grounding_utils.box_ops import np_box_iou

from chat import initialize_model_videogptplus, initialize_model_chatunivi, preprocess_vision, IMAGE_TOKEN_INDEX

def remove_small_blobs(binary_mask: np.ndarray, min_size: int = 0):
    """
    Removes from the input mask all the blobs having less than N adjacent pixels.
    We set the small objects to the background label 0.
    """
    if min_size > 0:
        dtype = binary_mask.dtype
        binary_mask = skimage.morphology.remove_small_objects(binary_mask.astype(bool), min_size=min_size)
        binary_mask = binary_mask.astype(dtype)
    return binary_mask


def parse_args():
    parser = argparse.ArgumentParser(description="Eval GCG")
    
    # Model parameters
    parser.add_argument("--llava_version_or_path", type=str, default="/home/shehan/workspace_grounding_lmm/LISA2/checkpoints_hf/ChatUniVi-SAM2-test")
    parser.add_argument("--vis_save_path", type=str, default="./vis_output/eval_mevis")
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
    parser.add_argument("--dataset_name", default="MEVIS|valid", type=str, choices=["MEVIS|valid", "MEVIS|valid_u", 
                                                                                    "ReferYouTubeVOS|valid", "ReferYouTubeVOS|test",
                                                                                    "ReferDAVIS|valid"])
                                                                                    
    
    return parser.parse_args()


if __name__ == "__main__":
    
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

    # Load dataset
    if args.dataset_name.split('|')[0] == "MEVIS":
        video_val_image_set = args.dataset_name.split('|')[-1] # 'valid_u' # 'valid'
        eval_dataset = MeVISBaseDataset(args.video_dataset_dir, image_set=video_val_image_set, num_frames=-1)
    elif args.dataset_name.split('|')[0] == "ReferYouTubeVOS":
        video_val_image_set = args.dataset_name.split('|')[-1]
        eval_dataset = ReferYouTubeVOSDataset(args.video_dataset_dir, split=video_val_image_set)
    elif args.dataset_name.split('|')[0] == "ReferDAVIS":
        video_val_image_set = args.dataset_name.split('|')[-1]
        eval_dataset = ReferDAVISDataset(args.video_dataset_dir, split=video_val_image_set)
    else:
        raise ValueError(f"Invalid dataset name: {args.dataset_name}")


    def clean_caption(text_output):
        text_output_ = text_output.replace("\n", "").replace("  ", " ")
        text_output_ = text_output_.split("ASSISTANT: ")[-1]
        # print('Output caption:', text_output_)
        cleaned_str = re.sub(r'<.*?>', '', text_output_)
        pattern = re.compile(r'<p>(.*?)<\/p>')
        phrases = pattern.findall(text_output_)
        phrases = [p.strip() for p in phrases]
        
        cleaned_str = cleaned_str.replace('[SEG]', '') # Remove the [SEG] token
        cleaned_str = ' '.join(cleaned_str.split()).strip("'") # Strip unnecessary spaces
        cleaned_str = cleaned_str.strip()
        
        return cleaned_str, phrases


    for idx in tqdm(range(len(eval_dataset))):
    # split in 4
    
    # for idx in tqdm(range(0 ,len(eval_dataset)//4)):
    # for idx in tqdm(range(len(eval_dataset)//4, 2*len(eval_dataset)//4)):
    # for idx in tqdm(range(2*len(eval_dataset)//4, 3*len(eval_dataset)//4)):
    # for idx in tqdm(range(3*len(eval_dataset)//4, len(eval_dataset))):
        
        try:
                                        
            np_images, target = eval_dataset[idx]
            pil_images = target["pil_images"]
            gt_caption = target['caption']
            video_path = target['video_path']
            
            # np_images : [T, H, W, C]
            np_images = [np_images] # Add batch dimension   # B x T x (H x W x C)

            ###
            # Prepare inputs
            prompt_text = "What is {phrase} in this video? Please respond with segmentation masks.".format(phrase=gt_caption.lower())
            enc_image, enc_context_image, image_sam, original_size_list, resize_list = preprocess_vision(np_images, type="video", 
                                                                                                enc_preprocessor=enc_preprocessor, 
                                                                                                sam_preprocessor=sam_preprocessor, 
                                                                                                conv_generator=conv_generator,
                                                                                                precision=args.precision)
            input_ids = conv_generator.apply_for_chat(prompt_text, type='video', tokenizer=tokenizer)
            
            #####

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
            # pred_masks: (batch_size x T_sam x [num_seg_tokens_per_sample, H, W])

            
            video_frames_np = np_images[0] # [T, H, W, C]  # numpy array
            video_segments = video_segments_batch[0]
            
            video_name, exp_id = video_path[0], video_path[1]
                
            for t, pred_mask in video_segments.items():
                for obj_id, pred_mask_i in pred_mask.items():
                    pred_mask_i = pred_mask_i > 0
                    
                    mevis_output_save_dir = os.path.join(args.vis_save_path, f"{args.dataset_name.split('|')[0]}____{video_val_image_set}_output", video_name, exp_id)
                    if not os.path.exists(mevis_output_save_dir):
                        os.makedirs(mevis_output_save_dir)
                    
                    
                    if args.dataset_name.split('|')[0] == "ReferYouTubeVOS" or args.dataset_name.split('|')[0] == "ReferDAVIS":
                        output_path = os.path.join(mevis_output_save_dir, f"{target['frame_ids'][t]}.png")
                    else:
                        output_path = os.path.join(mevis_output_save_dir, f"{t:05d}.png")
                    
                    mask_array = pred_mask_i
                    mask_array = (mask_array * 255).astype(np.uint8)
                    mask_image = Image.fromarray(mask_array)
                    mask_image.save(output_path)
                    
                    print(f"Saved {output_path}")
                    
                    break # NOTE: Only save the first mask for each frame
                                    
        

        except Exception as e:
            print("Error at idx:", idx)
            print("\033[91m\t\t\t", e, "\033[0m")
            continue