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
    parser.add_argument("--vis_save_path", type=str, default="./vis_output/eval_anet_entities")
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
    parser.add_argument("--dataset_name", default="anet_entities", type=str)
    
    return parser.parse_args()


from decord import VideoReader, cpu
from PIL import Image
import numpy as np

def load_frames(video_path, s_time, e_time, num_frames):
    # Initialize video reader
    vr = VideoReader(video_path, ctx=cpu(0))
    
    # Get video properties
    fps = vr.get_avg_fps()
    total_frames = len(vr)
    
    # Calculate start and end frames based on times and FPS
    start_frame = int(s_time * fps)
    end_frame = int(e_time * fps)
    
    # Ensure the start and end frames are within bounds
    start_frame = max(0, start_frame)
    end_frame = min(total_frames - 1, end_frame)  # Adjust end_frame to be within bounds
    
    # Calculate the indices of the frames to extract
    frame_indices = np.linspace(start_frame, end_frame, num_frames, dtype=int)
    
    # print(f"Extracting frames {frame_indices} from {start_frame} to {end_frame}")
    
    # Load frames as PIL images
    frames = [Image.fromarray(vr[i].asnumpy()) for i in frame_indices]
    
    return frames

import os
import json
import torch

class ANetEntitiesValDataset(torch.utils.data.DataLoader):
    def __init__(self, base_video_dataset_dir = './video_dataset'):
        self.base_video_dataset_dir = base_video_dataset_dir
        
        reference_file = os.path.join(base_video_dataset_dir, 'activitynet_entities/data/anet_entities_cleaned_class_thresh50_trainval.json')
        split_file = os.path.join(base_video_dataset_dir, 'activitynet_entities/data/split_ids_anet_entities.json')
        
        self.val_split = ['validation']
        
        self.anet_image_set = ['train']

        with open(split_file) as f:
            split_dict = json.load(f)
        split = {}
        for s in self.val_split:
            split.update({i:i for i in split_dict[s]})

        with open(reference_file) as f:
            ref = json.load(f)['annotations']
        ref = {k:v for k,v in ref.items() if k in split}
        self.refs = ref
        
        ls = []
        for vid, anns in self.refs.items():
            for seg, ann in anns['segments'].items():
                ls.append((vid, seg))
        self.ls = ls

    def __len__(self):
        return len(self.ls)

    def __getitem__(self, idx):
        vid, seg = self.ls[idx]
        
        
        extensions = ['.mp4', '.mkv', '.webm']
        exists = False
        for img_set in self.anet_image_set:
            video_root = os.path.join(self.base_video_dataset_dir, f"activitynet/videos/{img_set}")
            for ext in extensions:
                filename = vid+ext
                filepath = os.path.join(video_root, filename)
                if os.path.exists(filepath):
                    exists = True
                    break
            if exists:
                break
        if not exists:
            raise Exception(f"Video file not found for {vid}")
            return
            
        ann = self.refs[vid]['segments'][seg]
        s_t, e_t = ann['timestamps']
        
        frames = load_frames(filepath, s_t, e_t, num_frames=10)
        
        return vid, seg, frames
        


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
    if args.dataset_name == "anet_entities":        
        eval_dataset = ANetEntitiesValDataset(base_video_dataset_dir=args.video_dataset_dir)
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
        
        try:
        
            res = {}
            
            vid, seg, pil_images = eval_dataset[idx]
            

            save_dir_for_current_video = os.path.join(args.vis_save_path, args.dataset_name, vid, seg)
            if not os.path.exists(save_dir_for_current_video):
                os.makedirs(save_dir_for_current_video)
            else:
                saved_file = os.path.join(save_dir_for_current_video, "res.json")
                if os.path.exists(saved_file):
                    print(f"Skipping {idx} as it already exists.")
                    continue
            
            ###
        
            np_images = [np.array(image) for image in pil_images] # Tx(H, W, C)  # list of numpy arrays
            np_images = [np_images] # BxTx[H, W, C]
                        
            # Prepare inputs
            prompt_text = f"Could you please give me a detailed description of the video? Please respond with interleaved segmentation masks for the corresponding parts of the answer."
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
            text_output = text_output.replace("\n", "").replace("  ", " ")
            if '<|assistant|>' in text_output:
                text_output = text_output.split('<|assistant|>')[-1]
            else:
                text_output = text_output.split("ASSISTANT: ")[-1]
            # pred_masks: (batch_size x T_sam x [num_seg_tokens_per_sample, H, W])
            res["pred_text"] = text_output
            res["pred_text_cleaned"], res["pred_phrases"] = clean_caption(text_output)
            res["img_frames"] = {}
            res["pred_masks"] = defaultdict(list)
            
            
            video_frames_np = np_images[0] # [T, H, W, C]  # numpy array
            video_segments = video_segments_batch[0]
            
            for t, pred_mask in video_segments.items():
                res["img_frames"][t] = video_frames_np[t]
                for obj_id, pred_mask_i in pred_mask.items():
                    pred_mask_i = pred_mask_i > 0
                    pred_mask_i= remove_small_blobs(pred_mask_i, min_size=20) # NOTE
                    res["pred_masks"][obj_id].append(pred_mask_i)
            for obj_id in res["pred_masks"]:
                res["pred_masks"][obj_id] = np.stack(res["pred_masks"][obj_id])
                
            # Save results
            res_to_save = {}
            res_to_save["pred_text"] = res["pred_text"]
            res_to_save["pred_text_cleaned"] = res["pred_text_cleaned"]
            res_to_save["pred_phrases"] = res["pred_phrases"]
            
            # Save results as JSON
            json_path = os.path.join(save_dir_for_current_video, "res.json")
            with open(json_path, 'w') as f:
                json.dump(res_to_save, f)
            
            # Save images
            img_frames_dir = os.path.join(save_dir_for_current_video, "img_frames")
            os.makedirs(img_frames_dir, exist_ok=True)
            for frame_idx, image in res["img_frames"].items():
                image_path = os.path.join(img_frames_dir, f"frame_{frame_idx}.jpg")
                cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                
            # Save masks
            for obj_id, masks in res["pred_masks"].items():
                obj_dir = os.path.join(save_dir_for_current_video, f"pred_masks_{obj_id}")
                os.makedirs(obj_dir, exist_ok=True)
                for i, mask in enumerate(masks):
                    mask_path = os.path.join(obj_dir, f"mask_{i}.png")
                    skimage.io.imsave(mask_path, skimage.img_as_ubyte(mask), check_contrast=False)
            
            print(f"Saved idx:{idx} to {save_dir_for_current_video}")
        

        except Exception as e:
            print("Error at idx:", idx)
            print("\033[91m\t\t\t", e, "\033[0m")
            continue