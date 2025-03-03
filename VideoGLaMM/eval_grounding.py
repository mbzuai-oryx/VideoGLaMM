import cv2
import numpy as np
import torch
import os
from PIL import Image
import skimage
import json
import re
from tqdm import tqdm

from utils.vidstg_dataset import VideoModulatedSTGrounding
from utils.hcstvg_dataset import VideoModulatedSTGrounding_HCSTVGv2
from utils.grounding_utils.image_transforms import make_video_transforms

from utils.grounding_utils.box_ops import masks_to_boxes
from utils.grounding_utils.box_ops import np_box_iou

from chat import initialize_model_videogptplus, initialize_model_chatunivi, preprocess_vision, IMAGE_TOKEN_INDEX

iou_thresholds = [0.3, 0.5]

def summarize_metrics(results, tmp_loc):
    categories = set(x["qtype"] for x in results.values())
    metrics = {}
    counter = {}
    for category in categories:  # init metrics
        metrics[category] = {"gt_viou": 0}
        if tmp_loc:
            metrics[category].update({"tiou": 0, "viou": 0})
        for thresh in iou_thresholds:
            if tmp_loc:
                metrics[category][f"viou@{thresh}"] = 0
            metrics[category][f"gt_viou@{thresh}"] = 0
        counter[category] = 0
    for x in results.values():  # sum results
        qtype = x["qtype"]
        if tmp_loc:
            metrics[qtype]["tiou"] += x["tiou"]
            metrics[qtype]["viou"] += x["viou"]
        metrics[qtype]["gt_viou"] += x["gt_viou"]
        for thresh in iou_thresholds:
            if tmp_loc:
                metrics[qtype][f"viou@{thresh}"] += x[f"viou@{thresh}"]
            metrics[qtype][f"gt_viou@{thresh}"] += x[f"gt_viou@{thresh}"]
        counter[qtype] += 1
    for category in categories:  # average results per category
        for key in metrics[qtype]:
            metrics[category][key] = metrics[category][key] / counter[category]
            print(f"{category} {key}: {metrics[category][key]:.4f}")
    
    out = {}
    out["vid_metrics"] = results
    return out

def _calc_tiou(gt_sted, pred_sted, frame_ids):
    max_start = max(gt_sted[0], pred_sted[0])
    min_end = min(gt_sted[1], pred_sted[1])
    min_start = min(gt_sted[0], pred_sted[0])
    max_end = max(gt_sted[1], pred_sted[1])
    if min_end <= max_start:
        tiou = 0
    else:
        intersection = min_end - max_start
        gt_span = gt_sted[1] - gt_sted[0]
        pred_span = pred_sted[1] - pred_sted[0]
        union = gt_span + pred_span - intersection
        tiou = intersection / union
        
    union_predgt = [frame_id for frame_id in frame_ids if min_start <= frame_id < max_end]
    inter_predgt = set([frame_id for frame_id in frame_ids if max_start <= frame_id < min_end])
    
    return tiou, union_predgt, inter_predgt

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

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Eval grounding")
    
    # Model parameters
    parser.add_argument("--llava_version_or_path", type=str, default="/home/shehan/workspace_grounding_lmm/LISA2/checkpoints_hf/ChatUniVi-SAM2-test")
    parser.add_argument("--vis_save_path", type=str, default="./vis_output/eval_grounding")
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
    parser.add_argument("--dataset_name", default="vidstg", type=str, choices=["vidstg", "hcstvg"])
    parser.add_argument("--tmp_loc", action='store_true', default=False) #set this to False, if evaluating only on spatial localization performance
    
    return parser.parse_args()


if __name__=='__main__':
    
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
    base_video_dataset_dir = args.video_dataset_dir
    tmp_loc = args.tmp_loc
    print(f"tmp_loc: {tmp_loc}")
    
    if args.dataset_name == "vidstg":

        vidstg_vid_dir = os.path.join(base_video_dataset_dir, "vidstg/video")
        vidstg_ann_dir = os.path.join(base_video_dataset_dir,'processed/vidstg/vidstg_annotations')
        vidstg_ann_file = os.path.join(vidstg_ann_dir, "test.json")
        
        image_size = 224
        sample_fps = 1
        max_num_frames = 40 

        eval_dataset = VideoModulatedSTGrounding(
            vidstg_vid_dir,
            vidstg_ann_file,
            transforms=make_video_transforms("test", cautious=True, resolution=image_size, normalize=False), # make the test-transform without any randomization
            is_train=False, # Set this to false to prevent random downsampling of the video 
            video_max_len=max_num_frames, 
            video_max_len_train=max_num_frames, 
            fps=sample_fps, 
            tmp_crop=False, # No random temporal cropping
            tmp_loc=tmp_loc, # True: Need temporal localization timestamps.  (Set this to False, if evaluating only on spatial localization performance)
        )
    elif args.dataset_name == "hcstvg":
        vid_folder = os.path.join(base_video_dataset_dir,'hcstvg', "Video")
        processed_ann_dir = os.path.join(base_video_dataset_dir,'processed/hcstvg/hcstvg_annotations')

        image_set = "val"
        if image_set == "val": # HCSTVG-v2 has only a val set
            ann_file = os.path.join(processed_ann_dir, "val_v2_proc.json") 
        else:
            ann_file = os.path.join(processed_ann_dir, "train_v2_proc.json")
        
        image_size=224
        sample_fps = 1
        max_num_frames=40

        eval_dataset = VideoModulatedSTGrounding_HCSTVGv2(
            vid_folder,
            ann_file,
            transforms=make_video_transforms(image_set, cautious=True, resolution=image_size, normalize=False),
            is_train=image_set == "train",
            video_max_len=max_num_frames,
            video_max_len_train=max_num_frames,
            fps=sample_fps,
            tmp_crop=False, # No random temporal cropping
            tmp_loc=tmp_loc, # True: Need temporal localization timestamps.  (Set this to False, if evaluating only on spatial localization performance)
        )
    else:
        raise ValueError(f"Invalid dataset name: {args.dataset_name}")
    
    vid_metrics = {}
    
    # Loop through dataset
    for idx in tqdm(range(len(eval_dataset))):
        
        try:
    
            save_dir_for_current_video = os.path.join(args.vis_save_path, args.dataset_name, f"{idx:06d}")
            if not os.path.exists(save_dir_for_current_video):
                os.makedirs(save_dir_for_current_video)
            else:
                metrics_file = os.path.join(save_dir_for_current_video, "metrics.pt")
                if os.path.exists(metrics_file):
                    print(f"Skipping {idx} as it already exists.")
                    loaded_data = torch.load(metrics_file)
                    vid_metrics[idx] = loaded_data
                    continue
                            
            vid_path, images, targets, tmp_target = eval_dataset[idx]
            time_s, time_e = tmp_target['inter_idx']
            caption, qtype = tmp_target['caption'], tmp_target['qtype']
            
            gt_boxes_per_video = targets
            
            #image loading function
            np_images = [(img * 255).numpy().astype('uint8') for img in images] # Tx[H, W, C]
            np_images = [np_images] # BxTx[H, W, C]
            
            # NOTE: For HCSTVG, we need to load the question and answer from the json file, as in PG-Video-LLaVA
            if args.dataset_name == "hcstvg":
                # /home/shehan/workspace_grounding_lmm/LISA2/video_dataset/hcstvg/qa
                hcstvg_qa_dir = os.path.join(base_video_dataset_dir, 'hcstvg', 'qa')
                f = open( os.path.join(hcstvg_qa_dir, f"{idx}.json"))
                res_dict = json.load(f)
                f.close()
                question, answer = res_dict['Q'], res_dict['A']
                assert not(question=='') and not answer==''
                caption = question
                qtype = "interrogative"
            
            # Prepare inputs
            if qtype == "interrogative":
                prompt_text = f"{caption} Please respond with segmentation masks."
            else:
                prompt_text = f"Can you segment {caption} in this video?"
            # prompt_text = f"Can you spatiotemporally locate {caption} in this video?"
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
            print("text_output: ", text_output)
            # pred_masks: (batch_size x T_sam x [num_seg_tokens_per_sample, H, W])


            video_frames_np = np_images[0] # [T, H, W, C]  # numpy array
            video_segments = video_segments_batch[0]

            predictions = {}

            # 
            for t, pred_mask in video_segments.items():
                for obj_id, pred_mask_i in pred_mask.items():
                    pred_mask_i = pred_mask_i > 0
                    
                    pred_mask_i= remove_small_blobs(pred_mask_i, min_size=20) # NOTE
                    pred_boxes = masks_to_boxes(torch.tensor(np.expand_dims(pred_mask_i, axis=0)))
                    predictions[t] = {"boxes": pred_boxes}

                    save_path = "{}/mask_{}_{}.jpg".format(save_dir_for_current_video, t, obj_id)
                    cv2.imwrite(save_path, pred_mask_i * 100)
                    print("{} has been saved.".format(save_path))

                    save_path = "{}/masked_img_{}_{}.jpg".format(save_dir_for_current_video, t, obj_id)
                    save_img = video_frames_np[t].copy()
                    save_img[pred_mask_i] = (video_frames_np[t] * 0.5 + pred_mask_i[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5)[pred_mask_i]
                    
                    save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(save_path, save_img)
                    print("{} has been saved.".format(save_path))
            
            
                    
            ### Extract temporal timestamps
            if tmp_loc:
                match = re.search(r"frames:\((\d+),(\d+)\)", text_output)
                if match:
                    pred_t_start = int(match.group(1))
                    pred_t_end = int(match.group(2))
                    print("Start Time:", pred_t_start)
                    print("End Time:", pred_t_end)
                    
                    pred_sted = (pred_t_start, pred_t_end)
                    
                else:
                    print("No temporal localization predicted.")
                
            frame_ids = list(range(len(images))) 
            
            
            gt_sted = (time_s, time_e)
            inter_frames = list(range(gt_sted[0], gt_sted[1]+1))
            
            video_id = idx
            
            
            ### Calculate metrics
            
            # compute temporal iou
            if tmp_loc:
                
                tiou, union_predgt, inter_predgt = _calc_tiou(gt_sted, pred_sted, frame_ids)

                # compute viou and gt_viou
                curr_video_metrics = {
                    "gt_sted": gt_sted,
                    "pred_sted": pred_sted,
                    "tiou": tiou,
                    "qtype": qtype,
                    "img_metrics": {},
                }
                
                
                viou = 0
                
            else:
                curr_video_metrics = {
                    "qtype": qtype,
                    "img_metrics": {},
                }
                union_predgt = frame_ids
                inter_predgt = frame_ids


            gt_viou = 0

            for frame_id in (inter_frames):  # iterate on all frames of the annotated moment to update GT metrics
                
                if frame_id not in predictions:
                    raise RuntimeError(f"No prediction for frame {frame_id}")
                else:
                    pred_boxes = predictions[frame_id]["boxes"]
                
                gt_boxes = gt_boxes_per_video[frame_id]['boxes']
                
                iou = np_box_iou(np.array(pred_boxes), np.array(gt_boxes))[0][0]

                curr_video_metrics["img_metrics"][frame_id] = {
                    "iou": iou,
                    "pred_box": pred_boxes[0],
                    "gt_box": gt_boxes[0],
                }
                if (frame_id in inter_predgt and tmp_loc):  # update viou if this frame is in the intersection between the annotated moment and the predicted moment
                    viou += iou
                gt_viou += iou


            if tmp_loc:  # compute viou@R
                viou = viou / max(len(union_predgt), 1)
                curr_video_metrics["viou"] = viou
                recalls = {thresh: 0 for thresh in iou_thresholds}
                for thresh in iou_thresholds:
                    if viou > thresh:
                        recalls[thresh] += 1
                curr_video_metrics.update(
                    {
                        f"viou@{thresh}": recalls[thresh]
                        for thresh in iou_thresholds
                    }
                )

            # compute gt_viou@R
            gt_viou = gt_viou / max(len(inter_frames), 1)
            curr_video_metrics["gt_viou"] = gt_viou
            gt_recalls = {thresh: 0 for thresh in iou_thresholds}

            for thresh in iou_thresholds:
                if gt_viou > thresh:
                    gt_recalls[thresh] += 1

            curr_video_metrics.update(
                {
                    f"gt_viou@{thresh}": gt_recalls[thresh]
                    for thresh in iou_thresholds
                }
            )
            
            vid_metrics[video_id] = curr_video_metrics
            
            # Save curr_video_metrics as a JSON file
            metrics_file = os.path.join(save_dir_for_current_video, "metrics.pt")
            
            torch.save(curr_video_metrics, metrics_file)
            
        except Exception as e:
            print("Error at idx:", idx)
            print("\033[91m\t\t\t", e, "\033[0m")
            continue
        
    out = summarize_metrics(vid_metrics, tmp_loc=tmp_loc)
    
    