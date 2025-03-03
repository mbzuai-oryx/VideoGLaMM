import os
import argparse
from collections import defaultdict
import json
import re
from tqdm import tqdm
import cv2
import torch
import numpy as np
from PIL import Image
import skimage

from utils.dataset import ValGCGDataset

from model.chatunivi.constants import *
from model.chatunivi import conversation as conversation_lib
from model.chatunivi.mm_utils import tokenizer_image_token

from utils.grounding_utils.box_ops import masks_to_boxes
from utils.grounding_utils.box_ops import np_box_iou

    
##########################################################################
#### Calculate mask-mIoU

def compute_iou(mask1, mask2):
    '''
        mask1: (H,W) or (T,H,W)
        mask2: (H,W) or (T,H,W)
    '''
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)

    return iou


def compute_miou(pred_masks, gt_masks):
    '''
        pred_masks  : N1 x (H,W) or N1 x (T,H,W)
        gt_masks    : N2 x (H,W) or N2 x (T,H,W)
    '''
    # Computing mIoU between predicted masks and ground truth masks
    iou_matrix = np.zeros((len(pred_masks), len(gt_masks)))
    for i, pred_mask in enumerate(pred_masks):
        for j, gt_mask in enumerate(gt_masks):
            iou_matrix[i, j] = compute_iou(pred_mask, gt_mask)

    # One-to-one pairing and mean IoU calculation
    paired_iou = []
    while iou_matrix.size > 0 and np.max(iou_matrix) > 0:
        max_iou_idx = np.unravel_index(np.argmax(iou_matrix, axis=None), iou_matrix.shape)
        paired_iou.append(iou_matrix[max_iou_idx])
        iou_matrix = np.delete(iou_matrix, max_iou_idx[0], axis=0)
        iou_matrix = np.delete(iou_matrix, max_iou_idx[1], axis=1)

    return np.mean(paired_iou) if paired_iou else 0.0

def evaluate_mask_miou(all_pred_masks, all_gt_masks):
    # Load predictions

    mious = []

    for pred_masks, gt_masks in tqdm(zip(all_pred_masks, all_gt_masks), total=len(all_pred_masks)):
        # Compute and save the mIoU for the current image
        mious.append(compute_miou(pred_masks.values(), gt_masks.values()))

    print('mious', mious)
    
    # Report mean IoU across all videos
    mean_miou = np.mean(mious) if mious else 0.0  # If list is empty, return 0.0

    # print(f"Mean IoU (mIoU) across all videos: {mean_miou:.3f}")
    print(f"\033[92mMean IoU (mIoU) across all videos: {mean_miou}\033[0m")

    

##########################################################################
#### Calculate Recall


def compute_iou_matrix(pred_masks, gt_masks):
    '''
        pred_masks  : N1 x (H,W) or N1 x (T,H,W)
        gt_masks    : N2 x (H,W) or N2 x (T,H,W)
    '''
    iou_matrix = np.zeros((len(pred_masks), len(gt_masks)))
    for i, pred_mask in enumerate(pred_masks):
        for j, gt_mask in enumerate(gt_masks):
            iou_matrix[i, j] = compute_iou(pred_mask, gt_mask)

    return iou_matrix

# Load pre-trained model tokenizer and model for evaluation
from transformers import AutoTokenizer, AutoModel
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModel.from_pretrained("bert-base-uncased")

def get_bert_embedding(text):
    inputs = bert_tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    outputs = bert_model(**inputs)
    # Use the mean of the last hidden states as sentence embedding
    sentence_embedding = torch.mean(outputs.last_hidden_state[0], dim=0).detach().numpy()

    return sentence_embedding

from sklearn.metrics.pairwise import cosine_similarity
def text_similarity_bert(str1, str2):
    emb1 = get_bert_embedding(str1)
    emb2 = get_bert_embedding(str2)

    return cosine_similarity([emb1], [emb2])[0, 0]


def find_best_matches(gt_masks, gt_labels, pred_masks, pred_labels, iou_threshold=0.5, text_sim_threshold=0.5):
    '''
        gt_masks    : N1 x (H,W) or N1 x (T,H,W)
        gt_labels   : list of labels/phrases
        
        pred_masks  : N2 x (H,W) or N1 x (T,H,W)
        pred_labels : list of labels/phrases
        
        iou_threshold       :
        text_sim_threshold  : 
        
    '''
    best_matches = []

    # Compute pair - wise IoU
    # pred_masks = [maskUtils.decode(ann['segmentation']) for ann in dt_anns]
    # gt_masks = [maskUtils.decode(ann['segmentation']) for ann in gt_anns]
    ious = compute_iou_matrix(gt_masks, pred_masks)

    text_sims = np.zeros((len(gt_labels), len(pred_labels)))

    for i, gt_label in enumerate(gt_labels):
        for j, dt_label in enumerate(pred_labels):
            text_sims[i, j] = text_similarity_bert(gt_label, dt_label)

    # Find one-to-one matches satisfying both IoU and text similarity thresholds
    while ious.size > 0:
        max_iou_idx = np.unravel_index(np.argmax(ious), ious.shape)
        if ious[max_iou_idx] < iou_threshold or text_sims[max_iou_idx] < text_sim_threshold:
            break  # No admissible pair found

        best_matches.append(max_iou_idx)

        # Remove selected annotations from consideration
        ious[max_iou_idx[0], :] = 0
        ious[:, max_iou_idx[1]] = 0
        text_sims[max_iou_idx[0], :] = 0
        text_sims[:, max_iou_idx[1]] = 0

    return best_matches  # List of index pairs [(gt_idx, dt_idx), ...]


def evaluate_recall_with_mapping(all_gt_masks, all_gt_phrases, all_pred_masks, all_pred_phrases, 
                                 iou_threshold=0.5, text_sim_threshold=0.5):

    true_positives = 0
    actual_positives = 0

    for gt_masks, gt_labels, pred_masks, pred_labels in tqdm(zip(all_gt_masks, all_gt_phrases, all_pred_masks, all_pred_phrases), total=len(all_gt_masks)): 
        try:
            
            actual_positives += len(gt_labels)
            # Find best matching pairs
            best_matches = find_best_matches(gt_masks.values(), gt_labels, pred_masks.values(), pred_labels, iou_threshold, text_sim_threshold)
            true_positives += len(best_matches)
            
        except Exception as e:
            print(e)

    recall = true_positives / actual_positives if actual_positives > 0 else 0

    # print(f"Recall: {recall:.3f}")
    print(f"\033[92mRecall: {recall:.3f}\033[0m")

    



##########################################################################


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GCG Task")
    parser.add_argument("--video_dataset_dir", default='./video_dataset', type=str)
    
    parser.add_argument("--vis_save_path", default="./vis_output/eval_gcg", type=str)

    parser.add_argument("--dataset_name", default="video_gcg", type=str, choices=["video_gcg"])    
    
    parser.add_argument("--eval_miou", action="store_true", default=False)
    parser.add_argument("--eval_recall", action="store_true", default=False)
    parser.add_argument("--eval_caption", action="store_true", default=False)
    parser.add_argument("--use_clair", action="store_true", default=False)
    
    return parser.parse_args()

args = parse_args()

# Load dataset
base_video_dataset_dir = args.video_dataset_dir

if args.dataset_name == "video_gcg":

    image_size = 224
    sample_fps = 1
    
    eval_dataset = ValGCGDataset(args.video_dataset_dir, val_datasets = 'video_gcg||mevis_gcg||vidstg_gcg')

else:
    raise ValueError(f"Invalid dataset name: {args.dataset_name}")


##########################################################################
import json

all_res = []

print('eval_dataset', len(eval_dataset))
for idx in tqdm(range(len(eval_dataset))):
    
    try:
    
        save_dir_for_current_video = os.path.join(args.vis_save_path, args.dataset_name, f"{idx:06d}")
        print(save_dir_for_current_video)
        
        saved_file = os.path.join(save_dir_for_current_video, "res.json")
        
        if os.path.exists(saved_file):
            with open(saved_file, 'r') as file:
                res = json.load(file)
                all_res.append(res)
                
    
    except Exception as e:
        print(f"Error in processing {idx}: {e}")
        all_res.append(None)

print('all_res', len(all_res))

##########################################################################
#### Loop over and calculate metrics

import cv2
import os
import skimage
    
# miou
if args.eval_miou:
    mious = []
# recall
if args.eval_recall:
    iou_threshold=0.5
    text_sim_threshold=0.5
    true_positives = 0
    actual_positives = 0
# caption quality
if args.eval_caption:
    all_gt_references = []
    all_pred_captions = []

for idx in tqdm(range(len(all_res))):

    res = all_res[idx]

    try:

        gt_text = res['gt_text']
        pred_text = res['pred_text']
        
        gt_text_cleaned = res['gt_text_cleaned']
        pred_text_cleaned = res['pred_text_cleaned']
        
        gt_phrases = res['gt_phrases']
        pred_phrases = res['pred_phrases']

        
        save_dir_for_current_video = os.path.join(args.vis_save_path, args.dataset_name, f"{idx:06d}")

        img_frames_dir = os.path.join(save_dir_for_current_video, "img_frames")
        filenames = os.listdir(img_frames_dir)
        sorted_filenames = sorted(filenames, key=lambda x: int(re.search(r'\d+', x).group()))
        
        
        # Load images, masks if needed
        if args.eval_miou or args.eval_recall:
            
            images = []
            for filename in sorted_filenames:
                if filename.endswith(".jpg"):
                    image_path = os.path.join(img_frames_dir, filename)
                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images.append(image)
                    
            gt_masks = {}
            for obj_id in range(len(res['gt_phrases'])):
                obj_dir = os.path.join(save_dir_for_current_video, f"gt_masks_{obj_id}")
                gt_masks[obj_id] = []
                
                for ti in range(len(images)):
                    mask_path = os.path.join(obj_dir, f"mask_{ti}.png")
                    mask_img = skimage.io.imread(mask_path)
                    gt_masks[obj_id].append(mask_img)
            
            pred_masks = {}
            for obj_id in range(len(res['pred_phrases'])):
                obj_dir = os.path.join(save_dir_for_current_video, f"pred_masks_{obj_id}")
                pred_masks[obj_id] = []
                
                for ti in range(len(images)):
                    mask_path = os.path.join(obj_dir, f"mask_{ti}.png")
                    mask_img = skimage.io.imread(mask_path)
                    pred_masks[obj_id].append(mask_img)

        
        # miou
        if args.eval_miou:
            mious.append(compute_miou(pred_masks.values(), gt_masks.values()))
        # recall
        if args.eval_recall:
            actual_positives += len(gt_phrases)
            best_matches = find_best_matches(gt_masks.values(), gt_phrases, pred_masks.values(), pred_phrases, iou_threshold, text_sim_threshold)
            true_positives += len(best_matches)
        
        # caption
        if args.eval_caption:
            all_gt_references.append(gt_text_cleaned)
            all_pred_captions.append(pred_text_cleaned)
        
    except Exception as e:
        print(f"Error in processing {idx}: {e}")
        
        # miou
        if args.eval_miou:
            mious.append(0.0)
        # recall
        if args.eval_recall:
            actual_positives += len(gt_phrases)
            true_positives   += 0
        
        # caption
        if args.eval_caption:
            all_gt_references.append(gt_text_cleaned)
            all_pred_captions.append('')
        
        
# Report mean IoU across all videos
if args.eval_miou:
    mean_miou = np.mean(mious) if mious else 0.0  # If list is empty, return 0.0
    print(f"\033[92mMean IoU (mIoU) across all videos: {mean_miou}\033[0m")

# Recall
if args.eval_recall:
    recall = true_positives / actual_positives if actual_positives > 0 else 0
    print(f"\033[92mRecall: {recall:.3f}\033[0m")



##########################################################################
#### Calculate Caption Quality

def eval_caption_quality(all_gt_references, all_pred_captions):
    references = {}
    captions = {}

    k = 0
    for gt_ref, pred_caption in tqdm(zip(all_gt_references, all_pred_captions), total=len(all_gt_references)):
        if len(gt_ref) > 2000:
            gt_ref = gt_ref[:2000]
        if len(pred_caption) > 2000:
            pred_caption = pred_caption[:2000]
        references[str(k)] = [gt_ref]
        captions[str(k)] = [pred_caption]
        k += 1
        
    # Save them as correct json files
    import json

    new_cap = []
    for k, v in captions.items():
        new_cap.append({'image_id': k, 'caption': v[0]})

    new_ref = {'images': [], 'annotations': []}
    for k, refs in references.items():
        new_ref['images'].append({'id': k})
        for ref in refs:
            new_ref['annotations'].append({'image_id': k, 'id': k, 'caption': ref})

    with open('tmp_references.json', 'w') as fgts:
        json.dump(new_ref, fgts)
    with open('tmp_captions.json', 'w') as fres:
        json.dump(new_cap, fres)
    
    ##
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    annotation_file = 'tmp_references.json'
    results_file    = 'tmp_captions.json'

    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        # print(f'{metric}: {score:.3f}')
        print(f'\033[92m{metric}: {score:.3f}\033[0m')

        
def eval_caption_quality_with_clair(all_gt_references, all_pred_captions):
    from utils.clair import clair

    references = {}
    captions = {}

    k = 0
    for gt_ref, pred_caption in tqdm(zip(all_gt_references, all_pred_captions), total=len(all_gt_references)):
        if len(gt_ref) > 2000:
            gt_ref = gt_ref[:2000]
        if len(pred_caption) > 2000:
            pred_caption = pred_caption[:2000]
        references[str(k)] = [gt_ref]
        captions[str(k)] = [pred_caption]
        k += 1
        
    sum_ = 0
    count_ = 0
    for k in references:
        print(k, references[k], captions[k])
        
        clair_score, reason = clair(captions[k], references[k], model='chat-gpt')
        
        print(clair_score)
        print('-----------------')
        sum_ += clair_score
        count_ += 1
        
    avg_score = sum_ / count_
    
    print(f'\033[92m CLAIR Score: {avg_score:.3f}\033[0m')
    

if args.eval_caption and not args.use_clair:
    print('Evaluating caption quality...')
    eval_caption_quality(all_gt_references, all_pred_captions)
    
if args.use_clair:
    print('Evaluating caption quality with CLAIR...')
    eval_caption_quality_with_clair(all_gt_references, all_pred_captions)

##########################################################################


##########################################################################