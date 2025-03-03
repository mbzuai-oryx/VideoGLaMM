from torch.utils.data import Dataset
import numpy as np
import json
from pycocotools import mask as cocomask
from PIL import Image
from tqdm import tqdm
import os

def subsample_images(images, t):
    if isinstance(images, list):
        num_images = len(images)
        if t < num_images:
            indices = np.linspace(0, num_images - 1, num=t, dtype=int)
            return [images[i] for i in indices]
        else:
            return images
    elif isinstance(images, np.ndarray):
        T = images.shape[0]
        if t < T:
            indices = np.linspace(0, T - 1, num=t, dtype=int)
            return images[indices]
        else:
            return images
    else:
        raise ValueError("Input images must be either a list of PIL images or a numpy array.")

def get_imgs_and_masks_from_video(data_i):
    
    # video_annotations = data_i['annotations']
    
    # assert len(video_annotations)==data_i['length'], f"len(video_annotations): {len(video_annotations)}     data_i['length']:{data_i['length']}"
    vid_len = data_i['length']
    
    # h,w = data_i['annotations'][0][0]['segmentation']['size']
    imgs = []
    for frame_idx in range(vid_len):
        img_path = data_i['file_names'][frame_idx]
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        imgs.append(img)
    
    gt_masks = np.zeros((vid_len, h, w), dtype=np.uint8)
    # pred_masks = np.zeros((vid_len, h, w), dtype=np.uint8)

    if len(data_i['annotations'])==vid_len:
        for frame_idx in range(vid_len):
            num_annotations = len(data_i['annotations'][frame_idx])
            for anno_id in range(num_annotations):
                if data_i['annotations'][frame_idx]:
                    try:
                        mask_rle = data_i['annotations'][frame_idx][anno_id]['segmentation']
                        if mask_rle:
                            gt_masks[frame_idx] += cocomask.decode(mask_rle)
                    except:
                        print('frame_idx:', frame_idx, '    anno_id:', anno_id)
            
    return imgs, gt_masks

def load_mevis_json(mevis_root, image_set):
        
    image_root = os.path.join(mevis_root, image_set) # "./video_dataset/mevis/train"
    json_file = os.path.join(mevis_root, image_set, "meta_expressions.json") # "./video_dataset/mevis/train/meta_expressions.json"

    num_instances_without_valid_segmentation = 0
    num_instances_valid_segmentation = 0


    ann_file = json_file
    with open(str(ann_file), 'r') as f:
        subset_expressions_by_video = json.load(f)['videos']
    videos = list(subset_expressions_by_video.keys())
    print('number of video in the datasets:{}'.format(len(videos)))
    metas = []
    if image_set=='train' or image_set=='valid_u': # only train and valid_u sets have masks
        mask_json = os.path.join(image_root, 'mask_dict.json')
        print(f'Loading masks form {mask_json} ...')
        with open(mask_json) as fp:
            mask_dict = json.load(fp)

        for vid in videos:
            vid_data = subset_expressions_by_video[vid]
            vid_frames = sorted(vid_data['frames'])
            vid_len = len(vid_frames)
            if vid_len < 2:
                continue
            for exp_id, exp_dict in vid_data['expressions'].items():
                meta = {}
                meta['video'] = vid
                meta['exp'] = exp_dict['exp']
                meta['obj_id'] = [int(x) for x in exp_dict['obj_id']]
                meta['anno_id'] = [str(x) for x in exp_dict['anno_id']]
                meta['frames'] = vid_frames
                meta['exp_id'] = exp_id
                meta['category'] = 0
                meta['length'] = vid_len
                metas.append(meta)
    else: # valid set does not have masks
        for vid in videos:
            vid_data = subset_expressions_by_video[vid]
            vid_frames = sorted(vid_data['frames'])
            vid_len = len(vid_frames)
            for exp_id, exp_dict in vid_data['expressions'].items():
                meta = {}
                meta['video'] = vid
                meta['exp'] = exp_dict['exp']
                meta['obj_id'] = -1
                meta['anno_id'] = -1
                meta['frames'] = vid_frames
                meta['exp_id'] = exp_id
                meta['category'] = 0
                meta['length'] = vid_len
                metas.append(meta)

    dataset_dicts = []
    for vid_dict in tqdm(metas):
        record = {}
        record["file_names"] = [os.path.join(image_root, 'JPEGImages', vid_dict['video'], vid_dict["frames"][i]+ '.jpg') for i in range(vid_dict["length"])]
        record["length"] = vid_dict["length"]
        video_name, exp, anno_ids, obj_ids, category, exp_id = \
            vid_dict['video'], vid_dict['exp'], vid_dict['anno_id'], vid_dict['obj_id'], vid_dict['category'],  vid_dict['exp_id']

        exp = " ".join(exp.lower().split())
        if "eval_idx" in vid_dict:
            record["eval_idx"] = vid_dict["eval_idx"]

        video_objs = []
        if image_set=='train' or image_set=='valid_u': # only train and valid_u sets have masks
            for frame_idx in range(record["length"]):
                frame_objs = []
                for x, obj_id in zip(anno_ids, obj_ids):
                    obj = {}
                    segm = mask_dict[x][frame_idx]
                    if not segm:
                        num_instances_without_valid_segmentation += 1
                        continue
                    num_instances_valid_segmentation += 1
                    bbox = [0, 0, 0, 0]
                    obj["id"] = obj_id
                    obj["segmentation"] = segm
                    obj["category_id"] = category
                    obj["bbox"] = bbox
                    # obj["bbox_mode"] = BoxMode.XYXY_ABS
                    frame_objs.append(obj)
                video_objs.append(frame_objs)
        record["annotations"] = video_objs
        record["sentence"] = exp
        record["exp_id"] = exp_id
        record["video_name"] = video_name
        
        dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        print(
            "Total {} instance and Filtered out {} instances without valid segmentation. ".format(
                num_instances_valid_segmentation, num_instances_without_valid_segmentation
            )
        )
    return dataset_dicts

class MeVISBaseDataset(Dataset):
    def __init__(self, base_video_dataset_dir, image_set,
                 num_frames=-1
                 ):
        
        self.num_frames = num_frames
        self.image_set = image_set
        assert self.image_set in ["train", "valid", "valid_u"], f"invalid image_set:{self.image_set}"
        
        mevis_root = os.path.join(base_video_dataset_dir, "mevis")
        self.dataset = load_mevis_json(mevis_root, self.image_set)
        print("Done loading {} samples.".format(len(self.dataset)))
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data_i = self.dataset[idx]
        pil_images, gt_masks = get_imgs_and_masks_from_video(data_i)
        if not self.num_frames==-1:
            pil_images, gt_masks = subsample_images(pil_images, self.num_frames), subsample_images(gt_masks, self.num_frames) 
            
        caption = data_i['sentence']
        video_path = (data_i['video_name'], data_i['exp_id'])
        
        np_images = [np.array(image) for image in pil_images]
        np_images = np.stack(np_images, axis=0)
        
        target = {
                'masks': gt_masks,                          # [T, H, W]
                'caption': caption,
                'pil_images': pil_images,
                'video_path': video_path
            }
        
        return np_images, target