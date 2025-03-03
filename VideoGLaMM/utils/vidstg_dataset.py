import json
import os
import random

import numpy as np
import cv2
import torch
import torch.nn.functional as F

from PIL import Image

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

##################################################

import json
from torch.utils.data import Dataset
import time
import ffmpeg
import numpy as np
import random

from utils.grounding_utils.image_transforms import make_video_transforms, prepare

class VideoModulatedSTGrounding(Dataset):
    def __init__(
        self,
        vid_folder,
        ann_file,
        transforms,
        is_train=False,
        video_max_len=200,
        video_max_len_train=100,
        fps=5,
        tmp_crop=False,
        tmp_loc=True,
    ):
        """
        :param vid_folder: path to the folder containing a folder "video"
        :param ann_file: path to the json annotation file
        :param transforms: video data transforms to be applied on the videos and boxes
        :param is_train: whether training or not
        :param video_max_len: maximum number of frames to be extracted from a video
        :param video_max_len_train: maximum number of frames to be extracted from a video at training time
        :param fps: number of frames per second
        :param tmp_crop: whether to use temporal cropping preserving the annotated moment
        :param tmp_loc: whether to use temporal localization annotations
=        """
        self.vid_folder = vid_folder
        print("loading annotations into memory...")
        tic = time.time()
        self.annotations = json.load(open(ann_file, "r"))
        print("Done (t={:0.2f}s)".format(time.time() - tic))
        self._transforms = transforms
        self.is_train = is_train
        self.video_max_len = video_max_len
        self.video_max_len_train = video_max_len_train
        self.fps = fps
        self.tmp_crop = tmp_crop
        self.tmp_loc = tmp_loc
        self.vid2imgids = (
            {}
        )  # map video_id to [list of frames to be forwarded, list of frames in the annotated moment]
        for i_vid, video in enumerate(self.annotations["videos"]):
            video_fps = video["fps"]  # used for extraction
            sampling_rate = fps / video_fps
            assert sampling_rate <= 1  # downsampling at fps
            start_frame = (
                video["start_frame"] if self.tmp_loc else video["tube_start_frame"]
            )
            end_frame = video["end_frame"] if self.tmp_loc else video["tube_end_frame"]
            frame_ids = [start_frame]
            for frame_id in range(start_frame, end_frame):
                if int(frame_ids[-1] * sampling_rate) < int(frame_id * sampling_rate):
                    frame_ids.append(frame_id)

            if len(frame_ids) > video_max_len:  # subsample at video_max_len
                frame_ids = [
                    frame_ids[(j * len(frame_ids)) // video_max_len]
                    for j in range(video_max_len)
                ]

            inter_frames = set(
                [
                    frame_id
                    for frame_id in frame_ids
                    if video["tube_start_frame"] <= frame_id < video["tube_end_frame"]
                ]
            )  # frames in the annotated moment
            self.vid2imgids[video["video_id"]] = [frame_ids, inter_frames]
                    

    def __len__(self) -> int:
        return len(self.annotations["videos"])

    def __getitem__(self, idx):
        """
        :param idx: int
        :return:
        images: a CTHW video tensor
        targets: list of frame-level target, one per frame, dictionary with keys image_id, boxes, orig_sizes
        tmp_target: video-level target, dictionary with keys video_id, qtype, inter_idx, frames_id, caption
        """
        video = self.annotations["videos"][idx]
        caption = video["caption"]
        video_id = video["video_id"]
        video_original_id = video["original_video_id"]
        # clip_start = video["start_frame"]  # included
        # clip_end = video["end_frame"]  # excluded 
        clip_start = video["start_frame"] if self.tmp_loc else video["tube_start_frame"] #NOTE: important
        clip_end   = video["end_frame"] if self.tmp_loc else video["tube_end_frame"]
        
        frame_ids, inter_frames = self.vid2imgids[video_id]
        trajectory = self.annotations["trajectories"][video_original_id][
            str(video["target_id"])
        ]
        target_object_category = self.annotations["obj_categories"][video_original_id][ str(video["target_id"])]

        # ffmpeg decoding
        vid_path = os.path.join(self.vid_folder, video["video_path"])
        video_fps = video["fps"]
        ss = clip_start / video_fps
        t = (clip_end - clip_start) / video_fps
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
        inter_idx = []  # list of indexes of the frames in the annotated moment
        for i_img, img_id in enumerate(frame_ids):
            if img_id in inter_frames:
                anns = trajectory[
                    str(img_id)
                ]  # dictionary with bbox [left, top, width, height] key
                anns = [anns]
                inter_idx.append(i_img)
            else:
                anns = []
            target = prepare(w, h, anns)
            target["image_id"] = f"{video_id}_{img_id}"
            targets_list.append(target)

        # video spatial transform
        if self._transforms is not None:
            images, targets = self._transforms(images_list, targets_list)
        else:
            images, targets = images_list, targets_list

        if (
            inter_idx
        ):  # number of boxes should be the number of frames in annotated moment
            assert (
                len([x for x in targets if len(x["boxes"])])
                == inter_idx[-1] - inter_idx[0] + 1
            ), (len([x for x in targets if len(x["boxes"])]), inter_idx)

        # temporal crop
        if self.tmp_crop:
            p = random.random()
            if p > 0.5:  # random crop
                # list possible start indexes
                if inter_idx:
                    starts_list = [i for i in range(len(frame_ids)) if i < inter_idx[0]]
                else:
                    starts_list = [i for i in range(len(frame_ids))]

                # sample a new start index
                if starts_list:
                    new_start_idx = random.choice(starts_list)
                else:
                    new_start_idx = 0

                # list possible end indexes
                if inter_idx:
                    ends_list = [i for i in range(len(frame_ids)) if i > inter_idx[-1]]
                else:
                    ends_list = [i for i in range(len(frame_ids)) if i > new_start_idx]

                # sample a new end index
                if ends_list:
                    new_end_idx = random.choice(ends_list)
                else:
                    new_end_idx = len(frame_ids) - 1

                # update everything
                prev_start_frame = frame_ids[0]
                prev_end_frame = frame_ids[-1]
                frame_ids = [ x for i, x in enumerate(frame_ids) if new_start_idx <= i <= new_end_idx]
                images = images[:, new_start_idx : new_end_idx + 1]  # CTHW
                targets = [ x for i, x in enumerate(targets) if new_start_idx <= i <= new_end_idx]
                clip_start += frame_ids[0] - prev_start_frame
                clip_end += frame_ids[-1] - prev_end_frame
                if inter_idx:
                    inter_idx = [x - new_start_idx for x in inter_idx]

        if ( self.is_train and len(frame_ids) > self.video_max_len_train):  # densely sample video_max_len_train frames
            if inter_idx:
                starts_list = [
                    i
                    for i in range(len(frame_ids))
                    if inter_idx[0] - self.video_max_len_train < i <= inter_idx[-1]
                ]
            else:
                starts_list = [i for i in range(len(frame_ids))]

            # sample a new start index
            if starts_list:
                new_start_idx = random.choice(starts_list)
            else:
                new_start_idx = 0

            # select the end index
            new_end_idx = min(
                new_start_idx + self.video_max_len_train - 1, len(frame_ids) - 1
            )

            # update everything
            prev_start_frame = frame_ids[0]
            prev_end_frame = frame_ids[-1]
            frame_ids = [
                x for i, x in enumerate(frame_ids) if new_start_idx <= i <= new_end_idx
            ]
            images = images[:, new_start_idx : new_end_idx + 1]  # CTHW
            targets = [
                x for i, x in enumerate(targets) if new_start_idx <= i <= new_end_idx
            ]
            clip_start += frame_ids[0] - prev_start_frame
            clip_end += frame_ids[-1] - prev_end_frame
            if inter_idx:
                inter_idx = [
                    x - new_start_idx
                    for x in inter_idx
                    if new_start_idx <= x <= new_end_idx
                ]

        # video level annotations
        tmp_target = {
            "video_id": video_id,
            "qtype": video["qtype"],
            "inter_idx": [inter_idx[0], inter_idx[-1]] if inter_idx
                        else [-100,-100,],  # start and end (included) indexes for the annotated moment
            "frame_ids": frame_ids,
            "caption": caption,
            "target_object_category": target_object_category,
        }
        if not self._transforms:
            images = torch.tensor(images)
        else:
            images = torch.permute(images, (1,2,3,0)) # CTHW->THWC
        return vid_path, images, targets, tmp_target


##################################################    

class VidSTGDataset(torch.utils.data.Dataset):
    ignore_label = 255

    def __init__(
        self,
        base_video_dataset_dir,
        enc_preprocessor,
        sam_preprocessor,
        conversation_generator,
        image_set = 'train',
        num_frames_for_sam = -1,
    ):
        self.sam_preprocessor = sam_preprocessor
        self.enc_preprocessor = enc_preprocessor
        self.conversation_generator = conversation_generator
        
        self.base_video_dataset_dir = base_video_dataset_dir
        assert image_set in ["train", "test"], "Invalid image_set. Must be 'train' or 'test'."
        self.image_set = image_set
        self.num_frames_for_enc = self.enc_preprocessor.num_frames
        self.num_frames_for_sam = num_frames_for_sam
        
        self.masks_save_dir = os.path.join(self.base_video_dataset_dir, f"processed/vidstg/{image_set}/masks")
        
        self.video_dirs = sorted(os.listdir(self.masks_save_dir))
        
        # Remove ignore indices
        ignore_idxs_file = os.path.join(self.base_video_dataset_dir, f"processed/vidstg/{image_set}/ignore_idxs.txt")
        self.ignore_idxs = []
        with open(ignore_idxs_file, 'r') as file:
            for line in file:
                self.ignore_idxs.append(int(line.strip()))        
        self.video_dirs = [dir for i, dir in enumerate(self.video_dirs) if i not in self.ignore_idxs]
        
        # 
        self.DEFAULT_VIDEO_TOKEN = self.conversation_generator.DEFAULT_VIDEO_TOKEN
        self.QUESTION_LIST_FOR_INTERROGATIVE = [
            self.DEFAULT_VIDEO_TOKEN + "\n" + "Can you spatiotemporally locate {phrase} in this video?",
            self.DEFAULT_VIDEO_TOKEN + "\n" + "Please spatiotemporally locate {phrase} in this video.",
            self.DEFAULT_VIDEO_TOKEN + "\n" + "{phrase} Please respond with a segmentation masks and time interval.",
            self.DEFAULT_VIDEO_TOKEN + "\n" + "In the video, {phrase} Please include spatial locations and time duration in your answer.",
            self.DEFAULT_VIDEO_TOKEN + "\n" + "Perform spatiotemporal segmentation of {phrase}",
        ]
        self.QUESTION_LIST_FOR_DECLARATIVE = [
            self.DEFAULT_VIDEO_TOKEN + "\n" + "Can you spatiotemporally locate {phrase} in this video?",
            self.DEFAULT_VIDEO_TOKEN + "\n" + "Please spatiotemporally locate {phrase} in this video.",
            self.DEFAULT_VIDEO_TOKEN + "\n" + "Perform spatiotemporal segmentation of {phrase}",
        ]
        self.ANSWER_LIST = [
            "It is [SEG] in frames:({t_start},{t_end}).",
            "Sure, [SEG] frames:({t_start},{t_end}).",
            "Sure, it is [SEG] within frames:({t_start},{t_end}).",
            "Sure, the localization result is [SEG] in frames:({t_start},{t_end}).",
            "[SEG] frames:({t_start},{t_end}).",
        ]
        

    def __len__(self):
        return len(self.video_dirs)
    
    def generate_converation_from_template(self, caption, qtype, time_s, time_e):
                    
        conversations = []
        if qtype=="interrogative":
            conversations.append({'from': 'human', 
                              'value': random.choice(self.QUESTION_LIST_FOR_INTERROGATIVE).format(phrase=caption.lower())})
        elif qtype=="declarative":
            conversations.append({'from': 'human', 
                              'value': random.choice(self.QUESTION_LIST_FOR_DECLARATIVE).format(phrase=caption.lower())})
        else:
            raise Exception("Unsupported qtype")
        conversations.append({'from': 'gpt', 
                                'value': random.choice(self.ANSWER_LIST).format(t_start=time_s, t_end=time_e)})

        return conversations

    def get_from_idx(self, idx):
        video_dir_name = self.video_dirs[idx]
        
        video_dir = os.path.join(self.masks_save_dir, video_dir_name)
        # print(video_dir)
        # print(os.listdir(video_dir))
        # ['image_000006.jpg', 'image_000008.jpg', 'image_000012.jpg', 'image_000003.jpg', 'mask_000003.png', 'image_000010.jpg', 'image_000002.jpg', 'image_000009.jpg', 'image_000007.jpg', 'mask_000004.png', 'image_000005.jpg', 'image_000014.jpg', 'image_000013.jpg', 'metadata.json', 'image_000004.jpg', 'image_000001.jpg', 'image_000000.jpg', 'image_000011.jpg']
        metadata_file = os.path.join(video_dir, 'metadata.json')
        image_files = [f for f in os.listdir(video_dir) if f.startswith('image')]
        mask_files = [f for f in os.listdir(video_dir) if f.startswith('mask')]
        
        image_files = sorted(image_files)
        mask_files = sorted(mask_files)
        
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        # print(metadata)
        
        time_s = metadata['time_s']
        time_e = metadata['time_e']
        caption = metadata['caption']
        qtype = metadata['qtype']
        
        # 
        # image_files
        # ['image_000000.jpg', 'image_000001.jpg', 'image_000002.jpg', 'image_000003.jpg', 'image_000004.jpg', 'image_000005.jpg', 'image_000006.jpg', 'image_000007.jpg', 'image_000008.jpg', 'image_000009.jpg', 'image_000010.jpg', 'image_000011.jpg', 'image_000012.jpg', 'image_000013.jpg', 'image_000014.jpg']
        
        image_file_paths = [os.path.join(video_dir, f) for f in image_files]
        mask_file_paths = [os.path.join(video_dir, f) for f in mask_files]
        
        # All image frames
        pil_images = [Image.open(f) for f in image_file_paths]
        
        # For CLIP
        if not self.num_frames_for_enc==-1:
            pil_images_for_clip = subsample_images(pil_images, self.num_frames_for_enc)
        # preprocessed_for_clip = [self.clip_image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0] for img in pil_images_for_clip]
        enc_out = self.enc_preprocessor.preprocess(pil_images_for_clip)
        
        # For SAM
        gt_masks = []
        for mask_file_path in mask_file_paths:
            mask = cv2.imread(mask_file_path, cv2.IMREAD_GRAYSCALE)
            mask = mask.astype(np.uint8) / 255
            
            # mask_arrays.append(mask)
            if np.all(mask == 0):
                gt_masks.append(None)
            else:
                gt_masks.append(mask)
            
        # if not [time_s, time_e] == [-100, -100]:
        #     np_images_for_sam = np.array([np.array(img) for img in pil_images[time_s:time_e+1]])
        # else:
        #     np_images_for_sam = np.array([np.array(img) for img in pil_images])
        np_images_for_sam = np.array([np.array(img) for img in pil_images[time_s:time_e+1]])
        
        # Remove indices where gt_mask is None
        indices = [i for i, mask in enumerate(gt_masks) if mask is not None]
        np_images_for_sam = np_images_for_sam[indices]
        gt_masks = [mask for mask in gt_masks if mask is not None]
        gt_masks = np.array(gt_masks)
        
        if not self.num_frames_for_sam==-1:
            np_images_for_sam = subsample_images(np_images_for_sam, self.num_frames_for_sam)
            gt_masks          = subsample_images(gt_masks, self.num_frames_for_sam)


        # preprocess for sam
        preprocessed_for_sam_and_resize_shapes = [self.sam_preprocessor.preprocess(image) for image in np_images_for_sam]
        preprocessed_for_sam = [x[0] for x in preprocessed_for_sam_and_resize_shapes]
        resize = preprocessed_for_sam_and_resize_shapes[0][1]

        # 
        mask = torch.tensor(gt_masks)
        masks = [mask]
        masks = torch.stack(masks) # [N,T,H,W]

        # 
        label = torch.ones(masks.shape[2],  masks.shape[3]) * self.ignore_label # [H,W]

        ###
        source = self.generate_converation_from_template(caption, qtype, time_s, time_e)
        conversations = self.conversation_generator.apply(source)
        ### 
        
        data_dict = {
            'file_path': '',
            'preprocessed_for_sam': preprocessed_for_sam,
            'images': enc_out['images'],
            'context_images': enc_out['context_images'],
            'conversations': conversations,
            'masks': masks,
            'label': label,
            'resize': resize,
            'questions': None,
            'sampled_classes': None,
        }
        return data_dict
        
    def __getitem__(self, idx):
        # idx = random.randint(0, len(self.video_dirs) - 1)
        return self.get_from_idx(idx)
        # try:
        #     return self.get_from_idx(idx)
        # except Exception as e:
        #     print('Skipping vidstg idx:', idx, 'due to error:', e)
        #     idx = random.randint(0, len(self.video_dirs) - 1)
        #     return self.get_from_idx(idx)
