import os
import json
from torch.utils.data import Dataset
from utils.grounding_utils.image_transforms import make_video_transforms, prepare
import time
import ffmpeg
import numpy as np
import random
import torch


class VideoModulatedSTGrounding_HCSTVGv2(Dataset):
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
        stride=0,
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
        :param stride: temporal stride k
        """
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
        self.vid2imgids = {}
        self.stride = stride
        for i_vid, video in enumerate(self.annotations):
            video_num_images = video["frame_count"]
            video_fps = video_num_images / 20  # duration of videos in HC-STVG is 20s
            sampling_rate = fps / video_fps
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

            inter_frames = set(
                [
                    frame_id
                    for frame_id in frame_ids
                    if video["tube_start_frame"] <= frame_id < video["tube_end_frame"]
                ]
            )
            self.vid2imgids[video["video_id"]] = [frame_ids, inter_frames]

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        :param idx: int
        :return:
        images: a CTHW video tensor
        targets: list of frame-level target, one per frame, dictionary with keys image_id, boxes, orig_sizes
        tmp_target: video-level target, dictionary with keys video_id, inter_idx, frames_id, caption
        """
        video = self.annotations[idx]
        caption = video["caption"]
        video_id = video["video_id"]
        trajectory = video["trajectory"]
        frame_ids, inter_frames = self.vid2imgids[video_id]
        clip_start = 0
        clip_end = video["frame_count"] - 1

        # ffmpeg decoding
        vid_path = os.path.join(self.vid_folder, video["video_path"])
        ss = 0
        t = 20
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
                bbox = trajectory[
                    img_id - video["tube_start_frame"]
                ]  # dictionary with bbox [left, top, width, height] key
                anns = {"bbox": bbox}
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

        if inter_idx:
            assert (
                len([x for x in targets if len(x["boxes"])])
                == inter_idx[-1] - inter_idx[0] + 1
            ), (
                len([x for x in targets if len(x["boxes"])]),
                inter_idx,
            )  # , len([x for x in bis if len(x["boxes"])])

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
                frame_ids = [
                    x
                    for i, x in enumerate(frame_ids)
                    if new_start_idx <= i <= new_end_idx
                ]
                images = images[:, new_start_idx : new_end_idx + 1]  # CTHW
                targets = [
                    x
                    for i, x in enumerate(targets)
                    if new_start_idx <= i <= new_end_idx
                ]
                clip_start += frame_ids[0] - prev_start_frame
                clip_end += frame_ids[-1] - prev_end_frame
                if inter_idx:
                    inter_idx = [x - new_start_idx for x in inter_idx]

        if (
            self.is_train and len(frame_ids) > self.video_max_len_train
        ):  # densely sample video_max_len_train frames
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

        tmp_target = {
            "video_id": video_id,
            "inter_idx": [inter_idx[0], inter_idx[-1]] if inter_idx else [-100, -100],
            "frames_id": frame_ids,
            "caption": caption,
            "qtype": 'declarative',
        }
        if not self._transforms:
            images = torch.tensor(images)
        else:
            images = torch.permute(images, (1,2,3,0)) # CTHW->THWC
        return vid_path, images, targets, tmp_target


def build(image_set):

    # vid_dir = Path(args.hcstvg_vid_path)
    
    base_video_dataset_dir = '/home/shehan/workspace_grounding_lmm/LISA2/video_dataset'
    vid_folder = os.path.join(base_video_dataset_dir,'hcstvg', "Video")
    processed_ann_dir = os.path.join(base_video_dataset_dir,'processed/hcstvg/hcstvg_annotations')

    if image_set == "val": # HCSTVG-v2 has only a val set
        ann_file = os.path.join(processed_ann_dir, "valv2_proc.json") 
    else:
        ann_file = os.path.join(processed_ann_dir, "trainv2_proc.json")
    
    image_size=224
    sample_fps = 1
    max_num_frames=40
    tmp_loc = True #TODO: set this to False, if evaluating only on spatial localization performance

    dataset = VideoModulatedSTGrounding_HCSTVGv2(
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
    return dataset