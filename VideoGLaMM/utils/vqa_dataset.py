import json
import os
import random

import cv2
import torch
import torch.nn.functional as F


class VQADataset(torch.utils.data.Dataset):
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        enc_preprocessor,
        sam_preprocessor,
        conversation_generator,
        num_classes_per_sample: int = 3,
        vqa_data="llava_instruct_150k",
    ):
        self.enc_preprocessor = enc_preprocessor
        self.sam_preprocessor = sam_preprocessor
        self.conversation_generator = conversation_generator
        
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir

        DATA_DIR = os.path.join(base_image_dir, "llava_dataset")
        self.vqa_image_root = os.path.join(base_image_dir, "coco/train2017")
        with open(os.path.join(DATA_DIR, "{}.json".format(vqa_data))) as f:
            vqa_data = json.load(f)
        self.vqa_data = vqa_data

        print("vqa_data: ", len(self.vqa_data))

    def __len__(self):
        return len(self.vqa_data)

    def __getitem__(self, idx):
        # idx = random.randint(0, len(self.vqa_data) - 1)
        item = self.vqa_data[idx]
        image_path = os.path.join(self.vqa_image_root, item["image"])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # (224, 224, 3)
        ori_size = image.shape[:2]
        
        # preprocess image for clip # [3,224,224]
        enc_out = self.enc_preprocessor.preprocess(image)

        ###
        source = item["conversations"]
        conversations = self.conversation_generator.apply(source)


        # Preprocess image for SAM
        image_for_sam, resize_shape = self.sam_preprocessor.preprocess(image)

        masks = torch.rand(0, *ori_size) # random mask of size [0,224,224]
        label = torch.ones(ori_size) * self.ignore_label # array of size [224,224] with ignore label

        data_dict = {
            'file_path': image_path,
            'preprocessed_for_sam': image_for_sam,
            'images': enc_out['images'],
            'context_images': enc_out['context_images'],
            'conversations': conversations,
            'masks': masks,
            'label': label,
            'resize': resize_shape,
            'questions': None,
            'sampled_classes': None,
        }
        return data_dict
