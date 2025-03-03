import glob
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .data_processing import get_mask_from_json

class ReasonSegDataset(torch.utils.data.Dataset):
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        enc_preprocessor,
        sam_preprocessor,
        conversation_generator,
        num_classes_per_sample: int = 3,
        reason_seg_data="ReasonSeg|train",
        explanatory=0.1,
    ):
        self.sam_preprocessor = sam_preprocessor
        self.enc_preprocessor = enc_preprocessor
        self.conversation_generator = conversation_generator
        
        self.reason_seg_data = reason_seg_data
        self.explanatory = explanatory
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        
        self.default_image_token = DEFAULT_IMAGE_TOKEN = self.conversation_generator.DEFAULT_IMAGE_TOKEN
        self.short_question_list = [
                DEFAULT_IMAGE_TOKEN + "\n" + "Can you segment the {class_name} in this image?",
                DEFAULT_IMAGE_TOKEN + "\n" + "Please segment the {class_name} in this image.",
                DEFAULT_IMAGE_TOKEN
                + "\n"
                + "What is {class_name} in this image? Please respond with segmentation mask.",
                DEFAULT_IMAGE_TOKEN
                + "\n"
                + "What is {class_name} in this image? Please output segmentation mask.",
            ]
        self.long_question_list = [
                DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please respond with segmentation mask.",
                DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please output segmentation mask.",
            ]
        self.answer_list = [
                "It is [SEG].",
                "Sure, [SEG].",
                "Sure, it is [SEG].",
                "Sure, the segmentation result is [SEG].",
                "[SEG].",
            ]

        reason_seg_data, splits = reason_seg_data.split("|")
        splits = splits.split("_")
        images = []
        for split in splits:
            images_split = glob.glob(
                os.path.join(
                    base_image_dir, "reason_seg", reason_seg_data, split, "*.jpg"
                )
            )
            images.extend(images_split)
        jsons = [path.replace(".jpg", ".json") for path in images]
        self.reason_seg_data = (images, jsons)

        print("number of reason_seg samples: ", len(images))

        if explanatory != -1:
            self.explanatory_question_list = [
                "Please output segmentation mask and explain why.",
                "Please output segmentation mask and explain the reason.",
                "Please output segmentation mask and give some explanation.",
            ]
            self.img_to_explanation = {}
            with open(
                os.path.join(
                    base_image_dir,
                    "reason_seg",
                    reason_seg_data,
                    "explanatory",
                    "train.json",
                )
            ) as f:
                items = json.load(f)
            for item in items:
                img_name = item["image"]
                self.img_to_explanation[img_name] = {
                    "query": item["query"],
                    "outputs": item["outputs"],
                }

            print("len(self.img_to_explanation): ", len(self.img_to_explanation))

    def __len__(self):
        return len(self.reason_seg_data[0])

    def __getitem__(self, idx):
        images, jsons = self.reason_seg_data
        # idx = random.randint(0, len(images) - 1)
        image_path = images[idx]
        json_path = jsons[idx]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_size = image.shape[:2]
        # preprocess image for clip
        enc_img = self.enc_preprocessor.preprocess(image)

        mask, sents, is_sentence = get_mask_from_json(json_path, image)
        if len(sents) >= self.num_classes_per_sample:
            sampled_inds = np.random.choice(
                list(range(len(sents))), size=self.num_classes_per_sample, replace=False
            )
        else:
            sampled_inds = list(range(len(sents)))
        sampled_sents = np.vectorize(sents.__getitem__)(sampled_inds).tolist()
        sampled_masks = [
            (mask == 1).astype(np.float32) for _ in range(len(sampled_inds))
        ]

        image_name = image_path.split("/")[-1]
        if self.explanatory != -1 and image_name in self.img_to_explanation:
            if random.random() < self.explanatory:
                choice = 2
            else:
                choice = random.randint(0, 1)

        questions = []
        answers = []
        for text in sampled_sents:
            if is_sentence:
                question_template = random.choice(self.long_question_list)
                questions.append(question_template.format(sent=text))
            else:
                question_template = random.choice(self.short_question_list)
                questions.append(question_template.format(class_name=text.lower()))

            # add explanation if applicable
            img_name = image_path.split("/")[-1]
            if self.explanatory != -1 and img_name in self.img_to_explanation:
                if choice == 0:  # [SEG] token
                    answers.append(random.choice(self.answer_list))
                elif choice == 1:  # [SEG] token + text answer
                    image_name = image_path.split("/")[-1]
                    answer = self.img_to_explanation[image_name]["outputs"]
                    answer = random.choice(self.answer_list) + " {}".format(answer)
                    questions[-1] = (
                        self.default_image_token
                        + "\n"
                        + text
                        + " {}".format(random.choice(self.explanatory_question_list))
                    )
                    answers.append(answer)
                elif choice == 2:  # vanilla text answer
                    image_name = image_path.split("/")[-1]
                    answer = self.img_to_explanation[image_name]["outputs"]
                    questions[-1] = self.default_image_token + "\n" + text
                    answers.append(answer)
                else:
                    raise ValueError("Not implemented yet.")
            else:
                answers.append(random.choice(self.answer_list))

            ##
            conversation_source = []
            for i in range(len(questions)):
                conversation_source.append({'from': 'human', 'value': questions[i]})
                conversation_source.append({'from': 'gpt','value': answers[i]})
                
            conversation = self.conversation_generator.apply_on_semseg_dataset(conversation_source)
            

        # preprocess image for sam
        image, resize = self.sam_preprocessor.preprocess(image)

        image_name = image_path.split("/")[-1]
        if (
            self.explanatory != -1
            and image_name in self.img_to_explanation
            and choice == 2
        ):
            masks = torch.rand(0, *ori_size)
            label = torch.ones(ori_size) * self.ignore_label
        else:
            masks = np.stack(sampled_masks, axis=0)
            masks = torch.from_numpy(masks)
            label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

        data_dict = {
            'file_path': image_path,
            'preprocessed_for_sam': image,
            'images': enc_img['images'],
            'context_images': enc_img['context_images'],
            'conversations': conversation,
            'masks': masks,
            'label': label,
            'resize': resize,
            'questions': questions,
            'sampled_classes': sampled_sents,
        }
        return data_dict


class ReasonSegValDataset(torch.utils.data.Dataset):
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        enc_preprocessor,
        conversation_generator,
        # val_dataset, #"ReasonSeg|val"
    ):
        self.conversation_generator = conversation_generator
        self.default_image_token = self.conversation_generator.DEFAULT_IMAGE_TOKEN
        self.base_image_dir = base_image_dir
        val_dataset = "ReasonSeg|val"
        splits = val_dataset.split("|")
        if len(splits) == 2:
            ds, split = splits
            images = glob.glob(
                os.path.join(self.base_image_dir, "reason_seg", ds, split, "*.jpg")
            )
            self.images = images
            self.data_type = "reason_seg"

        self.ds = ds
        self.enc_preprocessor = enc_preprocessor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        json_path = image_path.replace(".jpg", ".json")
        mask_json, sampled_sents, is_sentence = get_mask_from_json(json_path, image)
        sampled_sents = [sampled_sents[0]]
        
        conversation_source = []
        for i in range(len(sampled_sents)):
            text = sampled_sents[i].strip()
            if is_sentence:
                conversation_source.append({'from': 'human', 
                                            'value': self.default_image_token + "\n {} Please output segmentation mask.".format(text),
                                            })
                conversation_source.append({'from': 'gpt','value': "[SEG]."})
            else:   
                conversation_source.append({'from': 'human', 
                                            'value': self.default_image_token + "\n What is {} in this image? Please output segmentation mask.".format(text),
                                            })
                conversation_source.append({'from': 'gpt','value': "[SEG]."})
                
        conversation = self.conversation_generator.apply_on_semseg_dataset(conversation_source)

        # preprocess image for clip
        enc_img = self.enc_preprocessor.preprocess(image)

        # preprocess image for sam
        image, resize = self.sam_preprocessor.preprocess(image)

        masks = [mask_json]

        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        labels = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

        data_dict = {
            'file_path': image_path,
            'preprocessed_for_sam': image,
            'images': enc_img['images'],
            'context_images': enc_img['context_images'],
            'conversations': conversation,
            'masks': masks,
            'label': labels,
            'resize': resize,
            'questions': None,
            'sampled_classes': sampled_sents,
            'inference': True,
        }
        return data_dict