import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask

from .grefer import G_REFER
from .refer import REFER

class ReferSegDataset(torch.utils.data.Dataset):
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        enc_preprocessor,
        sam_preprocessor,
        conversation_generator,
        num_classes_per_sample: int = 3,
        refer_seg_data="refclef||refcoco||refcoco+||refcocog",
    ):
        self.sam_preprocessor = sam_preprocessor
        self.enc_preprocessor = enc_preprocessor
        self.conversation_generator = conversation_generator
        
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir

        self.default_image_token = DEFAULT_IMAGE_TOKEN = conversation_generator.DEFAULT_IMAGE_TOKEN
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
        self.answer_list = [
                "It is [SEG].",
                "Sure, [SEG].",
                "Sure, it is [SEG].",
                "Sure, the segmentation result is [SEG].",
                "[SEG].",
            ]

        DATA_DIR = os.path.join(base_image_dir, "refer_seg")
        self.refer_seg_ds_list = refer_seg_data.split(
            "||"
        )  # ['refclef', 'refcoco', 'refcoco+', 'refcocog']
        self.refer_seg_data = {}
        for ds in self.refer_seg_ds_list:
            if ds == "refcocog":
                splitBy = "umd"
            else:
                splitBy = "unc"

            if ds == "grefcoco":
                refer_api = G_REFER(DATA_DIR, ds, splitBy)
            else:
                refer_api = REFER(DATA_DIR, ds, splitBy)
            ref_ids_train = refer_api.getRefIds(split="train")
            images_ids_train = refer_api.getImgIds(ref_ids=ref_ids_train)
            refs_train = refer_api.loadRefs(ref_ids=ref_ids_train)

            refer_seg_ds = {}
            refer_seg_ds["images"] = []
            loaded_images = refer_api.loadImgs(image_ids=images_ids_train)

            for item in loaded_images:
                item = item.copy()
                if ds == "refclef":
                    item["file_name"] = os.path.join(
                        DATA_DIR, "images/saiapr_tc-12", item["file_name"]
                    )
                else:
                    item["file_name"] = os.path.join(
                        DATA_DIR, "images/mscoco/images/train2014", item["file_name"]
                    )
                refer_seg_ds["images"].append(item)
            refer_seg_ds["annotations"] = refer_api.Anns  # anns_train

            print(
                "dataset {} (refs {}) (train split) has {} images and {} annotations.".format(
                    ds,
                    splitBy,
                    len(refer_seg_ds["images"]),
                    len(refer_seg_ds["annotations"]),
                )
            )

            img2refs = {}
            for ref in refs_train:
                image_id = ref["image_id"]
                img2refs[image_id] = img2refs.get(image_id, []) + [
                    ref,
                ]
            refer_seg_ds["img2refs"] = img2refs
            self.refer_seg_data[ds] = refer_seg_ds
            
        self.all_data = []
        for ds in self.refer_seg_ds_list:
            refer_seg_ds = self.refer_seg_data[ds]
            images = refer_seg_ds["images"]
            for idx in range(len(images)):
                image_info = images[idx]
                self.all_data.append((ds, image_info))

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        # ds = random.randint(0, len(self.refer_seg_ds_list) - 1)
        # ds = self.refer_seg_ds_list[ds]
        # refer_seg_ds = self.refer_seg_data[ds]
        # images = refer_seg_ds["images"]
        # annotations = refer_seg_ds["annotations"]
        # img2refs = refer_seg_ds["img2refs"]
        # idx = random.randint(0, len(images) - 1)
        # image_info = images[idx]
        
        ds, image_info = self.all_data[idx]
        refer_seg_ds = self.refer_seg_data[ds]
        annotations = refer_seg_ds["annotations"]
        img2refs = refer_seg_ds["img2refs"]
        image_path = image_info["file_name"]
        image_id = image_info["id"]
        refs = img2refs[image_id]
        
        if len(refs) == 0:
            return self.__getitem__(0)

        sents = []
        ann_ids = []
        for ref in refs:
            for sent in ref["sentences"]:
                text = sent["sent"]
                sents.append(text)
                ann_ids.append(ref["ann_id"])
        if len(sents) >= self.num_classes_per_sample:
            sampled_inds = np.random.choice(
                list(range(len(sents))), size=self.num_classes_per_sample, replace=False
            )
        else:
            sampled_inds = list(range(len(sents)))
        sampled_sents = np.vectorize(sents.__getitem__)(sampled_inds).tolist()
        # sampled_ann_ids = np.vectorize(ann_ids.__getitem__)(sampled_inds).tolist()
        sampled_ann_ids = [ann_ids[ind] for ind in sampled_inds]
        sampled_classes = sampled_sents
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # preprocess image for clip
        enc_img = self.enc_preprocessor.preprocess(image)

        questions = []
        answers = []
        for text in sampled_classes:
            text = text.strip()
            assert len(text.split("||")) == 1
            question_template = random.choice(self.short_question_list)
            questions.append(question_template.format(class_name=text.lower()))
            answers.append(random.choice(self.answer_list))
            
        ##
        conversation_source = []
        for i in range(len(questions)):
            conversation_source.append({'from': 'human', 'value': questions[i]})
            conversation_source.append({'from': 'gpt','value': answers[i]})
        ##

        conversations = self.conversation_generator.apply_on_semseg_dataset(conversation_source)

        # preprocess image for sam
        image, resize = self.sam_preprocessor.preprocess(image)

        flag = False
        masks = []
        for ann_id in sampled_ann_ids:
            if isinstance(ann_id, list):
                flag = True
                if -1 in ann_id:
                    assert len(ann_id) == 1
                    m = np.zeros((image_info["height"], image_info["width"])).astype(
                        np.uint8
                    )
                else:
                    m_final = np.zeros(
                        (image_info["height"], image_info["width"])
                    ).astype(np.uint8)
                    for ann_id_i in ann_id:
                        ann = annotations[ann_id_i]

                        if len(ann["segmentation"]) == 0:
                            m = np.zeros(
                                (image_info["height"], image_info["width"])
                            ).astype(np.uint8)
                        else:
                            if type(ann["segmentation"][0]) == list:  # polygon
                                rle = mask.frPyObjects(
                                    ann["segmentation"],
                                    image_info["height"],
                                    image_info["width"],
                                )
                            else:
                                rle = ann["segmentation"]
                                for i in range(len(rle)):
                                    if not isinstance(rle[i]["counts"], bytes):
                                        rle[i]["counts"] = rle[i]["counts"].encode()
                            m = mask.decode(rle)
                            m = np.sum(
                                m, axis=2
                            )  # sometimes there are multiple binary map (corresponding to multiple segs)
                            m = m.astype(np.uint8)  # convert to np.uint8
                        m_final = m_final | m
                    m = m_final
                masks.append(m)
                continue

            ann = annotations[ann_id]

            if len(ann["segmentation"]) == 0:
                m = np.zeros((image_info["height"], image_info["width"])).astype(
                    np.uint8
                )
                masks.append(m)
                continue

            if type(ann["segmentation"][0]) == list:  # polygon
                rle = mask.frPyObjects(
                    ann["segmentation"], image_info["height"], image_info["width"]
                )
            else:
                rle = ann["segmentation"]
                for i in range(len(rle)):
                    if not isinstance(rle[i]["counts"], bytes):
                        rle[i]["counts"] = rle[i]["counts"].encode()
            m = mask.decode(rle)
            m = np.sum(
                m, axis=2
            )  # sometimes there are multiple binary map (corresponding to multiple segs)
            m = m.astype(np.uint8)  # convert to np.uint8
            masks.append(m)

        masks = np.stack(masks, axis=0)

        # if ds == 'grefcoco' and flag:
        #     import shutil
        #     image_name = image_path.split("/")[-1]
        #     save_dir = os.path.join("/group/30042/xlai/LISA_refactor_final/debug", image_name.split(".")[0])
        #     os.makedirs(save_dir, exist_ok=True)
        #     shutil.copy(image_path, save_dir)
        #     for i in range(masks.shape[0]):
        #         cv2.imwrite(os.path.join(save_dir, "{}_{}_{}.jpg".format(image_name, i, sampled_classes[i])), masks[i].astype(np.int32) * 100)

        masks = torch.from_numpy(masks)
        label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

        data_dict = {
            'file_path': image_path,
            'preprocessed_for_sam': image,
            'images': enc_img['images'],
            'context_images': enc_img['context_images'],
            'conversations': conversations,
            'masks': masks,
            'label': label,
            'resize': resize,
            'questions': questions,
            'sampled_classes': sampled_classes,
        }
        return data_dict

class ReferSegValDataset(torch.utils.data.Dataset):
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        enc_preprocessor,
        conversation_generator,
        val_dataset, #"refcocog|umd|val"
    ):
        self.conversation_generator = conversation_generator
        self.base_image_dir = base_image_dir
        splits = val_dataset.split("|")

        if len(splits) == 3:
            ds, splitBy, split = splits
            refer_api = REFER(self.base_image_dir, ds, splitBy)
            ref_ids_val = refer_api.getRefIds(split=split)
            images_ids_val = refer_api.getImgIds(ref_ids=ref_ids_val)
            refs_val = refer_api.loadRefs(ref_ids=ref_ids_val)
            refer_seg_ds = {}
            refer_seg_ds["images"] = []
            loaded_images = refer_api.loadImgs(image_ids=images_ids_val)
            for item in loaded_images:
                item = item.copy()
                if ds == "refclef":
                    item["file_name"] = os.path.join(
                        base_image_dir, "images/saiapr_tc-12", item["file_name"]
                    )
                elif ds in ["refcoco", "refcoco+", "refcocog", "grefcoco"]:
                    item["file_name"] = os.path.join(
                        base_image_dir,
                        "images/mscoco/images/train2014",
                        item["file_name"],
                    )
                refer_seg_ds["images"].append(item)
            refer_seg_ds["annotations"] = refer_api.Anns  # anns_val

            img2refs = {}
            for ref in refs_val:
                image_id = ref["image_id"]
                img2refs[image_id] = img2refs.get(image_id, []) + [
                    ref,
                ]
            refer_seg_ds["img2refs"] = img2refs
            self.refer_seg_ds = refer_seg_ds
            self.data_type = "refer_seg"
        else:
            raise Exception(f'unsupported val_dataset type: {val_dataset}')

        self.ds = ds
        self.enc_preprocessor = enc_preprocessor
        self.default_image_token = self.conversation_generator.DEFAULT_IMAGE_TOKEN

    def __len__(self):
        return len(self.refer_seg_ds["images"])

    def __getitem__(self, idx):
        refer_seg_ds = self.refer_seg_ds
        images = refer_seg_ds["images"]
        annotations = refer_seg_ds["annotations"]
        img2refs = refer_seg_ds["img2refs"]

        image_info = images[idx]
        image_path = image_info["file_name"]
        image_id = image_info["id"]

        refs = img2refs[image_id]
        if len(refs) == 0:
            raise ValueError("image {} has no refs".format(image_id))

        sents = []
        ann_ids = []
        for ref in refs:
            for sent in ref["sentences"]:
                sents.append(sent["sent"].strip().lower())
                ann_ids.append(ref["ann_id"])

        sampled_sents = sents
        sampled_ann_ids = ann_ids
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        is_sentence = False
        
        ##
        conversation_source = []
        for i in range(len(sampled_sents)):
            text = sampled_sents[i].strip()
            conversation_source.append({'from': 'human', 
                                        'value': self.default_image_token
                                                + "\n What is {} in this image? Please output segmentation mask.".format(text)})
            conversation_source.append({'from': 'gpt','value': '[SEG].'})
        
        ###        
        conversations = self.conversation_generator.apply_on_semseg_dataset(conversation_source)

        # preprocess image for clip
        image_clip = self.enc_preprocessor.preprocess(image)

        # preprocess image for sam
        image, resize = self.sam_preprocessor.preprocess(image)

        masks = []
        for i, ann_id in enumerate(sampled_ann_ids):
            ann = annotations[ann_id]
            if len(ann["segmentation"]) == 0 and sampled_sents[i] != "":
                m = np.zeros((image_info["height"], image_info["width"], 1))
            else:
                if type(ann["segmentation"][0]) == list:  # polygon
                    rle = mask.frPyObjects(
                        ann["segmentation"],
                        image_info["height"],
                        image_info["width"],
                    )
                else:
                    rle = ann["segmentation"]
                    for i in range(len(rle)):
                        if not isinstance(rle[i]["counts"], bytes):
                            rle[i]["counts"] = rle[i]["counts"].encode()
                m = mask.decode(rle)
            m = np.sum(
                m, axis=2
            )  # sometimes there are multiple binary map (corresponding to multiple segs)
            m = m.astype(np.uint8)  # convert to np.uint8
            masks.append(m)


        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        labels = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        inference = True

        # return (
        #     image_path,
        #     image,
        #     image_clip,
        #     conversations,
        #     masks,
        #     labels,
        #     resize,
        #     None,
        #     None,
        #     inference,
        # )
        data_dict = {
            'file_path': image_path,
            'preprocessed_for_sam': image,
            'images': image_clip['images'],
            'context_images': image_clip['context_images'],
            'conversations': conversations,
            'masks': masks,
            'label': labels,
            'resize': resize,
            'questions': None,
            'sampled_classes': None,
            'inference': inference,
        }
        return data_dict
        
