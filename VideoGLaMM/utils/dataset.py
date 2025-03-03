import random
import numpy as np
import torch

from .reason_seg_dataset import ReasonSegDataset, ReasonSegValDataset
from .refer_seg_dataset import ReferSegDataset, ReferSegValDataset
from .sem_seg_dataset import SemSegDataset
from .refer_vos_dataset import ReferVOSDataset
from .video_vqa_dataset import VideoInstruct100kDataset
from .mevis_dataset import MEVISDataset
from .vqa_dataset import VQADataset
from .temporal_grounding_datasets import TemporalGroundingDataset
from .vidstg_dataset import VidSTGDataset
from .video_gcg_dataset import BURST_YTVIS_GCGDataset
from .grandf_dataset import GranDfAllDatasets
from .grounded_video_qa import GroundedVideoQADataset
from .video_gcg_anet import ANetEntitiesGCGDataset
from utils.ytvos_gcg import YTVOSGCGDataset
from utils.mevis_gcg import MevisGCGDataset
from utils.vidstg_hcstvg_gcg import VidSTG_HCSTVG_GCGDataset

from utils.itm_transforms import apply_augmentations_and_transforms


MASK_IGNORE_INDEX = -1
MAX_NUM_SEG_TOKENS_PER_SAMPLE = 4

def collate_fn(
    batch, tokenizer=None, local_rank=-1,
    conversation_generator=None,
):
    
    tokenizer_image_token = conversation_generator.tokenizer_image_token
    
    ###
    image_path_list = []
    images_sam_list = []
    images_list = []
    context_images_list = []
    conversation_list = []
    masks_list = []
    label_list = []
    resize_list = []
    questions_list = []
    sampled_classes_list = []
    offset_list = [0]
    cnt = 0
    inferences = []
    
    for sample in batch:
        image_path_list.append(sample['file_path'])
        images_sam_list.append(sample['preprocessed_for_sam'])
        images_list.append(sample['images'])
        context_images_list.append(sample['context_images'])
        conversation_list.extend(sample['conversations']) # NOTE: extend is used, not append
        masks_list.append(sample['masks'])
        label_list.append(sample['label'])
        resize_list.append(sample['resize'])
        questions_list.append(sample['questions'])
        sampled_classes_list.append(sample['sampled_classes'])
        cnt += len(sample['conversations'])
        offset_list.append(cnt)
        inferences.append(sample['inference'])

    ##############################
    
    # apply tokenizer with <image> converted to IMAGE_TOKEN_INDEX which is -200 in this case, and the rest of the text tokenized as usual
    input_ids = [ tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversation_list ]
    
    # convert a list of variable-length input_ids into a single tensor with padded sequences,
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    # create attention_masks to avoid the PAD tokens
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    targets = input_ids.clone()

    # apply preprocess_fn to the conversation_list, to mask targets accordingly
    conversation_generator.preprocess_fn(conversation_list, targets, tokenizer)

    # if training, truncate input_ids, targets, attention_masks to (model_max_length - 255) to accomodate the image_embedding_features 
    if inferences[0] == False:
        truncate_len = tokenizer.model_max_length - 255 #FIXME: replace 255 with an appropriate number to accomodate more number of embedding_features to facilitate videos

        if input_ids.shape[1] > truncate_len:
            input_ids = input_ids[:, :truncate_len]
            targets = targets[:, :truncate_len]
            attention_masks = attention_masks[:, :truncate_len]
            
    ##############################
    
    return {
        "image_paths": image_path_list,
        
        "images_for_sam": [torch.stack(img, dim=0) if type(img) is list else img for img in images_sam_list], # batch x [T_sam, 3, 1024,1024]
        "images": [torch.stack(img, dim=0) if type(img) is list else img for img in images_list], # batch x [T, 3, 224, 224]
        "context_images": [torch.stack(img, dim=0) if type(img) is list else img for img in context_images_list], # batch x [T, 3, 224, 224]
        "input_ids": input_ids,
        "labels": targets,
        "attention_masks": attention_masks,
        "masks_list": masks_list, #[m.unsqueeze(1) if len(m.shape) == 3 else m for m in masks_list], # batch_size x [num_seg_tokens_per_sample, T_sam, H, W] 
        
        "label_list": label_list,
        "resize_list": resize_list,
        "offset": torch.LongTensor(offset_list),
        
        # "questions_list": questions_list, # Is this ever used in train_ds.py or LISA.py ?
        # "sampled_classes_list": sampled_classes_list, #Is this ever used in train_ds.py or LISA.py ?
        "inference": inferences[0],
        # "conversation_list": conversation_list,
    }

class HybridDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        base_image_dir, base_video_dir,
        
        enc_preprocessor, 
        sam_preprocessor,
        conversation_generator,
        
        num_samples_per_epoch = 80000, num_classes_per_sample: int = 3,
        random_sampling = True,
        
        dataset=["sem_seg", "refer_seg", "vqa", "reason_seg"],
        sample_rate=[1,1,1,1],
        sem_seg_data="ade20k||cocostuff||partimagenet||pascal_part||paco_lvis||mapillary",
        refer_seg_data="refclef||refcoco||refcoco+||refcocog",
        vqa_data="llava_instruct_150k",
        reason_seg_data="ReasonSeg|train",
        refer_vos_data="ytvos||davis17||a2d||jhmdb",
        video_vqa_data="video_instruct_100k",
        video_tg_data="charades||anetcaps||qvh",
        
        num_frames_for_sam = -1,
        
        reason_seg_explanatory=0.1,
    ):
        
        self.num_frames_for_sam = num_frames_for_sam
        
        self.enc_preprocessor = enc_preprocessor
        self.sam_preprocessor = sam_preprocessor
        self.conversation_generator = conversation_generator
        
        self.random_sampling = random_sampling
                
        sample_rate = np.array(sample_rate)
        self.sample_rate = sample_rate / sample_rate.sum()
        
        self.dataset = dataset
        self.reason_seg_explanatory = reason_seg_explanatory
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.base_video_dir = base_video_dir

        self.datasets = dataset
        
        self.num_samples_per_epoch = num_samples_per_epoch

        self.all_datasets = []
        for dataset in self.datasets:
            
            ## Image datasets
            if dataset == "sem_seg":
                self.all_datasets.append(
                    SemSegDataset(
                        base_image_dir,
                        enc_preprocessor,
                        sam_preprocessor,
                        conversation_generator,
                        num_classes_per_sample,
                        sem_seg_data,
                    )
                )
            elif dataset == "refer_seg":
                self.all_datasets.append(
                    ReferSegDataset(
                        base_image_dir,
                        enc_preprocessor,
                        sam_preprocessor,
                        conversation_generator,
                        num_classes_per_sample,
                        refer_seg_data,
                    )
                )
            elif dataset == "vqa":
                self.all_datasets.append(
                    VQADataset(
                        base_image_dir,
                        enc_preprocessor,
                        sam_preprocessor,
                        conversation_generator,
                        num_classes_per_sample,
                        vqa_data,
                    )
                )
            elif dataset == "reason_seg":
                self.all_datasets.append(
                    ReasonSegDataset(
                        base_image_dir,
                        enc_preprocessor,
                        sam_preprocessor,
                        conversation_generator,
                        num_classes_per_sample,
                        reason_seg_data,
                        reason_seg_explanatory,
                    )
                )
            elif dataset == "grandf":
                self.all_datasets.append(
                    GranDfAllDatasets(
                        base_image_dir,
                        enc_preprocessor,
                        sam_preprocessor,
                        conversation_generator,
                        image_set="train",
                    )
                )
                
            ## video datasets
            elif dataset == "refer_vos":
                self.all_datasets.append(
                    ReferVOSDataset(
                        base_video_dataset_dir=base_video_dir,
                        enc_preprocessor=enc_preprocessor,
                        sam_preprocessor=sam_preprocessor,
                        conversation_generator=conversation_generator,
                        refer_vos_data=refer_vos_data,
                        image_set="train",
                        num_frames_for_sam=num_frames_for_sam,
                    )
                )
            elif dataset == "video_vqa":
                self.all_datasets.append(
                    VideoInstruct100kDataset(
                        base_video_dataset_dir=base_video_dir,
                        enc_preprocessor=enc_preprocessor,
                        sam_preprocessor=sam_preprocessor,
                        conversation_generator=conversation_generator,
                        image_set="train",
                        video_vqa_data=video_vqa_data,
                    )
                )
            elif dataset == "mevis":
                self.all_datasets.append(
                    MEVISDataset(
                        base_video_dataset_dir=base_video_dir,
                        enc_preprocessor=enc_preprocessor,
                        sam_preprocessor=sam_preprocessor,
                        conversation_generator=conversation_generator,
                        image_set="train",
                        num_frames_for_sam = num_frames_for_sam,
                    )
                )
            elif dataset == "temporal":
                self.all_datasets.append(
                    TemporalGroundingDataset(
                        base_video_dataset_dir=base_video_dir,
                        enc_preprocessor=enc_preprocessor,
                        sam_preprocessor=sam_preprocessor,
                        conversation_generator=conversation_generator,
                        tg_data=video_tg_data,
                        image_set="train",
                    )
                )
            elif dataset == "vidstg":
                self.all_datasets.append(
                    VidSTGDataset(
                        base_video_dataset_dir=base_video_dir,
                        enc_preprocessor=enc_preprocessor,
                        sam_preprocessor=sam_preprocessor,
                        conversation_generator=conversation_generator,
                        image_set="train",
                        num_frames_for_sam = num_frames_for_sam,
                    )
                )
            elif dataset == "video_gcg":
                self.all_datasets.append(
                    BURST_YTVIS_GCGDataset(
                        base_video_dir,
                        enc_preprocessor=enc_preprocessor,
                        sam_preprocessor=sam_preprocessor,
                        conversation_generator=conversation_generator,
                        image_set="train",
                        num_frames_for_sam = num_frames_for_sam,
                    )
                )
            elif dataset == "gvqa":
                self.all_datasets.append(
                    GroundedVideoQADataset(
                        base_video_dir,
                        enc_preprocessor=enc_preprocessor,
                        sam_preprocessor=sam_preprocessor,
                        conversation_generator=conversation_generator,
                        image_set="train",
                        num_frames_for_sam = num_frames_for_sam,
                    )
                )
            elif dataset == "anet_gcg":
                self.all_datasets.append(
                    ANetEntitiesGCGDataset(
                        base_video_dir,
                        enc_preprocessor=enc_preprocessor,
                        sam_preprocessor=sam_preprocessor,
                        conversation_generator=conversation_generator,
                        image_set="train",
                        num_frames_for_sam = num_frames_for_sam,
                    )
                )
            elif dataset == "ytvos_gcg":
                self.all_datasets.append(
                    YTVOSGCGDataset(
                        base_video_dir,
                        enc_preprocessor=enc_preprocessor,
                        sam_preprocessor=sam_preprocessor,
                        conversation_generator=conversation_generator,
                        image_set="train",
                        num_frames_for_sam = num_frames_for_sam,
                    )
                )
            elif dataset == "mevis_gcg":
                self.all_datasets.append(
                    MevisGCGDataset(
                        base_video_dir,
                        enc_preprocessor=enc_preprocessor,
                        sam_preprocessor=sam_preprocessor,
                        conversation_generator=conversation_generator,
                        image_set="train",
                        num_frames_for_sam = num_frames_for_sam,
                    )
                )
            elif dataset == "vidstg_gcg":
                self.all_datasets.append(
                    VidSTG_HCSTVG_GCGDataset(
                        base_video_dir,
                        enc_preprocessor=enc_preprocessor,
                        sam_preprocessor=sam_preprocessor,
                        conversation_generator=conversation_generator,
                        image_set="train",
                        num_frames_for_sam = num_frames_for_sam,
                        source_dataset='vidstg',
                    )
                )
            elif dataset == "hcstvg_gcg":
                self.all_datasets.append(
                    VidSTG_HCSTVG_GCGDataset(
                        base_video_dir,
                        enc_preprocessor=enc_preprocessor,
                        sam_preprocessor=sam_preprocessor,
                        conversation_generator=conversation_generator,
                        image_set="train",
                        num_frames_for_sam = num_frames_for_sam,
                        source_dataset='hcstvg',
                    )
                )
                
            else: 
                raise Exception(f'Unsupported dataset type: {dataset}')
        
        self.concatenated_dataset = torch.utils.data.ConcatDataset(self.all_datasets)

    def __len__(self):
        # return len(self.concatenated_dataset)
        return self.num_samples_per_epoch

    def __getitem__(self, idx):
        
        if self.random_sampling:
            ds_ind = np.random.choice(list(range(len(self.datasets))), p=self.sample_rate)
            ds = self.all_datasets[ds_ind]
            ind = np.random.randint(0, len(ds))
            data_sample = ds[ind]
        else:
            data_sample = self.concatenated_dataset[idx]
        
        
        data_sample_new = {
            "file_path": data_sample['file_path'],
            "conversations": data_sample['conversations'],
            "label": data_sample['label'],
            "resize": data_sample['resize'],
            "questions": data_sample['questions'],
            "sampled_classes": data_sample['sampled_classes'],        
        }
        
        preprocessed_for_sam = data_sample['preprocessed_for_sam'] # T_samx[3, 1024, 1024] (video) or [3, 1024, 1024]    (image)
        preprocessed_images = data_sample['images'] # Tx[3, 224, 224]  (video) or [3, 224, 224]      (image) 
        preprocessed_context_images = data_sample['context_images']
        masks = data_sample['masks'] # [num_masks, T_sam, h, w] (video) or [num_masks, h, w] (image)
        
        preprocessed_for_sam = preprocessed_for_sam.unsqueeze(0) if type(preprocessed_for_sam) is not list else torch.stack(preprocessed_for_sam, dim=0)     # [T_sam, 3, 1024,1024]
        preprocessed_images = preprocessed_images.unsqueeze(0) if type(preprocessed_images) is not list else torch.stack(preprocessed_images, dim=0) # [T, 3, 224, 224] 
        # preprocessed_context_images = preprocessed_context_images.unsqueeze(0) if (type(preprocessed_context_images) is not list and preprocessed_context_images is not None) else torch.stack(preprocessed_context_images, dim=0) # [T, 3, 224, 224]
        if preprocessed_context_images is not None:
            preprocessed_context_images = preprocessed_context_images.unsqueeze(0) if type(preprocessed_context_images) is not list else torch.stack(preprocessed_context_images, dim=0) # [T, 3, 224, 224]
            
        
        masks = masks.float()
        masks = masks.unsqueeze(1) if len(masks.shape) == 3 else masks # [num_seg_tokens_per_sample, T_sam, H, W]
                
        # 
        if masks.shape[0] > MAX_NUM_SEG_TOKENS_PER_SAMPLE:
            masks = masks[:MAX_NUM_SEG_TOKENS_PER_SAMPLE]
        if masks.shape[0] < MAX_NUM_SEG_TOKENS_PER_SAMPLE:
            # # add masks filled with MASK_IGNORE_INDEX
            _pad = torch.full((MAX_NUM_SEG_TOKENS_PER_SAMPLE - masks.shape[0], *masks.shape[1:]), MASK_IGNORE_INDEX, dtype=masks.dtype, device=masks.device)
            masks = torch.cat([masks, _pad], dim=0)
        
        # preprocessed_for_sam: [T_sam, 3, 1024,1024]
        # masks: [num_seg_tokens_per_sample, T_sam, H, W]
        # if T_sam != num_frames_for_sam, apply augmentations to preprocessed_for_sam and masks
        if preprocessed_for_sam.shape[0] != self.num_frames_for_sam:
            preprocessed_for_sam, masks = apply_augmentations_and_transforms(preprocessed_for_sam, masks, T_train=self.num_frames_for_sam)
        
        data_sample_new['preprocessed_for_sam'] = preprocessed_for_sam
        data_sample_new['images'] = preprocessed_images
        data_sample_new['context_images'] = preprocessed_context_images
        data_sample_new['masks'] = masks
        
        data_sample_new['inference'] = False
        
        return data_sample_new


class ValDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_image_dir,
        vision_tower,
        val_datasets = 'ReasonSeg|val', #"ReasonSeg|val||refcocog|umd|val",
    ):
        self.all_datasets = []
        
        for val_dataset in val_datasets.split("||"):
            splits = val_dataset.split("|")
            if len(splits) == 2: # ReasonSeg|val
                self.dataset = ReasonSegValDataset(base_image_dir,vision_tower)
            elif len(splits) == 3: # refcocog|umd|val
                self.dataset = ReferSegValDataset(base_image_dir,vision_tower, val_dataset="refcocog|umd|val")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


from utils.video_gcg_dataset import BURST_YTVIS_GCGBaseDataset
from utils.mevis_gcg import MevisGCGBaseDataset
from utils.vidstg_hcstvg_gcg import VidSTG_HCSTVG_GCGBaseDataset

class ValGCGDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_video_dir,
        val_datasets = 'video_gcg||mevis_gcg||vidstg_gcg', 
    ):
        self.all_datasets = []
        
        for val_dataset in val_datasets.split("||"):
            if val_dataset == "video_gcg":
                dataset = BURST_YTVIS_GCGBaseDataset(base_video_dir,
                                image_set="test", 
                                max_num_frames=40)
                self.all_datasets.append(dataset)
            elif val_dataset == "mevis_gcg":
                dataset = MevisGCGBaseDataset(base_video_dir,
                                image_set="valid_u")
                self.all_datasets.append(dataset)
            elif val_dataset == "vidstg_gcg":
                dataset = VidSTG_HCSTVG_GCGBaseDataset(base_video_dir,
                                image_set="val", 
                                source_dataset='vidstg')
                self.all_datasets.append(dataset)
                
        # concatenate all the datasets
        self.dataset = torch.utils.data.ConcatDataset(self.all_datasets)
                

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]