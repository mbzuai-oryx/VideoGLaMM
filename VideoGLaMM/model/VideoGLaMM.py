from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, CLIPVisionModel

from .videogpt_plus.model.language_model.phi3 import (VideoGPTPlusPhi3ForCausalLM, VideoGPTPlusPhi3Model)
from .segment_anything import build_sam_vit_h
from .segment_anything_2.sam2.build_sam import build_sam2, build_sam2_video_predictor

from utils.misc import print_dimensions 

MASK_IGNORE_INDEX = -1
MAX_NUM_SEG_TOKENS_PER_SAMPLE = 4

import torch
import torch.nn.functional as F

def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    ignore_index=None,
    scale=1000,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        ignore_index: A value in the target tensor to ignore during the loss calculation.
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)

    # Create mask to ignore specified index
    if ignore_index is not None:
        mask = targets != ignore_index
        inputs = inputs * mask
        targets = targets * mask

    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    ignore_index=None,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        ignore_index: A value in the target tensor to ignore during the loss calculation.
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    # Flatten and mask the loss values where targets match the ignore index
    loss = loss.flatten(1, 2)
    if ignore_index is not None:
        mask = targets.flatten(1, 2) != ignore_index
        loss = loss * mask

    loss = loss.mean(1).sum() / (num_masks + 1e-8)
    return loss


class VideoGLaMMMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(VideoGLaMMMetaModel, self).__init__(config)
        
        self.vision_pretrained = kwargs.get("sam_pretrained_path", None)

        self.config = config
        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
        if not hasattr(self.config, "mask_decoder_itm"):
            self.config.mask_decoder_itm = kwargs["mask_decoder_itm"]
        if not hasattr(self.config, "use_sam2"):
            self.config.use_sam2 = kwargs.get("use_sam2")
            
        use_sam2_video_branch = kwargs.get("use_sam2_video_branch", False)
            
        self.initialize_lisa_modules(self.config, use_sam2_video_branch=use_sam2_video_branch)

    def initialize_lisa_modules(self, config, use_sam2_video_branch=False):
        # SAM
        if config.use_sam2: # Use SAM2
            if not use_sam2_video_branch:
                print('\033[92m---Initialize SAM2 without video branch--\033[0m')
                self.visual_model = build_sam2("sam2_hiera_l.yaml", self.vision_pretrained, device=None)
            else:
                print('\033[92m---Initialize SAM2 with video branch--\033[0m')
                self.visual_model = build_sam2_video_predictor("sam2_hiera_l.yaml", self.vision_pretrained, device=None)
        elif config.mask_decoder_itm: # Use SAM_with_ITM
            self.visual_model = build_sam_vit_h(self.vision_pretrained, with_itm=True)
        else: # Use original SAM
            self.visual_model = build_sam_vit_h(self.vision_pretrained)
            
        for param in self.visual_model.parameters():
            param.requires_grad = False
        if config.train_mask_decoder:
            if config.use_sam2: # if using SAM2
                self.visual_model.sam_mask_decoder.train()
                for param in self.visual_model.sam_mask_decoder.parameters():
                    param.requires_grad = True
            else: # if using SAM
                self.visual_model.mask_decoder.train()
                for param in self.visual_model.mask_decoder.parameters():
                    param.requires_grad = True

        # Projection layer
        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True

    def postprocess_masks(self, masks: torch.Tensor, orig_hw) -> torch.Tensor:
        """
        Perform PostProcessing on output masks.
        """
        masks = masks.float()
        masks = F.interpolate(masks, orig_hw, mode="bilinear", align_corners=False)
        return masks
    
class VideoGLaMMModel(VideoGLaMMMetaModel, VideoGPTPlusPhi3Model):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(VideoGLaMMModel, self).__init__(config, **kwargs)

        self.config.use_cache = False
        # self.config.vision_tower = self.config.mm_vision_tower 
        self.config.vision_tower = getattr(self.config, "mm_vision_tower", None)
        # self.config.image_vision_tower = self.config.image_mm_vision_tower 
        self.config.image_vision_tower = getattr(self.config, "image_mm_vision_tower", None)
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.pretrain_mm_mlp_adapter = None
        self.config.pretrain_image_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False

        
class VideoGLaMM_SAM2():
    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        '''
            pixel_values : batch x [T, 3, 1024, 1024]
        '''
    
        # 
        images = pixel_values
        with torch.no_grad():
            image_embeddings_list = []
            batch_size = len(images)
            
            for i in range(batch_size): # for batch
                if images[i].shape[0]==1: # image
                    torch.cuda.empty_cache()
                    image_embeddings = self.model.visual_model.image_encoder(images[i])
                    image_embeddings_list.append([image_embeddings])
                else: # video
                    ###
                    t = images[i].shape[0]
                    image_embeddings_i = []
                    for ti in range(t):
                        torch.cuda.empty_cache()
                        image_embeddings = self.model.visual_model.image_encoder(images[i][ti].unsqueeze(0))
                        image_embeddings_i.append(image_embeddings)
                    image_embeddings_list.append(image_embeddings_i)
            torch.cuda.empty_cache()
        return image_embeddings_list # B x T x [1, 256, 64, 64]
    
    def get_visual_embs_sam2(self, images_for_sam_all: torch.FloatTensor):
        '''
            images_for_sam_all : batch x [T, 3, 1024, 1024] : (list of tensors)
        '''
        
        batch_size = len(images_for_sam_all)
        _features_all = []
        
        with torch.no_grad():
            for batch_idx in range(len(images_for_sam_all)):
                features_in_batch = []
                for t in range(len(images_for_sam_all[batch_idx])):
                    images_for_sam = images_for_sam_all[batch_idx][t].unsqueeze(0) # [1, 3, 1024, 1024]
                    
                    backbone_out = self.model.visual_model.forward_image(images_for_sam) 
                    _, image_embeddings, _, _ = self.model.visual_model._prepare_backbone_features(backbone_out)
                    image_embeddings = [_.to(images_for_sam.dtype) for _ in image_embeddings]
                    
                    bs = images_for_sam.shape[0]
                    
                    if self.model.visual_model.directly_add_no_mem_embed:
                        image_embeddings[-1] = image_embeddings[-1] + self.model.visual_model.no_mem_embed

                    _bb_feat_sizes = [(256, 256),(128, 128),(64, 64),]
                    feats = [
                        feat.permute(1, 2, 0).view(bs, -1, *feat_size)
                        for feat, feat_size in zip(image_embeddings[::-1], _bb_feat_sizes[::-1])
                    ][::-1]
                    _features = {
                        "image_embed": feats[-1], # [bs, 256, 64, 64] # bs=1 in this case
                        "high_res_feats": feats[:-1]} #   # 2 x [ bs, 32, 256, 256] # bs=1 in this case
                    
                    features_in_batch.append(_features)
                _features_all.append(features_in_batch)
            torch.cuda.empty_cache()
        # 
        return batch_size, _features_all
    
    # def forward(self, **kwargs):
    #     if "past_key_values" in kwargs:
    #         return super().forward(**kwargs)
    #     return self.model_forward(**kwargs)

    def __inference_path(self, input_ids, images, context_images, attention_masks):

        # length = input_ids.shape[0]

        assert len(images) == 1 # batch size is 1
        assert input_ids.shape[0] == 1 # batch size is 1
        
        # images_clip_extend = images * length # 1x[1, 3, 224, 224] -> lengthx[1, 3, 224, 224] 
        images_clip_extend = images # 1x[1, 3, 224, 224]

        # print_dimensions('context_images', context_images) 
        
        # if context_images is not None:
        #     context_images_clip_extend = context_images * length # 1x[1, 3, 224, 224] -> lengthx[1, 3, 224, 224]
        # else:
        #     context_images_clip_extend = None
        
        context_images_clip_extend = context_images

        output_hidden_states = []
        # for i in range(n_batch):
        start_i, end_i   =  0, 1 #i * length   ,  min((i + 1) * length, input_ids.shape[0]) # 0, 1
        output_i = self.super_forward(
            # images=images_clip_extend[: 1], 
            images=images_clip_extend,
            # context_images=context_images_clip_extend[: 1] if context_images_clip_extend is not None else [None]*length,
            context_images=context_images_clip_extend if context_images_clip_extend is not None else [None],
            attention_mask=attention_masks[0:1] if attention_masks is not None else None,
            input_ids=input_ids[0:1],
            output_hidden_states=True,
        )
        output_hidden_states.append(output_i.hidden_states)
        seg_token_mask = output_i.seg_token_mask
        torch.cuda.empty_cache()
        
        output_hidden_states = [torch.cat(output_hidden_states, dim=0)]
        
        output = None
        
        return output, output_hidden_states, seg_token_mask
        
    def __training_path(self, images, context_images, input_ids, labels, attention_masks, offset):
        
        # print('In __training_path')
        # print_dimensions('images', images) # [B, T, 3, 224, 224]
        # print_dimensions('context_images', context_images)
        # print('context_images:', context_images)
        
        # prepare images
        images_clip_list = []
        for i in range(len(offset) - 1): # batch_size = len(offset) - 1
            start_i, end_i = offset[i], offset[i + 1]
            for j in range(end_i - start_i):
                images_clip_list.append(images[i])
        images = images_clip_list # len(all_conversations) x [T, 3, 224, 224] 
        
        # prepare context images
        context_images_clip_list = []
        for i in range(len(offset) - 1): # batch_size = len(offset) - 1
            start_i, end_i = offset[i], offset[i + 1]
            for j in range(end_i - start_i):
                context_images_clip_list.append(context_images[i])
        context_images = context_images_clip_list # len(all_conversations) x [T, 3, 224, 224]
        
        output = self.super_forward(
            images=images,
            context_images=context_images,
            attention_mask=attention_masks,
            input_ids=input_ids,
            labels=labels,
            output_hidden_states=True,
        )
        output_hidden_states = output.hidden_states
        seg_token_mask = output.seg_token_mask
        
        return output, output_hidden_states, seg_token_mask
    
    def model_forward(
        self,
        images_for_sam: List[torch.FloatTensor],      # preprocessed image for SAM   # batchx[T_sam, 3, 1024, 1024]
        images: List[torch.FloatTensor],              # preprocessed image           # batchx[T, 3, 224, 224]
        context_images: List[torch.FloatTensor],      # preprocessed context image   # batchx[T, 3, 224, 224]
        input_ids: torch.LongTensor,                  # [num_conversations, length_of_sequence]
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        masks_list: List[torch.FloatTensor],          # ground truth masks
        label_list: List[torch.Tensor], #  a pseudo label of which the shape indicates the original frame dimensions
        resize_list: List[tuple],
        inference: bool = False,
        **kwargs,
    ):
        
        if self.config.use_sam2:
            USE_SAM2 = True
            USE_SAM1 = False
        else:
            USE_SAM2 = False
            USE_SAM1 = True
            
        ### 1 - Extract grounding encoder image embeddings
        if USE_SAM1:
            image_embeddings_for_sam = self.get_visual_embs(images_for_sam) # get SAM embeddings
            batch_size = len(image_embeddings_for_sam)
        elif USE_SAM2:
            batch_size, _features_all = self.get_visual_embs_sam2(images_for_sam) # get SAM2 embeddings
            
                
        assert batch_size == len(offset) - 1        
        
        ### 3 - Handle inference or training path
        if inference:
            output, output_hidden_states, seg_token_mask = self.__inference_path(input_ids, images, context_images, attention_masks)
        else:
            output, output_hidden_states, seg_token_mask = self.__training_path(images, context_images, input_ids, labels, attention_masks, offset)
            
        # output_hidden_states: [num_hidden_layers, num_conversations, length_of_sequence, 4096]
        # seg_token_mask : [num_conversations, length_of_sequence]
                
        ### 4 - Process hidden states
        hidden_states = []
        assert len(self.model.text_hidden_fcs) == 1
        
        # Pass the output_hidden_states[-1] through the fully connected layer (text_hidden_fcs)
        hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))  # ([num_conversations, length_of_sequence, 4096]) -> ([batch_size, length_of_sequence, 256])
        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1) # ([num_conversations, length_of_sequence, 256])
                
        # obtain the output of the fully connected layer corresponding to [SEG] token
        # this stores the embeddings of all the [SEG] tokens present in all the conversations in the batch
        pred_embeddings = last_hidden_state[seg_token_mask] # [num_seg_tokens_present, 256]
        
                
        # stores how many [SEG] tokens in each conversation in each conversation
        seg_token_counts = seg_token_mask.int().sum(-1)  # shape: [num_conversations] # e.g.: [1,2,1,1,3,1,...]
                
        seg_token_offset_ = seg_token_counts.cumsum(-1) 
        seg_token_offset_ = torch.cat([torch.zeros(1).long().cuda(), seg_token_offset_], dim=0) 
        seg_token_offset = seg_token_offset_[offset] # [batch_size+1]
        
        pred_embeddings_ = []
        for i in range(len(seg_token_offset) - 1): # for i in batch_size (i.e. for each sample in the batch)
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_.append(pred_embeddings[start_i:end_i]) # append a tensor of shape: [num_seg_tokens_per_sample, 256]
        pred_embeddings = pred_embeddings_ # [batch_size, num_seg_tokens_per_sample, 256]
        
                
        ### 5 - Generate and post-process masks
        pred_masks = []
        for batch_idx in range(len(pred_embeddings)): # for batch_idx in batch_size (i.e. for each sample in the batch)
            
            # Apply SAM prompt encoder
            if USE_SAM1:
                (sparse_embeddings,dense_embeddings,) = self.model.visual_model.prompt_encoder(
                                                            points=None, boxes=None, masks=None, 
                                                            text_embeds=pred_embeddings[batch_idx].unsqueeze(1),
                                                        ) 
                                    # text_embeds passed in have the shape: [num_seg_tokens_per_sample, 1, 256]
                                    # sparse_embeddings: [num_seg_tokens_per_sample,1,256], 
                                    # dense_embeddings: [num_seg_tokens_per_sample,256,64,64]
            elif USE_SAM2:
                (sparse_embeddings,dense_embeddings,) = self.model.visual_model.sam_prompt_encoder(
                                                            points=None,boxes=None,masks=None,
                                                            text_embeds=pred_embeddings[batch_idx].unsqueeze(1),
                                                        )
                        
            ### 
            num_seg_tokens_per_sample = len(pred_embeddings[batch_idx])
            # max_num_seg_tokens_per_sample = max([len(_) for _ in pred_embeddings])
            max_num_seg_tokens_per_sample = MAX_NUM_SEG_TOKENS_PER_SAMPLE
            
            sparse_embeddings = sparse_embeddings.to(pred_embeddings[batch_idx].dtype) # [num_seg_tokens_per_sample,1,256]
            
            # masks_list_b = masks_list[batch_idx] # [num_seg_tokens_per_sample, T_sam, H, W] if video, [num_seg_tokens_per_sample, H, W] if image
            if num_seg_tokens_per_sample == 0:
                pad_ = torch.zeros(max_num_seg_tokens_per_sample, 1, 256).to(sparse_embeddings.dtype).to(sparse_embeddings.device)
                sparse_embeddings = torch.cat([sparse_embeddings, pad_], dim=0)
                pad_ = torch.zeros(max_num_seg_tokens_per_sample, 256, 64, 64).to(dense_embeddings.dtype).to(dense_embeddings.device)
                dense_embeddings = torch.cat([dense_embeddings, pad_], dim=0)
                # pad_ = MASK_IGNORE_INDEX * torch.ones(max_num_seg_tokens_per_sample, *masks_list_b.shape[1:]).to(masks_list_b.dtype).to(masks_list_b.device)
                # masks_list_b = torch.cat([masks_list_b, pad_], dim=0)
            
            if num_seg_tokens_per_sample < max_num_seg_tokens_per_sample:
                pad_ = torch.zeros(max_num_seg_tokens_per_sample - num_seg_tokens_per_sample, 1, 256).to(sparse_embeddings.dtype).to(sparse_embeddings.device)
                sparse_embeddings = torch.cat([sparse_embeddings, pad_], dim=0)
                pad_ = torch.zeros(max_num_seg_tokens_per_sample - num_seg_tokens_per_sample, 256, 64, 64).to(dense_embeddings.dtype).to(dense_embeddings.device)
                dense_embeddings = torch.cat([dense_embeddings, pad_], dim=0)
                # pad_ = MASK_IGNORE_INDEX * torch.ones(max_num_seg_tokens_per_sample - num_seg_tokens_per_sample, *masks_list_b.shape[1:]).to(masks_list_b.dtype).to(masks_list_b.device)
                # masks_list_b = torch.cat([masks_list_b, pad_], dim=0)
            # new_masks_list.append(masks_list_b) # append [num_seg_tokens_per_sample, T_sam, H, W] if video, [num_seg_tokens_per_sample, H, W] if image
            ###
            
            ##
            
            if USE_SAM1:
                image_embeddings_for_sam_i = image_embeddings_for_sam[batch_idx] # [T_sam, 1, 256, 64, 64] if video or [1, 1, 256, 64, 64] if image
                t = len(image_embeddings_for_sam_i) # T_sam
            elif USE_SAM2:
                t = len(_features_all[batch_idx])
                
            pred_masks_ti = []
            track_token = None
            # num_seg_tokens_per_sample = len(pred_embeddings[batch_idx])
            # track_token = torch.ones(num_seg_tokens_per_sample, 4, 256).cuda() if self.model.config.mask_decoder_itm else None
            for ti in range(t):
                
                # Apply SAM mask decoder
                if USE_SAM1:
                    low_res_masks, iou_predictions, track_token = self.model.visual_model.mask_decoder(
                        image_embeddings=image_embeddings_for_sam_i[ti], # [1, 256, 64, 64]
                        image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings, # [num_seg_tokens_per_sample,1,256]
                        dense_prompt_embeddings=dense_embeddings, # [num_seg_tokens_per_sample,256,64,64]
                        multimask_output=False,
                        track_token_in=track_token,
                    ) # low_res_masks: [num_seg_tokens_per_sample, 1, 256, 256]
                    
                    pred_mask = self.model.visual_model.postprocess_masks(
                            low_res_masks,                     # Batched masks from the mask_decoder in BxCxHxW format.
                            input_size=resize_list[batch_idx],         # The size of the image input to the model, in (H, W) format
                            original_size=label_list[batch_idx].shape, # The original size of the image before resizing for input to the model, in (H, W) format
                        ) # Returns: (torch.Tensor): Batched masks in BxCxHxW format, where (H, W) is given by original_size.
                        # pred_mask: [num_seg_tokens_per_sample, 1, H, W]
                        
                elif USE_SAM2:
                    
                    if sparse_embeddings.shape[0] == 0 or dense_embeddings.shape[0] == 0: # if no SEG tokens are present
                        h_,w_ = label_list[batch_idx].shape
                        pred_mask = torch.zeros(0, 1, h_, w_).cuda()
                        
                    else:
                        _features = _features_all[batch_idx][ti]
                        low_res_masks, iou_predictions, _, _ = self.model.visual_model.sam_mask_decoder(
                            image_embeddings=_features["image_embed"], # [1, 256, 64, 64]
                            image_pe=self.model.visual_model.sam_prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=sparse_embeddings,  # [num_seg_tokens_per_sample,1,256]
                            dense_prompt_embeddings=dense_embeddings, # [num_seg_tokens_per_sample,256,64,64]
                            multimask_output=False,
                            repeat_image = True,
                            # high_res_features=[feat_level for feat_level in _features["high_res_feats"]] # [2, 1, 32, 256, 256]
                            high_res_features=_features["high_res_feats"] # [2, 1, 32, 256, 256]
                        )
                        pred_mask = self.model.postprocess_masks(
                            low_res_masks,
                            orig_hw=label_list[batch_idx].shape,
                        )
                
                pred_masks_ti.append(pred_mask[:, 0]) # append tensor of shape : [num_seg_tokens_per_sample, H, W]
            pred_masks.append(pred_masks_ti) 
            # pred_masks will have dimensions: (batch_size x T_sam x [num_seg_tokens_per_sample, H, W])

        ###
        
        model_output = output
        gt_masks = masks_list # [batch_size, num_seg_tokens_per_sample, T_sam, H, W] if video, [batch_size, num_seg_tokens_per_sample, H, W] if image
        # gt_masks = new_masks_list # [batch_size, num_seg_tokens_per_sample, T_sam, H, W] if video, [batch_size, num_seg_tokens_per_sample, H, W] if image
        
        if inference:
            return {
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
            }

        ### 6 - Calculate losses
        output = model_output.logits

        # Text generation losss (Cross Entropy Loss)
        ce_loss = model_output.loss
        ce_loss = ce_loss * self.ce_loss_weight
        
        # Mask BCE & DICE losses
        mask_bce_loss = torch.tensor(0.0, device=ce_loss.device, dtype=ce_loss.dtype)
        mask_dice_loss = torch.tensor(0.0, device=ce_loss.device, dtype=ce_loss.dtype)
        
        num_masks = 0
        for batch_idx in range(len(pred_masks)): # for batch_idx in batch_size: (i.e. for each sample in the batch)
            
            gt_mask = gt_masks[batch_idx]       # [num_seg_tokens_per_sample, T_sam, H, W] if video, [num_seg_tokens_per_sample, H, W] if image
            pred_mask = pred_masks[batch_idx]   # (T_sam x [num_seg_tokens_per_sample, H, W]) if video, [num_seg_tokens_per_sample, H, W] if image
            
            gt_mask = [gt_mask[:,tn,:,:]  for tn in range(gt_mask.shape[1]) ]# [num_conversations, T_sam, H, W] -> # T_sam x [num_conversations, H, W]
            t = len(pred_mask)
            assert ( len(gt_mask) == len(pred_mask)), "len(gt_mask): {}, len(pred_mask): {}".format(len(gt_mask), len(pred_mask))
            for ti in range(t):
                pred_mask_i = pred_mask[ti]
                gt_mask_i   = gt_mask[ti] # [num_seg_tokens_per_sample, H, W]
                min_len = min(gt_mask_i.shape[0], pred_mask_i.shape[0])
                gt_mask_i = gt_mask_i[:min_len]
                pred_mask_i = pred_mask_i[:min_len]
                assert ( gt_mask_i.shape[0] == pred_mask_i.shape[0]), "gt_mask_i.shape: {}, pred_mask_i.shape: {}".format(gt_mask_i.shape, pred_mask_i.shape)
                mask_bce_loss += ( sigmoid_ce_loss(pred_mask_i, gt_mask_i, num_masks=gt_mask_i.shape[0], ignore_index=MASK_IGNORE_INDEX) * gt_mask_i.shape[0])
                mask_dice_loss += ( dice_loss(pred_mask_i, gt_mask_i, num_masks=gt_mask_i.shape[0], ignore_index=MASK_IGNORE_INDEX) * gt_mask_i.shape[0])
                num_masks += gt_mask_i.shape[0]
        
        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss

        loss = ce_loss + mask_loss
        
        # print_dimensions('gt_masks', gt_masks) # shape: [batch_size, num_seg_tokens_per_sample, T_sam, H, W] if video, [batch_size, num_seg_tokens_per_sample, H, W] if image
        # print_dimensions('pred_masks', pred_masks) # shape: [batch_size, T_sam, num_seg_tokens_per_sample, H, W] if video, [batch_size, num_seg_tokens_per_sample, H, W] if image
        
        # print('ce_loss:', ce_loss, '    mask_bce_loss:', mask_bce_loss, '   mask_dice_loss:', mask_dice_loss, ' mask_loss:', mask_loss)
        
        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
        }

    def inference(
        self,
        images,
        context_images,
        images_for_sam,
        input_ids,
        resize_list,
        original_size_list,
        max_new_tokens=32,
        use_sam2_video_branch=False,
    ):

        if use_sam2_video_branch:
            if self.config.use_sam2:
                print('\033[92m---Inference with video branch---\033[0m')
                return self.inference_video_branch(
                    images,
                    context_images,
                    images_for_sam,
                    input_ids,
                    resize_list,
                    original_size_list,
                    max_new_tokens,
                )
            else:
                raise ValueError("use_sam2_video_branch is True, but model is not configured to use SAM2")
        else:
            print('\033[92m---Inference without video branch---\033[0m')
            return self.inference_framewise(
                images,
                context_images,
                images_for_sam,
                input_ids,
                resize_list,
                original_size_list,
                max_new_tokens,
            )
            
    def inference_framewise(
        self,
        images,
        context_images,
        images_for_sam,
        input_ids,
        resize_list,
        original_size_list,
        max_new_tokens=32,
    ):
        
        ### Find the number of newly added tokens
        with torch.no_grad():
            _, _, seg_token_mask = self.__inference_path(input_ids, images, context_images, None)
            
        num_newly_added_tokens = (seg_token_mask.shape[1] - input_ids.shape[1]) # 111 or 255
        
        
        with torch.no_grad():
            outputs = self.generate(
                images=images,
                context_images=context_images,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
                use_cache=False,
            )
            output_hidden_states = outputs.hidden_states
            output_ids = outputs.sequences

            seg_token_mask = output_ids[:, 1:] == self.config.seg_token_idx
            # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
            # seg_token_mask = torch.cat([torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(),seg_token_mask], dim=1)
            seg_token_mask = torch.cat([torch.zeros((seg_token_mask.shape[0], num_newly_added_tokens)).bool().cuda(),seg_token_mask], dim=1)
            
            # seg_token_mask = outputs.seg_token_mask
            hidden_states = []

            assert len(self.model.text_hidden_fcs) == 1

            hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1])) #(33, 1, 1, 4096) -> (33, 1, 1, 256)

            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1) #(33, 1, 1, 256)
            pred_embeddings = last_hidden_state[seg_token_mask] # [??, 256]
            
            seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
            seg_token_offset = seg_token_counts.cumsum(-1)
            seg_token_offset = torch.cat(
                [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
            )

            pred_embeddings_ = []
            for i in range(len(seg_token_offset) - 1):
                start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                pred_embeddings_.append(pred_embeddings[start_i:end_i])
            pred_embeddings = pred_embeddings_
            
            # print_dimensions('**pred_embeddings_1', pred_embeddings) #[1, 0, 256] # [batch_size, num_seg_tokens_per_sample, 256]

            if self.config.use_sam2:
                USE_SAM2 = True
                USE_SAM1 = False
            else:
                USE_SAM2 = False
                USE_SAM1 = True
                                
            if USE_SAM1:
                image_embeddings_for_sam = self.get_visual_embs(images_for_sam) # get SAM embeddings
                batch_size = len(image_embeddings_for_sam)
            elif USE_SAM2:
                batch_size, _features_all = self.get_visual_embs_sam2(images_for_sam) # get SAM2 embeddings
                            
            # print_dimensions('**image_embeddings_for_sam', image_embeddings_for_sam) #  [1, 1, 256, 64, 64]
            # [batch, T_sam, 1, 256, 64, 64] if video or [batch, 1, 256, 64, 64] if image
            
            
            pred_masks_batch = []
            video_segments_batch = []
            for batch_idx in range(len(pred_embeddings)): # for batch_idx in batch_size (i.e. for each sample in the batch)
                
                # Apply SAM prompt encoder
                if USE_SAM1:
                    (sparse_embeddings,dense_embeddings,) = self.model.visual_model.prompt_encoder(
                                                                points=None, boxes=None, masks=None, 
                                                                text_embeds=pred_embeddings[batch_idx].unsqueeze(1),
                                                            ) 
                                        # text_embeds passed in have the shape: [num_seg_tokens_per_sample, 1, 256]
                                        # sparse_embeddings: [num_seg_tokens_per_sample,1,256], 
                                        # dense_embeddings: [num_seg_tokens_per_sample,256,64,64]
                elif USE_SAM2:
                    (sparse_embeddings,dense_embeddings,) = self.model.visual_model.sam_prompt_encoder(
                                                                points=None,boxes=None,masks=None,
                                                                text_embeds=pred_embeddings[batch_idx].unsqueeze(1),
                                                            )
                    
                # (sparse_embeddings,dense_embeddings,) = self.model.visual_model.prompt_encoder(points=None, boxes=None, masks=None, 
                #                                         text_embeds=pred_embeddings[batch_idx].unsqueeze(1),) # text_embeds passed in have the shape: [num_seg_tokens_per_sample, 1, 256]
                # sparse_embeddings: [num_seg_tokens_per_sample,1,256], dense_embeddings: [num_seg_tokens_per_sample,256,64,64]

                sparse_embeddings = sparse_embeddings.to(pred_embeddings[batch_idx].dtype)
                
                if USE_SAM1:
                    image_embeddings_for_sam_i = image_embeddings_for_sam[batch_idx] # [T_sam, 1, 256, 64, 64] if video or [1, 1, 256, 64, 64] if image
                    t = len(image_embeddings_for_sam_i) # T_sam
                elif USE_SAM2:
                    t = len(_features_all[batch_idx])
                
                
                pred_masks_ti = []
                track_token = None
                for ti in range(t):
                    
                    # Apply SAM mask decoder
                    if USE_SAM1:
                        low_res_masks, iou_predictions, track_token = self.model.visual_model.mask_decoder(
                            image_embeddings=image_embeddings_for_sam_i[ti], # [1, 256, 64, 64]
                            image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=sparse_embeddings, # [num_seg_tokens_per_sample,1,256]
                            dense_prompt_embeddings=dense_embeddings, # [num_seg_tokens_per_sample,256,64,64]
                            multimask_output=False,
                            track_token_in=track_token,
                        ) # low_res_masks: [num_seg_tokens_per_sample, 1, 256, 256]
                        pred_mask = self.model.visual_model.postprocess_masks(
                                low_res_masks,                     # Batched masks from the mask_decoder in BxCxHxW format.
                                input_size=resize_list[batch_idx],         # The size of the image input to the model, in (H, W) format
                                original_size=original_size_list[batch_idx], # The original size of the image before resizing for input to the model, in (H, W) format
                            ) # Returns: (torch.Tensor): Batched masks in BxCxHxW format, where (H, W) is given by original_size.
                            # pred_mask: [num_seg_tokens_per_sample, 1, H, W]
                            
                    elif USE_SAM2:
                        
                        if sparse_embeddings.shape[0] == 0 or dense_embeddings.shape[0] == 0: # if no SEG tokens are present
                            h_,w_ = original_size_list[batch_idx].shape
                            pred_mask = torch.zeros(0, 1, h_, w_).cuda()
                            
                        else:
                            _features = _features_all[batch_idx][ti]
                            low_res_masks, iou_predictions, _, _ = self.model.visual_model.sam_mask_decoder(
                                image_embeddings=_features["image_embed"], # [1, 256, 64, 64]
                                image_pe=self.model.visual_model.sam_prompt_encoder.get_dense_pe(),
                                sparse_prompt_embeddings=sparse_embeddings,  # [num_seg_tokens_per_sample,1,256]
                                dense_prompt_embeddings=dense_embeddings, # [num_seg_tokens_per_sample,256,64,64]
                                multimask_output=False,
                                repeat_image = True,
                                high_res_features=_features["high_res_feats"] # [2, 1, 32, 256, 256]
                            )
                            pred_mask = self.model.postprocess_masks(
                                low_res_masks,
                                orig_hw=original_size_list[batch_idx],
                            )
                    
                    
                    pred_masks_ti.append(pred_mask[:, 0]) # append tensor of shape : [num_seg_tokens_per_sample, H, W]

                pred_masks_batch.append(pred_masks_ti) 
                # pred_masks_batch will have dimensions: (batch_size x T_sam x [num_seg_tokens_per_sample, H, W])

                # create dictionary of video segments
                video_segments = {} # keys of video_segments are timestamps # keys of video_segments[t] are object IDs
                pred_mask_of_video = pred_masks_ti # tensor of shape : tx[num_seg_tokens_per_sample, H, W]
                for t in range(len(pred_mask_of_video)):
                    pred_mask_of_frame = pred_mask_of_video[t] # tensor of shape : [num_seg_tokens_per_sample, H, W]
                    pred_mask_of_frame = pred_mask_of_frame.detach().cpu().numpy()
                    pred_mask_of_frame = pred_mask_of_frame > 0
                    for seg_token_id in range(len(pred_mask_of_frame)):
                        video_segments.setdefault(t, {})[seg_token_id] = pred_mask_of_frame[seg_token_id]
                video_segments_batch.append(video_segments)
                    
        return output_ids, video_segments_batch

    def inference_video_branch(
        self,
        images,
        context_images,
        images_for_sam,
        input_ids,
        resize_list,
        original_size_list,
        max_new_tokens=32,
    ):
        
        ### Find the number of newly added tokens
        with torch.no_grad():
            _, _, seg_token_mask = self.__inference_path(input_ids, images, context_images, None)
            
        
        num_newly_added_tokens = (seg_token_mask.shape[1] - input_ids.shape[1]) # 111 or 255
        
        
        with torch.no_grad():
            outputs = self.generate(
                images=images,
                context_images=context_images,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
                use_cache=False,
            )
            output_hidden_states = outputs.hidden_states
            output_ids = outputs.sequences

            seg_token_mask = output_ids[:, 1:] == self.config.seg_token_idx
            # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
            # seg_token_mask = torch.cat([torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(),seg_token_mask], dim=1)
            seg_token_mask = torch.cat([torch.zeros((seg_token_mask.shape[0], num_newly_added_tokens)).bool().cuda(),seg_token_mask], dim=1)
            
            # seg_token_mask = outputs.seg_token_mask
            hidden_states = []

            assert len(self.model.text_hidden_fcs) == 1

            hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1])) #(33, 1, 1, 4096) -> (33, 1, 1, 256)

            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1) #(33, 1, 1, 256)
            # print_dimensions('**last_hidden_state', last_hidden_state) # [1, 166, 256]
            # print_dimensions('**seg_token_mask', seg_token_mask) # [1, 166]
            pred_embeddings = last_hidden_state[seg_token_mask] # [??, 256]
            

            seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
            seg_token_offset = seg_token_counts.cumsum(-1)
            seg_token_offset = torch.cat(
                [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
            )

            pred_embeddings_ = []
            for i in range(len(seg_token_offset) - 1):
                start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                pred_embeddings_.append(pred_embeddings[start_i:end_i])
            pred_embeddings = pred_embeddings_
                        
            video_segments_batch = []
            for batch_idx in range(len(pred_embeddings)): # for idx in batch_size (i.e. for each sample in the batch)
                
                feat = pred_embeddings[batch_idx] # [num_seg_tokens_per_sample, 256]
                num_seg_tokens_per_sample = len(feat)
                feat = feat.unsqueeze(1) # [num_seg_tokens_per_sample, 1, 256]
                
                if num_seg_tokens_per_sample==0:
                    video_segments_batch.append({})
                    continue
                
                # SAM2 video model
                video_height, video_width = original_size_list[batch_idx]
                inference_state = self.model.visual_model.init_state_from_tensor(images_for_sam[batch_idx], video_height, video_width)
                self.model.visual_model.reset_state(inference_state)

                # begin video segmentation
                ann_frame_idx = 0  # the frame index we interact with
                
                # ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
                # _, out_obj_ids, out_mask_logits = self.model.visual_model.add_new_text(
                #     inference_state=inference_state,
                #     frame_idx=ann_frame_idx,
                #     obj_id=ann_obj_id,
                #     text=feat
                # )
                
                for ann_obj_id in range(0, feat.shape[0]):
                    _, out_obj_ids, out_mask_logits = self.model.visual_model.add_new_text(
                        inference_state=inference_state,
                        frame_idx=ann_frame_idx,
                        obj_id=ann_obj_id,
                        text=feat[ann_obj_id].unsqueeze(0)
                    )
                
                # run propagation throughout the video and collect the results in a dict
                video_segments = {}  # video_segments contains the per-frame segmentation results
                for out_frame_idx, out_obj_ids, out_mask_logits in self.model.visual_model.propagate_in_video(inference_state):
                    video_segments[out_frame_idx] = {
                        # out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()[0] # select only one mask per object
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }
                # video_segments is supposed to contain the per-frame segmentation results per each seg-token in the sample
                video_segments_batch.append(video_segments)
                
        return output_ids, video_segments_batch


class VideoGLaMMForCausalLM(VideoGPTPlusPhi3ForCausalLM, VideoGLaMM_SAM2):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
        self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
        self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
        ##
        super().__init__(config)
        self.model = VideoGLaMMModel(config, **kwargs)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
    
    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return VideoGPTPlusPhi3ForCausalLM.forward(self, **kwargs)
        return VideoGLaMM_SAM2.model_forward(self, **kwargs)
    
    def super_forward(self, **kwargs):
        return VideoGPTPlusPhi3ForCausalLM.forward(self, **kwargs)

