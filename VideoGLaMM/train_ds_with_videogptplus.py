import argparse
import os
import sys
from functools import partial


import deepspeed
import numpy as np
import torch
import transformers
from peft import LoraConfig, get_peft_model

from VideoGLaMM.model.VideoGLaMM import VideoGLaMMForCausalLM

from utils.dataset import HybridDataset, ValDataset, collate_fn

from utils.trainer import LISATrainer, LISAValidator, get_ds_config

from dataclasses import dataclass, field
from transformers import HfArgumentParser
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


from model.videogpt_plus.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN


@dataclass
class AllArguments:
    ############ ModelArguments ######
    videogptplus_path:str = field(default="/home/shehan/workspace_grounding_lmm/VideoGPT-plus/MBZUAI/VideoGPT-plus_Phi3-mini-4k/mvbench")
    base_llm_path:str = field(default="microsoft/Phi-3-mini-4k-instruct")
    base_type:str=field(default="phi3") # phi3|llama3_1
    pretrain_mm_mlp_adapter:str=field(default="")
    pretrain_image_mm_mlp_adapter:str=field(default="")
    precision:str=field(default="fp16",metadata={"help": "fp32|fp16|bf16"})
    image_size:int=field(default=1024)
    model_max_length:int=field(default=2048)
    vision_tower:str=field(default="OpenGVLab/InternVideo2-Stage2_1B-224p-f4/InternVideo2-stage2_1b-224p-f4.pt")
    image_vision_tower:str=field(default="openai/clip-vit-large-patch14-336")
    load_in_8bit:bool=field(default=False)
    load_in_4bit:bool=field(default=False)
    
    sam_pretrained_path:str=field(default="./checkpoints/sam_vit_h_4b8939.pth")
    out_dim:int=field(default=256)
    
    # use_mm_start_end:bool=field(default=True)
    
    tune_mm_mlp_adapter:bool=field(default=False)
    train_mask_decoder:bool=field(default=True)
    use_sam_version:str=field(default="v1", metadata={"help": "v1|v1_itm|v2"})

    ############ DataArguments ######
    dataset:str=field(default="sem_seg||refer_seg||vqa||reason_seg")    
    # sample_rates_for_datasets:str=field(default="9,3,3,1")
    sample_rates_for_datasets:str=field(default="")
    # dataset="refer_seg||refer_vos",
    # dataset="sem_seg||refer_seg||vqa||reason_seg||refer_vos||video_vqa||mevis||temporal||vidstg",
    sem_seg_data:str=field(default="ade20k||cocostuff||pascal_part||paco_lvis||mapillary")
    refer_seg_data:str=field(default="refclef||refcoco||refcoco+||refcocog")
    vqa_data:str=field(default="llava_instruct_150k")
    reason_seg_data:str=field(default="ReasonSeg|train")
    reason_seg_explanatory:float=field(default=0.1)
    # refer_vos_data="ytvos||davis17||a2d||jhmdb", #NOTE
    refer_vos_data:str=field(default="ytvos||davis17")
    video_vqa_data:str=field(default="video_instruct_100k")
    video_tg_data:str=field(default="charades||anetcaps||qvh")
    
    val_dataset:str=field(default="ReasonSeg|val") #"MEVIS|valid" # "MEVIS|valid_u"
    
    num_frames_for_sam:int=field(default=1) #number of frames/masks per video, considered during training
    
    dataset_dir:str=field(default="./dataset")
    video_dataset_dir:str=field(default="./video_dataset/")
    
    num_classes_per_sample:int=field(default=3)

    ############ TrainingArguments ######
    local_rank:int=field(default=0)
    lora_r:int=field(default=8)
    
    vis_save_path:str = field(default="./vis_output")
    
    epochs:int=field(default=10)
    steps_per_epoch:int=field(default=500)
    batch_size:int=field(default=2)
    grad_accumulation_steps:int=field(default=10)
    val_batch_size:int=field(default=1)
    
    workers:int=field(default=4)
    lr:float=field(default=0.0003)
    ce_loss_weight:float=field(default=1.0)
    dice_loss_weight:float=field(default=0.5)
    bce_loss_weight:float=field(default=2.0)

    lora_alpha:int=field(default=16)
    lora_dropout:float=field(default=0.05)
    lora_target_modules:str=field(default="q_proj,v_proj")
    
    
    beta1:float=field(default=0.9)
    beta2:float=field(default=0.95)
    
    eval_only:bool=field(default=False)
    eval_num_frames:int=field(default=-1)
    
    gradient_checkpointing:bool=field(default=True)
    
    resume_dir:str=field(default="")
    auto_resume:bool=field(default=False)
    start_epoch:int=field(default=0)
    print_freq:int=field(default=1)
    
    exp_name:str=field(default='dev_train')
    
    logs_base_dir:str=field(default='/opt/ml/output/tensorboard/')
    ckpt_base_dir:str=field(default='/opt/ml/checkpoints/')
    
    ############ Other ######
    save_hf_model:bool=field(default=False)
    intermediate_weight:str=field(default="")
    hf_save_path:str=field(default="./hf_model")

if __name__=='__main__':

    parser = HfArgumentParser(AllArguments)
        
    args, unknown_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)
            
    
    ### Load model
    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "mask_decoder_itm": (True if args.use_sam_version=="v1_itm" else False),
        "use_sam2": (True if args.use_sam_version=="v2" else False),
        "sam_pretrained_path": args.sam_pretrained_path,
    }
    if not (args.eval_only or args.save_hf_model):
        model_args.update({
            "ce_loss_weight": args.ce_loss_weight,
            "dice_loss_weight": args.dice_loss_weight,
            "bce_loss_weight": args.bce_loss_weight,
        })
    torch_dtype = torch.bfloat16 if args.precision == "bf16" else (torch.half if args.precision == "fp16" else torch.float32)
    
    def load_videogptplus_model_from_pretrained(model_path, model_base, base_type="phi3"):
        
        ## Load model
        lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
        
        print('Loading VideoGPT+ from base model...')
        if base_type=="phi3":
            model = VideoGLaMMForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=False, config=lora_cfg_pretrained, torch_dtype=torch_dtype,
                                                            **model_args)
        else:
            raise ValueError("Invalid base_type. Should be either phi3 or llama3_1")
        token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
        if model.lm_head.weight.shape[0] != token_num:
            model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
            model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

        print('Loading additional VideoGPT+ weights...')
        if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
            non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
        else:
            raise FileNotFoundError("Lora is specified in the model path, however could not find non-LORA trainables weights.")

        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
        model.load_state_dict(non_lora_trainables, strict=False)

        from peft import PeftModel
        print('Loading LoRA weights...')
        model = PeftModel.from_pretrained(model, model_path)
        print('Merging LoRA weights...')
        model = model.merge_and_unload()
        print('Model is loaded...')
        
        ## Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
        
        ##
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", False)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        
        tokenizer.pad_token = tokenizer.unk_token
        num_added_tokens = tokenizer.add_tokens("[SEG]")
        seg_token_idx = tokenizer.convert_tokens_to_ids("[SEG]")
        print("seg_token_idx: ", seg_token_idx)
        model.resize_token_embeddings(len(tokenizer))
        
        if hasattr(model.config, "max_sequence_length"):
            context_len = model.config.max_sequence_length
        else:
            context_len = 2048
            
        # set seg_token_idx in model config
        model.config.seg_token_idx = seg_token_idx
        
        # for llama3_1
        if tokenizer.pad_token_id == None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            
        return model, tokenizer, context_len, seg_token_idx, mm_use_im_start_end, mm_use_im_patch_token
    
    
    # 
    if args.base_type=="phi3":
        model, tokenizer, context_len, seg_token_idx, use_mm_start_end, mm_use_im_patch_token = load_videogptplus_model_from_pretrained(args.videogptplus_path, args.base_llm_path, 'phi3')
    else:
        raise ValueError("Invalid base_type")
    
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # If training, enable gradient checkpointing
    if not (args.eval_only or args.save_hf_model):
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        
    # conversation_generator
    from utils.conv_generator import ConvGenerator_VideoGPTPlus
    conversation_generator = ConvGenerator_VideoGPTPlus(use_mm_start_end=use_mm_start_end, base_type=args.base_type)
    
    
    # enc_preprocessor
    from utils.enc_preprocessors import EncPreprocessor_VideoGPTPlus
    enc_preprocessor = EncPreprocessor_VideoGPTPlus()
    
    ### Initialize encoder modules
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)
    image_vision_tower = model.get_model().get_image_vision_tower()
    image_vision_tower.to(dtype=torch_dtype, device=args.local_rank)

    # Initialize grounding modules from pretrained-SAM, during training. Otherwise, it will be initialised from the saved LISA-Huggingface weights
    load_sam_from_original= not args.eval_only
    if not (args.eval_only or args.save_hf_model): # if training
        model.get_model().initialize_lisa_modules(model.get_model().config)
    # If training, make the vision tower frozen and the projector trainable, as needed
    if not (args.eval_only or args.save_hf_model):
        # Make CLIP vision encoder frozen
        for p in vision_tower.parameters():
            p.requires_grad = False
        for p in image_vision_tower.parameters(): #NOTE
            p.requires_grad = False
        # Make projectors frozen
        for p in model.get_model().mm_projector.parameters():
            if args.tune_mm_mlp_adapter:
                p.requires_grad = True
            else:
                p.requires_grad = False
        for p in model.get_model().image_mm_projector.parameters(): #NOTE
            if args.tune_mm_mlp_adapter:
                p.requires_grad = True
            else:
                p.requires_grad = False
            
        if args.tune_mm_mlp_adapter:
            model.initialize_vision_tokenizer(
                mm_use_im_patch_token = mm_use_im_patch_token,
                mm_use_im_start_end = use_mm_start_end,
                tune_mm_mlp_adapter = args.tune_mm_mlp_adapter,
                pretrain_mm_mlp_adapter = args.pretrain_mm_mlp_adapter,
                tokenizer=tokenizer
            )

    
    ### Setup LoRA settings
    def get_model_with_lora(model, lora_r):
        def find_linear_layers(model, lora_target_modules):
            '''
            - lora_target_modules: default=["q_proj","v_proj"]
            '''
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (isinstance(module, cls) and 
                    all([ x not in name for x in [
                                                    "visual_model", # ignore grounding model (encoder-decoder)
                                                    "vision_tower", "image_vision_tower", # ignore vision tower
                                                    "mm_projector", "image_mm_projector", # ignore projector
                                                    "text_hidden_fcs", # ignore output projection
                                                    ]])
                    and any([x in name for x in lora_target_modules]) # include all target modules (q_proj,v_proj)
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_target_modules = args.lora_target_modules.split(",")
        lora_target_modules = find_linear_layers(
            model, lora_target_modules
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        return model
    
    lora_r = args.lora_r
    if lora_r > 0:
        model = get_model_with_lora(model, lora_r)

    # Save model and tokenizer in Huggingface Format
    if args.save_hf_model:
        
        state_dict = torch.load(args.intermediate_weight, map_location="cpu")
        
        model.load_state_dict(state_dict, strict=False)

        model = model.merge_and_unload()
        state_dict = {}
        for k, v in model.state_dict().items():
            if "vision_tower" not in k or "image_vision_tower" not in k:
                state_dict[k] = v
                
        # Fix the generation config based on the warning
        # if not model.generation_config.do_sample:
        #     model.generation_config.temperature = None  # Unset temperature
        #     model.generation_config.top_p = None        # Unset top_p

        # Alternatively, if you do want to sample, set do_sample=True
        model.generation_config.do_sample = True
        
        model.save_pretrained(args.hf_save_path, state_dict=state_dict, safe_serialization=False)
        tokenizer.save_pretrained(args.hf_save_path)
        print("Saved model in Huggingface format at: ", args.hf_save_path)
        sys.exit(0)
        
    ### make text_hidden_fcs, mask_decoder, lm_head, embed_tokens trainable
    for n, p in model.named_parameters():
        if any([x in n for x in ["lm_head", "embed_tokens", "text_hidden_fcs"]]):
            print("n: ", n, "p.shape: ", p.shape)
            p.requires_grad = True
        if "mask_decoder" in n:
            print("n: ", n, "p.shape: ", p.shape)
            if args.train_mask_decoder:
                p.requires_grad = True
            else:
                p.requires_grad = False
                
    ### SAM preprocessor 
    if args.use_sam_version=="v1" or args.use_sam_version=="v1_itm":
        from utils.sam_transforms import SAM_v1_Preprocess
        sam_preprocessor = SAM_v1_Preprocess()
    elif args.use_sam_version=="v2":
        from utils.sam_transforms import SAM_v2_Preprocess
        sam_preprocessor = SAM_v2_Preprocess()

    ### Load train and validation datasets
    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1
    
    dataset_names_list = args.dataset.split("||")
    if args.sample_rates_for_datasets=="": #default
        sample_rates = [1.0] * len(dataset_names_list)
    else:
        sample_rates = [float(x) for x in args.sample_rates_for_datasets.split(",")]
    assert len(dataset_names_list) == len(sample_rates), "Number of datasets and sample rates should be equal."
    num_samples_per_epoch=args.batch_size * args.grad_accumulation_steps * args.steps_per_epoch * world_size
    

    # Training   
    if not args.eval_only:
        
        # Train dataset
        train_dataset = HybridDataset(
            base_image_dir = args.dataset_dir,
            base_video_dir = args.video_dataset_dir,

            enc_preprocessor = enc_preprocessor, 
            sam_preprocessor = sam_preprocessor,
            conversation_generator = conversation_generator,
                        
            num_samples_per_epoch=num_samples_per_epoch,
            num_classes_per_sample=args.num_classes_per_sample,
            
            dataset=dataset_names_list,
            sample_rate=sample_rates,
            sem_seg_data=args.sem_seg_data,
            refer_seg_data=args.refer_seg_data,
            vqa_data=args.vqa_data,
            reason_seg_data=args.reason_seg_data,
            refer_vos_data=args.refer_vos_data,
            video_vqa_data=args.video_vqa_data,
            video_tg_data=args.video_tg_data,
            
            num_frames_for_sam=args.num_frames_for_sam,
            reason_seg_explanatory=args.reason_seg_explanatory,
        )

        # DeepSpeed 
        print("Initializing DeepSpeed")
        model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            training_data=train_dataset,
            collate_fn=partial(
                collate_fn,
                tokenizer=tokenizer,
                local_rank=args.local_rank,
                conversation_generator=conversation_generator
            ),
            config=get_ds_config(args),
        )
        
        # Trainer
        print("Initializing Trainer")
        trainer = LISATrainer(model_engine, train_loader, scheduler, args, exp_name=args.exp_name)

        # Resume training
        if args.auto_resume:
            if os.path.exists(os.path.join(trainer.ckpt_save_dir, "latest")):
                trainer.resume(trainer.ckpt_save_dir)
        elif args.resume_dir:
            trainer.resume(args.resume_dir)
        
        # Start training
        print("Starting training")
        trainer.train(args.epochs)

    # Validation
    elif args.eval_only:
        validator = LISAValidator(model_engine=model_engine, args=args, exp_name=args.exp_name)
        if args.val_dataset=="ReasonSeg|val":      
                  
            val_dataset = ValDataset(
                args.dataset_dir,
                tokenizer,
                args.vision_tower,
                args.val_dataset,
                args.image_size,
            )
            print(f"Validating with {len(val_dataset)} examples.")
            
            assert args.val_batch_size == 1
            val_sampler = torch.utils.data.distributed.DistributedSampler( val_dataset, shuffle=False, drop_last=False)
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=args.val_batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=False,
                sampler=val_sampler,
                collate_fn=partial(
                    collate_fn,
                    tokenizer=tokenizer,
                    local_rank=args.local_rank,
                    conversation_generator=conversation_generator
                ),
            )
            validator.validate_on_reasonseg(val_loader=val_loader)
            
        elif args.val_dataset=="MEVIS|valid" or args.val_dataset=="MEVIS|valid_u":    
            video_val_image_set = args.val_dataset.split('|')[-1]#'valid_u' # 'valid'
            from utils.mevis_dataset import MEVISDataset
            video_val_dataset = MEVISDataset(base_video_dataset_dir=args.video_dataset_dir,
                        tokenizer=tokenizer,
                        vision_tower=args.vision_tower,
                        image_size=args.image_size,
                        image_set=video_val_image_set, 
                        num_frames_for_clip=args.eval_num_frames,
                        num_frames_for_sam=-1,
                        debug_mode=False)
            print(f"Validating with {len(video_val_dataset)} examples.")
            
            assert args.val_batch_size == 1
            video_val_sampler = torch.utils.data.distributed.DistributedSampler(
                video_val_dataset, shuffle=False, drop_last=False
            )
            video_val_loader = torch.utils.data.DataLoader(
                video_val_dataset,
                batch_size=args.val_batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=False,
                sampler=video_val_sampler,
                collate_fn=partial(
                    collate_fn,
                    tokenizer=tokenizer,
                    local_rank=args.local_rank,
                    conversation_generator=conversation_generator
                ),
            )
            if video_val_image_set=='valid_u':
                validator.validate_on_mevis(val_loader=video_val_loader, save_masks_for_bench=False)
            else: # 'valid'
                validator.validate_on_mevis(val_loader=video_val_loader, save_masks_for_bench=True)
             