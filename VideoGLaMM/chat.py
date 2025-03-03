import argparse
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

import transformers
from transformers import AutoTokenizer, BitsAndBytesConfig

import matplotlib.pyplot as plt
from decord import VideoReader, cpu
from PIL import Image

DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
DEFAULT_VID_START_TOKEN = "<vid_start>"
DEFAULT_VID_END_TOKEN = "<vid_end>"
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"


def write_masks(video_segments, video_frames_np, save_dir):
    ''' Write masks to disk 
    Args:
    - video_segments: dictionary with keys being frame indices, and values being dictionaries with keys being segment indices
    - video_frames_np: numpy array of video frames # [T, H, W, C]  # numpy array
    '''
    
    # video_segments is a dictionary with keys being frame indices
    # video_segments[t] is a dictionary with keys being segment indices
    
    for t, pred_mask in video_segments.items():
        
        # save image frame
        save_img = video_frames_np[t].copy()
        img_dir = os.path.join(save_dir, "img_frames")
        os.makedirs(img_dir, exist_ok=True)
        img_save_path = os.path.join(img_dir, f"frame_{t}.jpg")
        cv2.imwrite(img_save_path, cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR))        
        
        # save mask for each object
        for obj_id, pred_mask_i in pred_mask.items():
            pred_mask_i = pred_mask_i > 0

            # save mask
            obj_dir = os.path.join(save_dir, f"pred_masks_{obj_id}")
            os.makedirs(obj_dir, exist_ok=True)
            mask_path = os.path.join(obj_dir, f"mask_{t}.png")
            cv2.imwrite(mask_path, pred_mask_i * 255)
            print("{} has been saved.".format(mask_path))
            
            # save masked image frame
            save_path = "{}/masked_images/masked_img_{}_{}.jpg".format(save_dir, t, obj_id)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            save_img = video_frames_np[t].copy()
            save_img[pred_mask_i] = (video_frames_np[t] * 0.5 + pred_mask_i[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5)[pred_mask_i]
            save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, save_img)
            print("{} has been saved.".format(save_path))

def _get_rawvideo_dec(video_path, video_framerate=1, s=None, e=None):
    if s is None:
        start_time, end_time = None, None
    else:
        start_time = int(s)
        end_time = int(e)
        start_time = start_time if start_time >= 0. else 0.
        end_time = end_time if end_time >= 0. else 0.
        if start_time > end_time:
            start_time, end_time = end_time, start_time
        elif start_time == end_time:
            end_time = start_time + 1

    if os.path.exists(video_path):
        vreader = VideoReader(video_path, ctx=cpu(0))
    else:
        print(video_path)
        raise FileNotFoundError

    fps = vreader.get_avg_fps()
    f_start = 0 if start_time is None else int(start_time * fps)
    f_end = int(min(1000000000 if end_time is None else end_time * fps, len(vreader) - 1))
    num_frames = f_end - f_start + 1
    if num_frames > 0:
        # T x 3 x H x W
        sample_fps = int(video_framerate)
        t_stride = int(round(float(fps) / sample_fps))
        sample_pos = list(range(f_start, f_end + 1, t_stride))
        np_images = [f for f in vreader.get_batch(sample_pos).asnumpy()]
    else:
        print("video path: {} error.".format(video_path))

    return np_images

#  get args from cli
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llava_version_or_path", type=str, default="/home/shehan/workspace_grounding_lmm/LISA2/checkpoints_hf/ChatUniVi-SAM2-test")
    parser.add_argument("--vis_save_path", type=str, default="./vis_output/chat_output")
    parser.add_argument("--precision", type=str, default="fp16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--model_max_length", type=int, default=2048)
    parser.add_argument("--vision_tower", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--use_mm_start_end", action="store_true")
    parser.add_argument("--use_sam2_video_branch", action="store_true")
    parser.add_argument("--base_model_type", type=str, default="vgpt|phi3", choices=["vgpt|phi3","vgpt|llama3_1", "chatunivi"])
    parser.add_argument("--prompt_text", type=str, default="")
    args = parser.parse_args()
    return args

def initialize_model_chatunivi(
    llava_version_or_path, 
    precision, 
    model_max_length, 
    vision_tower, 
    local_rank, 
    load_in_8bit, 
    load_in_4bit, 
    use_mm_start_end,
    use_sam2_video_branch):

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        llava_version_or_path,
        cache_dir=None,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=False,
        legacy=False
    )

    # Add new tokens to tokenizer
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_tokens("[SEG]")
    seg_token_idx = tokenizer.convert_tokens_to_ids("[SEG]")
    print('seg_token_idx:', seg_token_idx)
    
    if use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )
        tokenizer.add_tokens(
            [DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN], special_tokens=True
        )

    torch_dtype = torch.bfloat16 if precision == "bf16" else (torch.half if precision == "fp16" else torch.float32)
    model_args = {"torch_dtype": torch_dtype}
    if load_in_4bit:
        model_args.update(
            {
                "torch_dtype": torch.half,
                "load_in_4bit": True,
                "quantization_config": transformers.BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif load_in_8bit:
        model_args.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": transformers.BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )

    from model.LISA_with_chatunivi import LISAForCausalLM
        
    model = LISAForCausalLM.from_pretrained(
        llava_version_or_path, low_cpu_mem_usage=True, 
        vision_tower=vision_tower, 
        seg_token_idx=seg_token_idx,
        use_sam2_video_branch=use_sam2_video_branch,
        **model_args)
    
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    
    
    ### Initialize modules
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)

    ### Model dtype
    if precision == "bf16":
        model = model.bfloat16().cuda()
    elif (
        precision == "fp16" and (not load_in_4bit) and (not load_in_8bit)
    ):
        vision_tower = model.get_model().get_vision_tower()
        model.model.vision_tower = None
        import deepspeed

        model_engine = deepspeed.init_inference(
            model=model,
            dtype=torch.half,
            replace_with_kernel_inject=False, #NOTE
            replace_method="auto",
        )
        model = model_engine.module
        model.model.vision_tower = vision_tower.half().cuda()
    elif precision == "fp32":
        model = model.float().cuda()

    model.eval()
    
    # enc_preprocessor
    from utils.enc_preprocessors import EncPreprocessor_ChatUniVi
    enc_preprocessor = EncPreprocessor_ChatUniVi()
    
    # conversation_generator
    from utils.conv_generator import ConvGenerator_ChatUniVi
    conv_generator = ConvGenerator_ChatUniVi(use_mm_start_end=use_mm_start_end)
    
    # sam preprocessor
    if model.config.use_sam2:
        from utils.sam_transforms import SAM_v2_Preprocess
        sam_preprocessor = SAM_v2_Preprocess()
    else:
        from utils.sam_transforms import SAM_v1_Preprocess
        sam_preprocessor = SAM_v1_Preprocess()
    
    return model, tokenizer, enc_preprocessor, conv_generator, sam_preprocessor

def initialize_model_videogptplus(
    model_base,
    precision,
    local_rank,
    load_in_8bit,
    load_in_4bit,
    use_sam2_video_branch,
    base_llm_type
):    
    # model_args
    torch_dtype = torch.bfloat16 if precision == "bf16" else (torch.half if precision == "fp16" else torch.float32)
    model_args = {"torch_dtype": torch_dtype}
    if load_in_4bit:
        model_args.update(
            {
                "torch_dtype": torch.half,
                "load_in_4bit": True,
                "quantization_config": transformers.BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif load_in_8bit:
        model_args.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": transformers.BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )

    # Load model
    from model.VideoGLaMM import VideoGLaMMForCausalLM
    if base_llm_type == "phi3":
        model = VideoGLaMMForCausalLM.from_pretrained(
            model_base, low_cpu_mem_usage=False, 
            use_sam2_video_branch=use_sam2_video_branch,
            **model_args)
    else:
        raise ValueError("Invalid base_llm_type")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
    # Add new tokens to tokenizer
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
    
    # set seg_token_idx in model config
    model.config.seg_token_idx = seg_token_idx
    # set model configs
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # for llama3_1
    if tokenizer.pad_token_id == None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Initialize encoder modules
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=local_rank)
    image_vision_tower = model.get_model().get_image_vision_tower()
    image_vision_tower.to(dtype=torch_dtype, device=local_rank)
    
    # 
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048
        
    
    ### Model dtype
    if precision == "bf16":
        model = model.bfloat16().cuda()
    elif (
        precision == "fp16" and (not load_in_4bit) and (not load_in_8bit)
    ):
        vision_tower = model.get_model().get_vision_tower()
        model.model.vision_tower = None
        image_vision_tower = model.get_model().get_image_vision_tower()
        model.model.image_vision_tower = None
        
        import deepspeed
        model_engine = deepspeed.init_inference(
            model=model,
            dtype=torch.half,
            replace_with_kernel_inject=False, #NOTE
            replace_method="auto",
        )
        model = model_engine.module
        model.model.vision_tower = vision_tower.half().cuda()
        model.model.image_vision_tower = image_vision_tower.half().cuda()
    elif precision == "fp32":
        model = model.float().cuda()
        
    model.eval()
        
    # enc_preprocessor
    from utils.enc_preprocessors import EncPreprocessor_VideoGPTPlus
    enc_preprocessor = EncPreprocessor_VideoGPTPlus()
    
    # conversation_generator
    from utils.conv_generator import ConvGenerator_VideoGPTPlus
    conv_generator = ConvGenerator_VideoGPTPlus(use_mm_start_end=mm_use_im_start_end, base_type=base_llm_type)
    
    # sam preprocessor
    if model.config.use_sam2:
        from utils.sam_transforms import SAM_v2_Preprocess
        sam_preprocessor = SAM_v2_Preprocess()
    else:
        from utils.sam_transforms import SAM_v1_Preprocess
        sam_preprocessor = SAM_v1_Preprocess()
    
    return model, tokenizer, enc_preprocessor, conv_generator, sam_preprocessor

def load_image(image_path):
    ''' Returns: numpy array of image # B x T x (H x W x C)
    '''
    image_np = cv2.imread(image_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    
    image_np= [[image_np]] # B x T x (H x W x C)
    
    return image_np

def load_video(video_path):
    ''' Returns: list of numpy arrays of video frames # T x (H x W x C) '''
    ### Load video
    video_framerate = 1
    max_num_frames = 64
    np_images = _get_rawvideo_dec(
        video_path, 
        video_framerate=video_framerate, 
        s=None, e=None)
    
    if len(np_images) > max_num_frames:
        new_np_images_idxs = np.linspace(0, len(np_images)-1, max_num_frames, dtype=int)
        new_np_images = [np_images[i] for i in new_np_images_idxs]
        np_images = new_np_images
    
    np_images = [np_images] # B x T x (H x W x C) # Add batch dimension
    
    return np_images
    

def preprocess_vision(np_images, type="video", enc_preprocessor=None, sam_preprocessor=None, conv_generator=None, precision="fp16"):
    
    '''
    Args:
    - np_images: list of numpy arrays of video frames : B x T x (H x W x C)
    
    Returns: Dictionary with keys:
    - enc_image: list of tensors of video frames preprocessed for CLIP : B x (T x 3 x 224 x 224)
    - image_sam: tensor of video frames preprocessed for SAM : B x (T x 3 x 1024 x 1024)
    - original_size_list: list of original sizes of video frames: B x (2)
    - resize_list: list of resized sizes of video frames: B x (2)
    
    '''
    
    if type == "video":
        
        assert len(np_images) == 1, "Batch size must be 1"
        np_images_ = np_images[0] # T x (H x W x C)
        
        # Subsample video frames if longer than max_num_frames
        max_num_frames = conv_generator.NUM_FRAMES
        if len(np_images_) > max_num_frames:
            new_np_images_idxs = np.linspace(0, len(np_images_)-1, max_num_frames, dtype=int)
            np_images_for_enc = [np_images_[i] for i in new_np_images_idxs]
        else:
            np_images_for_enc = np_images_
        
        # Preprocess image for encoder
        pil_images = [Image.fromarray(img) for img in np_images_for_enc]
        image_enc_dict = enc_preprocessor.preprocess(pil_images) # Tx(3 x 224 x 224)
        enc_image = image_enc_dict['images']
        enc_image = torch.stack(enc_image, dim=0) # (T x 3 x 224 x 224)
        enc_image = enc_image.bfloat16() if precision == "bf16" else (enc_image.half() if precision == "fp16" else enc_image.float())
        enc_image = enc_image.cuda(non_blocking=True)
        enc_image = [enc_image] # 1x(T x 3 x 224 x 224) # Add batch dimension
        
        enc_context_image = image_enc_dict['context_images']
        if enc_context_image is not None:
            enc_context_image = torch.stack(enc_context_image, dim=0) # (T x 3 x 224 x 224)
            enc_context_image = enc_context_image.bfloat16() if precision == "bf16" else (enc_context_image.half() if precision == "fp16" else enc_context_image.float())
            enc_context_image = enc_context_image.cuda(non_blocking=True)
            enc_context_image = [enc_context_image] # 1x(T x 3 x 224 x 224) # Add batch dimension
            
        
        ### Preprocess image for SAM
        original_size_list = [np_images_[0].shape[:2]]
        preprocessed_for_sam_and_resize_shapes = [sam_preprocessor.preprocess(image) for image in np_images_]
        image_sam = [x[0] for x in preprocessed_for_sam_and_resize_shapes]
        resize_shape = preprocessed_for_sam_and_resize_shapes[0][1]
        resize_list = [resize_shape]
        image_sam = torch.stack(image_sam, dim=0).cuda() # (T x 3 x 1024 x 1024)
        
        image_sam = image_sam.bfloat16() if precision == "bf16" else (image_sam.half() if precision == "fp16" else image_sam.float())
        image_sam = image_sam.cuda(non_blocking=True)
        image_sam = [image_sam] # 1x (T x 3 x 1024 x 1024) # Add batch dimension
        
    elif type == "image":
        
        assert len(np_images) == 1, "Batch size must be 1"
        assert len(np_images[0]) == 1, "Time dimension must be 1"
        image_np = np_images[0][0] # (H x W x C)
        
        # Preprocess image for encoder
        image_enc_dict = enc_preprocessor.preprocess(image_np) # (3, 224, 224)
        enc_image = image_enc_dict['images']
        enc_image = enc_image.unsqueeze(0) # (1, 3, 224, 224) # Add time dimension
        enc_image = enc_image.bfloat16() if precision == "bf16" else (enc_image.half() if precision == "fp16" else enc_image.float())
        enc_image = enc_image.cuda(non_blocking=True)
        enc_image = [enc_image] # 1x(1, 3, 224, 224) # Add batch dimension
        
        enc_context_image = image_enc_dict['context_images']
        if enc_context_image is not None:
            enc_context_image = enc_context_image.unsqueeze(0) # (1, 3, 224, 224) # Add time dimension
            enc_context_image = enc_context_image.bfloat16() if precision == "bf16" else (enc_context_image.half() if precision == "fp16" else enc_context_image.float())
            enc_context_image = enc_context_image.cuda(non_blocking=True)
            enc_context_image = [enc_context_image] # 1x(1, 3, 224, 224) # Add batch dimension
        
        ### Preprocess image for SAM
        original_size_list = [image_np.shape[:2]]
        
        image_sam, resize_shape = sam_preprocessor.preprocess(image_np)
        resize_list = [resize_shape]
        
        image_sam = image_sam.bfloat16() if precision == "bf16" else (image_sam.half() if precision == "fp16" else image_sam.float())
        image_sam = image_sam.unsqueeze(0).cuda() # (1, 3, 1024, 1024) # Add time dimension
        image_sam = [image_sam] # 1x(1, 3, 1024, 1024) # Add batch dimension
        
    return enc_image, enc_context_image, image_sam, original_size_list, resize_list

if __name__=='__main__':
    # args = parse_args(args)
    args = get_args()

    # Load model, tokenizer, and image processor
    if args.base_model_type.split('|')[0] == "vgpt":
        model, tokenizer, enc_preprocessor, conv_generator, sam_preprocessor = initialize_model_videogptplus(
            model_base=args.llava_version_or_path,
            precision=args.precision,
            local_rank=args.local_rank,
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit,
            use_sam2_video_branch=args.use_sam2_video_branch,
            base_llm_type=args.base_model_type.split('|')[1]
        )
    elif args.base_model_type.split('|')[0] == "chatunivi":
        model, tokenizer, enc_preprocessor, conv_generator, sam_preprocessor = initialize_model_chatunivi(
            llava_version_or_path=args.llava_version_or_path,
            precision=args.precision,
            model_max_length=args.model_max_length,
            vision_tower=args.vision_tower,
            local_rank=args.local_rank,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            use_mm_start_end=args.use_mm_start_end,
            use_sam2_video_branch=args.use_sam2_video_branch
        )

    
    while True:

    # prompt_text = 'segment the singer'
    # image_or_video_path = '/home/shehan/workspace_grounding_lmm/LISA2/video_dataset/vidstg/video/0001/3058613626.mp4'

        print('\033[92m----------------------------------------\033[0m')
        image_or_video_path = input("Please input the image/video path: ")
        if args.prompt_text == "":
            prompt_text = input("Please input your prompt: ")
        else:
            # prompt_text = args.prompt_text
            # dispay press enter to continue with args.prompt_text, or enter your own prompt
            prompt_text = input("Press enter to continue with the following prompt: \n" + args.prompt_text + "\n\nOr enter new prompt: ")
            if prompt_text == "":
                prompt_text = args.prompt_text
            
        print('\033[92m----------------------------------------\033[0m')

        #####

        if os.path.splitext(image_or_video_path)[1].lower() in ['.jpg', '.jpeg', '.png']: # image
            np_images = load_image(image_or_video_path)
            enc_image, enc_context_image, image_sam, original_size_list, resize_list = preprocess_vision(np_images, type="image", 
                                                                                                        enc_preprocessor=enc_preprocessor, 
                                                                                                        sam_preprocessor=sam_preprocessor, 
                                                                                                        conv_generator=conv_generator,
                                                                                                        precision=args.precision)
            input_ids = conv_generator.apply_for_chat(prompt_text, type='image', tokenizer=tokenizer)
        else: # video
            np_images = load_video(image_or_video_path)
            enc_image, enc_context_image, image_sam, original_size_list, resize_list = preprocess_vision(np_images, type="video", 
                                                                                                        enc_preprocessor=enc_preprocessor, 
                                                                                                        sam_preprocessor=sam_preprocessor, 
                                                                                                        conv_generator=conv_generator,
                                                                                                        precision=args.precision)
            input_ids = conv_generator.apply_for_chat(prompt_text, type='video', tokenizer=tokenizer)

        # if video, do inference frame-wise or using video-branch
        with torch.cuda.amp.autocast():
            output_ids_batch, video_segments_batch = model.inference(
                images=enc_image,
                context_images=enc_context_image,
                images_for_sam=image_sam,
                input_ids=input_ids,
                resize_list=resize_list,
                original_size_list=original_size_list,
                max_new_tokens=1024,
                use_sam2_video_branch=args.use_sam2_video_branch,
            )
        
        assert len(output_ids_batch) == 1 and len(video_segments_batch) == 1, "Batch size must be 1"
        
        output_ids = output_ids_batch[0]
        output_ids = output_ids[output_ids != IMAGE_TOKEN_INDEX]
        
        text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
        # text_output = text_output.replace("\n", "").replace("  ", " ")
        print("text_output: ", text_output)
        
        # save video segments
        batch_idx = 0

        video_frames_np = np_images[batch_idx] # [T, H, W, C]  # numpy array
        video_segments = video_segments_batch[batch_idx]
        
        save_name = image_or_video_path.split("/")[-1].split(".")[0]
        save_dir = os.path.join(args.vis_save_path, save_name)
        # os.makedirs(save_dir, exist_ok=True)
        # if save_dir exists, delete it
        if os.path.exists(save_dir):
            import shutil
            shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        print('save_dir:', save_dir)
        write_masks(video_segments, video_frames_np, save_dir)
        # save caption.txt at save_dir
        with open(os.path.join(save_dir, "caption.txt"), "w") as f:
            f.write(text_output)