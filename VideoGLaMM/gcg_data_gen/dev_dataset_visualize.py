from utils.mevis_gcg import MevisGCGBaseDataset
from utils.vidstg_hcstvg_gcg import VidSTG_HCSTVG_GCGBaseDataset
############################################################################################################


import gradio as gr
import random
import numpy as np
import torch
import os
import json
from torchvision import transforms
from PIL import Image

# Instantiate the dataset
video_dataset_dir = './video_dataset'

# add argparse argument to select dataset
import argparse

parser = argparse.ArgumentParser(description='Visualize GCG Dataset Samples')
parser.add_argument('--dataset', type=str, default='vidstg', help='Dataset to visualize: mevis, vidstg', choices=['mevis', 'vidstg'])
args = parser.parse_args()

if args.dataset == 'mevis':
    # Mevis GCG Dataset
    DATASET_NAME = "Mevis GCG"
    image_set = "valid_u" # "train", "valid_u"
    dataset = MevisGCGBaseDataset(base_video_dataset_dir=video_dataset_dir, image_set=image_set)
    modified_jsons_save_dir = f"gcg_saved_jsons_mevis/{image_set}"
elif args.dataset == 'vidstg':
    # VidSTG GCG Dataset
    DATASET_NAME = "VidSTG GCG"
    image_set = "val" # "train", "val", "test"
    dataset = VidSTG_HCSTVG_GCGBaseDataset(base_video_dataset_dir=video_dataset_dir, image_set=image_set, source_dataset="vidstg")
    modified_jsons_save_dir = f"gcg_saved_jsons_vidstg/{image_set}"

# Define function to retrieve and visualize a random sample from the dataset
def visualize_sample(idx=None):
    if idx is None:
        idx = random.randint(0, len(dataset) - 1)
    
    video_name, json_file, pil_images, all_gt_masks, caption, phrases = dataset[idx]

    # Select multiple frames for visualization
    num_frames_for_sam = 5
    frame_indices = np.linspace(0, len(pil_images) - 1, num_frames_for_sam, dtype=int)
    selected_pil_images = [pil_images[i] for i in frame_indices]  # T_samx()
    gt_masks = [mask[frame_indices] for mask in all_gt_masks]  # num_objsx(T_samxHxW)
    gt_masks_ = [torch.from_numpy(image).float() for image in gt_masks]  # num_objsx(1xHxW) or num_objsx(T_samxHxW)
    gt_masks_ = torch.stack(gt_masks_)  # (num_objs, T_sam, H, W)
    masks = gt_masks_.to(torch.float32)  # [num_objects, T_sam, H, W]

    # Prepare display of images (one row of 5 frames)
    combined_image = Image.new('RGB', (selected_pil_images[0].width * num_frames_for_sam, selected_pil_images[0].height))
    for i, img in enumerate(selected_pil_images):
        combined_image.paste(img, (i * img.width, 0))

    # Prepare display of masks overlayed on images (each object gets one row of 5 frames)
    mask_rows = []
    for obj_idx in range(masks.shape[0]):
        combined_mask_overlay = Image.new('RGB', (selected_pil_images[0].width * num_frames_for_sam, selected_pil_images[0].height))
        for i in range(num_frames_for_sam):
            img = selected_pil_images[i].copy()
            mask_frame = transforms.ToPILImage()(masks[obj_idx, i])
            mask_frame = mask_frame.convert("RGBA")
            img = img.convert("RGBA")
            mask_overlay = Image.blend(img, mask_frame, alpha=0.5)
            combined_mask_overlay.paste(mask_overlay.convert("RGB"), (i * img.width, 0))
        mask_rows.append((combined_mask_overlay, phrases[obj_idx]))

    # Load json file content
    with open(os.path.join(dataset.captions_dir, json_file), 'r') as f:
        json_content = json.load(f)

    return combined_image, mask_rows, caption, json.dumps(json_content, indent=4), video_name, idx

# Define function to save modified json content
def save_json_content(json_content, video_name):
    # save_path = f"saved_jsons/{video_name}.json"
    save_path = os.path.join(modified_jsons_save_dir, f"{video_name}.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(json.loads(json_content), f, indent=4)
    return f"Saved JSON to {save_path}"

# Define function to increment/decrement index and visualize
def navigate_dataset(current_idx, direction):
    if direction == "next":
        new_idx = min(current_idx + 1, len(dataset) - 1)
    elif direction == "prev":
        new_idx = max(current_idx - 1, 0)
    else:
        new_idx = current_idx
    return visualize_sample(new_idx)

# Create a Gradio interface
with gr.Blocks() as demo:
    gr.Markdown(f"# {DATASET_NAME} Dataset : {image_set}\nThis tool visualizes random samples from the dataset, including frames, masks, and corresponding captions.")

    with gr.Row():
        random_sample_btn = gr.Button("Get Random Sample")
        idx_input = gr.Number(label="Enter Index for Sample (optional)")
        video_name_box = gr.Textbox(label="Video Name", interactive=False)
        sample_by_idx_btn = gr.Button("Get Sample by Index")
    
    image_output = gr.Image(type="pil")
    mask_outputs = gr.Gallery(label="Masks Overlayed on Images (Each Row Represents One Object)",
                              object_fit="cover" ,
                              selected_index=0)
    caption_output = gr.Textbox(label="Caption")
    json_content_output = gr.Textbox(label="JSON File Content", lines=10)
    save_json_btn = gr.Button("Save JSON", variant="primary")

    with gr.Row():
        prev_btn = gr.Button("Previous")
        next_btn = gr.Button("Next")

    random_sample_btn.click(visualize_sample, [], [image_output, mask_outputs, caption_output, json_content_output, video_name_box, idx_input])
    sample_by_idx_btn.click(visualize_sample, [idx_input], [image_output, mask_outputs, caption_output, json_content_output, video_name_box, idx_input])
    save_json_btn.click(save_json_content, [json_content_output, video_name_box], gr.Textbox(label="Save Status"))
    prev_btn.click(lambda idx: navigate_dataset(idx, 'prev'), [idx_input], [image_output, mask_outputs, caption_output, json_content_output, video_name_box, idx_input])
    next_btn.click(lambda idx: navigate_dataset(idx, 'next'), [idx_input], [image_output, mask_outputs, caption_output, json_content_output, video_name_box, idx_input])

# Launch the Gradio app
demo.launch(share=True)
