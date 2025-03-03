import json
import os
import ffmpeg
import numpy as np
from tqdm import tqdm

# Function to load JSON data from a file
def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Function to load annotations from the dataset directory
def load_annotations(vidstg_ann_dir, split='train'):
    # Define file paths for the current split
    file_ids_path = os.path.join(vidstg_ann_dir, f"{split}_files.json")
    annotations_path = os.path.join(vidstg_ann_dir, f"{split}_annotations.json")
    
    # Load the file ids and annotations
    file_ids = load_json(file_ids_path)
    annotations = load_json(annotations_path)
    
    # Create a dictionary with video ID as keys for easier access
    annotations_dict = {annotation['vid']: annotation for annotation in annotations}
    
    return file_ids, annotations_dict

# Main function to load all splits
def load_vidstg_data(vidstg_ann_dir):
    splits = ['train', 'val', 'test']
    data = {}
    
    for split in splits:
        file_ids, annotations = load_annotations(vidstg_ann_dir, split)
        data[split] = {
            'file_ids': file_ids,
            'annotations': annotations
        }
    
    return data

vidor_ann_dir = './video_dataset/vidstg/vidor_annotations'
vidor_video_dir = os.path.join('./video_dataset/', 'vidstg', 'video')
vidstg_ann_dir = './video_dataset/vidstg/vidstg_annotations'
vidstg_data = load_vidstg_data(vidstg_ann_dir)

image_set = 'test' # 'train', 'val', 'test'
if image_set == 'train':
    vidor = json.load(open(os.path.join(vidstg_ann_dir, "vidor_training.json"), "r")) # train
    chosen_file_ids = vidstg_data['train']['file_ids']
    chosen_annotations = vidstg_data['train']['annotations']
elif image_set == 'val':
    vidor = json.load(open(os.path.join(vidstg_ann_dir, "vidor_training.json"), "r"))
    chosen_file_ids = vidstg_data['val']['file_ids']
    chosen_annotations = vidstg_data['val']['annotations']
elif image_set == 'test':
    vidor = json.load(open(os.path.join(vidstg_ann_dir, "vidor_validation.json"), "r"))
    chosen_file_ids = vidstg_data['test']['file_ids']
    chosen_annotations = vidstg_data['test']['annotations']

# VidOR contains 7,000, 835 and 2,165 videos for training, validation and testing, respectively. 
# Since box annota-tions of testing videos are unavailable yet, VidSTG omit testing videos, split 10% training videos as our validation data and regard original validation videos as the testing data


def get_video_relation_data(ann):
    caption_dict = ann["captions"][0]
    description = caption_dict["description"]

    subject_tid = ann["used_relation"]['subject_tid']
    object_tid = ann["used_relation"]['object_tid']

    # search ann["subject/objects"] for subject_category and object_category
    subject_category = None
    object_category = None
    for obj in ann["subject/objects"]:
        if obj["tid"] == subject_tid:
            subject_category = obj["category"]
        if obj["tid"] == object_tid:
            object_category = obj["category"]
            
    # print the following . e.g.
    # # subject : 
    #   target_id :0, category : rabbit
    # object :
    #   target_id : 1, category : adult
    # relation : lean_on

    string = ''
    string += (f"\tsubject : \n  \t\ttarget_id : {subject_tid}, category : {subject_category}\n")
    string += (f"\tobject : \n  \t\ttarget_id : {object_tid}, category : {object_category}\n")
    string += (f"\trelation : {ann['used_relation']['predicate']}\n")
    string += (f"\tdescription : {description}\n")
    
    return string



_EXAMPLES = """\
Example 1: 

Input :
  subject : 
    target_id :0, category : rabbit
  object :
    target_id : 1, category : adult
  relation : lean_on
  description: "there is a white rabbit leaning on an adult by the water".
  
Output:
{'caption': 'there is a [white rabbit](0) leaning on an [adult](1) by the water'}


"""

_PROMPT = """\
Your task is to generate annotated video captions, given original unannotated video descriptions, the lists of subjects/objects in the video and the relation between them.

For each video, you are given a relation between a subject and an object, along with the categories and target IDs of the subject and object.
Your task is to generate a new caption annotating the subject and object in the caption with the corresponding target IDs.

You may look at the following examples:
{examples}

Now please process the following.

{video_relation_data}

In the new caption, the noun phrases should be included within square brackets and object ID/IDs should be included within paranthesis. E.g. [noun phrase](object ID/IDs) .

Please provide the generated caption in JSON format, with a key "caption".
"""


from openai import OpenAI
import ast

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

def get_caption(ann):
    """
    """
    formatted_prompt = _PROMPT.format(
        examples = _EXAMPLES,
        video_relation_data = get_video_relation_data(ann)
    )
    
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": formatted_prompt}
        ],
        response_format={ "type": "json_object" }
    )

    response_message = completion.choices[0].message.content
    response_dict = ast.literal_eval(response_message)
    return response_dict

output_dir = f'vidstg_gcg_captions/{image_set}'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def process_idx(idx):
    try:
        video_id = chosen_file_ids[idx]
        ann = chosen_annotations[video_id]
    except KeyError:
        print(f"Video ID: {video_id} not found in annotations")
        return
    
    caption = get_caption(ann)
    
    output_path = os.path.join(output_dir, f"{video_id}.json")
    with open(output_path, 'w') as f:
        json.dump(caption, f)

    print(f"Processed video ID: {video_id}", " Caption saved at:", output_path)
    

from concurrent.futures import ThreadPoolExecutor, as_completed

with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(process_idx, idx) for idx in range(len(chosen_file_ids))]
    
    for future in tqdm(as_completed(futures), total=len(chosen_file_ids), desc="Processing ..."):
        future.result()