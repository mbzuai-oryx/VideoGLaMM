json_path = '/home/shehan/workspace_grounding_lmm/LISA2/archive/anet_entities_gcg/hanan/all_captions.json'

import json
import os
import sys

# read json file
with open(json_path) as f:
    data = json.load(f)


import numpy as np
import random
from PIL import Image
import copy
from tqdm import tqdm
import argparse
import openai
import ast
import os

openai.api_key = os.environ.get("OPENAI_API_KEY")
openai_model_name = "gpt-3.5-turbo"


_EXAMPLES = """\
Example 1:

Ground truth caption:
A <p> weight </p> [SEG:1] lifter is in a <p> gym </p> [SEG:2] , and <p> he </p> [SEG:1] lifts a <p> barbell </p> [SEG:0]

Reference captions:
In the video, a man is lifting weights in a gym. He lifts the weights over his head and then drops them on the ground.
In the video, a person is seen lifting weights in a gym setting. The individual is focused on performing the weightlifting exercise, and their posture indicates a controlled and deliberate movement. The gym environment is equipped with various weightlifting equipment, and there are other people present in the background, suggesting a shared workout space. The person's attire and the equipment indicate that this is a dedicated space for physical fitness and strength training. The video captures a moment of physical exertion and dedication to fitness.

Output:
{"refined_caption": "In the video, <p> A man </p> [SEG:1] is lifting weights in a <p> gym </p> [SEG:2]. <p> He </p> [SEG:1] is lifting a <p> barbell </p> [SEG:0] over his head and then drops them on the ground."}

Example 2:

Ground truth caption:
The <p> man </p> [SEG:1] stands while holding onto the <p> swing </p> [SEG:0]

Reference captions: 
In the video, a man is swinging on a swing set in a park. He is wearing a black shirt and is swinging back and forth while looking towards the camera.
In the video, a person is standing in a park, wearing a black shirt and dark pants. The individual appears to be posing or standing still, possibly enjoying the surroundings or waiting for someone. The park features a playground with visible equipment, such as a swing set, indicating a recreational area for children and families. The person is standing on a concrete surface, and there are trees and other greenery in the background, suggesting a peaceful and natural setting. The individual's pose and the environment create a calm and leisurely atmosphere.

Output:
{"refined_caption": "In the video, <p> a man </p> [SEG:1] is swinging on a <p> swing set </p> [SEG:0] in a park. He is wearing a black shirt and is swinging back and forth while looking towards the camera."}

Example 3:

Ground truth caption:
<p> She </p> [SEG:1] puts shaving <p> cream </p> [SEG:2] on <p> her </p> [SEG:1] <p> leg </p> [SEG:0] and shaves <p> her </p> [SEG:1] <p> leg </p> [SEG:0]

Reference captions: 
In the video, a person is seen sitting on a tub and shaving their legs with a razor.
In the video, a person is seen sitting in a bathtub, and their legs are being shaved with a razor. The individual appears to be focused on the shaving process, and there are no other significant actions or events occurring in the video. The person's posture and the position of the razor suggest a careful and deliberate approach to shaving their legs. The setting appears to be a private bathroom, and there are no other people or objects visible in the frame.

Output:
{"refined_caption": "In the video, <p> a woman </p> [SEG:1] is seen sitting in a bathtub, shaving <p> her </p> [SEG:1] <p> legs </p> [SEG:0] with a razor. <p> She </p> [SEG:1] is applying <p> shaving cream </p> [SEG:2] on <p> her </p> [SEG:1] <p> leg </p> [SEG:0]."}

Example 4:

Ground truth caption:
One last <p> woman </p> [SEG:1] speaks to the camera and the <p> girl </p> [SEG:0] cheers and smiles again

Reference captions:
In the video, a woman is seen holding a contact lens case and a booklet. She opens the case and takes out a lens, which she then puts on her finger. She then proceeds to put the lens in her eye. After that, she takes out another lens and puts it on her other eye.
In the video, a person's hand is seen holding a bottle of Dailies Aqua Comfort Plus contact lens solution. The hand appears to be in the process of either opening the bottle or pouring the solution into a contact lens case. The person's fingers are visible, and they seem to be manipulating the bottle cap or the bottle itself. The action is focused and precise, indicating that the person is familiar with handling the product. The video does not show any other significant actions or events, and the main focus is on the interaction with the contact lens solution bottle.

Output:
{"refined_caption": "<p> A woman </p> [SEG:1] speaks to the camera, while <p> a girl </p> [SEG:0] cheers and smiles."}

Example 5:

Ground truth caption:
One of the <p> girls </p> [SEG:0] is wearing a red <p> shirt </p> [SEG:2] and the other is wearing a dark blue <p> shirt </p> [SEG:1]

Reference captions:
In the video, two girls are seen dancing in a room. They are wearing casual clothes and are dancing in front of a television. The girls are moving their arms and legs in a coordinated manner, and they seem to be enjoying themselves. The room is well-lit, and there are no other people or objects visible in the frame. The girls continue dancing for a while, and then they stop and look at each other.
In the video, two individuals are engaged in a lively activity, possibly dancing or playing a motion-controlled video game. They are standing in a room with a television set, which is displaying a vibrant screen that could be a game or a music video. The individuals are dressed in casual attire, with one wearing a red top and the other in beige pants. Their movements are energetic and synchronized, suggesting a shared experience or performance. The atmosphere appears to be joyful and interactive, with the focus on the television screen and their coordinated actions.

Output:
{"refined_caption": "In the video, two girls are seen dancing in a room. <p> One of the girls </p> [SEG:0] is wearing <p> a red shirt </p> [SEG:2], while the other is wearing <p> a dark blue shirt </p> [SEG:1]. They are dancing in front of a television, moving their arms and legs in a coordinated manner, and they seem to be enjoying themselves. The room is well-lit, and there are no other people or objects visible in the frame. The girls continue dancing for a while, and then they stop and look at each other."}


"""

_PROMPT = """\
Your task is to process video captions to make them more detailed and explanatory.
You are given a ground truth caption and a set of dense noisy captions.
Ground truth caption contains a description of the objects visible in a video, with noun phrases of significant objects surrounded by <p> and </p> tags, followed by a [SEG:x] tag.
Dense noisy captions contain additional information about the video, but they may be redundant or less precise than the ground truth caption.

Your task is to paraphrase the ground truth caption by incorporating relevant information from the dense noisy captions.
The refined caption should be more detailed and explanatory than the ground truth caption.
The refined caption should preserve the original <p>, </p>, and [SEG:x] tags.
The refined caption should also preserve the identity of [SEG:x] tags, given by a unique identifier x.

You may look at the following examples:
{examples}


Now please refine the following caption:


Ground truth caption:
{gt_caption}

Reference captions:

{reference_captions}

Please provide the refined caption in \
(JSON format, with a key "refined_caption".)
"""

def refine(gt_caption, reference_captions):
    """
    """
    formatted_prompt = _PROMPT.format(
        gt_caption=gt_caption,
        reference_captions="\n".join(reference_captions),
        examples=_EXAMPLES
    )
        
    # Compute the noun phrase and referring expression
    completion = openai.ChatCompletion.create(
        model=openai_model_name,
        messages=[
            {"role": "system","content": "You are a helpful assistant."},
            {"role": "user","content": formatted_prompt},
        ]
    )
    # Convert response to a Python dictionary.
    response_message = completion["choices"][0]["message"]["content"]
    # print(response_message)
    response_dict = ast.literal_eval(response_message)
    return response_dict




c_ = 0

# res_data = {}

res_data = json.load(open('res_data.json'))

for vid in data:
    for seg in data[vid]:
        try:
            if vid in res_data and seg in res_data[vid]:
                print('Already processed', vid, seg)
                continue
            
            gt_caption, reference_captions = data[vid][seg]['gt_caption'], data[vid][seg]['references']
            response_dict = refine(gt_caption, reference_captions)
            refined_caption = response_dict['refined_caption']
            # print(refined_caption)
            print('GT       :', gt_caption)
            print('Reference:', reference_captions)
            print('Refined  :', refined_caption)
            print('-'*50)
            c_ += 1
            
            res_data[vid] = res_data.get(vid, {})
            res_data[vid][seg] = res_data[vid].get(seg, {})
            res_data[vid][seg]['gt_caption'] = gt_caption
            res_data[vid][seg]['references'] = reference_captions
            res_data[vid][seg]['refined_caption'] = refined_caption
            
        except Exception as e:
            print('error', e)
            
# Save res_data to a JSON file
with open('res_data_new.json', 'w') as f:
    json.dump(res_data, f)