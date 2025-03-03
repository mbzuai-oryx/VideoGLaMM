from utils.refer_datasets.ytvos import YTVOSDataset
from utils.refer_datasets.transforms import make_coco_transforms

from pathlib import Path
import os
import json
from tqdm import tqdm

base_video_dataset_dir = './video_dataset'
image_set = 'train'
num_frames_for_clip = 5

# Refer-YTVOS
ytvos_root = Path(os.path.join(base_video_dataset_dir, "refer_youtube_vos"))
assert ytvos_root.exists(), f'provided YTVOS path {ytvos_root} does not exist'
PATHS = {
    "train": (ytvos_root / "train", ytvos_root / "meta_expressions" / "train" / "meta_expressions.json"),
    "val": (ytvos_root / "valid", ytvos_root / "meta_expressions" / "val" / "meta_expressions.json"),    # not used actually
}
img_folder, ann_file = PATHS[image_set]
# dataset = YTVOSDataset(img_folder, ann_file, 
#         transforms=make_coco_transforms(image_set="val", do_not_normalize=True), # val transform to avoid random transforms
#         return_masks=True, num_frames=num_frames_for_clip)


def prepare_metas_2(img_folder, ann_file, num_frames):
    # read object information
    with open(os.path.join(str(img_folder), 'meta.json'), 'r') as f:
        subset_metas_by_video = json.load(f)['videos']
    
    # read expression data
    with open(str(ann_file), 'r') as f:
        subset_expressions_by_video = json.load(f)['videos']
    videos = list(subset_expressions_by_video.keys())

    # metas = []
    metas_videowise = []
    for vid in videos:
        vid_meta = subset_metas_by_video[vid]
        vid_data = subset_expressions_by_video[vid]
        vid_frames = sorted(vid_data['frames'])
        vid_len = len(vid_frames)
        
        meta_vid = {}
        meta_vid['video'] = vid
        meta_vid['frames'] = vid_frames
        meta_vid['objs'] = []
        
        for exp_id, exp_dict in vid_data['expressions'].items():
            # for frame_id in range(0, vid_len, num_frames):
            meta = {}
            # meta['video'] = vid
            meta['exp'] = exp_dict['exp']
            meta['obj_id'] = int(exp_dict['obj_id'])
            # meta['frames'] = vid_frames
            # meta['frame_id'] = frame_id
            # get object category
            obj_id = exp_dict['obj_id']
            meta['category'] = vid_meta['objects'][obj_id]['category']
            meta_vid['objs'].append(meta)
        metas_videowise.append(meta_vid)
    
    ###
    dataset_videowise = []
    for meta_vid in metas_videowise:
        record = {}
        # record["file_names"] = #TODO
        record["video_name"] = meta_vid['video']
        record["objs"] = []
        
        for meta in meta_vid['objs']:
            exp = meta['exp']
            obj_ids = meta['obj_id']
            category = meta['category']
            
            record["objs"].append({'sentence':exp, 'obj_ids':obj_ids,
                                #    'annotations':video_objs #TODO
                                })
        dataset_videowise.append(record)
        
    
    
    print('\n video num: ', len(videos), ' clip num: ', len(metas_videowise))  
    print('video wise dataset: ', len(dataset_videowise))
    return metas_videowise, dataset_videowise

###########################################
from openai import OpenAI
import ast

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

_EXAMPLES = """\
Example 1:

Input (The given object IDs and Referring Expressions):
(1) : ['panda rolling around', 'panda roll on the ground']
(0, 2) : ['panda climbing branch']
(3) : ['panda falling off branch', 'panda fall down from the tree']
(0) : ['panda climb up and then falls down']
(2) : ['panda climb to tree', 'a panda that successfully climbed to a higher place in a tree']
(0, 1, 2, 3) : ['four pandas', 'the four pandas playing and frolicking.']
(0, 1) : ['roll on the ground']

Output (Generated caption):
{'caption': "In this scene, there are [four pandas](0, 1, 2, 3) engaging in fun activities.\
    Two [pandas are climbing a branch](0, 2) finding its way up, while another [panda is falling off a branch](3). One more [panda is seen rolling on the ground](1) with joy.}

Example 2:

Input (The given object IDs and Referring Expressions):
(0) : ['shirtless man standing', 'a man waiting for hair removal']
(1) : ['woman in red dress turning around and putting white tape on chest of shirtless man']

Output (Generated caption):
{"caption": "In this scene, there is a [shirtless man standing](0) who appears to be waiting for a hair removal procedure. \
    Nearby, a [woman in a red dress](1) is turning around and putting white tape on the chest of the shirtless man, assisting him with the preparation."}

Example 3:

Input (The given object IDs and Referring Expressions):
(1) : ['a white cowboy had on a man on a horse second from the left', 'a hat worn by the man riding the horse that s second to the right']
(2) : ['a white cowboy hat on a man fourth from the left', 'a hat being worn by the man riding the horse located the fourth from the left']

Output (Generated caption):
{"caption": "In this video, men are seen riding horses. [A white cowboy hat worn is by the man riding the horse that is second to the right](1).\
    [Another white cowboy hat is being worn by the man riding the horse located the fourth from the left](2)."}

"""

_PROMPT = """\
Your task is to generate meaningful video captions, given the list of objects in the video and referring expressions describing each object.
The following is a list of mapping of object IDs in a video and corresponding referring expressions. 
Please generate a video caption that includes these objects and referring expressions. 
In the generated caption, the noun phrases should be included within square brackets and object ID/IDs should be included within paranthesis. E.g. [noun phrase](object ID/IDs) .

You may look at the following examples:
{examples}

Now please process the following.

The given object IDs and Referring Expressions are:
{mapping_string}


Please provide the generated caption in \
(JSON format, with a key "caption".)
"""

def get_caption(mapping_string):
    """
    """
    formatted_prompt = _PROMPT.format(
        mapping_string = mapping_string,
        examples = _EXAMPLES
    )
        
    # Compute the noun phrase and referring expression
    # completion = openai.ChatCompletion.create(
    #     model=openai_model_name,
    #     messages=[
    #         {"role": "system","content": "You are a helpful assistant."},
    #         {"role": "user","content": formatted_prompt},
    #     ]
    # )
    
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": formatted_prompt}
        ],
        response_format={ "type": "json_object" }
    )

    # print(completion.choices[0].message.content)
    # Convert response to a Python dictionary.
    response_message = completion.choices[0].message.content #completion["choices"][0]["message"]["content"]
    response_dict = ast.literal_eval(response_message)
    return response_dict


###########################################
_, dataset = prepare_metas_2(img_folder, ann_file, num_frames_for_clip)

output_dir = 'ytvos_captions'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
def process_data(data):
    try:
        video_name = data['video_name']
        # file_names = data['file_names']
        # print(f'video_name: {video_name}')
        
        json_filepath = os.path.join(output_dir, f'{video_name}.json')
        if os.path.exists(json_filepath):
            print(f'Caption already exists at {json_filepath}')
            return

        obj_ids_and_expressions = {}
        for obj in data['objs']:
            sentence = obj['sentence']
            # exp_id = obj['exp_id']
            obj_ids = obj['obj_ids']
            obj_ids = (obj_ids,) # tuple
            # print(f'{sentence} - {obj_ids}')
            
            if obj_ids in obj_ids_and_expressions:
                obj_ids_and_expressions[obj_ids].append(sentence)
            else:
                obj_ids_and_expressions[obj_ids] = [sentence]
        
        mapping_string = ''
        for obj_ids, expressions in obj_ids_and_expressions.items():
            obj_ids = str(obj_ids) if len(obj_ids) > 1 else f'({str(obj_ids[0])})'
            expressions = expressions[:2] if len(expressions) > 2 else expressions
            mapping_string += f'{obj_ids} : {expressions}'
            mapping_string += '\n'

        # print(mapping_string)
        
        response_dict = get_caption(mapping_string)
        
        
        
        with open(json_filepath, 'w') as f:
            json.dump(response_dict, f)
            print(f'Caption saved at {json_filepath}')
            print('-----------------------------------')
            
    except Exception as e:
        print(f'\033[91mError: {e}\033[0m')
        
###########################################


from concurrent.futures import ThreadPoolExecutor, as_completed

# Assuming dataset is a list or iterable containing the data
with ThreadPoolExecutor(max_workers=5) as executor:
    # Submit all tasks to the executor
    futures = {executor.submit(process_data, data): data for data in dataset}
    
    # Use tqdm to track the progress
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing videos"):
        # Optionally, handle the result if needed
        result = future.result()