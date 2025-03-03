import json
import os
from tqdm import tqdm

base_video_dataset_dir = './video_dataset'
mevis_root = os.path.join(base_video_dataset_dir, "mevis")

def load_mevis_json_2(mevis_root, image_set):
        
    image_root = os.path.join(mevis_root, image_set) # "./video_dataset/mevis/train"
    json_file = os.path.join(mevis_root, image_set, "meta_expressions.json") # "./video_dataset/mevis/train/meta_expressions.json"

    num_instances_without_valid_segmentation = 0
    num_instances_valid_segmentation = 0


    ann_file = json_file
    with open(str(ann_file), 'r') as f:
        subset_expressions_by_video = json.load(f)['videos']
    videos = list(subset_expressions_by_video.keys())
    print('number of video in the datasets:{}'.format(len(videos)))
    
    # 
    # metas = []
    metas_videowise = []
    if image_set=='train' or image_set=='valid_u': # only train and valid_u sets have masks
        mask_json = os.path.join(image_root, 'mask_dict.json')
        print(f'Loading masks form {mask_json} ...')
        with open(mask_json) as fp:
            mask_dict = json.load(fp)

        for vid in videos:
            vid_data = subset_expressions_by_video[vid]
            vid_frames = sorted(vid_data['frames'])
            vid_len = len(vid_frames)
            if vid_len < 2:
                continue
            
            meta_vid = {}
            meta_vid['video'] = vid
            meta_vid['frames'] = vid_frames
            meta_vid['length'] = vid_len
            meta_vid['objs'] = []
            
            for exp_id, exp_dict in vid_data['expressions'].items():
                meta = {}
                # meta['video'] = vid
                meta['exp'] = exp_dict['exp']
                meta['obj_id'] = [int(x) for x in exp_dict['obj_id']]
                meta['anno_id'] = [str(x) for x in exp_dict['anno_id']]
                # meta['frames'] = vid_frames
                meta['exp_id'] = exp_id
                meta['category'] = 0
                # meta['length'] = vid_len
                meta_vid['objs'].append(meta)
            metas_videowise.append(meta_vid)
            
    else: # valid set does not have masks
        for vid in videos:
            vid_data = subset_expressions_by_video[vid]
            vid_frames = sorted(vid_data['frames'])
            vid_len = len(vid_frames)
            
            
            meta_vid = {}
            meta_vid['video'] = vid
            meta_vid['frames'] = vid_frames
            meta_vid['length'] = vid_len
            meta_vid['objs'] = []
            
            for exp_id, exp_dict in vid_data['expressions'].items():
                meta = {}
                # meta['video'] = vid
                meta['exp'] = exp_dict['exp']
                meta['obj_id'] = -1
                meta['anno_id'] = -1
                # meta['frames'] = vid_frames
                meta['exp_id'] = exp_id
                meta['category'] = 0
                # meta['length'] = vid_len
                meta_vid['objs'].append(meta)
            metas_videowise.append(meta_vid)
            
    ###
    dataset_videowise = []
    for meta_vid in metas_videowise:
        record = {}
        # video_name = meta_vid['video']
        # file_names = [os.path.join(image_root, 'JPEGImages', meta_vid['video'], meta_vid['frames'][i] + '.jpg') for i in range(meta_vid["length"])]
        record["file_names"] = [os.path.join(image_root, 'JPEGImages', meta_vid['video'], meta_vid["frames"][i]+ '.jpg') for i in range(meta_vid["length"])]
        record["length"] = meta_vid["length"]
        record["video_name"] = meta_vid['video']
        record["objs"] = []
        
        for meta in meta_vid['objs']:
            exp = meta['exp']
            obj_ids = meta['obj_id']
            anno_ids = meta['anno_id']
            category = meta['category']
            exp_id = meta['exp_id']
            
            exp = " ".join(exp.lower().split())
            if "eval_idx" in meta:
                eval_idx = meta["eval_idx"]
            
            video_objs = []
            if image_set=='train' or image_set=='valid_u':
                for frame_idx in range(meta_vid["length"]): # for time
                    frame_objs = []
                    for x, obj_id in zip(anno_ids, obj_ids): # for objects
                        obj = {}
                        segm = mask_dict[x][frame_idx]
                        if not segm:
                            num_instances_without_valid_segmentation += 1
                            continue
                        num_instances_valid_segmentation += 1
                        bbox = [0, 0, 0, 0]
                        obj["id"] = obj_id
                        obj["segmentation"] = segm
                        obj["category_id"] = category
                        obj["bbox"] = bbox
                        frame_objs.append(obj)
                    video_objs.append(frame_objs)            
            record["objs"].append({'sentence':exp, 'obj_ids':obj_ids, 'exp_id':exp_id, 'annotations':video_objs})
        
        dataset_videowise.append(record)

    if num_instances_without_valid_segmentation > 0:
        print(
            "Total {} instance and Filtered out {} instances without valid segmentation. ".format(
                num_instances_valid_segmentation, num_instances_without_valid_segmentation
            )
        )
    return dataset_videowise

########################################
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
image_set = 'valid_u' # 'train', 'valid_u', 'valid'
dataset = load_mevis_json_2(mevis_root, image_set)

output_dir = 'mevis_captions'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def process_data(data):
    try:
        video_name = data['video_name']
        file_names = data['file_names']
        # print(f'video_name: {video_name}')
        
        json_filepath = os.path.join(output_dir, f'{video_name}.json')
        if os.path.exists(json_filepath):
            print(f'Caption already exists at {json_filepath}')
            return

        obj_ids_and_expressions = {}
        for obj in data['objs']:
            sentence = obj['sentence']
            exp_id = obj['exp_id']
            obj_ids = obj['obj_ids']
            # print(f'{sentence} - {obj_ids}')
            obj_ids = tuple(obj_ids)
            
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

# for data in tqdm(dataset, desc="Processing videos"):
#     process_data(data)

# from concurrent.futures import ThreadPoolExecutor

# # Assuming dataset is a list or iterable containing the data
# num_threads = 10  # Define the number of threads

# # Run the processing in parallel and track progress with tqdm
# with ThreadPoolExecutor(max_workers=num_threads) as executor:
#     list(tqdm(executor.map(process_data, dataset), total=len(dataset), desc="Processing videos"))


from concurrent.futures import ThreadPoolExecutor, as_completed

# Assuming dataset is a list or iterable containing the data
with ThreadPoolExecutor(max_workers=10) as executor:
    # Submit all tasks to the executor
    futures = {executor.submit(process_data, data): data for data in dataset}
    
    # Use tqdm to track the progress
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing videos"):
        # Optionally, handle the result if needed
        result = future.result()