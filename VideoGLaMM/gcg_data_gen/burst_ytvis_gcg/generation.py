import os
import json
import argparse
from PIL import Image,ImageDraw,ImageFont
from pycocotools import mask as maskUtils
import logging
import sys
from datetime import datetime
import google.generativeai as genai
import time




color_map = [(20, 255, 20), (20, 20, 255), (255, 20, 20), (20, 255, 255), (255, 20, 255), (255, 255, 20), (42, 42, 128), (165, 42, 42), (134, 134, 103), (0, 0, 142), (255, 109, 65), \
        (0, 226, 252), (5, 121, 0), (0, 60, 100), (250, 170, 30), (100, 170, 30), (179, 0, 194), (255, 77, 255), (120, 166, 157), \
        (73, 77, 174), (0, 80, 100), (182, 182, 255), (0, 143, 149), (174, 57, 255), (0, 0, 230), (72, 0, 118), (255, 179, 240), \
        (0, 125, 92), (209, 0, 151), (188, 208, 182), (145, 148, 174), (106, 0, 228), (0, 0, 70), (199, 100, 0), (166, 196, 102), \
        (110, 76, 0), (133, 129, 255), (0, 0, 192), (183, 130, 88), (130, 114, 135), (107, 142, 35), (0, 228, 0), (174, 255, 243), (255, 208, 186)]


def convert_txt2json(ori_txt_path,empty_obj=None):
    print('Generation complete, conversion format ...')
    merge_lines=[]
    total_cap={}
    total_list=[]
    with open(ori_txt_path, 'r') as file:
        for line in file:
            if line =='/n' or 'block_reason: OTHER' in line:
                continue
            
            elif not line.startswith('2024'):
                if merge_lines:
                    merge_lines[-1]=merge_lines[-1].strip() + ' ' + line.strip()
                else:
                    merge_lines.append(line.strip())
            else:
                merge_lines.append(line.strip())
        for_ind=[]
        for inds in merge_lines:
            if len(inds.strip().split(' '))<6:
                continue
            if inds.strip().split(' ')[6]=='0' or inds.strip().split(' ')[6]=='1' or inds.strip().split(' ')[6]=='2' or inds.strip().split(' ')[6]=='3'or inds.strip().split(' ')[6]=='5':
                for_ind.append(int(inds.strip().split(' ')[5].split('/')[0]))
                print(inds.strip().split(' ')[5])
                continue
            ind=inds.strip().split(' ')[5]
            idx=inds.strip().index(ind)
            if inds.strip().split(' ')[5].split('/')[0] in total_cap:
                print(inds.strip().split(' ')[5].split('/')[0])
            if empty_obj is not None and inds.strip().split(' ')[5].split('/')[0] in empty_obj:
                continue
            total_cap[inds.strip().split(' ')[5].split('/')[0]]=inds.strip()[idx+len(ind)+1:]
            total_list.append(inds)


    json_out_path=ori_txt_path.replace('.txt','.json')
    with open(json_out_path, 'w') as json_file:
        json.dump(total_cap, json_file)
    
    mod_txt_out_path =ori_txt_path.replace('.txt','mod.txt')
    with open(mod_txt_out_path, 'w') as txt_file:
        for key in total_list:
            txt_file.write(f"{key} \n")
            txt_file.write(f" \n")
    print('Finished')

def crop_by_box(image: Image.Image, bbox: tuple) -> Image.Image:
    """
    Crop the given image based on the provided bounding box, adjusting the size 
    and position as described.

    Parameters:
    image (PIL.Image.Image): The image to crop.
    bbox (tuple): A tuple of (x, y, width, height) where (x, y) is the center of 
                  the bounding box and (width, height) are its dimensions.

    Returns:
    PIL.Image.Image: The cropped image.
    """
    x, y, width, height = bbox
    
    # Calculate the new top-left corner (new_x, new_y) and the size (new_w, new_h)
    new_x = max(x - width / 2, 0)
    new_y = max(y - height / 2, 0)
    new_w = min(2*width , image.width) 
    new_h = min(2*height , image.height) 
    
    # Define the new bounding box based on calculated values
    new_bbox = (new_x, new_y, new_x + new_w, new_y + new_h)
    
    # Crop the image using the new bounding box
    return image.crop(new_bbox)

def gen_step1(instruction,model,ann_path,video_path,output_file,logger):
    max_images_length = 40
    frame_count=0

    ann_file=json.load(open(ann_path))
    video_name={}
    cat_name={}
    for video in ann_file['videos']:
        id =video['id']
        video_name[id]=video['file_names']
    for cat in ann_file['categories']:
        id =cat['id']
        cat_name[id]=cat['name']
    annos=ann_file['annotations']
    empty_obj=[]
    iii=0
    for ann in annos:
        if iii>2:
            break
        iii+=1
        count_box=0
        file_names=video_name[ann['video_id']]
        cate=cat_name[ann['category_id']]
        mod_instruction=instruction.replace('<cls>',cate)


        total_num_frames=len(file_names)
        sampling_interval = int(total_num_frames / max_images_length)
        if sampling_interval == 0:
            sampling_interval = 1
        base64Frames = []
        for fid,frame_name in enumerate(file_names):
            
            frame_path=os.path.join(video_path,frame_name)
            frame=Image.open(frame_path)
            if ann['segmentations'][fid] is not None:
                if ann['bboxes'][fid] is not None:
                    bbox=ann['bboxes'][fid]

                else:
                    bbox = maskUtils.toBbox(ann['segmentations'][fid])
                if (bbox[2]*bbox[3])>2500:
                    count_box+=1
                    frame=crop_by_box(frame,bbox)
                    # save_path=os.path.join('out_crop/',str(ann['id'])+frame_name.split('/')[-2],frame_name.split('/')[-1])
                    # save_dir=os.path.dirname(save_path)
                    # os.makedirs(save_dir,exist_ok=True)
                    # frame.save(save_path)
                
                    if frame_count % sampling_interval == 0:  
                        base64Frames.append(frame)
                    frame_count += 1
                else:
                    continue

                if len(base64Frames) >= max_images_length:
                    break
            else:
                continue
        if count_box==0:
            empty_obj.append(str(ann['id']))
        if len(base64Frames) ==0:
            continue
        time.sleep(20)
        response = model.generate_content(base64Frames+[mod_instruction],
                                        safety_settings=[
                                                {
                                                    "category": "HARM_CATEGORY_HARASSMENT",
                                                    "threshold": "BLOCK_NONE",
                                                },
                                                {
                                                    "category": "HARM_CATEGORY_HATE_SPEECH",
                                                    "threshold": "BLOCK_NONE",
                                                },
                                                {
                                                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                                                    "threshold": "BLOCK_NONE",
                                                },
                                                {
                                                    "category": "HARM_CATEGORY_DANGEROUS",
                                                    "threshold": "BLOCK_NONE",
                                                },
                                                ],request_options={"timeout": 600}
                                            )
        if 'block_reason' in response.prompt_feedback:
            logger.info(f"{str(ann['id'])}/{os.path.dirname(file_names[0])}: {response.prompt_feedback}")
        elif response.parts==[]:
            logger.info(f"{str(ann['id'])}/{os.path.dirname(file_names[0])}: {response.candidates[0].finish_reason}")
        else:
            logger.info(f"{str(ann['id'])}/{os.path.dirname(file_names[0])}: {response.text}")
    convert_txt2json(output_file,empty_obj)    
    
       
def gen_step2(instruction,model,ann_path,video_path,caption_file,output_file,logger):
    max_images_length = 40
    frame_count=0
 
    ann_file=json.load(open(ann_path))
    caption_file=json.load(open(caption_file))
    video_name={}
    cat_name={}
    for video in ann_file['videos']:
        id =video['id']
        video_name[id]=video['file_names']
    for cat in ann_file['categories']:
        id =cat['id']
        cat_name[id]=cat['name']
    annos=ann_file['annotations']
    empty_obj=[]
    iii=0
    for ann in annos:
        if iii>2:
            break
        iii+=1
        count_box=0
        file_names=video_name[ann['video_id']]
        cate=cat_name[ann['category_id']]
        
        obj_caption=caption_file[str(ann['id'])]
        instruction=instruction.replace('<cap>',obj_caption)
        instruction=instruction.replace('<obj_id>',str(ann['id']))
        mod_instruction=instruction.replace('<cls>',cate)
   

        total_num_frames=len(file_names)
        sampling_interval = int(total_num_frames / max_images_length)
        if sampling_interval == 0:
            sampling_interval = 1
        base64Frames = []
        for fid,frame_name in enumerate(file_names):
            frame_path=os.path.join(video_path,frame_name)
            frame=Image.open(frame_path)
            draw = ImageDraw.Draw(frame)
            img_width, img_height = frame.size
            if ann['segmentations'][fid] is not None:
                if ann['bboxes'][fid] is not None:
                    bbox=ann['bboxes'][fid]

                else:
                    bbox = maskUtils.toBbox(ann['segmentations'][fid])
                if (bbox[2]*bbox[3])>2500:
                    count_box+=1
            
                    x, y, w, h = bbox
                    rectangle_coordinates = [(x, y), (x+w, y+h)] 
                    color = (255, 20, 20) 
                    text_content = str(ann['id']) 
                    draw.rectangle(rectangle_coordinates, outline=color,width=2)
                    font_size = 20
                    font = ImageFont.load_default().font_variant(size=font_size)
                    text_position=(x+5,y)
                    draw.text(text_position, text_content, fill=color, font=font)
                    frame=frame.resize((int(img_width/2), int(img_height/2)))
                    # save_path=os.path.join('out_draw/',str(ann['id'])+frame_name.split('/')[-2],frame_name.split('/')[-1])
                    # save_dir=os.path.dirname(save_path)
                    # os.makedirs(save_dir,exist_ok=True)
                    # frame.save(save_path)
                    if frame_count % sampling_interval == 0:
                        base64Frames.append(frame)
                    frame_count += 1
                if len(base64Frames) >= max_images_length:
                    break
            else:
                continue
        if count_box==0:
            empty_obj.append(str(ann['id']))
        if len(base64Frames) ==0:
            continue
        time.sleep(20)
        response = model.generate_content(base64Frames+[mod_instruction],
                                        safety_settings=[
                                                {
                                                    "category": "HARM_CATEGORY_HARASSMENT",
                                                    "threshold": "BLOCK_NONE",
                                                },
                                                {
                                                    "category": "HARM_CATEGORY_HATE_SPEECH",
                                                    "threshold": "BLOCK_NONE",
                                                },
                                                {
                                                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                                                    "threshold": "BLOCK_NONE",
                                                },
                                                {
                                                    "category": "HARM_CATEGORY_DANGEROUS",
                                                    "threshold": "BLOCK_NONE",
                                                },
                                                ],request_options={"timeout": 600}
                                            )
        if 'block_reason' in response.prompt_feedback:
            logger.info(f"{str(ann['id'])}/{os.path.dirname(file_names[0])}: {response.prompt_feedback}")
        elif response.parts==[]:
            logger.info(f"{str(ann['id'])}/{os.path.dirname(file_names[0])}: {response.candidates[0].finish_reason}")
        else:
            logger.info(f"{str(ann['id'])}/{os.path.dirname(file_names[0])}: {response.text}")
    convert_txt2json(output_file,empty_obj)  


def gen_step3(instruction,model,ann_path,video_path,caption_file,output_file,logger):
    max_images_length = 40
    frame_count=0
    font_size = 20
    ann_file=json.load(open(ann_path))
    caption_file=json.load(open(caption_file))
    video_name={}
    cat_name={}
    video_cap={}
    for ann in ann_file['annotations']:
        if ann['video_id'] not in video_cap.keys():
            video_cap[ann['video_id']]=[]
        if str(ann['id']) in caption_file.keys():
            video_cap[ann['video_id']].append(dict(cls_id=ann['category_id'],seg=ann['segmentations'],bboxes=ann['bboxes'],cap=caption_file[str(ann['id'])],obj_id=ann['id'],ann_id=len(video_cap[ann['video_id']])))
    
    for video in ann_file['videos']:
        id =video['id']
        video_name[id]=video['file_names']
    for cat in ann_file['categories']:
        id =cat['id']
        cat_name[id]=cat['name']
    
    for vid, (video_id,frame_path) in enumerate(video_name.items()):
        if vid>1:
            break
        caps=''
        bbox=[]
        for cap in video_cap[video_id]:
            caps+='The obj_'+str(cap['ann_id'])+' must be surrounded by a rectangular box with color number '+ str(cap['ann_id'])+'. It is a'+cat_name[cap['cls_id']]+'. '+ cap['cap']+' '
            bbox.append(cap['bboxes'])
        prepared_instruction=instruction.replace('<cap>',caps)
        total_num_frames=len(frame_path)
        sampling_interval = int(total_num_frames / max_images_length)
        if sampling_interval == 0:
            sampling_interval = 1
        base64Frames = []
        for f_id,frame_name in enumerate(frame_path):
            frame_path=os.path.join(video_path,frame_name)
            frame=Image.open(frame_path)
            draw = ImageDraw.Draw(frame)
            for box_id,box in enumerate(bbox):
                    if box[f_id]==None:
                        continue
                    else:
                        x,y,w,h=box[f_id]
                        rectangle_coordinates = [(x, y), (x+w, y+h)] 
                        color = color_map[box_id]  
                        text_content = str(box_id) 
                        draw.rectangle(rectangle_coordinates, outline=color,width=2)
                        
                        font = ImageFont.load_default().font_variant(size=font_size)
                        text_position=(x+5,y)
                        draw.text(text_position, text_content, fill=color, font=font)
            img_width, img_height = frame.size

            # out_frame_path=os.path.join('output_f',frame_name)
            # os.makedirs(os.path.dirname(out_frame_path),exist_ok=True)
            # frame.save(out_frame_path)
            frame=frame.resize((int(img_width/2), int(img_height/2)))
            if frame_count % sampling_interval == 0:
                        base64Frames.append(frame)  
            frame_count += 1
            if len(base64Frames) >= max_images_length:
                    break

        if len(base64Frames) ==0:
            continue
        time.sleep(20)
        response = model.generate_content(base64Frames+[prepared_instruction],
                                            safety_settings=[
                                                {
                                                    "category": "HARM_CATEGORY_HARASSMENT",
                                                    "threshold": "BLOCK_NONE",
                                                },
                                                {
                                                    "category": "HARM_CATEGORY_HATE_SPEECH",
                                                    "threshold": "BLOCK_NONE",
                                                },
                                                {
                                                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                                                    "threshold": "BLOCK_NONE",
                                                },
                                                {
                                                    "category": "HARM_CATEGORY_DANGEROUS",
                                                    "threshold": "BLOCK_NONE",
                                                },
                                                ],request_options={"timeout": 600}
                                            )
        if 'block_reason' in response.prompt_feedback:
            logger.info(f"{str(vid)}/{os.path.dirname(frame_path[0])}: {response.prompt_feedback}")
        elif response.parts==[]:
            logger.info(f"{str(vid)}/{os.path.dirname(frame_path[0])}: {response.candidates[0].finish_reason}")
        else:
            logger.info(f"{str(vid)}/{os.path.dirname(frame_path[0])}: {response.text}")
    convert_txt2json(output_file) 

    
def get_arguments():
    parser = argparse.ArgumentParser(description="Inference parameters")

    parser.add_argument("--ann_path", type=str,default='path to annotations file', help="path to annotations")
    parser.add_argument("--question", type=str, help="question to ask")
    parser.add_argument("--video_path", type=str, help="path to the video file")
    parser.add_argument("--step", type=str, default='step1', help="generation step")
    parser.add_argument("--output_file", type=str, default='results.txt', help="file name of results")
    parser.add_argument("--caption_file", type=str, help="path to generated caption file")

    return parser.parse_args()

if __name__ == "__main__":
    
    os.environ["GOOGLE_API_KEY"] = ""##gemini

    args=get_arguments()
    output_file = os.path.join('output', args.output_file)
    
    logging.basicConfig(filename=output_file, level=logging.INFO, filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logger = logging.getLogger(__name__)

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    video_path=args.video_path
    instruction=args.question
    ann_path=args.ann_path
  
    step=args.step
    caption_file=args.caption_file
  
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    if step=='step1':
        pred=gen_step1(instruction,model,ann_path,video_path,output_file,logger)
    elif step=='step2':
        pred=gen_step2(instruction,model,ann_path,video_path,caption_file,output_file,logger)
    elif step=='step3':
        pred=gen_step3(instruction,model,ann_path,video_path,caption_file,output_file,logger)
    else:
        raise ValueError(f"Wrong step")
