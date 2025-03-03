import json 
import re
import argparse


def get_arguments():
    parser = argparse.ArgumentParser(description="Inference parameters")

    parser.add_argument("--ann_file", type=str, help="path to annotations")
    parser.add_argument("--obj_cap", type=str, help="the generated object-level caption")
    parser.add_argument("--dense_cap", type=str, help="the generated video-level caption")
    parser.add_argument("--out_ann_file", type=str, help="path to the final annotations file")
  
    return parser.parse_args()

if __name__ == "__main__":
    args=get_arguments()
    ann_file= json.load(open(args.ann_file))
    obj_cap=json.load(open(args.obj_cap))
    dense_cap=json.load(open(args.dense_cap))
    out_ann_file= args.out_ann_file


    for ann in ann_file['annotations']:
        if str(ann['id']) in obj_cap.keys():
            ann['cap']=obj_cap[str(ann['id'])]
        else:
            ann['cap']=None

    video_cap={}
    for ann in ann_file['annotations']:
            if ann['video_id'] not in video_cap.keys():
                video_cap[ann['video_id']]=[]
            if str(ann['id']) in obj_cap.keys():
                video_cap[ann['video_id']].append(dict(cls_id=ann['category_id'],seg=ann['segmentations'],bboxes=ann['bboxes'],cap=obj_cap[str(ann['id'])],obj_id=ann['id'],ann_id=len(video_cap[ann['video_id']])))
    for vid,video in enumerate(ann_file['videos']):
        if str(vid) in dense_cap.keys():
            if len(video_cap[video['id']])==0:
                video['dense_cap']={}
                video['dense_cap']['v_id2o_id']=None
                video['dense_cap']['token_pos']=None
                video['dense_cap']['mask_id']=None
                video['dense_cap']['caption']=None
            else:
                video['dense_cap']={}
                video['dense_cap']['v_id2o_id']={}
                video['dense_cap']['token_pos']=[]
                video['dense_cap']['mask_id']=[]
                for an in video_cap[video['id']]:
                    video['dense_cap']['v_id2o_id'][an['ann_id']]=an['obj_id']
                spl_dense_cap=dense_cap[str(vid)].split(' ')
                me_cap=[]
                for wid,word in enumerate(spl_dense_cap):
                    if '{obj_' in word :  
                        video['dense_cap']['token_pos'].append(len(me_cap)-1)
                        m_id=int(re.findall(r'\d+',word)[0])
                        if m_id<len(video['dense_cap']['v_id2o_id']):
                            video['dense_cap']['mask_id'].append(video['dense_cap']['v_id2o_id'][m_id])
                        # spl_dense_cap.remove(word)
                        # wid-=1
                    else:
                        me_cap.append(word)
                ori_dense=' '.join(me_cap) 
                video['dense_cap']['caption']=ori_dense
        else:
            video['dense_cap']={}
            video['dense_cap']['v_id2o_id']=None
            video['dense_cap']['token_pos']=None
            video['dense_cap']['mask_id']=None
            video['dense_cap']['caption']=None

    with open(out_ann_file, 'w') as json_file:
        json.dump(ann_file, json_file)
    print('Finished')
            
         