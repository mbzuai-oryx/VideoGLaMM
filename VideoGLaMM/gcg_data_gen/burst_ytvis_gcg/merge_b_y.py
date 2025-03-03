import json 
import copy
import argparse
import os


def get_arguments():
    parser = argparse.ArgumentParser(description="Inference parameters")

    parser.add_argument("--burst_train", type=str, help="path to burst train generated annotations")
    parser.add_argument("--burst_val", type=str, help="path to burst val generated annotations")
    parser.add_argument("--yt19_train", type=str, help="path to yt19 train generated annotations")
    parser.add_argument("--hq_ann_file", type=str, help="path to ytvis_hq annotations file")
    parser.add_argument("--out_ann_path", type=str, help="path to the final annotations file")
  
    return parser.parse_args()

if __name__ == "__main__":
    args=get_arguments()
    burst_train= json.load(open(args.burst_train))
    burst_val=json.load(open(args.burst_val))
    yt19_train=json.load(open(args.yt19_train))
    hq_ann_file= args.hq_ann_file
    out_ann_path= args.out_ann_path
    

    b_trian=copy.deepcopy(burst_train)
    b_val=copy.deepcopy(burst_val)

    v_train_max=2988
    a_train_max=2551

    v_val_max=2991
    a_val_max=5479
    split=['ytvis_hq-train.json','ytvis_hq-val.json','ytvis_hq-test.json']
    f1 = json.load(open(os.path.join(hq_ann_file,split[0])))
    f2 = json.load(open(os.path.join(hq_ann_file,split[1])))
    f3 = json.load(open(os.path.join(hq_ann_file,split[2])))
    v1_id=[]
    v2_id=[]
    for v1 in f1['videos']:
        v1_id.append(v1['id'])

    for v2 in f2['videos']:
        v2_id.append(v2['id'])
    for v3 in f3['videos']:
        v2_id.append(v3['id'])

    v_1_re=[]
    v_2_re=[55,624,1985,1032,163,2150,1162,921,460,378,743,1412,182,
            717,296,237,561,308,2181,75,694,3,779,1375,1198,1474,123,
            165,847,2055,1905,482,1582,426,1570,617,730,585,1171,2212,
            1794,2189,119,788,2112,358,2074,1277,1891,1587574,1593,1006,##
            1929,1554,2065,345,1694,868,599,895,204,1527,190,906,533,1858,535,
            1864,1917,302,1442,1430,918,1804,1466,461,514,2131,411,2089,1406,733,
            1189,2058,913,1954,958,85,1328,212,1004,1507,935,407,445,353,202,2049,
            1400,2086,1680,830,1102,1518,839,449,753,765,1514,312,1724,2075,989,1571,16,2197,301,957,363,128,219,##
            60,350,1163,591,1272,1241,360,2094,201,473,
            646,440,1614,213,586,539,1716,384,1101,1733,
            1446,1020,917,442,325,1633,608,1106,1612,1849,
            1297,1337,524,1429,1399,1105,1519,237,911,270,
            686,1827,173,2165,1215,175,512,103,38,471,
            638,1803,1165,1517,1784,366,29,291,540,636,
            24,728,365,836,803,1525]

    v1_del = [x for x in v1_id if x not in v_2_re]
    v2_del = [x for x in v2_id if x not in v_1_re]

    for v1re in v_1_re:
        if v1re not in v1_del:
            v1_del.append(v1re)
    for v2re in v_2_re:
        if v2re not in v2_del:
            v2_del.append(v2re)


    a1_id=[]
    a2_id=[]
    for a1 in yt19_train['annotations']:
        if a1['video_id'] in v1_del:
            a1_id.append(a1['id'])
        if a1['video_id'] in v2_del:
            a2_id.append(a1['id'])

    cat_dic={}

    for cat in burst_train['categories']:
        cat_dic[cat['name']]=cat['id']

    id_map={}
    manul_map={6:86,7:270,11:305,17:446,21:273,23:483,24:484,26:44,33:485,36:274,39:486,9:487}
    new_manul_map={23:483,24:484,33:485,39:486,9:487}
    for yt_cat in yt19_train['categories']:
        if yt_cat['name'] in cat_dic.keys():
            id_map[yt_cat['id']]=cat_dic[yt_cat['name']]
    id_map.update(manul_map)

    new_category=burst_train['categories']
    for yt_cid in yt19_train['categories']:
        if yt_cid['id'] in new_manul_map.keys():
            yt_cid['id']=new_manul_map[yt_cid['id']]
            new_category.append(yt_cid)
    b_trian['categories']=new_category
    b_val['categories']=new_category
    for b_t_v in b_trian['videos']:
        b_t_v['dataset_split']='burst'
    for b_v_v in b_val['videos']:
        b_v_v['dataset_split']='burst'

    for vid,video in enumerate(yt19_train['videos']):
        if video['id'] in v2_del:
            video['id']=video['id']+v_train_max
            for idxx,m_id in enumerate(video['dense_cap']['mask_id']):
                video['dense_cap']['mask_id'][idxx]=m_id+a_train_max
            video['dataset_split']='yt19'
            b_trian['videos'].append(video)
        else:
            video['id']=video['id']+v_val_max
            for idxx,m_id in enumerate(video['dense_cap']['mask_id']):
                video['dense_cap']['mask_id'][idxx]=m_id+a_val_max
            video['dataset_split']='yt19'
            b_val['videos'].append(video)

    for ann_id,ann in enumerate(yt19_train['annotations']):
        if ann['id'] in a2_id:
            ann['id']=ann['id']+a_train_max
            ann['video_id']= ann['video_id']+v_train_max
            ann['category_id']=id_map[ann['category_id']]
            b_trian['annotations'].append(ann)
        else:
            ann['id']=ann['id']+a_val_max
            ann['video_id']= ann['video_id']+v_val_max
            ann['category_id']=id_map[ann['category_id']]
            b_val['annotations'].append(ann)


    merge_train_json_path=os.path.join(out_ann_path,'bay_train_add_cap_re_2.json')
    with open(merge_train_json_path, 'w') as t_json_file:
        json.dump(b_trian, t_json_file)

    merge_val_json_path=os.path.join(out_ann_path,'bay_val_add_cap_re_2.json')
    with open(merge_val_json_path, 'w') as v_json_file:
        json.dump(b_val, v_json_file)
    ################################################
        
    print('Finished')