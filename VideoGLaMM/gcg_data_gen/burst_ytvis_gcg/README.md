
**Run the demo** <br>
Take YouTubeVIS2019 for example
```bash
# Generation step1 --rough description of each object
python generation.py --video_path video_data/youtube2019/train/JPEGImages --question These are frames from a video that I want to upload. What does the <cls> look like and what is the <cls> doing? --ann_path video_data/youtube2019/train.json --output_file generated_step1.txt --step step1


# Generation step2 --corrected description of the object
python generation.py --video_path video_data/youtube2019/train/JPEGImages --question These are frames from a video that I want to upload. Please modify this caption: <cap> The instance in the video is surrounded by a rectangular box with color number <obj_id>. The output caption must include what the <cls> looks like and what the <cls> is doing. Please do not mention any information about the bbox in the output. --ann_path video_data/youtube2019/train.json --output_file generated_step2.txt --step step2 --caption_file output/generated_step1.json

# Generation step3 --comprehensive description of the video
python generation.py --video_path video_data/youtube2019/train/JPEGImages --question These are frames from a video that I want to upload. In the video, the ID number of the box is on the top left of the box. There are some instance captions: '<cap>' Generate a dense caption that describes the video in detail based on the video and instance captions, including all of the instances mentioned in the instance captions and other instances in the video. Ensure that each instance mentioned in the instance caption appears exactly once in the dense caption, followed by the format {obj_} to indicate which instance caption the mentioned instance corresponds to. The {obj_} must directly follow the noun representing the instance.Please do not mention any information about the bbox in the output. --ann_path video_data/youtube2019/train.json --output_file generated_step3.txt --step step3 --caption_file output/generated_step2.json

# Manually review the {obj_id} in the generated video captions based on the video content

# Generate annotation file with caption
python generate_annotations.py --ann_file video_data/youtube2019/train.json --obj_cap output/generated_step2.json --dense_cap output/manual_generated_step3.json --out_ann_file generated_annotation.json


# Merge BURST and YouTubeVIS2019 annotation files
python merge_b_y.py --burst_train video_data/burst/train/b2y_train_add_cap_del_filtered_ann.json --burst_val video_data/burst/val/b2y_val_add_cap_del_filtered_ann.json --yt19_train video_data/ytvis_2019/train_add_cap_filtered_ann.json --hq_ann_file video_data/ytvis_2019/ --out_ann_path output
```
