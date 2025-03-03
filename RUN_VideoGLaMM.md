# Checkpoints

* Download SAM2 checkpoints from [here](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt)

* Download InternVideo2 checkpoints from [here](https://huggingface.co/OpenGVLab/InternVideo2-Stage2_1B-224p-f4)

* Download VideoGLaMM checkpoints from [here](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/shehan_munasinghe_mbzuai_ac_ae/Etucj3LuqdRDocrle_8eJbcB8C11u-020AX7fwIYWJh-dg?e=uPanYM)

# Command Line Demo

    python chat.py \
        --llava_version_or_path="<path_to_ckpts>" \
        --use_sam2_video_branch \
        --base_model_type="vgpt|phi3"

# Evaluation


## GCG Task

    python eval_gcg_infer.py \
        --llava_version_or_path="<path_to_ckpts>" \
        --use_sam2_video_branch \
        --base_model_type="vgpt|phi3" \
        --dataset_name='video_gcg'\
        --vis_save_path="./vis_output_path"

    export OPENAI_API_KEY='<YOUR KEY>'

    python eval_gcg_metrics.py \
        --vis_save_path="<path_to_ckpts>" \
        --eval_miou --eval_recall --eval_caption --use_clair

## MeViS

    python eval_mevis.py \
        --llava_version_or_path="<path_to_ckpts>" \
        --use_sam2_video_branch \
        --base_model_type="vgpt|phi3" \
        --dataset_name="MEVIS|valid"\
        --vis_save_path="./vis_output_path"

You can use following command to prepare .zip submission file

    cd [vis_output_path]
    zip -r ../mevis_out.zip *

## VidSTG

    python eval_grounding.py \
        --llava_version_or_path="<path_to_ckpts>" \
        --use_sam2_video_branch \
        --base_model_type="vgpt|phi3" \
        --dataset_name="vidstg"\
        --vis_save_path="./vis_output_path"


## HCSTVG


    python eval_grounding.py \
        --llava_version_or_path="<path_to_ckpts>" \
        --use_sam2_video_branch \
        --base_model_type="vgpt|phi3" \
        --dataset_name="hcstvg"\
        --vis_save_path="./vis_output_path"

## ReferYTVOS

    python eval_mevis.py \
        --llava_version_or_path="<path_to_ckpts>" \
        --use_sam2_video_branch \
        --base_model_type="vgpt|phi3" \
        --dataset_name="ReferYouTubeVOS|valid" \
        --vis_save_path="./vis_output_path"


## ReferDAVIS17


    python eval_referdavis_infer.py \
        --llava_version_or_path="<path_to_ckpts>" \
        --use_sam2_video_branch \
        --base_model_type="vgpt|phi3" \
        --dataset_name="ReferDAVIS|valid" \
        --vis_save_path="./vis_output_path"

    python eval_referdavis_metrics.py --output_dir \
        "./vis_output_path"