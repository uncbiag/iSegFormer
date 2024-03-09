# export CUDA_VISIBLE_DEVICES=0

# python3 interactive_gui.py \
# --images /playpen-raid2/qinliu/projects/iSegFormer/maskprop/XMem/workspace/lung_001/images/ \
# --num_objects=1 \
# --prop_model=/playpen-raid2/qinliu/projects/iSegFormer/maskprop/Med-STCN/saves/Aug01_22.03.33_retrain_s4_ft_from_med_10000.pth 

export CUDA_VISIBLE_DEVICES=0

MODEL_FOLDER=/playpen-raid2/qinliu/projects/iSegFormer/maskprop/Med-STCN/saves
MODEL_NAME=Aug01_22.03.33_retrain_s4_ft_from_med_10000.pth 
IMAGE_PATH=/playpen-raid2/qinliu/projects/iSegFormer/maskprop/XMem/workspace/lung_001/images/
VOLUME_PATH=/playpen-raid2/qinliu/data/OAI-ZIB/test_volumes/9102858_image.nii.gz

python3 interactive_gui.py \
--volume ${VOLUME_PATH} \
--num_objects=1 \
--prop_model=${MODEL_FOLDER}/${MODEL_NAME}