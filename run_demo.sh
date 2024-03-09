export CUDA_VISIBLE_DEVICES=0

MODEL_PATH=./saves/stcn_ft_with_cycle.pth 
VOLUME_PATH=./saves/image.nii.gz

python3 ./maskprop/MiVOS/interactive_gui.py --volume ${VOLUME_PATH} --num_objects=1 --prop_model=${MODEL_PATH}