export CUDA_VISIBLE_DEVICES=0

python3 interactive_gui.py \
--images /playpen-raid2/qinliu/projects/iSegFormer/maskprop/XMem/workspace/lung_001/images/ \
--num_objects=1 \
--prop_model=/playpen-raid2/qinliu/projects/iSegFormer/maskprop/Med-STCN/saves/Aug01_22.03.33_retrain_s4_ft_from_med_10000.pth 