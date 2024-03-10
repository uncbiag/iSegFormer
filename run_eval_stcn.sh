
python3 ./maskprop/Med-STCN/eval_med_stcn.py \
  --model ./saves/stcn_ft_with_cycle.pth \
  --med_path ./data/AbdomenCT-1K/Organ-12-Subset_frames \
  --output ./results \
  --include_last

# calcualte the segmentation metrics for an assigned label
python ./maskprop/Med-STCN/abdomen1k-evaluation/evaluation_method.py \
--data_folder ./data/AbdomenCT-1K/Organ-12-Subset_frames \
--seg_folder ./results \
--label 1 # Set label from 1 to 12 for 12 organs AbdomenCT-1K