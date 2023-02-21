# MODEL=./saves/stcn.pth

# Finetune model on videos
#MODEL=/work/projects/STCN/saves/Jul20_19.26.32_ft_s012/Jul20_19.26.32_ft_s012_400000.pth
#OUTPUT=/work/results/stcn_med_abdomen1k_Jul20_19.26.32_ft_s012_400k_no_cc

# Finetune model on Abdomen-1k without cc
#MODEL=/work/projects/STCN/saves/Aug01_15.34.08_retrain_s4_ft_from_med/Aug01_15.34.08_retrain_s4_ft_from_med_10000.pth
#OUTPUT=/work/results/stcn_med_abdomen1k_Aug01_15.34.08_ft_s012_10k_no_cc

# Finetune model on Abdomen-1k with cc
MODEL=/work/projects/STCN/saves/Aug01_22.03.33_retrain_s4_ft_from_med/Aug01_22.03.33_retrain_s4_ft_from_med_10000.pth
OUTPUT=/work/results/stcn_med_abdomen1k_Aug01_22.03.33_ft_s012_10k_cc

python3 eval_med.py \
  --model ${MODEL} \
  --output ${OUTPUT} \
  --include_last
