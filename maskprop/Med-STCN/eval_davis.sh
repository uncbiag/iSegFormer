#MODEL=./saves/stcn.pth
#OUTPUT=/work/results/stcn

#MODEL=./saves/Jul12_21.14.02_retrain_s012/Jul12_21.14.02_retrain_s012_100000.pth
#OUTPUT=/work/results/Jul12_21.14.02_retrain_s012_100000

#MODEL=./saves/Jul12_21.13.52_retrain_s012_ft/Jul12_21.13.52_retrain_s012_ft_100000.pth
#OUTPUT=/work/results/Jul12_21.13.52_retrain_s012_ft_100000

#MODEL=./saves/Jul15_18.23.03_retrain_s012/Jul15_18.23.03_retrain_s012_50000.pth
#OUTPUT=/work/results/Jul15_18.23.03_retrain_s012/500000

#MODEL=./saves/Jul18_15.14.54_retrain_s012/Jul18_15.14.54_retrain_s012_300000.pth
#OUTPUT=/work/results/Jul18_15.14.54_retrain_s012/300000

#MODEL=./saves/Jul18_18.26.08_retrain_s012/Jul18_18.26.08_retrain_s012_300000.pth
#OUTPUT=/work/results/Jul18_18.26.08_retrain_s012/300000

#MODEL=./saves/Jul18_21.19.02_retrain_s012/Jul18_21.19.02_retrain_s012_300000.pth
#OUTPUT=/work/results/Jul18_21.19.02_retrain_s012/300000

#MODEL=./saves/Jul19_20.48.22_retrain_s012/Jul19_20.48.22_retrain_s012_300000.pth
#OUTPUT=/work/results/Jul19_20.48.22_retrain_s012/300000

#MODEL=./saves/Jul20_19.26.32_ft_s012/Jul20_19.26.32_ft_s012_350000.pth
#OUTPUT=/work/results/Jul20_19.26.32_ft_s012/350000

#MODEL=./saves/Jul20_19.28.06_ft_s012/Jul20_19.28.06_ft_s012_450000.pth
#OUTPUT=/work/results/Jul20_19.28.06_ft_s012/450000

MODEL=/work/projects/STCN/saves/Aug01_15.34.08_retrain_s4_ft_from_med/Aug01_15.34.08_retrain_s4_ft_from_med_10000.pth
OUTPUT=/work/results/Aug01_15.34.08_retrain_s4_ft_from_med_10000_davis

#python3 eval_davis.py \
#  --model ./saves/Jul07_22.46.10_retrain_s4/Jul07_22.46.10_retrain_s4_50000.pth \
#  --output /work/results/Jul07_22.46.10_retrain_s4_50000

python3 eval_davis.py \
  --model ${MODEL} \
  --output ${OUTPUT} \
  --include_last

python3 ./davis2017-evaluation/evaluation_method.py --task semi-supervised --results_path ${OUTPUT}

