# finetune a video-pretrained STCN model on medical images for 10k iterations
# without cycle consistency loss

torchrun ./maskprop/Med-STCN/train.py \
--id retrain_s4_ft_from_med \
--load_network ./saves/stcn.pth \
--abd1k_root ./data/AbdomenCT-1K/Organ-12-Subset_finetune \
--stage 4 \
--batch_size 10 \
--iterations 10000 \
--save_model_interval 10000 