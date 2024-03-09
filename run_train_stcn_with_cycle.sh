# finetune a video-pretrained STCN model on medical images for 10k iterations
# with cycle consistency loss

torchrun ./maskprop/Med-STCN/train.py \
--id retrain_s4_ft_from_med \
--load_network ./saves/stcn.pth \
--stage 4 \
--batch_size 10 \
--iterations 10000 \
--save_model_interval 10000 \
--use_cycle_loss
