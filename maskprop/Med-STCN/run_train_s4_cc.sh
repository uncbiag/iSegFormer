git config --global --add safe.directory /playpen-raid2/qinliu/models/model_0224_2023

CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=2 python3 -m torch.distributed.launch \
--master_port 9843 --nproc_per_node=2 train.py --id retrain_s4_ft_from_med \
--load_network /playpen-raid2/qinliu/projects/iSegFormer/maskprop/Med-STCN/saves/stcn.pth \
--stage 4 \
--use_cycle_loss
