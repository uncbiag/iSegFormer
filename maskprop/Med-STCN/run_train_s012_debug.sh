git config --global --add safe.directory /work/projects/STCN

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python3 -m torch.distributed.launch \
--master_port 9842 --nproc_per_node=1 train.py --id retrain_s012_debug \
--load_network /work/projects/STCN/saves/stcn_s01.pth \
--stage 2 \
--debug \
--use_cycle_loss
