git config --global --add safe.directory /work/projects/STCN

sota_s012_model=/work/projects/STCN/saves/Jul18_15.14.54_retrain_s012/Jul18_15.14.54_retrain_s012_checkpoint.pth

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python3 -m torch.distributed.launch \
--master_port 9846 --nproc_per_node=1 train.py --id ft_s012 \
--load_model ${sota_s012_model} \
--stage 5 \
--use_const_skip_values \
--save_model_interval 10000
