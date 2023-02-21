# docker_folder=/playpen-raid2/qinliu/projects/nvidia-docker
# ${docker_folder}/nvidia-docker run \
#                             --name="qin_0_1_stcn_dev" \
#                             --gpus all \
#                             -e DISPLAY=${DISPLAY} \
#                             -v ${HOME}/.Xauthority:/root/.Xauthority:rw \
#                             --net host \
#                             -v /playpen-raid2/qinliu/data/:/work/data \
#                             -v /playpen-raid2/qinliu/projects/:/work/projects \
#                             --rm -it stcn bash

docker run -e DISPLAY=$DISPLAY \
           -v /tmp/Xauthority-qinliu19:/root/.Xauthority:rw \
           --shm-size 16G \
           --gpus all \
           -v /playpen-raid2/qinliu/data:/work/data \
           -v /playpen-raid2/qinliu/projects:/work/projects \
           --net host --rm -it stcn bash