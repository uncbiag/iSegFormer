FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
	    git \
	    curl \
        libglib2.0-0 \
        software-properties-common \
        python3.6-dev \
        python3-pip \
        python3-tk \
        nano \
        ffmpeg \
        libsm6 \
        libxext6 \
        libxcb-icccm4 \
        libxcb-image0 \
        libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-xinerama0 libxcb-xkb-dev libxkbcommon-x11-0

RUN pip3 install --upgrade pip
RUN pip3 install setuptools
RUN pip3 install matplotlib numpy pandas scipy tqdm pyyaml easydict scikit-image bridson Pillow ninja
RUN pip3 install imgaug mxboard graphviz
RUN pip3 install albumentations --no-deps
RUN pip3 install opencv-python-headless
RUN pip3 install Cython
RUN pip3 install torch==1.8.1
RUN pip3 install torchvision==0.9.1
RUN pip3 install scikit-learn
RUN pip3 install tensorboard
RUN pip3 install PyQt5 davisinteractive progressbar2 opencv-python networkx gitpython gdown Cython
RUN pip3 install SimpleITK
RUN pip3 install nano
RUN pip3 install git+https://github.com/cheind/py-thin-plate-spline

RUN apt-get install firefox -y

RUN mkdir /work
WORKDIR /work
RUN chmod -R 777 /work && chmod -R 777 /root

ENV TINI_VERSION v0.18.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash" ]
