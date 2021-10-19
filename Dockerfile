FROM nvidia/cudagl:11.0-base-ubuntu18.04
WORKDIR /workspace
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        cmake \
        curl \
        gcc-8 \
        g++-8 \
        wget \
        bzip2 \
        git \
        vim \
        tmux \
        git \
        unzip \
        vulkan-utils \
        mesa-common-dev \
        mesa-vulkan-drivers \
        libosmesa6-dev \
        libgl1-mesa-glx \
        libglfw3 \
        patchelf \
        libglu1-mesa \
        libxext6 \
        graphviz \
        libxtst6 \
        libxrender1 \
        libxi6 \
        libegl1 \
        libxrandr-dev \
        libxinerama-dev \
        libxcursor-dev

RUN apt-get clean && rm -rf /var/lib/apt/lists/*

COPY isaacloco /workspace

RUN curl -o ~/miniconda.sh -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

RUN apt-get update \
    && apt-get install -y mpich && \ 
    cd /workspace/isaacloco && \
    bash create_conda_env_rlgpu.sh && \
    /opt/conda/bin/conda clean -ya

RUN rm -rf /root/.cache/pip

RUN /opt/conda/envs/rlgpu/bin/pip install pytorch3d

RUN apt-get update && \
    apt-get install libxi6

RUN rm -rf /root/.cache/pip

RUN apt-get clean && rm -rf /var/lib/apt/lists/*
