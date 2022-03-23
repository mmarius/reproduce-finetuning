# Base image must at least have pytorch and CUDA installed.
# We are using NVIDIA NGC's PyTorch image here, see: https://ngc.nvidia.com/catalog/containers/nvidia:pytorch for latest version
# See https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html#framework-matrix-2021 for installed python, pytorch, etc. versions
FROM nvcr.io/nvidia/pytorch:21.10-py3

# Set path to CUDA
ENV CUDA_HOME=/usr/local/cuda

# Install additional programs
RUN apt update && \
    apt install -y build-essential \
    sudo \
    lsb-release \
    htop \
    wget \
    gnupg \
    git \
    curl \
    ca-certificates \
    vim \
    tmux && \
    rm -rf /var/lib/apt/lists

# Update pip
RUN SHA=ToUcHMe which python3
RUN python3 -m pip install --upgrade pip

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

# Install dependencies 
RUN python3 -m pip install numpy
RUN python3 -m pip install matplotlib
RUN python3 -m pip install pandas
RUN python3 -m pip install ipywidgets
RUN python3 -m pip install transformers
RUN python3 -m pip install accelerate
RUN python3 -m pip install datasets
RUN python3 -m pip install wandb

# Install additional dependencies
RUN python3 -m pip install autopep8
RUN python3 -m pip install attrdict
RUN python3 -m pip install pylint

# Specify a new user (USER_NAME and USER_UID are specified via --build-arg)
ARG USER_UID
ARG USER_NAME
ENV USER_GID=$USER_UID
ENV USER_GROUP="users"

# Create the user
RUN mkdir /home/$USER_NAME 
RUN useradd -l -d /home/$USER_NAME -u $USER_UID -g $USER_GROUP $USER_NAME

# Setup VSCode stuff (comment when not using vscode)
RUN mkdir /home/$USER_NAME/.vscode-server 
RUN mkdir /home/$USER_NAME/.vscode-server-insiders

# Change owner of home dir
RUN chown -R ${USER_UID}:${USER_GID} /home/$USER_NAME/

# Change owner of conda/python dir
# - This is needed to be able to install transformers inside the container using pip -e 
RUN chown ${USER_UID}:${USER_GID} /opt/conda/lib/python3.8/site-packages/
RUN chown -R ${USER_UID}:${USER_GID} /opt/conda/bin/

# Set workdir when starting container
WORKDIR /reproduce-finetuning

# Add workdir to PYTHONPATH
ENV PYTHONPATH="$PYTHONPATH:/reproduce-finetuning"

CMD ["/bin/bash"]
