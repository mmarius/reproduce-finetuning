# Reproducing MRPC fine-tuning results of [Mosbach et al. (2021)](https://arxiv.org/abs/2006.04884)

This code runs with:

- Python 3.8
- NVIDIA CUDA 11.4.2 with cuBLAS 11.6.5.2
- PyTorch 1.10.0a0+0aef44c
- huggingface transformers 4.17.0

## Setup

- See `/docker/run_docker.txt` for instructions to create a docker image and docker container.

## Run fine-tuning

- Attach to the docker container and run: `bash /scripts/finetune.sh`.

## Results

- Performance on the MRPC development set are shown in `plot_results.ipynb`.