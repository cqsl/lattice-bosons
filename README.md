# lattice-bosons
Accompanying code for "Accurate neural quantum states for interacting lattice bosons" (https://arxiv.org/abs/2404.07869).

[![Slack](https://img.shields.io/badge/slack-chat-green.svg)](https://join.slack.com/t/mlquantum/shared_invite/zt-19wibmfdv-LLRI6i43wrLev6oQX0OfOw)


## Content of the repository

- `vmc.py`: script for running a VMC simulation from a json file containing all hyperparameters.
- `example.json`: example of a parameter file for a 8x8 periodic lattice at unit filling.

- `bnqs/`: folder containing all classes relevant for simulating Bose Hubbard.
    - `models/`: folder containing the flax linen modules.
    - `operators/`: folder containing the definition of some custom jax operators.
    - `sampler/`: folder containing our custom Metropolis-Hastings rule.

## Simple example

Quick example (8x8 at $U=8J$):

`python vmc.py --jobid 1234 --parameters example.json`

## Software version

This code was successfully tested on GPU with python 3.11.7, CUDA V12.4.131 and the following environment (see `requirements.txt`):

| package | version |
| --- | --- |
| `flax` | `0.10.2` |
| `jax` | `0.4.37` |
| `jax-cuda12-pjrt` | `0.4.36` |
| `jax-cuda12-plugin` | `0.4.36` |
| `jaxlib` | `0.4.36` |
| `jaxtyping` | `0.2.36` |
| `NetKet` | `3.15.1` |
| `nvidia-cublas-cu12` | `12.6.4.1` |
| `nvidia-cuda-cupti-cu12` | `12.6.80` |
| `nvidia-cuda-nvcc-cu12` | `12.6.85` |
| `nvidia-cuda-runtime-cu12` | `12.6.77` |
| `nvidia-cudnn-cu12` | `9.6.0.74` |
| `nvidia-cufft-cu12` | `11.3.0.4` |
| `nvidia-cusolver-cu12` | `11.7.1.2` |
| `nvidia-cusparse-cu12` | `12.5.4.2` |
| `nvidia-nccl-cu12` | `2.23.4` |
| `nvidia-nvjitlink-cu12` | `12.6.85` |
