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

## Content of the repository

Quick example (8x8 at $U=8J$):

`python vmc.py --jobid 1234 --parameters example.json`