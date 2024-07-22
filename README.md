# Language Group
Our code is mainly based on [wenet](https://github.com/wenet-e2e/wenet) version 2.0

- The configuration file for the experiment is in `./conf`
- The package used for the experimental conda environment is shown in `./environment_packages.txt`
- The source code for this experiment is located at `./src/Group-MoE`

## Introduction
We implement a highly flexible MoE model, based on the proposed dynamic language expert group, which allows us to flexibly carry out the design of the expert group according to the actual needs and to choose different topk for reasoning in order to realize the trade-off between performance and speed. And since we are based on the U2++ architecture, we also support streaming inference with different chunksizes.
## Train && Infer
If you want to reproduce our experiment, you just need to prepare the dataset and place it in the `./data` and run the following command
```
bash train.sh
bash infer.sh
```