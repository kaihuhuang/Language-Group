# Language Group
Our code was mainly modified from , we are mainly based on [wenet](https://github.com/wenet-e2e/wenet) version 2.0

- All experiments are configured as shown in ./conf
- The package used for the experimental conda environment is shown in ./environment_packages.txt
## Train && Infer && Export
All you need to reproduce our experiment is just
```
bash train.sh
bash infer.sh
```
You can export a JIT model using the following command
```
bash export.sh
```