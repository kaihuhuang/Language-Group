# Language Group
Our code is mainly based on [wenet](https://github.com/wenet-e2e/wenet) version 2.0

- All experiments are configured as shown in `./conf`
- The package used for the experimental conda environment is shown in `./environment_packages.txt`
## Train && Infer && Export
If you want to reproduce our experiment, you just need to process the dataset and place it in the `./data` and run the following command
```
bash train.sh
bash infer.sh
```
You can export a JIT model using the following command
```
bash export.sh
```