# ShiftAddNet

[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](https://opensource.org/licenses/MIT)

This is a PyTorch implementation of ***ShiftAddNet: A Hardware-Inspired Deep Network*** published on the NeurIPS 2020


---

### Prerequisite

* GCC >= 5.4.0
* PyTorch == 1.4
* Other common library are included in `requirements.txt`


### Compile Adder Cuda Kernal

The original [AdderNet Repo](https://github.com/huawei-noah/AdderNet) considers using PyTorch for implementing add absed convolution, however it remains slow and requires much more runtime memory costs as compared to the variant with CUDA acceleration.

We here provide one kind of CUDA implementation, please follow the intruction below to compile and check that the `forwad/backward` results are consistent with the original version.

#### Step 1: modify PyTorch before launch (for solving compiling issue)

Change lines:57-64 in `anaconda3/lib/python3.7/site-packages/torch/include/THC/THCTensor.hpp`
from:
````
#include <THC/generic/THCTensor.hpp>
#include <THC/THCGenerateAllTypes.h>

#include <THC/generic/THCTensor.hpp>
#include <THC/THCGenerateBoolType.h>

#include <THC/generic/THCTensor.hpp>
#include <THC/THCGenerateBFloat16Type.h>
````
to:
````
#include <THC/generic/THCTensor.h>
#include <THC/THCGenerateAllTypes.h>

#include <THC/generic/THCTensor.h>
#include <THC/THCGenerateBoolType.h>

#include <THC/generic/THCTensor.h>
#include <THC/THCGenerateBFloat16Type.h>
````

#### Step 2: launch command to make sure you can successfully compile


````
python check.py
````

You should be able to successfully compile and see the runtime speed comparisons in the toy cases.

---

### Reproduce Resutlts in Paper

We release the pretrained checkpoints in [Google Drive](https://drive.google.com/drive/folders/1nON7w5-y40PPGT1NCh_n_h3RLFwP8DO6?usp=sharing). To evaluate the inference accuracy of test set, we provide evaluation scripts shown below for your convenience. If you want to train your own model, the only change should be removing `--eval_only` option in the commands.

* Examples for training of AdderNet

````
# CIFAR-10
    bash ./scripts/addernet/cifar10/FP32.sh
    bash ./scripts/addernet/cifar10/FIX8.sh

# CIFAR-100
    bash ./scripts/addernet/cifar100/FP32.sh
    bash ./scripts/addernet/cifar100/FIX8.sh
````

* Examples for training of DeepShift

````
# CIFAR-10
    bash ./scripts/deepshift/cifar10.sh

# CIFAR-100
    bash ./scripts/deepshift/cifar100.sh
````

* Examples for training of ShiftAddNet

````
# CIFAR-10
    bash ./scripts/shiftaddnet/cifar10/FP32.sh
    bash ./scripts/shiftaddnet/cifar10/FIX8.sh

# CIFAR-100
    bash ./scripts/shiftaddnet/cifar100/FP32.sh
    bash ./scripts/shiftaddnet/cifar100/FIX8.sh
````

* Examples for training of ShiftAddNet (FIX shift variant)

````
# CIFAR-10
    bash ./scripts/shiftaddnet_fix/cifar10/FP32.sh
    bash ./scripts/shiftaddnet_fix/cifar10/FIX8.sh

# CIFAR-100
    bash ./scripts/shiftaddnet_fix/cifar100/FP32.sh
    bash ./scripts/shiftaddnet_fix/cifar100/FIX8.sh
````

---

### T-SNE Visualization

Reproduce the T-SNE visualization of the class divergences in AdderNet, and the proposed ShiftAddNet, using ResNet-20 on CIFAR-10 as an example.

````
python ./scripts/gen_feat.sh # generate the features that will be used for visualization

cd tsne_vis &&
python visual_tsne.py --save_dir resnet20_add_FP32
python visual_tsne.py --save_dir resnet20_add_FIX8
python visual_tsne.py --save_dir resnet20_shiftadd_FP32
python visual_tsne.py --save_dir resnet20_shiftadd_FIX8

python visual_tsne.py --save_dir resnet20_add_FP32 --scratch --dim_3d
python visual_tsne.py --save_dir resnet20_add_FIX8 --scratch --dim_3d
python visual_tsne.py --save_dir resnet20_shiftadd_FP32 --scratch --dim_3d
python visual_tsne.py --save_dir resnet20_shiftadd_FIX8 --scratch --dim_3d
````

The output figure should look like below:
<br><br>
![](./tsne_vis/comp.png)


---

### Citation

If you find this codebase is useful for your research, please cite:

````
@inproceedings{ShiftAddNet,
title={ShiftAddNet: A Hardware-Inspired Deep Network},
author={Haoran You, Xiaohan Chen, Yongan Zhang, Chaojian Li, Sicheng Li, Zihao Liu, Zhangyang Wang, Yingyan Lin},
booktitle={Thirty-fourth Conference on Neural Information Processing Systems},
year={2020},
}
````