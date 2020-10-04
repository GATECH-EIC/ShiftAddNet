# ShiftAddNet

[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](https://opensource.org/licenses/MIT)

This is PyTorch implementation of ***ShiftAddNet: A Hardware-Inspired Deep Network*** published on the NeurIPS 2020


---

### Prerequisite

* GCC >= 5.4.0
* PyTorch == 1.4
* other common library are included in requirements.txt


### Compile Adder Cuda Kernal

#### Step 1: modify pytorch before launch (for compiling issue)

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
python adder/check.py
````

---

### Reproduce Resutlts in Paper

We release the pretrained checkpoints in [Google Drive](https://drive.google.com/drive/folders/1nON7w5-y40PPGT1NCh_n_h3RLFwP8DO6?usp=sharing). To evaluate the inference accuracy of test set, we provide evaluation scripts shown below for your convenience. If you want to train your own model, the only change should be removing `--eval_only` option in the commands.

* Examples for training of AdderNet

````
bash ./scripts/addernet/cifar10/FP32.sh
bash ./scripts/addernet/cifar10/FIX8.sh

bash ./scripts/addernet/cifar100/FP32.sh
bash ./scripts/addernet/cifar100/FIX8.sh
````

* Examples for training of DeepShift

````
bash ./scripts/deepshift/cifar10.sh
bash ./scripts/deepshift/cifar100.sh
````

* Examples for training of ShiftAddNet

````
bash ./scripts/shiftaddnet/cifar10/FP32.sh
bash ./scripts/shiftaddnet/cifar10/FIX8.sh

bash ./scripts/shiftaddnet/cifar100/FP32.sh
bash ./scripts/shiftaddnet/cifar100/FIX8.sh
````

* Examples for training of ShiftAddNet (FIX shift variant)

````
bash ./scripts/shiftaddnet_fix/cifar10/FP32.sh
bash ./scripts/shiftaddnet_fix/cifar10/FIX8.sh

bash ./scripts/shiftaddnet_fix/cifar100/FP32.sh
bash ./scripts/shiftaddnet_fix/cifar100/FIX8.sh
````

### Citation

If you find this codebase is useful for your research, please cite:

````
@inproceedings{ShiftAddNet,
title={ShiftAddNet: A Hardware-Inspired Deep Network},
author={Haoran You, Xiaohan Che, Yongan Zhang, Chaojian Li, Sicheng Li, Zihao Liu, Zhangyang Wang, Yingyan Lin},
booktitle={Thirty-fourth Conference on Neural Information Processing Systems},
year={2020},
}
````