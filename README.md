# ShiftAddNet
[NeurIPS 2020] ShiftAddNet: A Hardware-Inspired Deep Network


### Prerequisite

* GCC >= 5.4.0
* PyTorch == 1.4
* pip install git+https://github.com/cmpark0126/pytorch-polynomial-lr-decay.git
* other common library are included in requirements.txt

---

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

````

---