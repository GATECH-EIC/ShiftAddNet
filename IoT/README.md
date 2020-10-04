# ShiftAddNet on IoT Datasets

## IoT Datasets

* MHEALTH datasets can be found at [Google Drive](https://drive.google.com/file/d/1n2zJ9rljwURo8N35f9u3jBpGZ8mIwuD4/view?usp=sharing)

* USCHAD datasets can be found at [Google Drive](https://drive.google.com/file/d/15uAXufMIPhXZqAxbicm18xdIO5mUULUw/view?usp=sharing)

## Training Scripts

* MHEALTH - all training scripts can be found at `./MHEALTH/train.sh`, below lists few examples for your reference:

````
# Mult
    CUDA_VISIBLE_DEVICES=0 python MHEALTH.py --arch CNN_mult --lr 0.01 --epochs 40 --schedule 20 30 --save ./rebuttal/CNN_mult/FP32/ --log_interval 10

# AdderNet
    CUDA_VISIBLE_DEVICES=0 python MHEALTH.py --arch CNN_add --lr 0.01 --epochs 40 --schedule 20 30 --save ./rebuttal/CNN_add/FP32/ --log_interval 10

# DeepShift
    CUDA_VISIBLE_DEVICES=0 python MHEALTH.py --arch CNN_shift --lr 0.01 --epochs 40 --schedule 20 30 --shift_quant_bits 16 --save ./rebuttal/CNN_shift/FIX16/ --log_interval 10

# ShiftAddNet
    CUDA_VISIBLE_DEVICES=0 python MHEALTH.py --arch CNN_shiftadd --lr 0.01 --epochs 40 --schedule 20 30 --save ./rebuttal/CNN_shiftadd/FP32/ --log_interval 10

# ShiftAddNet (Fixed Shift)
    CUDA_VISIBLE_DEVICES=0 python MHEALTH.py --arch CNN_shiftadd_se --lr 0.01 --epochs 40 --schedule 20 30 --save ./rebuttal/CNN_shiftadd_se/FP32/ --log_interval 10
````

* USCHAD - all training scripts can be found at `./USCHAD/train.sh`, below lists few examples for your reference:

````
# Mult
    CUDA_VISIBLE_DEVICES=0 python USCHAD.py --arch CNN_mult --lr 0.01 --epochs 40 --schedule 20 30 --save ./rebuttal/CNN_mult/FP32/ --log_interval 10

# AdderNet
    CUDA_VISIBLE_DEVICES=0 python USCHAD.py --arch CNN_add --lr 0.01 --epochs 40 --schedule 20 30 --save ./rebuttal/CNN_add/FP32/ --log_interval 10

# DeepShift
    CUDA_VISIBLE_DEVICES=0 python USCHAD.py --arch CNN_shift --lr 0.01 --epochs 40 --schedule 20 30 --shift_quant_bits 16 --save ./rebuttal/CNN_shift/FIX16/ --log_interval 10

# ShiftAddNet
    CUDA_VISIBLE_DEVICES=0 python USCHAD.py --arch CNN_shiftadd --lr 0.01 --epochs 40 --schedule 20 30 --save ./rebuttal/CNN_shiftadd/FP32/ --log_interval 10

# ShiftAddNet (Fixed Shift)
    CUDA_VISIBLE_DEVICES=0 python USCHAD.py --arch CNN_shiftadd_se --lr 0.01 --epochs 40 --schedule 20 30 --save ./rebuttal/CNN_shiftadd_se/FP32/ --log_interval 10
````

## PreTrained Checkpoints

* Pretrained checkpoints for experiments on MHEALTH can be found at [Google Drive](https://drive.google.com/drive/folders/1Crmmgd-CRzL-zFl860F-exvQVvlr52CZ?usp=sharing)

* Pretrained checkpoints for experiments on USCHAD can be found at [Google Drive](https://drive.google.com/drive/folders/1C0-uUiy0cwxvs8GW-lasqkQbZ0uITZHt?usp=sharing)