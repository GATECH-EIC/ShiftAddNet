CUDA_VISIBLE_DEVICES=0 python train.py \
--dataset cifar10 \
--arch resnet20_add \
--save ./temp \
--eval_only \
--add_quant True \
--add_bits 32 \
--quantize_v wageubn \
--resume ./ShiftAddNet_ckpt/addernet/resnet20-cifar10-FIX32.pth.tar