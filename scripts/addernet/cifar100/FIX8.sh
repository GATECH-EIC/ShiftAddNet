CUDA_VISIBLE_DEVICES=0 python train.py \
--dataset cifar100 \
--arch resnet20_add \
--save ./temp \
--eval_only \
--add_quant True \
--add_bits 8 \
--quantize_v wageubn \
--resume ./ShiftAddNet_ckpt/addernet/resnet20-cifar100-FIX8.pth.tar