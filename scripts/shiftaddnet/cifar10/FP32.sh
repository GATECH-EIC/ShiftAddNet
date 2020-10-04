CUDA_VISIBLE_DEVICES=0 python train.py \
--dataset cifar10 \
--arch resnet20_shiftadd \
--save ./temp \
--shift_depth 100 \
--shift_type PS \
--rounding deterministic \
--weight_bits 5 \
--eval_only \
--resume ./ShiftAddNet_ckpt/shiftaddnet/resnet20-cifar10-FP32.pth.tar