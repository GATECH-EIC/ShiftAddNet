CUDA_VISIBLE_DEVICES=0 python train.py \
--dataset cifar100 \
--arch resnet20_add \
--save ./temp \
--eval_only \
--resume ./ShiftAddNet_ckpt/addernet/resnet20-cifar100-FP32.pth.tar