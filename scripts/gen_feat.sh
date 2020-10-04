CUDA_VISIBLE_DEVICES=0 python generate_feature.py \
--dataset cifar10 \
--arch resnet20_add \
--save ./tsne_vis/resnet20_add_FP32 \
--eval_only \
--resume ./ShiftAddNet_ckpt/addernet/resnet20-cifar10-FP32.pth.tar


CUDA_VISIBLE_DEVICES=0 python generate_feature.py \
--dataset cifar10 \
--arch resnet20_shiftadd \
--save ./tsne_vis/resnet20_shiftadd_FP32 \
--shift_depth 100 \
--shift_type PS \
--rounding deterministic \
--weight_bits 5 \
--eval_only \
--resume ./ShiftAddNet_ckpt/shiftaddnet/resnet20-cifar10-FP32.pth.tar

CUDA_VISIBLE_DEVICES=0 python generate_feature.py \
--dataset cifar10 \
--arch resnet20_add \
--save ./tsne_vis/resnet20_add_FIX8 \
--eval_only \
--add_quant True \
--add_bits 8 \
--quantize_v wageubn \
--resume ./ShiftAddNet_ckpt/addernet/resnet20-cifar10-FIX8.pth.tar

CUDA_VISIBLE_DEVICES=0 python generate_feature.py \
--dataset cifar10 \
--arch resnet20_shiftadd \
--save ./tsne_vis/resnet20_shiftadd_FIX8 \
--shift_depth 100 \
--shift_type PS \
--rounding deterministic \
--weight_bits 5 \
--eval_only \
--add_quant True \
--add_bits 8 \
--quantize_v wageubn \
--resume ./ShiftAddNet_ckpt/shiftaddnet/resnet20-cifar10-FIX8.pth.tar