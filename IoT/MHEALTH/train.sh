---
CUDA_VISIBLE_DEVICES=0 python MHEALTH.py --arch CNN_mult --lr 0.01 --epochs 40 --schedule 20 30 --save ./rebuttal/CNN_mult/FP32/ --log_interval 10
CUDA_VISIBLE_DEVICES=0 python MHEALTH.py --arch CNN_mult --lr 0.01 --epochs 40 --schedule 20 30 --add_quant True --add_bits 32 --save ./rebuttal/CNN_mult/FIX32/ --log_interval 10
CUDA_VISIBLE_DEVICES=0 python MHEALTH.py --arch CNN_mult --lr 0.01 --epochs 40 --schedule 20 30 --add_quant True --add_bits 16 --save ./rebuttal/CNN_mult/FIX16/ --log_interval 10
CUDA_VISIBLE_DEVICES=0 python MHEALTH.py --arch CNN_mult --lr 0.01 --epochs 40 --schedule 20 30 --add_quant True --add_bits 8 --save ./rebuttal/CNN_mult/FIX8/ --log_interval 10

CUDA_VISIBLE_DEVICES=0 python MHEALTH.py --arch CNN_add --lr 0.01 --epochs 40 --schedule 20 30 --save ./rebuttal/CNN_add/FP32/ --log_interval 10
CUDA_VISIBLE_DEVICES=0 python MHEALTH.py --arch CNN_add --lr 0.01 --epochs 40 --schedule 20 30 --add_quant True --add_bits 32 --save ./rebuttal/CNN_add/FIX32/ --log_interval 10
CUDA_VISIBLE_DEVICES=0 python MHEALTH.py --arch CNN_add --lr 0.01 --epochs 40 --schedule 20 30 --add_quant True --add_bits 16 --save ./rebuttal/CNN_add/FIX16/ --log_interval 10
CUDA_VISIBLE_DEVICES=0 python MHEALTH.py --arch CNN_add --lr 0.01 --epochs 40 --schedule 20 30 --add_quant True --add_bits 8 --save ./rebuttal/CNN_add/FIX8/ --log_interval 10

CUDA_VISIBLE_DEVICES=1 python MHEALTH.py --arch CNN_shift --lr 0.01 --epochs 40 --schedule 20 30 --shift_quant_bits 32 --save ./rebuttal/CNN_shift/FIX32/  --log_interval 10
CUDA_VISIBLE_DEVICES=1 python MHEALTH.py --arch CNN_shift --lr 0.01 --epochs 40 --schedule 20 30 --shift_quant_bits 16 --save ./rebuttal/CNN_shift/FIX16/ --log_interval 10
CUDA_VISIBLE_DEVICES=1 python MHEALTH.py --arch CNN_shift --lr 0.001 --epochs 40 --schedule 20 30 --shift_quant_bits 8 --save ./rebuttal/CNN_shift/FIX8/ --log_interval 10

CUDA_VISIBLE_DEVICES=2 python MHEALTH.py --arch CNN_shiftadd --lr 0.01 --epochs 40 --schedule 20 30 --save ./rebuttal/CNN_shiftadd/FP32/ --log_interval 10
CUDA_VISIBLE_DEVICES=2 python MHEALTH.py --arch CNN_shiftadd --lr 0.01 --epochs 40 --schedule 20 30 --add_quant True --add_bits 32 --save ./rebuttal/CNN_shiftadd/FIX32/ --log_interval 10
CUDA_VISIBLE_DEVICES=2 python MHEALTH.py --arch CNN_shiftadd --lr 0.01 --epochs 40 --schedule 20 30 --add_quant True --add_bits 16 --save ./rebuttal/CNN_shiftadd/FIX16/ --log_interval 10
CUDA_VISIBLE_DEVICES=2 python MHEALTH.py --arch CNN_shiftadd --lr 0.01 --epochs 40 --schedule 20 30 --add_quant True --add_bits 8 --save ./rebuttal/CNN_shiftadd/FIX8/ --log_interval 10

--
CUDA_VISIBLE_DEVICES=3 python MHEALTH.py --arch CNN_shiftadd_se --lr 0.01 \
--epochs 40 --schedule 20 30 \
--save ./rebuttal/CNN_shiftadd_se/FP32/ --log_interval 10

CUDA_VISIBLE_DEVICES=3 python MHEALTH.py --arch CNN_shiftadd_se --lr 0.01 \
--epochs 40 --schedule 20 30 \
--add_quant True --add_bits 32 --save ./rebuttal/CNN_shiftadd_se/FIX32/ --log_interval 10

CUDA_VISIBLE_DEVICES=3 python MHEALTH.py --arch CNN_shiftadd_se --lr 0.01 \
--epochs 40 --schedule 20 30 \
--add_quant True --add_bits 16 --save ./rebuttal/CNN_shiftadd_se/FIX16/ --log_interval 10

CUDA_VISIBLE_DEVICES=3 python MHEALTH.py --arch CNN_shiftadd_se --lr 0.01 \
--epochs 40 --schedule 20 30 \
--add_quant True --add_bits 8 --save ./rebuttal/CNN_shiftadd_se/FIX8/ --log_interval 10