#!/bin/bash


for trial in 0 1 2 3 4
do
python train_student_cmo_attention_clean.py --dataset prostate_hv --n_cls 4  --image_size 512  \
        --model_s effiB0 --distill cmo -c 1 -d 1 -b 1   --num_workers 8 \
        --batch_size 64  --epochs 50  --cosine --aug_train RA --feat_dim 512 \
        --std_pre PANDA --tec_pre PANDA  --head mlp --attn self  \
        --gpu_id 0,1 --dist-url tcp://127.0.0.1:23347 --multiprocessing-distributed --trial ${trial}

done
