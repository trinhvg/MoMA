#!/bin/bash

for trial in 0 1 2 3 4
do
python train_teacher.py --dataset prostate_hv \
        --cosine --aug_train RA --pretrain PANDA\
        --batch_size 64 --epochs 50 --n_cls 4 \
        --image_size 512 --model effiB0  \
        --num_workers 8 --gpu_id 0,1 --dist-url tcp://127.0.0.1:23345 \
        --multiprocessing-distributed --trial ${trial}
done