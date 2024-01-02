#!/bin/bash


for trial in 0 1 2 3 4
do
python train_student_comparison.py \
    --cosine --aug_train RA \
      --model_s effiB0 --distill kd  -c 1 -d 0 -b 1    --num_workers 8  \
     --dataset prostate_hv --n_cls 4  --image_size 512  \
     --std_pre PANDA  --tec_pre PANDA   --batch_size 64  --epochs 50 \
   --gpu_id 0,1 --dist-url tcp://127.0.0.1:23348 --multiprocessing-distributed  --trial  ${trial}

python train_student_comparison.py \
--cosine --aug_train RA \
    --model_s effiB0 --distill hint -c 1 -d 1 -b 100    --num_workers 8 \
   --dataset prostate_hv --n_cls 4  --image_size 512  \
   --std_pre PANDA  --tec_pre PANDA   --batch_size 64  --epochs 50 \
   --gpu_id 0,1 --dist-url tcp://127.0.0.1:23348 --multiprocessing-distributed  --trial  ${trial}

python train_student_comparison.py \
  --cosine --aug_train RA \
      --model_s effiB0 --distill correlation -c 1 -d 1 -b 0.02     --num_workers 8 \
     --dataset prostate_hv --n_cls 4  --image_size 512  \
     --std_pre PANDA  --tec_pre PANDA   --batch_size 64  --epochs 50 \
   --gpu_id 0,1 --dist-url tcp://127.0.0.1:23348 --multiprocessing-distributed  --trial  ${trial}

python train_student_comparison.py \
    --cosine --aug_train RA \
      --model_s effiB0 --distill  crd -c 1 -d 1 -b 0.8     --num_workers 8 \
     --dataset prostate_hv --n_cls 4  --image_size 512  \
     --std_pre PANDA  --tec_pre PANDA   --batch_size 64  --epochs 50 \
     --gpu_id 0,1 --dist-url tcp://127.0.0.1:23348 --multiprocessing-distributed  --trial  ${trial}

python train_student_comparison.py \
  --cosine --aug_train RA \
    --model_s effiB0 --distill  attention -c 1 -d 1 -b 1000       --num_workers 8 \
   --dataset prostate_hv --n_cls 4  --image_size 512  \
   --std_pre PANDA  --tec_pre PANDA   --batch_size 64  --epochs 50 \
   --gpu_id 0,1 --dist-url tcp://127.0.0.1:23348 --multiprocessing-distributed  --trial  ${trial}

python train_student_comparison.py \
  --cosine --aug_train RA \
    --model_s effiB0 --distill semckd -c 1 -d 0 -b 50      --num_workers 8 \
   --dataset prostate_hv --n_cls 4  --image_size 512   \
   --std_pre PANDA  --tec_pre PANDA   --batch_size 64  --epochs 50 \
   --gpu_id 0,1 --dist-url tcp://127.0.0.1:23348 --multiprocessing-distributed  --trial  ${trial}

python train_student_comparison.py \
  --cosine --aug_train RA \
    --model_s effiB0 --distill simkd  -c 0 -d 0 -b 1    --num_workers 8  \
   --dataset prostate_hv --n_cls 4  --image_size 512  \
   --std_pre PANDA  --tec_pre PANDA   --batch_size 64  --epochs 50 \
   --gpu_id 0,1 --dist-url tcp://127.0.0.1:23348 --multiprocessing-distributed  --trial  ${trial}

done