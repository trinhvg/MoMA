
## MoMA

Implementation of paper [[arXiv]](): 

"MoMA: Momentum Contrastive Learning with Multi-head Attention-based Knowledge Distillation for Histopathology Image Analysis"
Trinh Thi Le Vuong and Jin Tae Kwak. 

<p align="center">
  <img src="figures/overview.png" width="600">
</p>

Overview of distillation flow across different tasks and datasets. 1) Supervised task is always conducted, 2) Feature distillation is applied if a well-trained teacher model is available, and 3) Vanilla ${L}_{KD}$ is employed if teacher and student models conduct the same task.


<p align="center">
  <img src="figures/KD_dataset_v2.png" width="600">
</p>

Overview of distillation flow across different tasks and datasets. 1) Supervised task is always conducted, 2) Feature distillation is applied if a well-trained teacher model is available, and 3) Vanilla ${L}_{KD}$ is employed if teacher and student models conduct the same task.

## Train the teacher network (optional)
 
```
python train_teacher.py \
 --dataset 'PANDA' 
```



## Train the student network

```
python train_student.py \
 --dataset 'prostate_hv' \
 --method MoMA \
 --tec_pre 'PANDA' \
 --std_pre 'PANDA' \
 --loss_st NCE \

```

## Inference on independent dataset (optional)

```
python inference.py \
 --dataset 'prostate_kbsmc' \
 --ckpt ./save/ckpt.pth\
```

