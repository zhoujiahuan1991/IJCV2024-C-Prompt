#!/bin/bash

output_path=your_path
resize_test=224
resize_train=224
lr=0.0006

current_datetime=$(TZ="Asia/Shanghai" date +"%Y-%m-%d-%H:%M:%S")

python main.py --info=${current_datatime}-lr=${lr}-resize=${resize_train} --dataset=domain-net \
    --pool_size=150 --prompt_num=8 --topN=3 --prompt_comp --prompt_per_task=25 --use_prompt_penalty_3 \
    --fuse_prompt --use_ema_c  --output_path=$output_path --adapt_ema_c  --adapt_h=10 --lr=${lr} \
    --resize_test=$resize_test --resize_train=$resize_train


