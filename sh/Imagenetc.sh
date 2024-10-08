#!/bin/bash


output_path=your_path
resize_test=224
resize_train=224
lr=0.005

current_datetime=$(TZ="Asia/Shanghai" date +"%Y-%m-%d-%H:%M:%S")


python main.py --info=${current_datatime}-lr=${lr}-resize=${resize_train} --dataset=imagenet-c \
    --pool_size=15 --prompt_num=20 --topN=1 --prompt_comp --prompt_per_task=1 --use_prompt_penalty_3  \
    --output_path=$output_path   --use_ema_c --adapt_ema_c --adapt_h=8.5 \
    --resize_test=$resize_test --resize_train=$resize_train


