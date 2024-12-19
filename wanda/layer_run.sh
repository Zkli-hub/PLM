#!/bin/bash

# Loop through layer IDs 0 to 31
for layer_id in {3..31}
do
    echo "Running for layer_id=$layer_id"
    CUDA_VISIBLE_DEVICES=0 python main.py \
        --model NousResearch/llama-2-7b-hf \
        --prune_method wanda \
        --sparsity_ratio 0.5 \
        --sparsity_type 2:4 \
        --save out/llama2_7b/unstructured/wanda/ \
        --eval_zero_shot \
        --module_name gate \
        --permutate_mode full \
        --layer_id $layer_id
done
