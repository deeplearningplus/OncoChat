export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node=4 --master_port=20002 train_mamba.py \
    --fsdp "full_shard auto_wrap" --fsdp_transformer_layer_cls_to_wrap 'MambaBlock' \
    --model_name_or_path state-spaces/mamba-790m-hf \
    --data_path data/CKP-train.json \
    --bf16 True \
    --output_dir oncochat-mamba-790m-v2 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --model_max_length 4096 \
    --lazy_preprocess True

