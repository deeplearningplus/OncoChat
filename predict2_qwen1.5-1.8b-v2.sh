
sleep 6000

python predict2_qwen.py \
    --model_name_or_path oncochat-qwen1.5-1.8b-v2 \
    --data_path data/CKP_test.no_mutation_signature.json \
    --bf16 True \
    --tf32 True \
    --lazy_preprocess True \
    --model_max_length 4096 \
    --device "cuda:3" \
    --per_device_eval_batch_size 2 \
    --output_dir predictions2 \
    --output_file predictions2/oncochat-qwen1.5-1.8b-CKP_test.no_mutation_signature.predictions.json

python predict2_qwen.py \
    --model_name_or_path oncochat-qwen1.5-1.8b-v2 \
    --data_path data/CUP.no_mutation_signature.json \
    --bf16 True \
    --tf32 True \
    --lazy_preprocess True \
    --model_max_length 4096 \
    --device "cuda:3" \
    --per_device_eval_batch_size 2 \
    --output_dir predictions2 \
    --output_file predictions2/oncochat-qwen1.5-1.8b-CUP.no_mutation_signature.predictions.json

python predict2_qwen.py \
    --model_name_or_path oncochat-qwen1.5-1.8b-v2 \
    --data_path data/UNKNOWN.no_mutation_signature.json \
    --bf16 True \
    --tf32 True \
    --lazy_preprocess True \
    --model_max_length 4096 \
    --device "cuda:3" \
    --per_device_eval_batch_size 2 \
    --output_dir predictions2 \
    --output_file predictions2/oncochat-qwen1.5-1.8b-UNKNOWN.no_mutation_signature.predictions.json

