
python predict2_qwen.py \
    --model_name_or_path oncochat-qwen1.5-4b-v2 \
    --data_path data/CKP_test.no_mutation_signature.json \
    --bf16 True \
    --tf32 True \
    --bfloat16 True \
    --lazy_preprocess True \
    --model_max_length 4096 \
    --device "cuda:1" \
    --per_device_eval_batch_size 1 \
    --output_dir predictions2 \
    --output_file predictions2/oncochat-qwen1.5-4b-CKP_test.no_mutation_signature.predictions.json

python predict2_qwen.py \
    --model_name_or_path oncochat-qwen1.5-4b-v2 \
    --data_path data/CUP.no_mutation_signature.json \
    --bf16 True \
    --tf32 True \
    --bfloat16 True \
    --lazy_preprocess True \
    --model_max_length 4096 \
    --device "cuda:1" \
    --per_device_eval_batch_size 1 \
    --output_dir predictions2 \
    --output_file predictions2/oncochat-qwen1.5-4b-CUP.no_mutation_signature.predictions.json

python predict2_qwen.py \
    --model_name_or_path oncochat-qwen1.5-4b-v2 \
    --data_path data/UNKNOWN.no_mutation_signature.json \
    --bf16 True \
    --tf32 True \
    --bfloat16 True \
    --lazy_preprocess True \
    --model_max_length 4096 \
    --device "cuda:1" \
    --per_device_eval_batch_size 1 \
    --output_dir predictions2 \
    --output_file predictions2/oncochat-qwen1.5-4b-UNKNOWN.no_mutation_signature.predictions.json

