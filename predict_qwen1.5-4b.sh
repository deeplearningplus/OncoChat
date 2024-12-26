python predict_qwen.py \
    --model_name_or_path oncochat-qwen1.5-4b-v2 \
    --data_path data/CKP-test.json \
    --bf16 True \
    --tf32 True \
    --bfloat16 True \
    --lazy_preprocess True \
    --model_max_length 4096 \
    --device "cuda:2" \
    --per_device_eval_batch_size 1 \
    --output_dir predictions \
    --output_file oncochat-qwen1.5-4b-v2-CKP-test-predictions.json


