
python predict_qwen.py \
    --model_name_or_path oncochat-qwen1.5-1.8b-v2 \
    --data_path data/CKP-test.json \
    --bf16 True \
    --tf32 True \
    --lazy_preprocess True \
    --model_max_length 4096 \
    --device "cuda:1" \
    --per_device_eval_batch_size 2 \
    --output_dir predictions \
    --output_file oncochat-qwen1.5-1.8b-v2-CKP-test-predictions.json


