export CUDA_VISIBLE_DEVICES=7
python predict_mamba.py \
    --model_name_or_path oncochat-mamba-790m-v2 \
    --data_path data/CKP-test.json \
    --lazy_preprocess True \
    --model_max_length 4096 \
    --float16 True \
    --device "cuda:0" \
    --per_device_eval_batch_size 1 \
    --output_dir predictions \
    --output_file oncochat-mamba-790m-v2-CKP-test-predictions.json


