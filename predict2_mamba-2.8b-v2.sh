export CUDA_VISIBLE_DEVICES=5
python predict2_mamba.py \
    --model_name_or_path oncochat-mamba-2.8b-v2 \
    --data_path data/CKP_test.no_mutation_signature.json \
    --lazy_preprocess True \
    --model_max_length 4096 \
    --float16 True \
    --device "cuda:0" \
    --per_device_eval_batch_size 1 \
    --output_dir predictions2 \
    --output_file data/oncochat-mamba-2.8b-v2-CKP_test.no_mutation_signature-predictions.json

python predict2_mamba.py \
    --model_name_or_path oncochat-mamba-2.8b-v2 \
    --data_path data/UNKNOWN.no_mutation_signature.json \
    --lazy_preprocess True \
    --model_max_length 4096 \
    --float16 True \
    --device "cuda:0" \
    --per_device_eval_batch_size 1 \
    --output_dir predictions2 \
    --output_file data/oncochat-mamba-2.8b-v2-UNKNOWN.no_mutation_signature-predictions.json

python predict2_mamba.py \
    --model_name_or_path oncochat-mamba-2.8b-v2 \
    --data_path data/CUP.no_mutation_signature.json \
    --lazy_preprocess True \
    --model_max_length 4096 \
    --float16 True \
    --device "cuda:0" \
    --per_device_eval_batch_size 1 \
    --output_dir predictions2 \
    --output_file data/oncochat-mamba-2.8b-v2-CUP.no_mutation_signature-predictions.json

