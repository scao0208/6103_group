export CUBLAS_WORKSPACE_CONFIG=:16:8

python vllm_inference.py \
--model_path ./model/Qwen1.5-4B-Chat \
--adapter_path ./saves/4b_lora_ckpt_1epoch \
--data_path ./test_data \
--data_name mit-movie \
--output_path ./output/results
