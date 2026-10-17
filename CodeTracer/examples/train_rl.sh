
export PYTHONPATH=.
export HF_ALLOW_CODE_EVAL=1

CUDA_VISIBLE_DEVICES=1 ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file src/configs/accelerate_configs/ddp.yaml \
    --num_processes=1 \
    --main_process_port 29515 \
    scripts/run_train.py --config src/configs/config_demo_code.yaml \
    --use_vllm False \
    --num_generations 8 \
    --output_dir outputs/orm-masked-adv-clamp2 \
    --overwrite_output_dir false \
    --save_steps 1 \
    --watermark_context_width 2 \
    --reward_funcs watermark code \
    --reward_weights 100.0 1.0 
