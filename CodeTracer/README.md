# Code Watermark for LLMs Training and Evaluation Framework

This repository contains a comprehensive Python-based framework for training and evaluating code language models with watermarking capabilities. The project supports supervised fine-tuning (SFT), reinforcement learning (RL) training, and watermarked inference across multiple programming languages.

## Project Structure

```
.
├── src/                    # Source code directory
│   ├── lm_eval/           # Language model evaluation framework
│   │   ├── tasks/         # Evaluation tasks (HumanEval, MBPP, DS1000, etc.)
│   │   └── ...            # Task-specific implementations
│   ├── rl/                # Reinforcement learning training modules
│   │   ├── grpo_trainer.py    # GRPO trainer implementation
│   │   ├── rewards.py         # Reward function definitions
│   │   ├── configs.py         # RL training configurations
│   │   └── utils/             # RL utilities and callbacks
│   ├── sft/               # Supervised fine-tuning components
│   │   ├── data.py            # SFT data processing
│   │   └── sft_trainer.py     # SFT trainer implementation
│   ├── watermark/         # Code watermarking for inference
│   │   ├── coderl.py          # CodeRL watermarking scheme
│   │   ├── wllm.py            # WLLM watermarking scheme
│   │   ├── models.py          # Watermarked model implementations
│   │   ├── data.py            # Watermark data utilities
│   │   └── utils.py           # Watermark detection utilities
│   └── configs/           # Configuration files
├── scripts/               # Training and evaluation scripts
│   ├── run_train.py      # Main RL training script
│   ├── run_eval.py       # Evaluation script with watermarking
│   └── run_sft.py        # SFT training script
├── examples/              # Example shell scripts
│   ├── train_sft.sh      # SFT training example
│   ├── train_rl.sh       # RL training example
│   └── eval.sh           # Evaluation example
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies
```

## Main Components

- **lm_eval/**: Multi-language code evaluation framework supporting various benchmarks including HumanEval, MBPP, DS1000, APPS, CodeXGLUE, and HumanEvalPack. Includes custom metrics for multiple programming languages (Python, C++, Java, Rust, etc.)
- **rl/**: GRPO (Generalized Reward-based Policy Optimization) based reinforcement learning training with process reward modeling and advanced training utilities
- **sft/**: Supervised fine-tuning implementation for initial model training with data augmentation and distillation capabilities
- **watermark/**: Code watermarking schemes including CodeRL and WLLM approaches for inference-time watermark injection and detection

## Getting Started

### Prerequisites

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Training Workflow

This framework follows a **one-time training pipeline** where models are trained once and then used with watermarking for all inference tasks:

#### 1. Supervised Fine-Tuning (SFT)

First, perform supervised fine-tuning on your base model:

```bash
bash examples/train_sft.sh
```

Example SFT training configuration:
```bash
accelerate launch \
    --num_processes 1 \
    scripts/run_sft.py \
    --task_name humaneval \
    --model "deepseek-ai/deepseek-coder-1.3b-instruct" \
    --output_dir outputs/sft-model \
    --train_batch_size 64 \
    --num_epochs 3 \
    --alpha_ce 0.2 \
    --alpha_switch 0.2 \
    --context_width 2
```

#### 2. Reinforcement Learning Training

After SFT, enhance the model with RL training using GRPO:

```bash
bash examples/train_rl.sh
```

Example RL training configuration:
```bash
accelerate launch \
    --num_processes 1 \
    scripts/run_train.py \
    --model "outputs/sft-model" \
    --task_name humaneval \
    --output_dir outputs/rl-model \
    --train_batch_size 512 \
    --num_epochs 1000 \
    --alpha_distill 0.1 \
    --alpha_ce 0.1 \
    --alpha_switch 8.0 \
    --context_width 2
```

### Inference with Watermarking

Once training is complete, use the trained models with watermarking for all evaluation tasks:

```bash
bash examples/eval.sh
```

Example evaluation with watermarking:
```bash
python scripts/run_eval.py \
    --model "infly/OpenCoder-1.5B-Instruct" \
    --task_name humaneval \
    --output_dir ./outputs/eval \
    --batch_size 20 \
    --max_length 2048 \
    --temperature 0.2 \
    --n_samples 20 \
    --wm coderl \
    --gamma 0.5 \
    --delta 2.0 \
    --context_width 2 \
    --switch_threshold 0.5 \
    --entropy_threshold 1.2 \
    --code_model outputs/rl-model/checkpoint-300
```

## Supported Tasks and Languages

### Evaluation Benchmarks
- **HumanEval**: Python code generation
- **MBPP**: Python programming problems
- **DS1000**: Data science code generation
- **APPS**: Programming contest problems
- **CodeXGLUE**: Code understanding and generation
- **HumanEvalPack**: Multi-language code generation

### Programming Languages
- Python, C++, Java, JavaScript
- Rust, Go, Shell, Perl, Lua, R

### Watermarking Schemes
- **CodeRL**: Context-aware watermarking with learned green lists
- **WLLM**: Watermarking for large language models

## Key Features

- **One-time Training**: Train models once with SFT and RL, then use for all inference tasks
- **Multi-language Support**: Comprehensive evaluation across multiple programming languages
- **Watermark Integration**: Seamless watermarking during inference without retraining
- **Advanced Training**: GRPO-based RL with process reward modeling
- **Flexible Evaluation**: Support for various code generation and understanding tasks

For detailed configuration options and advanced usage, please refer to the individual script files and configuration modules.