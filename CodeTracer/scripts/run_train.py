# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys

import datasets
import transformers
from src.rl.configs import GRPOConfig, GRPOScriptArguments
from src.rl.data import extended_load_dataset
from src.rl.grpo_trainer import GRPOTrainer
from src.rl.prm_grpo_trainer import PRMGRPOTrainer
from src.rl.rewards import get_reward_funcs
from src.rl.utils import get_tokenizer
from src.rl.utils.callbacks import get_callbacks
from src.rl.utils.wandb_logging import init_wandb_training
from src.watermark.models import get_watermarked_model
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import ModelConfig, TrlParser, get_peft_config

logger = logging.getLogger(__name__)


def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, training_args)
    if not tokenizer.eos_token:
        if tokenizer.bos_token:
            tokenizer.eos_token = tokenizer.bos_token
            print("bos_token used as eos_token")
        else:
            raise ValueError("No eos_token or bos_token found")
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    ##############
    # Load model #
    ##############
    logger.info("*** Loading model ***")
    # model = get_model(model_args, training_args)
    model = get_watermarked_model(model_args, training_args, vocab_size=len(tokenizer))
    # Get reward functions from the registry
    reward_funcs = get_reward_funcs(script_args)

    train_dataset, eval_dataset = extended_load_dataset(
        dataset_name=script_args.dataset_name,
        column=script_args.dataset_prompt_column,
        system_prompt=training_args.system_prompt,
        train_split=script_args.dataset_train_split,
        eval_split=script_args.dataset_test_split,
    )

    #############################
    # Initialize the GRPO trainer
    #############################
    if script_args.prm:
        trainer = PRMGRPOTrainer(
            model=model,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=get_peft_config(model_args),
            callbacks=get_callbacks(training_args, model_args),
            processing_class=tokenizer,
        )
    else:
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=get_peft_config(model_args),
            callbacks=get_callbacks(training_args, model_args),
            processing_class=tokenizer,
        )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
