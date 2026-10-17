import math
import logging
from typing import Dict, Any, List, Optional, Tuple
from itertools import chain, tee

import torch
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer
from datasets import load_dataset

from lm_eval import tasks

logger = logging.getLogger(__name__)


def extended_load_dataset(path, column, system_prompt, train_split, eval_split):
    """
    Create a dataloader for task evaluation
    """
    if path in ['humaneval', 'mbpp']:
        return load_local_dataset(path)
    else:
        return load_remote_dataset(path, column, system_prompt, train_split, eval_split)

def load_remote_dataset(path, column, system_prompt, train_split, eval_split):

    # Load the dataset
    dataset = load_dataset(path, name=None)

    # Format into conversation
    def make_conversation(example, prompt_column: str = column):
        prompt = []

        if system_prompt is not None:
            prompt.append({"role": "system", "content": system_prompt})

        if prompt_column not in example:
            raise ValueError(f"Dataset Question Field Error: {prompt_column} is not supported.")

        prompt.append({"role": "user", "content": example[prompt_column]})
        return {"prompt": prompt}

    dataset = dataset.map(make_conversation)

    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")
            
    return dataset[train_split], None

def load_local_dataset(path):

    logger.info(f"Loading and formatting local dataset for task: {path}")
    task_obj = tasks.get_task(path)
    raw_dataset = task_obj.get_dataset()

    def format_prompt_for_local_task(item: Dict[str, Any]) -> Dict[str, List[Dict[str, str]]]:
        actual_prompt_string = task_obj.get_prompt(item)
        return {
            "prompt": [{"role": "user", "content": actual_prompt_string}],
            "dataset": path
        }

    formatted_dataset = raw_dataset.map(format_prompt_for_local_task)

    return formatted_dataset, None
