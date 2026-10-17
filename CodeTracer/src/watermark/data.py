import math
import logging
from typing import Dict, Any, List, Optional, Tuple
from itertools import chain, tee
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.lm_eval import tasks

logger = logging.getLogger(__name__)

NUM_AUGMENTATIONS = 3

def ngrams(sequence, n: int, pad_left: bool = False, pad_right: bool = False, pad_symbol: Any = None):
    """Generate n-grams from a sequence"""
    sequence = iter(sequence)
    if pad_left:
        sequence = chain((pad_symbol,) * (n - 1), sequence)
    if pad_right:
        sequence = chain(sequence, (pad_symbol,) * (n - 1))
    iterables = tee(sequence, n)
    
    for i, sub_iterable in enumerate(iterables):
        for _ in range(i):
            next(sub_iterable, None)
    return zip(*iterables)

class TaskDataset(IterableDataset):
    """Dataset class that handles loading, tokenization and iteration of task data"""
    
    def __init__(self, 
                 task_name: str,
                 tokenizer,
                 prefix: str = "",
                 n_samples: int = 1,
                 batch_size: int = 1,
                 max_length: int = 512,
                 limit: Optional[int] = None):
        """
        Initialize dataset with task configuration
        
        Args:
            task_name: Name of the task to load
            tokenizer: Tokenizer to use for processing prompts
            prefix: Optional prefix to add to prompts
            n_samples: Number of samples to generate per prompt
            batch_size: Batch size for determining copies needed
            max_length: Maximum sequence length for tokenization
            limit: Optional limit on number of examples to load
        """
        self.task_name = task_name
        self.tokenizer = tokenizer
        self.prefix = prefix
        self.n_copies = math.ceil(n_samples / batch_size)
        self.max_length = max_length
        
        # Load task data
        task_data, task = self._load_task_data(limit)
        self._process_prompts(task_data)
        
        self.data = task_data
        self.task = task
        
        logger.info(f"Created TaskDataset with {len(self.data)} items and {self.n_copies} copies each")

    def _load_task_data(self, limit: Optional[int]) -> Tuple[List[Dict], Any]:
        """Load and prepare task data"""
        logger.info(f"Loading task data for {self.task_name}")
        try:
            task = tasks.get_task(self.task_name)
            dataset = task.get_dataset()
        except Exception as e:
            logger.error(f"Failed to load task {self.task_name}: {str(e)}")
            raise

        logger.info(f"Loading {len(dataset)} items from dataset")
        
        task_data = []
        for idx, item in enumerate(dataset):
            task_data.append({
                'task_id': idx,
                'full_data': task.get_full_data(item),
                'reference': task.get_reference(item),
                'solution': task.get_solutions(item),
                'prompt': task.get_prompt(item)
            })
            
        if limit:
            task_data = task_data[10:30]
            
        return task_data, task

    def _process_prompts(self, task_data):
        """Process and tokenize prompts"""
        # Add prefix to prompts if they're strings
        prompts = [
            self.prefix + item['prompt'] if isinstance(item['prompt'], str) else item['prompt']
            for item in task_data
        ]
        
        # Tokenize prompts
        self.tokenized_prompts = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
            return_attention_mask=True
        )

    def __iter__(self):
        """Iterate over dataset items with proper copies"""
        for i in range(len(self.data)):
            for _ in range(self.n_copies):
                yield {
                    "task_id": self.data[i]['task_id'],
                    "ids": self.tokenized_prompts.input_ids[i],
                    "input_len": self.tokenized_prompts.attention_mask[i].sum(),
                    "attention_mask": self.tokenized_prompts.attention_mask[i],
                    "prompt": self.data[i]['prompt'],
                    "reference": self.data[i]['reference'],
                    "full_data": self.data[i]['full_data'],
                    "solution": self.data[i]['solution']
                }

def create_task_dataloader(task_name: str, tokenizer: AutoTokenizer, args) -> Tuple[DataLoader, Any, List[Dict]]:
    """
    Create a dataloader for task evaluation
    
    Args:
        task_name: Name of the task to load
        tokenizer: Tokenizer to use for processing prompts
        prefix: Optional prefix to add to prompts
        n_samples: Number of samples to generate per prompt
        batch_size: Batch size for the dataloader
        max_length: Maximum sequence length for tokenization
        limit: Optional limit on number of examples to load
    
    Returns:
        Tuple of (DataLoader, task, data)
    """
    # Create dataset
    dataset = TaskDataset(
        task_name=task_name,
        tokenizer=tokenizer,
        prefix=args.prefix,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        max_length=args.max_length,
        limit=args.limit
    )
    
    # Define collate function
    def collate_fn(batch):
        return {
            "task_id": torch.tensor([int(item["task_id"]) for item in batch]),
            "ids": torch.stack([item["ids"] for item in batch]),
            "input_len": torch.tensor([item["input_len"] for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
            "prompt": [item["prompt"] for item in batch],
            "reference": [item["reference"] for item in batch],
            "full_data": [item["full_data"] for item in batch],
            "solution": [item["solution"] for item in batch]
        }
    
    # Create and return dataloader
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    
    return dataloader, dataset.task, dataset.data


def prepare_and_generate_data(
    task_name: str,
    model_name: str = "infly/OpenCoder-1.5B-Instruct",
    output_dir: str = "outputs/generated_data",
    batch_size: int = 20,
    n_samples: int = 20,
    max_length: int = 2048,
    precision: str = "bf16",
    seed: int = 42,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Prepare environment and generate data for training with enhanced logging.
    
    Args:
        task_name: Name of the task (humaneval or mbpp)
        model_name: Name of the model to use
        output_dir: Directory to save outputs
        batch_size: Batch size for generation
        n_samples: Number of samples to generate per prompt
        max_length: Maximum length of generated sequences
        precision: Model precision (bf16, fp16, fp32)
        seed: Random seed
        limit: Limit on number of examples to process
    
    Returns:
        Dictionary containing all generated data and metadata
    """
    # Setup basic configuration
    from src.utils import setup_seed, setup_logging, get_tokenizer
    from src.main import generate_code
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_seed(seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    setup_logging(
        os.path.join(output_dir, 'generation.log'),
        level=logging.INFO,
    )
    
    # Log configuration
    logger.info("Starting data generation with configuration:")
    logger.info(f"Task: {task_name}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Device: {device}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Samples per prompt: {n_samples}")
    logger.info(f"Precision: {precision}")
    
    # Setup model precision
    dict_precisions = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32
    }
    
    # Initialize tokenizer and model
    logger.info(f"Loading tokenizer and model from {model_name}")
    tokenizer = get_tokenizer(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dict_precisions[precision],
        trust_remote_code=True
    ).to(device)
    
    # Adjust vocab size if needed
    if len(tokenizer) != model.vocab_size:
        logger.info("Resizing model embeddings to match tokenizer")
        model.resize_token_embeddings(len(tokenizer))
    
    # Create arguments namespace
    class Args:
        pass
    args = Args()
    args.model = model_name
    args.batch_size = batch_size
    args.max_length = max_length
    args.n_samples = n_samples
    args.temperature = 0.2
    args.top_p = 0.95
    args.top_k = 0
    args.do_sample = True
    args.prefix = ""
    args.postprocess = False
    args.limit = limit
    args.wm = "no"
    
    # Load data
    logger.info("Loading task data")
    dataloader, task, data = create_task_dataloader(task_name, tokenizer, args)
    
    # Generate data
    results = {}
    total_batches = len(data) * math.ceil(args.n_samples / args.batch_size)
    logger.info(f"Starting generation for {total_batches} batches")
    
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), 
                           total=total_batches,
                           desc="Generating data"):
            # Generate code samples
            gen_codes = generate_code(
                tokenizer=tokenizer,
                model=model,
                processor=None,
                task=task,
                batch=batch,
                device=device,
                args=args
            )
            
            # Store results
            task_id = batch['task_id'].item()
            results[f"{task_id}"] = {
                "task_id": task_id,
                "prompt": batch['prompt'][0],
                "reference": batch['reference'][0],
                "generations": gen_codes,
                "full_data": gen_codes
            }
            
            # Log progress periodically
            if (i + 1) % 10 == 0:
                logger.info(f"Completed batch {i + 1}/{total_batches}")
    
    # Save results
    output_file = os.path.join(output_dir, f'{task_name}_generated_data.pt')
    torch.save(results, output_file)
    logger.info(f"Saved generated data to {output_file}")
    
    # Log completion statistics
    logger.info("Generation completed:")
    logger.info(f"Total tasks processed: {len(results)}")
    logger.info(f"Total generations: {sum(len(data['generations']) for data in results.values())}")
    
    return results

if __name__ == "__main__":
    # Set environment variables for better performance
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1" 
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["RAYON_NUM_THREADS"] = "1"
    
    # Generate data
    generated_data = prepare_and_generate_data(
        task_name="mbpp",
        model_name="deepseek-ai/deepseek-coder-1.3b-instruct",
        output_dir="outputs/dp_generation",
        batch_size=20,
        n_samples=20,
        limit=0
    )
    
    # Process and display results
    print("\nGeneration Results Summary:")
    print("-" * 50)
    for task_id, data in generated_data.items():
        print(f"\nTask {task_id}:")
        print(f"Number of generations: {len(data['generations'])}")
        print(f"First generation preview: {data['generations'][0][:100]}...")