# coding=utf-8
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

"""Reward functions for GRPO training."""

import asyncio
import json
import math
import re
import traceback
from functools import partial, update_wrapper
from typing import Callable, Dict, Optional

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

import torch
from src.rl.utils import is_e2b_available
from src.rl.utils.ioi import SubtaskResult, add_includes, get_piston_client_from_env, score_subtask
from src.lm_eval import tasks

if is_e2b_available():
    from dotenv import load_dotenv
    from e2b_code_interpreter import AsyncSandbox

    from src.rl.utils.routed_sandbox import RoutedSandbox

    load_dotenv()
else:
    AsyncSandbox = None


def accuracy_reward(completions: list[list[dict[str, str]]], solution: list[str], **kwargs) -> list[Optional[float]]:
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
        )
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # Compute binary rewards if verifiable, `None` otherwise to skip this example
            try:
                reward = float(verify(gold_parsed, answer_parsed))
            except Exception as e:
                print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
                reward = None
        else:
            # If the gold solution is not parseable, we assign `None` to skip this example
            reward = None
            print("Failed to parse gold solution: ", sol)
        rewards.append(reward)

    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def tag_count_reward(completions, **kwargs) -> list[float]:
    """Reward function that checks if we produce the desired number of think and answer tags associated with `format_reward()`.

    Adapted from: https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb#file-grpo_demo-py-L90
    """

    def count_tags(text: str) -> float:
        count = 0.0
        if text.count("<think>\n") == 1:
            count += 0.25
        if text.count("\n</think>\n") == 1:
            count += 0.25
        if text.count("\n<answer>\n") == 1:
            count += 0.25
        if text.count("\n</answer>") == 1:
            count += 0.25
        return count

    contents = [completion[0]["content"] for completion in completions]
    return [count_tags(c) for c in contents]


def reasoning_steps_reward(completions, **kwargs):
    r"""Reward function that checks for clear step-by-step reasoning.
    Regex pattern:
        Step \d+: - matches "Step 1:", "Step 2:", etc.
        ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
        \n- - matches bullet points with hyphens
        \n\* - matches bullet points with asterisks
        First,|Second,|Next,|Finally, - matches transition words
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [len(re.findall(pattern, content)) for content in completion_contents]

    # Magic number 3 to encourage 3 steps and more, otherwise partial reward
    return [min(1.0, count / 3) for count in matches]


def len_reward(completions: list[Dict[str, str]], solution: list[str], **kwargs) -> float:
    """Compute length-based rewards to discourage overthinking and promote token efficiency.

    Taken from the Kimi 1.5 tech report: https://arxiv.org/abs/2501.12599

    Args:
        completions: List of model completions
        solution: List of ground truth solutions

    Returns:
        List of rewards where:
        - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
        - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """
    contents = [completion[0]["content"] for completion in completions]

    # First check correctness of answers
    correctness = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) == 0:
            # Skip unparseable examples
            correctness.append(True)  # Treat as correct to avoid penalizing
            print("Failed to parse gold solution: ", sol)
            continue

        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        correctness.append(verify(answer_parsed, gold_parsed))

    # Calculate lengths
    lengths = [len(content) for content in contents]
    min_len = min(lengths)
    max_len = max(lengths)

    # If all responses have the same length, return zero rewards
    if max_len == min_len:
        return [0.0] * len(completions)

    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)

        if is_correct:
            reward = lambda_val
        else:
            reward = min(0, lambda_val)

        rewards.append(float(reward))

    return rewards


def get_cosine_scaled_reward(
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
):
    def cosine_scaled_reward(completions, solution, **kwargs):
        """Reward function that scales based on completion length using a cosine schedule.

        Shorter correct solutions are rewarded more than longer ones.
        Longer incorrect solutions are penalized less than shorter ones.

        Args:
            completions: List of model completions
            solution: List of ground truth solutions

        This function is parameterized by the following arguments:
            min_value_wrong: Minimum reward for wrong answers
            max_value_wrong: Maximum reward for wrong answers
            min_value_correct: Minimum reward for correct answers
            max_value_correct: Maximum reward for correct answers
            max_len: Maximum length for scaling
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol in zip(contents, solution):
            gold_parsed = parse(sol, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
            if len(gold_parsed) == 0:
                rewards.append(1.0)  # Skip unparseable examples
                print("Failed to parse gold solution: ", sol)
                continue

            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed=True,
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

            is_correct = verify(answer_parsed, gold_parsed)
            gen_len = len(content)

            # Apply cosine scaling based on length
            progress = gen_len / max_len
            cosine = math.cos(progress * math.pi)

            if is_correct:
                min_value = min_value_correct
                max_value = max_value_correct
            else:
                # Swap min/max for incorrect answers
                min_value = max_value_wrong
                max_value = min_value_wrong

            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))

        return rewards

    return cosine_scaled_reward


def get_repetition_penalty_reward(ngram_size: int, max_penalty: float, language: str = "en"):
    """
    Computes N-gram repetition penalty as described in Appendix C.2 of https://arxiv.org/abs/2502.03373.
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    Args:
    ngram_size: size of the n-grams
    max_penalty: Maximum (negative) penalty for wrong answers
    language: Language of the text, defaults to `en`. Used to choose the way to split the text into n-grams.
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    if language == "en":

        def zipngram(text: str, ngram_size: int):
            words = text.lower().split()
            return zip(*[words[i:] for i in range(ngram_size)]), words
    elif language == "zh":
        from transformers.utils.import_utils import _is_package_available

        if not _is_package_available("jieba"):
            raise ValueError("Please install jieba to use Chinese language")

        def zipngram(text: str, ngram_size: int):
            import jieba

            seg_list = list(jieba.cut(text))
            return zip(*[seg_list[i:] for i in range(ngram_size)]), seg_list
    else:
        raise ValueError(
            f"Word splitting for language `{language}` is not yet implemented. Please implement your own zip-ngram function."
        )

    def repetition_penalty_reward(completions, **kwargs) -> float:
        """
        reward function the penalizes repetitions
        ref implementation: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

        Args:
            completions: List of model completions
        """

        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion in contents:
            if completion == "":
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            ngram_array, words = zipngram(completion, ngram_size)

            if len(words) < ngram_size:
                rewards.append(0.0)
                continue

            for ng in ngram_array:
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty
            rewards.append(reward)
        return rewards

    return repetition_penalty_reward


def _init_event_loop():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def ioi_code_reward(completions, test_batch_size: int = 1, **kwargs) -> list[float]:
    """Reward function that evaluates IOI problems using Piston+our IOI package.

    Assumes the dataset has the same format as hf.co/datasets/open-r1/ioi

    test_batch_size: evaluate these many test cases in parallel, then check if any of them failed (0 score): if so stop evaluating; otherwise continue with the next batch of test cases.
    """
    # for info on setting up piston workers, see slurm/piston/README.md
    piston_client = get_piston_client_from_env()

    code_snippets = [
        # note: grading is automatically skipped if no code is extracted
        add_includes(extract_code(completion[-1]["content"], "cpp"), problem_id)
        for completion, problem_id in zip(completions, kwargs["id"])
    ]

    async def run_catch_exceptions(task):
        try:
            return await task
        except Exception as e:
            print(f"Error from Piston worker: {e}")
            return SubtaskResult()  # score 0.0

    # load problem data. undo separating kwargs by column
    problems_data = [dict(zip(kwargs.keys(), values)) for values in zip(*kwargs.values())]

    loop = _init_event_loop()
    evals = [
        loop.create_task(
            run_catch_exceptions(score_subtask(piston_client, problem_data, code, test_batch_size=test_batch_size))
        )
        for problem_data, code in zip(problems_data, code_snippets)
    ]
    results = loop.run_until_complete(asyncio.gather(*evals))

    return [result.score for result in results]


def extract_code(completion: str, language: str = "python") -> str:
    # Case-insensitive pattern for language
    language_pattern = re.escape(language)
    pattern = re.compile(rf"```(?i:{language_pattern})[ \t]*\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    
    if not matches:
        # Fallback for any code block if specific language not found
        fallback_pattern = re.compile(r"```.*?\n(.*?)```", re.DOTALL)
        matches = fallback_pattern.findall(completion)
    
    # If no matches found, return empty string
    if not matches:
        return ""
    
    # Check the last match first
    if matches[-1].strip():
        return matches[-1].strip()
    
    # Then check the first match
    if len(matches) > 1 and matches[0].strip():
        return matches[0].strip()
    
    # Then check the remaining matches in order
    for i in range(1, len(matches) - 1):
        if matches[i].strip():
            return matches[i].strip()
    
    # If all matches are empty, return an empty string
    return ""


def binary_code_reward(completions, num_parallel: int = 2, e2b_router_url=None, **kwargs) -> list[float]:
    rewards = code_reward(completions, num_parallel=num_parallel, e2b_router_url=e2b_router_url, **kwargs)
    BINARY_THRESHOLD = 0.99

    output = []
    for reward in rewards:
        if reward is None:
            output.append(None)
        else:
            output.append(1.0 if reward > BINARY_THRESHOLD else 0.0)

    return output


def code_reward(completions, num_parallel: int = 2, e2b_router_url=None, **kwargs) -> list[float]:
    """Reward function that evaluates code snippets using the E2B code interpreter.

    Assumes the dataset contains a `verification_info` column with test cases.
    """
    if not is_e2b_available():
        raise ImportError(
            "E2B is not available and required for this reward function. Please install E2B with "
            "`pip install e2b-code-interpreter` and add an API key to a `.env` file."
        )

    # TODO: add support for other languages in E2B: https://e2b.dev/docs/code-interpreting/supported-languages
    """Returns a reward function that evaluates code snippets in a sandbox."""
    evaluation_script_template = """
    import subprocess
    import json

    def evaluate_code(code, test_cases):
        passed = 0
        total = len(test_cases)
        exec_timeout = 5

        for case in test_cases:
            process = subprocess.run(
                ["python3", "-c", code],
                input=case["input"],
                text=True,
                capture_output=True,
                timeout=exec_timeout
            )

            if process.returncode != 0:  # Error in execution
                continue

            output = process.stdout.strip()

            # TODO: implement a proper validator to compare against ground truth. For now we just check for exact string match on each line of stdout.
            all_correct = True
            for line1, line2 in zip(output.split('\\n'), case['output'].split('\\n')):
                all_correct = all_correct and line1.strip() == line2.strip()

            if all_correct:
                passed += 1

        success_rate = (passed / total)
        return success_rate

    code_snippet = {code}
    test_cases = json.loads({test_cases})

    evaluate_code(code_snippet, test_cases)
    """
    code_snippets = [extract_code(completion[-1]["content"]) for completion in completions]
    verification_info = kwargs["verification_info"]
    scripts = [
        evaluation_script_template.format(code=json.dumps(code), test_cases=json.dumps(json.dumps(info["test_cases"])))
        for code, info in zip(code_snippets, verification_info)
    ]

    language = verification_info[0]["language"]
    if not all(v["language"] == language for v in verification_info):
        raise ValueError("All verification_info must have the same language", verification_info)

    if e2b_router_url is not None:
        routed_sandbox = RoutedSandbox(router_url=e2b_router_url)

        executions = routed_sandbox.run_code(
            scripts=scripts,
            language=language,
            timeout=30,
            request_timeout=28,
        )

        rewards = []
        for execution in executions:
            try:
                reward = float(execution.text)
                rewards.append(reward)
            except Exception:
                rewards.append(None)
        return rewards

    try:
        rewards = run_async_from_sync(scripts, language, num_parallel)
    except Exception as e:
        print(f"Error from E2B executor: {e}")
        rewards = [0.0] * len(completions)

    return rewards


def get_code_format_reward(language: str = "python"):
    """Format reward function specifically for code responses.

    Args:
        language: Programming language supported by E2B https://e2b.dev/docs/code-interpreting/supported-languages
    """
    pattern = rf"^<think>\n.*?\n</think>\n<answer>\n.*?```{language}.*?```.*?\n</answer>$"

    def code_format_reward(completions, **kwargs):
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
        return [1.0 if match else 0.0 for match in matches]

    return code_format_reward


def run_async_from_sync(scripts: list[str], language: str, num_parallel: int) -> list[float]:
    """Function wrapping the `run_async` function."""
    # Create a new event loop and set it
    try:
        # Run the async function and get the result
        rewards = asyncio.run(run_async(scripts, language, num_parallel))
    except Exception as e:
        print(f"Error from E2B executor async: {e}")
        raise e

    return rewards


async def run_async(scripts: list[str], language: str, num_parallel: int) -> list[float]:
    # Limit the number of concurrent tasks
    semaphore = asyncio.Semaphore(num_parallel)

    # Create a list of tasks for running scripts concurrently
    tasks = [run_script(script, language, semaphore) for script in scripts]

    # Wait for all tasks to complete and gather their results as they finish
    results = await asyncio.gather(*tasks)
    rewards = list(results)  # collect results

    return rewards


async def run_script(script: str, language: str, semaphore: asyncio.Semaphore) -> float:
    # We set a timeout margin, as the AsyncSandbox timeout does not seem to work
    # These values are based on running 256 examples with the gold solution
    # from open-r1/verifiable-coding-problems-python_decontaminated
    # see scripts/benchmark_e2b.py

    SANDBOX_TIMEOUT = 30
    MARGIN = 2
    REQUEST_TIMEOUT = SANDBOX_TIMEOUT - MARGIN
    ASYNCIO_TIMEOUT = SANDBOX_TIMEOUT + MARGIN

    async with semaphore:
        try:
            sandbox = await AsyncSandbox.create(timeout=SANDBOX_TIMEOUT, request_timeout=REQUEST_TIMEOUT)
            execution = await asyncio.wait_for(sandbox.run_code(script, language=language), timeout=ASYNCIO_TIMEOUT)
            return float(execution.text)
        except (TypeError, ValueError):
            return 0.0
        except asyncio.TimeoutError:
            print("Operation timed out")
            return 0.0
        except Exception as e:
            print(f"Error in `run_script` from E2B sandbox ID {sandbox.sandbox_id} : {e}")
            return 0.0
        finally:
            try:
                await sandbox.kill()
            except Exception as e:
                print(f"Error from E2B executor kill with sandbox ID {sandbox.sandbox_id} : {e}")


def get_reward_funcs(script_args) -> list[Callable]:
    REWARD_FUNCS_REGISTRY = {
        "accuracy": accuracy_reward,
        "format": format_reward,
        "reasoning_steps": reasoning_steps_reward,
        "cosine": get_cosine_scaled_reward(
            min_value_wrong=script_args.cosine_min_value_wrong,
            max_value_wrong=script_args.cosine_max_value_wrong,
            min_value_correct=script_args.cosine_min_value_correct,
            max_value_correct=script_args.cosine_max_value_correct,
            max_len=script_args.cosine_max_len,
        ),
        "repetition_penalty": get_repetition_penalty_reward(
            ngram_size=script_args.repetition_n_grams,
            max_penalty=script_args.repetition_max_penalty,
        ),
        "length": len_reward,
        "code": update_wrapper(
            partial(
                code_reward,
                num_parallel=script_args.parallel_code_exec_per_proc,
                e2b_router_url=script_args.e2b_router_url,
            ),
            code_reward,
        ),
        "functionality": functionality_reward,
        "watermark": good_watermark_detection_reward,
        "per_token_watermark": per_token_watermark_detection_reward,
        "binary_code": update_wrapper(
            partial(
                binary_code_reward,
                num_parallel=script_args.parallel_code_exec_per_proc,
                e2b_router_url=script_args.e2b_router_url,
            ),
            binary_code_reward,
        ),
        "ioi_code": update_wrapper(
            partial(ioi_code_reward, test_batch_size=script_args.code_eval_test_batch_size), ioi_code_reward
        ),
        "code_format": get_code_format_reward(language=script_args.code_language),
        "tag_count": tag_count_reward,
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    return reward_funcs


def watermark_detection_reward(completions, **kwargs) -> list[float]:
    """Reward function that evaluates the strength of watermark in generated code.
    
    This function uses watermark detection to measure how strongly watermarked the generated
    completions are, providing a reward signal that encourages the model to produce
    properly watermarked text.
    
    Args:
        prompts: List of prompts
        completions: List of model completions 
        model: The model to use for scoring logits (passed from the trainer)
        tokenizer: Tokenizer for processing text
        device: Device to use for processing
        context_width: Context width for detector (default 4)
        
    Returns:
        List of rewards based on watermark detection strength (0.0-1.0 range)
    """
    import math
    from src.watermark.coderl import CodeDetector
    
    # Extract the content from completions if they're in the chat format
    if isinstance(completions[0], list) and "content" in completions[0][0]:
        code_snippets = [extract_code(completion[-1]["content"]) for completion in completions]
    else:
        code_snippets = extract_code(completions)
    
    # Get required parameters from kwargs
    model = kwargs.get("model").watermark_model
    tokenizer = kwargs.get("tokenizer")
    device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    max_length = kwargs.get("max_length", 2048)
    
    # Watermark detector parameters
    context_width = kwargs.get("context_width", 4)
    gamma = kwargs.get("gamma", 0.5)
    z_threshold = kwargs.get("z_threshold", 4.0)
    switch_threshold = kwargs.get("switch_threshold", 0.5)
    
    # Initialize detector - only create it once if possible
    detector = CodeDetector(
        vocab_size=len(tokenizer),
        context_width=context_width,
        device=device,
        gamma=gamma,
        tokenizer=tokenizer,
        z_threshold=z_threshold,
        model=model,
        switch_threshold=switch_threshold
    )
    
    rewards = []
    
    # Calculate rewards for each prompt-completion pair
    for code_snippet in code_snippets:
        try:
            # Process completion tokenization
            content_tokenized = tokenizer(
                code_snippet,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=max_length
            )["input_ids"].squeeze().to(device)
            
            # Detect watermark
            detection_result = detector.detect(
                tokenized_text=content_tokenized,
            )
                
            # Skip invalid detections
            if detection_result.pop('invalid', False):
                rewards.append(0.0)
                continue
                
            # Extract z-score or other watermark strength metric
            z_score = detection_result.get("z_score", 0.0)
            
            # Normalize watermark strength to 0-1 range
            if z_score > 2:
                watermark_reward = 1.0
            elif z_score < 0:
                watermark_reward = 0
            else:
                watermark_reward = z_score / 2
            rewards.append(watermark_reward)
            
        except Exception as e:
            print(f"Error in watermark detection: {e}")
            rewards.append(0.0)
    
    return rewards

def functionality_reward(completions: list[list[dict[str, str]]], **kwargs) -> list[Optional[float]]:

    task = kwargs.get("task")
    task_obj = tasks.get_task(task)

    completions = [completion[-1]["content"] for completion in completions]
    # prompts = [prompt[0]["content"] for prompt in kwargs["prompts"]]
    # task_ids = kwargs["task_id"]  # Remove this line if unused
    # canonical_solutions = kwargs["canonical_solution"]
    references = kwargs["test"]

    if task == "humaneval":
        stop_words = task_obj.stop_words
        processed_completions = [[task_obj._stop_at_stop_token(completion, stop_words)] for completion in completions]
    elif task == "mbpp":
        stop_words = task_obj.stop_words
        processed_completions = [[task_obj.first_block(completion, stop_words)] for completion in completions]

    pass_info, pass_results = task_obj.process_results(processed_completions, references)

    # Convert pass_results (indexed by task_id) into a dense list of rewards
    num_examples = len(processed_completions)
    rewards: list[Optional[float]] = []
    for idx in range(num_examples):
        # pass_results[idx] is a list like [(completion_id, {...})]; take the first result_dict.
        _, result_dict = pass_results[idx][0]
        try:
            passed_flag = result_dict.get("passed", False)
        except Exception as e:
            print(f"Error in functionality reward: {e}")
            import ipdb; ipdb.set_trace()
            

        # Binary reward: 1.0 for pass, 0.0 for fail
        rewards.append(1.0 if passed_flag else 0.0)
    return rewards

def identify_code_token_indices(tokenizer, text: str, token_ids: list[int]) -> list[bool]:
    """
    Identifies which tokens in the tokenized text represent code.
    
    Args:
        tokenizer: The tokenizer used to convert the token IDs to tokens
        text: The original text string
        token_ids: List of token IDs from the tokenizer
        
    Returns:
        A list of booleans where True indicates the token is part of code
    """
    # First identify code spans in the original text
    code_spans = identify_code_spans(text)
    
    # Convert token IDs to tokens for easier debugging
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    
    # Initialize all tokens as non-code
    is_code_token = [False] * len(tokens)
    
    # If no code spans found, return the result
    if not code_spans:
        return is_code_token
    
    # Map token positions to character positions in original text
    token_to_char_map = []
    offset = 0
    
    # Approximate token position mapping
    # This is a simplified approach that may need refinement for your specific tokenizer
    for i, token in enumerate(tokens):
        # Skip special tokens and handle them specially
        if token.startswith("<") and token.endswith(">"):
            token_to_char_map.append((-1, -1))  # Special token
            continue
            
        # Try to find the token in the text starting from the current offset
        token_text = token
        # Handle subword tokens (depending on tokenizer)
        if token.startswith("##") or token.startswith("Ġ") or token.startswith("▁"):
            token_text = token[2:] if token.startswith("##") else token[1:]
        
        # Find position in text
        pos = text.find(token_text, offset)
        if pos >= 0:
            token_to_char_map.append((pos, pos + len(token_text)))
            offset = pos + len(token_text)
        else:
            # Fallback if token not found
            token_to_char_map.append((offset, offset))
    
    # Mark tokens as code if they fall within a code span
    for i, (start, end) in enumerate(token_to_char_map):
        if start == -1:  # Skip special tokens
            continue
            
        # Check if token overlaps with any code span
        for code_start, code_end in code_spans:
            if (start <= code_end and end >= code_start):
                is_code_token[i] = True
                break
    
    return is_code_token

def identify_code_spans(text: str) -> list[tuple[int, int]]:
    spans = []
    for match in re.finditer(r"```(?:[a-zA-Z0-9_]+)?\s*\n(.*?)\n\s*```", text, re.DOTALL):
        spans.append((match.start(1), match.end(1)))
    for match in re.finditer(r"`([^`\n]+?)`", text):
        if not any(s_start <= match.start(1) <= s_end for s_start, s_end in spans):
             spans.append((match.start(1), match.end(1)))
    return spans

def per_token_watermark_detection_reward(completions, **kwargs) -> tuple[list[float], list[list[int]]]:
    from src.watermark.coderl import CodeDetector
    import torch

    # Extract text content from completions
    completions = [completion[-1]["content"] for completion in completions]

    # Get required parameters from kwargs
    model = kwargs.get("model").watermark_model
    tokenizer = kwargs.get("tokenizer")
    device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    max_length = kwargs.get("max_length", 2048)
    
    # Watermark detector parameters
    context_width = kwargs.get("context_width", 4)
    gamma = kwargs.get("gamma", 0.5)
    z_threshold = kwargs.get("z_threshold", 4.0)
    switch_threshold = kwargs.get("switch_threshold", 0.5)

    # Initialize detector
    detector = CodeDetector(
        vocab_size=tokenizer.vocab_size,
        context_width=context_width,
        device=device,
        gamma=gamma,
        tokenizer=tokenizer,
        z_threshold=z_threshold,
        model=model,
        switch_threshold=switch_threshold
    )

    rewards = []
    token_annotations_list = []
    per_token_rewards_list = []

    for text_content in completions:
        # Tokenize content
        encoding = tokenizer(
            text_content,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=max_length,
        )
        content_tokenized = encoding["input_ids"].squeeze(0).to(device)
        token_count = content_tokenized.size(0)

        # Handle empty content
        if token_count == 0:
            rewards.append(0.0)
            token_annotations_list.append([])
            continue
        
        # Handle content with insufficient tokens
        if token_count < context_width + 1:
            rewards.append(0.0)
            token_annotations_list.append([-1] * token_count)
            continue

        # Initialize all annotations as -1 (non-code)
        per_token_annotations = [-1] * token_count
        
        # First identify which tokens are code
        is_code_flags = identify_code_token_indices(tokenizer, text_content, content_tokenized.tolist())
        
        # Find first contiguous code segment to process
        first_code_segment = None
        in_code_segment = False
        code_start = None
        
        for i, is_code in enumerate(is_code_flags):
            if is_code and not in_code_segment:
                # Found start of first code segment
                in_code_segment = True
                # Include context_width tokens before if available
                code_start = max(0, i - context_width)
            elif not is_code and in_code_segment:
                # Found end of first code segment
                first_code_segment = {
                    "start": code_start, 
                    "end": i  # End is exclusive
                }
                break
                
        # If we reached the end still in a code segment
        if in_code_segment and first_code_segment is None:
            first_code_segment = {
                "start": code_start,
                "end": token_count
            }
        
        if first_code_segment:
            start, end = first_code_segment["start"], first_code_segment["end"]
            
            # Extract this segment
            segment_tokens = content_tokenized[start:end]
            
            # Only process if segment has enough tokens
            if len(segment_tokens) > context_width:
                # Get watermark status only for this segment
                segment_statuses = detector.get_token_watermark_status(segment_tokens)
                per_token_annotations[start:end] = segment_statuses

        
            def transform_list(per_token_annotations, max_length):
                mapping = {
                    1: 1,   # if element is 1, return 1, green token
                    2: -3,  # if element is 2, return -1, red token
                    -1: 2,  # if element is -1, return 0, not code
                    0: 0    # if element is 0, return 0, no watermark
                }
                
                # Create a list first, then convert to tensor
                transformed_list = [mapping[elem] for elem in per_token_annotations]
                
                # Create a padded tensor filled with zeros
                padded_tensor = torch.ones(max_length, dtype=torch.long, device=device)*2
                
                # Fill in the values up to the length of the transformed list
                padded_tensor[:len(transformed_list)] = torch.tensor(transformed_list, device=device)
                
                return padded_tensor

            # Assuming we have a list of per_token_annotations
            per_token_rewards = transform_list(per_token_annotations, max_length)
            per_token_rewards_list.append(per_token_rewards)
        else:
            per_token_rewards_list.append(torch.ones(max_length, dtype=torch.long, device=device)*2)
    
    rewards_tensor = torch.stack(per_token_rewards_list)
    return rewards_tensor

def good_watermark_detection_reward(completions, **kwargs) -> list[float]:
    from src.watermark.coderl import CodeDetector
    import torch

    # Extract text content from completions
    completions = [completion[-1]["content"] for completion in completions]

    # Get required parameters from kwargs
    model = kwargs.get("model").watermark_model
    tokenizer = kwargs.get("tokenizer")
    device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    max_length = kwargs.get("max_length", 2048)
    
    # Watermark detector parameters
    context_width = kwargs.get("context_width", 4)
    gamma = kwargs.get("gamma", 0.5)
    z_threshold = kwargs.get("z_threshold", 4.0)
    switch_threshold = kwargs.get("switch_threshold", 0.5)

    # Initialize detector
    detector = CodeDetector(
        vocab_size=tokenizer.vocab_size,
        context_width=context_width,
        device=device,
        gamma=gamma,
        tokenizer=tokenizer,
        z_threshold=z_threshold,
        model=model,
        switch_threshold=switch_threshold
    )

    rewards = []
    token_annotations_list = []
    per_token_rewards_list = []

    for text_content in completions:
        # Tokenize content
        encoding = tokenizer(
            text_content,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=max_length,
        )
        content_tokenized = encoding["input_ids"].squeeze(0).to(device)
        token_count = content_tokenized.size(0)

        # Handle empty content
        if token_count == 0:
            rewards.append(0.0)
            token_annotations_list.append([])
            continue
        
        # Handle content with insufficient tokens
        if token_count < context_width + 1:
            rewards.append(0.0)
            token_annotations_list.append([-1] * token_count)
            continue

        # Initialize all annotations as -1 (non-code)
        per_token_annotations = [-1] * token_count
        
        # First identify which tokens are code
        is_code_flags = identify_code_token_indices(tokenizer, text_content, content_tokenized.tolist())
        
        # Find first contiguous code segment to process
        first_code_segment = None
        in_code_segment = False
        code_start = None
        
        for i, is_code in enumerate(is_code_flags):
            if is_code and not in_code_segment:
                # Found start of first code segment
                in_code_segment = True
                # Include context_width tokens before if available
                code_start = max(0, i - context_width)
            elif not is_code and in_code_segment:
                # Found end of first code segment
                first_code_segment = {
                    "start": code_start, 
                    "end": i  # End is exclusive
                }
                break
                
        # If we reached the end still in a code segment
        if in_code_segment and first_code_segment is None:
            first_code_segment = {
                "start": code_start,
                "end": token_count
            }
        
        if first_code_segment:
            start, end = first_code_segment["start"], first_code_segment["end"]
            
            # Extract this segment
            segment_tokens = content_tokenized[start:end]
            
            # Only process if segment has enough tokens
            if len(segment_tokens) > context_width:
                # Get watermark status only for this segment
                segment_statuses = detector.get_token_watermark_status(segment_tokens)
                per_token_annotations[start:end] = segment_statuses
                
                # Detect watermark
                detection_result = detector.detect(
                    tokenized_text=content_tokenized,
                )
                    
                # Skip invalid detections
                if detection_result.pop('invalid', False):
                    rewards.append(0.0)
                    continue
                    
                # Extract z-score or other watermark strength metric
                z_score = detection_result.get("z_score", 0.0)
                
                # Normalize watermark strength to 0-1 range
                if z_score > 3:
                    watermark_reward = 1.0
                elif z_score < 0:
                    watermark_reward = 0
                else:
                    watermark_reward = z_score / 3
                rewards.append(watermark_reward)

        
            def transform_list(per_token_annotations, max_length):
                mapping = {
                    1: 1,   # if element is 1, return 1, green token
                    2: -3,  # if element is 2, return -1, red token
                    -1: 2,  # if element is -1, return 0, not code
                    0: 0    # if element is 0, return 0, no watermark
                }
                
                # Create a list first, then convert to tensor
                transformed_list = [mapping[elem] for elem in per_token_annotations]
                
                # Create a padded tensor filled with zeros
                padded_tensor = torch.ones(max_length, dtype=torch.long, device=device)*2
                
                # Fill in the values up to the length of the transformed list
                padded_tensor[:len(transformed_list)] = torch.tensor(transformed_list, device=device)
                
                return padded_tensor

            # Assuming we have a list of per_token_annotations
            per_token_rewards = transform_list(per_token_annotations, max_length)
            per_token_rewards_list.append(per_token_rewards)
        else:
            rewards.append(0.0)
            per_token_rewards_list.append(torch.ones(max_length, dtype=torch.long, device=device)*2)
    
    rewards_tensor = torch.stack(per_token_rewards_list)
    try:
        assert rewards_tensor.size(0) == len(rewards)
    except:
        import ipdb; ipdb.set_trace()
    return rewards_tensor, rewards