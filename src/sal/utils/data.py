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
import time
from pathlib import Path
import sys
import os

import pandas as pd
from datasets import Dataset, load_dataset, concatenate_datasets
from huggingface_hub import (
    create_branch,
    list_repo_commits,
    repo_exists,
)

from sal.config import Config

logger = logging.getLogger()

# Add evaluation path to import bbh_format_guide
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../evaluation'))
try:
    from bbh_format_guide import add_format_instruction_to_prompt
except ImportError:
    logger.warning("Failed to import bbh_format_guide, format instructions will not be added")
    def add_format_instruction_to_prompt(prompt, bbh_subset):
        return prompt


def get_dataset(config: Config) -> Dataset:
    if config.dataset_name == "sampled_math500":
        dataset = pd.read_csv(config.dataset_name + ".csv")
        dataset = Dataset.from_pandas(dataset)
    elif config.dataset_name == "prm_math500":
        # load from jsonl file
        dataset = pd.read_json("train.jsonl", lines=True)
        dataset = Dataset.from_pandas(dataset)
    elif "mmlu_pro" in config.dataset_name.lower() or "MMLU-Pro" in config.dataset_name:
        # MMLU-Pro dataset
        dataset = load_dataset(
            "TIGER-Lab/MMLU-Pro" if config.dataset_name.lower() == "mmlu_pro" else config.dataset_name,
            split=config.dataset_split,
            trust_remote_code=True
        )
        
        # If dataset_start and dataset_end are specified, sample evenly from each category
        if config.dataset_start is not None and config.dataset_end is not None:
            # Group by category and sample evenly
            import numpy as np
            
            # Get all unique categories
            categories = set(dataset['category'])
            logger.info(f"Found {len(categories)} categories in MMLU-Pro dataset")
            
            # Calculate how many samples per category to select
            num_categories = len(categories)
            total_samples_needed = config.dataset_end - config.dataset_start
            samples_per_category = total_samples_needed // num_categories
            remainder = total_samples_needed % num_categories
            
            logger.info(f"Selecting {samples_per_category} samples per category (+ {remainder} extra)")
            
            # Select samples from each category
            selected_indices = []
            for i, category in enumerate(sorted(categories)):
                # Get indices for this category
                category_indices = [j for j, cat in enumerate(dataset['category']) if cat == category]
                
                # Calculate how many to take from this category
                num_to_take = samples_per_category + (1 if i < remainder else 0)
                
                # Sample evenly across the category
                if num_to_take > 0 and len(category_indices) > 0:
                    # Use numpy to evenly distribute indices
                    step = max(1, len(category_indices) // num_to_take)
                    sampled_indices = category_indices[::step][:num_to_take]
                    selected_indices.extend(sampled_indices)
                    
                    logger.info(f"  Category '{category}': selected {len(sampled_indices)} samples")
            
            # Select the samples from the dataset
            dataset = dataset.select(selected_indices)
            logger.info(f"Total samples selected: {len(dataset)}")
        
        # Format questions with options for MMLU-Pro
        # Define formatting function inline to avoid import issues
        def format_mmlu_pro_question(example):
            """Format MMLU-Pro question with options."""
            question = example.get("question", "").strip()
            options = example.get("options", example.get("choices", []))
            
            if not options:
                return question
            
            # Format options as (A) option1 (B) option2 ...
            formatted_options = []
            option_letters = "ABCDEFGHIJ"[:len(options)]  # Support up to 10 options
            
            for letter, option in zip(option_letters, options):
                formatted_options.append(f"({letter}) {str(option).strip()}")
            
            options_str = " ".join(formatted_options)
            full_question = f"{question}\nAnswer Choices: {options_str}"
            
            return full_question
        
        # Format each example to include options in the problem field
        def format_mmlu_pro_example(example):
            formatted_question = format_mmlu_pro_question(example)
            example['problem'] = formatted_question
            return example
        
        dataset = dataset.map(format_mmlu_pro_example)
        
        # Keep the original question and options fields for reference
        # No need to rename, just added formatted 'problem' field
    elif config.dataset_name == "lukaemon/bbh":
        # BBH dataset requires config names (subsets)
        bbh_subsets = [
            'boolean_expressions', 'causal_judgement', 'date_understanding', 
            'disambiguation_qa', 'dyck_languages', 'formal_fallacies', 
            'geometric_shapes', 'hyperbaton', 'logical_deduction_five_objects', 
            'logical_deduction_seven_objects', 'logical_deduction_three_objects', 
            'movie_recommendation', 'multistep_arithmetic_two', 'navigate', 
            'object_counting', 'penguins_in_a_table', 'reasoning_about_colored_objects', 
            'ruin_names', 'salient_translation_error_detection', 'snarks', 
            'sports_understanding', 'temporal_sequences', 'tracking_shuffled_objects_five_objects', 
            'tracking_shuffled_objects_seven_objects', 'tracking_shuffled_objects_three_objects', 
            'web_of_lies', 'word_sorting'
        ]
        
        # Load samples from each subset
        subset_datasets = []
        start_idx = config.dataset_start if config.dataset_start is not None else 0
        end_idx = config.dataset_end if config.dataset_end is not None else None
        
        for subset in bbh_subsets:
            try:
                subset_data = load_dataset(
                    config.dataset_name, 
                    subset, 
                    split=config.dataset_split, 
                    trust_remote_code=False  # trust_remote_code is deprecated
                )
                
                # Select range from this subset
                if end_idx is not None:
                    num_samples = min(end_idx - start_idx, len(subset_data))
                    if num_samples > 0:
                        subset_data = subset_data.select(range(start_idx, min(start_idx + num_samples, len(subset_data))))
                        # Add subset name and map 'input' field to 'problem' field for BBH
                        def format_bbh_example(x):
                            result = {**x, 'bbh_subset': subset}
                            # BBH uses 'input' field for the question, map it to 'problem'
                            if 'input' in x and 'problem' not in x:
                                # Add format instruction to the problem prompt
                                result['problem'] = add_format_instruction_to_prompt(x['input'], subset)
                            return result
                        subset_data = subset_data.map(format_bbh_example)
                        subset_datasets.append(subset_data)
                else:
                    if start_idx < len(subset_data):
                        subset_data = subset_data.select(range(start_idx, len(subset_data)))
                        # Add subset name and map 'input' field to 'problem' field for BBH
                        def format_bbh_example(x):
                            result = {**x, 'bbh_subset': subset}
                            # BBH uses 'input' field for the question, map it to 'problem'
                            if 'input' in x and 'problem' not in x:
                                # Add format instruction to the problem prompt
                                result['problem'] = add_format_instruction_to_prompt(x['input'], subset)
                            return result
                        subset_data = subset_data.map(format_bbh_example)
                        subset_datasets.append(subset_data)
                        
                logger.info(f"Loaded {len(subset_data)} samples from BBH subset: {subset}")
            except Exception as e:
                logger.warning(f"Failed to load BBH subset {subset}: {e}")
        
        # Concatenate all subset datasets
        if subset_datasets:
            dataset = concatenate_datasets(subset_datasets)
            logger.info(f"Total BBH samples loaded: {len(dataset)} from {len(subset_datasets)} subsets")
        else:
            raise ValueError("No BBH subsets could be loaded")
    elif config.dataset_name == "mbpp":
        # MBPP (Mostly Basic Programming Problems) dataset for code generation
        dataset = load_dataset(
            "mbpp",
            split=config.dataset_split,
            trust_remote_code=True
        )
        
        # MBPP has the following fields:
        # - task_id: problem ID
        # - text: problem description
        # - code: reference solution code
        # - test_list: list of test cases (assertions)
        # - test_setup_code: setup code to run before tests
        # - challenge_test_list: additional challenging test cases
        
        # Map text field to problem field for consistency
        def format_mbpp_example(example):
            result = {**example}
            # Map text to problem field and add format instruction
            problem_text = example.get('text', '')
            # Add format instruction to guide model output with clear structure
            formatted_problem = (
                f"{problem_text}\n\n"
                "Please solve this problem following these steps:\n"
                "1. First, provide your reasoning and approach to solve the problem\n"
                "2. Then, implement your solution as a complete Python function\n\n"
                "IMPORTANT: You must write your Python code inside a markdown code block like this:\n"
                "```python\n"
                "def your_function_name(parameters):\n"
                "    # Your implementation here\n"
                "    return result\n"
                "```\n\n"
                "Start with your reasoning, then provide the code."
            )
            result['problem'] = formatted_problem
            # Keep test_list for evaluation
            # Keep code as ground truth answer
            result['answer'] = example.get('code', '')
            return result
        
        dataset = dataset.map(format_mbpp_example)
        logger.info(f"Loaded {len(dataset)} samples from MBPP dataset")
    else:
        dataset = load_dataset(config.dataset_name, split=config.dataset_split, trust_remote_code=True)

    # Apply dataset_start and dataset_end for non-BBH and non-MMLU-Pro datasets
    # (MMLU-Pro already handles this with category-based sampling)
    is_mmlu_pro = "mmlu_pro" in config.dataset_name.lower() or "MMLU-Pro" in config.dataset_name
    if config.dataset_name != "lukaemon/bbh" and not is_mmlu_pro:
        if config.dataset_start is not None and config.dataset_end is not None:
            dataset = dataset.select(range(config.dataset_start, config.dataset_end))
        if config.num_samples is not None:
            dataset = dataset.select(range(min(len(dataset), config.num_samples)))

    return dataset


def save_dataset(dataset, config):
    def _extract_model_tag(model_path: str) -> str:
        if not model_path:
            return "unknown"
        name = os.path.basename(model_path.rstrip("/"))
        if name.startswith("models--"):
            name = name[len("models--"):]
        for suffix in ["-Instruct", "-instruction", "-chat", "-Chat", "-base", "-Base"]:
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        return name or "unknown"

    model_tag = _extract_model_tag(getattr(config, "model_path", ""))
    draft_model_tag = (
        _extract_model_tag(getattr(config, "draft_model_path", ""))
        if getattr(config, "draft_model_path", None)
        else None
    )
    draft_suffix = f"_draft-{draft_model_tag}" if draft_model_tag else ""

    if config.push_to_hub:
        # Since concurrent pushes can get rejected by the Hub, we make several attempts to push the dataset with try/except
        for _ in range(20):
            try:
                # Create branch from the repo's initial commit.
                # This is needed to avoid branching from a commit on main that already has data
                if repo_exists(config.hub_dataset_id, repo_type="dataset"):
                    initial_commit = list_repo_commits(
                        config.hub_dataset_id, repo_type="dataset"
                    )[-1]
                    create_branch(
                        repo_id=config.hub_dataset_id,
                        branch=config.revision,
                        revision=initial_commit.commit_id,
                        exist_ok=True,
                        repo_type="dataset",
                    )
                url = dataset.push_to_hub(
                    config.hub_dataset_id,
                    revision=config.revision,
                    split="train",
                    private=config.hub_dataset_private,
                    commit_message=f"Add {config.revision}",
                )
                break
            except Exception as e:
                logger.error(f"Error pushing dataset to the Hub: {e}")
                time.sleep(5)
        logger.info(f"Pushed dataset to {url}")
    else:
        if config.output_dir is None:
            # Set specific output directory for special score methods
            if getattr(config, 'run_slm_only_baseline', False):
                config.output_dir = "/storage/ukp/work/xie12/uncertainty-guided-reasoning/UQ_Guided_Router/outputs/smart/slm_only"
            elif getattr(config, 'run_llm_only_baseline', False):
                config.output_dir = "/storage/ukp/work/xie12/uncertainty-guided-reasoning/UQ_Guided_Router/outputs/smart/llm_only"
            elif config.score_method == 'random_score':
                config.output_dir = "/storage/ukp/work/xie12/uncertainty-guided-reasoning/UQ_Guided_Router/outputs/smart/random_score"
            elif config.score_method == 'token_sar':
                config.output_dir = "/storage/ukp/work/xie12/uncertainty-guided-reasoning/UQ_Guided_Router/outputs/smart/token_sar"
            elif config.score_method == 'token_sar_conf_margin':
                config.output_dir = "/storage/ukp/work/xie12/uncertainty-guided-reasoning/UQ_Guided_Router/outputs/smart/token_sar_conf_margin"
            elif config.score_method == 'hybrid_prm':
                config.output_dir = "/storage/ukp/work/xie12/uncertainty-guided-reasoning/UQ_Guided_Router/outputs/smart/hybrid_prm"
            else:
                config.output_dir = f"/storage/ukp/work/xie12/uncertainty-guided-reasoning/UQ_Guided_Router/outputs/smart/{config.score_method}"
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Name the folder based on the approach used
        if getattr(config, 'run_slm_only_baseline', False):
            folder_name = "smart_slm_only"
        elif getattr(config, 'run_llm_only_baseline', False):
            folder_name = "smart_llm_only"
        elif config.draft_model_path is not None:
            if config.score_method == 'prm':
                folder_name = "smart_prm"
            elif config.score_method == 'conf':
                folder_name = "smart_conf"
            elif config.score_method == 'perplexity':
                folder_name = "smart_perplexity"
            elif config.score_method == 'msp':
                folder_name = "smart_msp"
            elif config.score_method == 'top2_margin':
                folder_name = "smart_margin"
            elif config.score_method == 'token_entropy':
                folder_name = "smart_token_entropy"
            elif config.score_method == 'token_sar':
                folder_name = "smart_token_sar"
            elif config.score_method == 'token_sar_conf_margin':
                folder_name = "smart_token_sar_conf_margin"
            elif config.score_method == 'random_score':
                folder_name = "smart_random_score"
            elif config.score_method == 'hybrid_prm':
                folder_name = "smart_hybrid_prm"
            elif config.score_method == 'uhead':
                folder_name = "smart_uhead"
            # if score_methode starts with cocoa, then folder_name is smart_cocoa
            elif config.score_method.startswith('cocoa'):
                folder_name = "smart_cocoa"
        else: 
            if config.score_method == 'prm':
                folder_name = "base_prm"
            elif config.score_method == 'conf':
                folder_name = "base_conf"
            elif config.score_method == 'perplexity':
                folder_name = "base_perplexity"
            elif config.score_method == 'msp':
                folder_name = "base_msp"
            elif config.score_method == 'top2_margin':
                folder_name = "base_margin"
            elif config.score_method == 'token_entropy':
                folder_name = "base_token_entropy"
            elif config.score_method == 'token_sar':
                folder_name = "base_token_sar"
            elif config.score_method == 'token_sar_conf_margin':
                folder_name = "base_token_sar_conf_margin"
            elif config.score_method == 'hybrid_prm':
                folder_name = "base_hybrid_prm"
            elif config.score_method == 'uhead':
                folder_name = "base_uhead"
        # Name the appoarch in likelihood score
        if config.beam_width == 1:
            approach_fn = "best_of_n"
        else: 
            approach_fn = config.approach
        
        # Clean dataset name for filename (replace slashes and special chars)
        dataset_name_clean = config.dataset_name.replace('/', '_').replace('\\', '_')
        
        # Create folder_name subdirectory if it doesn't exist
        folder_path = Path(config.output_dir) / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)

        # Include seed in filename when available
        seed = getattr(config, 'seed', None)
        seed_suffix = f"_seed-{seed}" if seed is not None else ""
        
        # Save the dataset to a jsonl file by splitting the dataset or not
        if getattr(config, 'run_slm_only_baseline', False):
            # SLM-only baseline: use threshold format but with None/0 values since no correction is used
            # Get threshold values from config (use defaults if not set)
            threshold_val = getattr(config, 'threshold', 0.1)
            uq_threshold_val = getattr(config, 'uq_threshold', 0.15)
            if config.dataset_start is not None and config.dataset_end is not None:
                filename = f"beam_search_dataset-{dataset_name_clean}_completions_T-{config.temperature}--top_p-{config.top_p}--n-{config.n}--m-{config.beam_width}--iters-{config.num_iterations}--look-{config.lookahead}--seed-{config.seed}--agg_strategy--{config.agg_strategy}_threshold-{threshold_val}_threshold_uq-{uq_threshold_val}_None_datasplit_{config.dataset_start}-{config.dataset_end}_method-slm_only_use_default_beam_search-False{seed_suffix}_model-{model_tag}{draft_suffix}.jsonl"
            else:
                filename = f"beam_search_dataset-{dataset_name_clean}_completions_T-{config.temperature}--top_p-{config.top_p}--n-{config.n}--m-{config.beam_width}--iters-{config.num_iterations}--look-{config.lookahead}--seed-{config.seed}--agg_strategy--{config.agg_strategy}_threshold-{threshold_val}_threshold_uq-{uq_threshold_val}_None_datasplit_None_method-slm_only_use_default_beam_search-False{seed_suffix}_model-{model_tag}{draft_suffix}.jsonl"
            dataset.to_json(
                f"{config.output_dir}/{folder_name}/{filename}", lines=True
            )
            logger.info(
                f"Saved completions to {config.output_dir}/{folder_name}/{filename}"
            )
        elif getattr(config, 'run_llm_only_baseline', False):
            # LLM-only baseline: use threshold format but with None/0 values since no correction is used
            # Get threshold values from config (use defaults if not set)
            threshold_val = getattr(config, 'threshold', 0.1)
            uq_threshold_val = getattr(config, 'uq_threshold', 0.15)
            if config.dataset_start is not None and config.dataset_end is not None:
                filename = f"beam_search_dataset-{dataset_name_clean}_completions_T-{config.temperature}--top_p-{config.top_p}--n-{config.n}--m-{config.beam_width}--iters-{config.num_iterations}--look-{config.lookahead}--seed-{config.seed}--agg_strategy--{config.agg_strategy}_threshold-{threshold_val}_threshold_uq-{uq_threshold_val}_None_datasplit_{config.dataset_start}-{config.dataset_end}_method-llm_only_use_default_beam_search-False{seed_suffix}_model-{model_tag}{draft_suffix}.jsonl"
            else:
                filename = f"beam_search_dataset-{dataset_name_clean}_completions_T-{config.temperature}--top_p-{config.top_p}--n-{config.n}--m-{config.beam_width}--iters-{config.num_iterations}--look-{config.lookahead}--seed-{config.seed}--agg_strategy--{config.agg_strategy}_threshold-{threshold_val}_threshold_uq-{uq_threshold_val}_None_datasplit_None_method-llm_only_use_default_beam_search-False{seed_suffix}_model-{model_tag}{draft_suffix}.jsonl"
            dataset.to_json(
                f"{config.output_dir}/{folder_name}/{filename}", lines=True
            )
            logger.info(
                f"Saved completions to {config.output_dir}/{folder_name}/{filename}"
            )
        elif config.dataset_start is not None and config.dataset_end is not None:
            # Filename: dataset_method_uqthreshold_th{threshold}_start-end
            temperature = getattr(config, "temperature", None)
            temp_suffix = f"_T-{temperature}" if temperature is not None else ""
            threshold_val = getattr(config, 'threshold', None)
            threshold_suffix = f"_th{threshold_val}" if threshold_val is not None else ""
            filename = (
                f"{dataset_name_clean}_{config.score_method}_uq{config.uq_threshold}"
                f"{threshold_suffix}{temp_suffix}{seed_suffix}_{config.dataset_start}-{config.dataset_end}_model-{model_tag}.jsonl"
            )
            filename = filename.replace(".jsonl", f"{draft_suffix}.jsonl")
            dataset.to_json(
                f"{config.output_dir}/{folder_name}/{filename}", lines=True
            )
            logger.info(
                f"Saved completions to {config.output_dir}/{folder_name}/{filename}"
            )
        else:
            # Filename without start-end: dataset_method_uqthreshold_th{threshold}
            temperature = getattr(config, "temperature", None)
            temp_suffix = f"_T-{temperature}" if temperature is not None else ""
            threshold_val = getattr(config, 'threshold', None)
            threshold_suffix = f"_th{threshold_val}" if threshold_val is not None else ""
            filename = (
                f"{dataset_name_clean}_{config.score_method}_uq{config.uq_threshold}"
                f"{threshold_suffix}{temp_suffix}{seed_suffix}_model-{model_tag}{draft_suffix}.jsonl"
            )
            dataset.to_json(
                f"{config.output_dir}/{folder_name}/{filename}", lines=True
            )
            logger.info(
                f"Saved completions to {config.output_dir}/{folder_name}/{filename}"
            )