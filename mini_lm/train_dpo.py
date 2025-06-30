#!/usr/bin/env python
# Direct Preference Optimization (DPO) training script for SmolLMForCausalLM on UltraFeedback dataset

import os
import json
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    set_seed,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)
from trl import DPOTrainer
from accelerate import Accelerator
import logging
import sys
import gc
import argparse
from transformers import PreTrainedTokenizerFast

# Import the custom model implementation
from smol_model import SmolLMConfig, SmolLMForCausalLM

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Set seed for reproducibility
set_seed(42)

# Constants
BASE_MODEL_PATH = "HuggingFaceTB/SmolLM-135M-Instruct"  # Path to SFT model
TOKENIZER_PATH = "tokenizer"  # Path to tokenizer directory
TOKENIZER_TYPE = "qwen"  # Type of tokenizer to use ('qwen' or 'custom')
DATASET_NAME = "HuggingFaceTB/smoltalk-preference"
MAX_LENGTH = 1024
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 16
LEARNING_RATE = 5e-7
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.03
NUM_TRAIN_EPOCHS = 3
LOGGING_STEPS = 10
SAVE_STEPS = 200
OUTPUT_DIR = "./dpo_output"
USE_FLASH_ATTENTION = True
USE_MOE = False  # Whether to use Mixture of Experts
NUM_EXPERTS = 8  # Number of experts for MOE
NUM_EXPERTS_PER_TOKEN = 2  # Number of experts to route each token to
DATA_PATH = "data/dpo/dpo_data.jsonl"  # Default path to DPO data

# DPO specific parameters
BETA = (
    0.1  # Controls how much to optimize for preferences vs staying close to reference
)

# LoRA specific parameters
USE_LORA = True
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def load_model_and_tokenizer(tokenizer_path=TOKENIZER_PATH, tokenizer_type=TOKENIZER_TYPE):
    """Load the SFT model and tokenizer for DPO training."""
    logger.info(f"Loading model from {BASE_MODEL_PATH}")

    # Set up custom config
    custom_config = SmolLMConfig(
        use_flashattention=USE_FLASH_ATTENTION,
        use_moe=USE_MOE,
        num_experts=NUM_EXPERTS,
        num_experts_per_token=NUM_EXPERTS_PER_TOKEN,
        tokenizer_path=tokenizer_path,
        tokenizer_type=tokenizer_type,
    )

    # Log configuration
    if USE_MOE:
        logger.info(
            f"MOE is enabled with {NUM_EXPERTS} experts, {NUM_EXPERTS_PER_TOKEN} experts per token"
        )

    # Load model with custom config
    model = SmolLMForCausalLM(custom_config)

    # Initialize model with pretrained weights
    # This step assumes the weights exist and are compatible with the MOE configuration

    # Load tokenizer
    if tokenizer_type == "qwen":
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    else:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

    # Handle missing pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")

    return model, tokenizer


def apply_lora_if_needed(model):
    """Apply LoRA adapters to the model if USE_LORA is True."""
    if not USE_LORA:
        logger.info("Using full-parameter DPO training (LoRA disabled)")
        return model

    logger.info("Configuring LoRA adapters for DPO training")

    # Define LoRA configuration
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        inference_mode=False,
    )

    # Prepare model for training if needed
    model = prepare_model_for_kbit_training(model)

    # Apply LoRA to model
    logger.info(f"Applying LoRA to target modules: {TARGET_MODULES}")
    model = get_peft_model(model, peft_config)

    # Print trainable parameters
    model.print_trainable_parameters()

    return model


def process_ultrafeedback_dataset(tokenizer, split="train"):
    """Process UltraFeedback dataset for DPO training."""
    logger.info(f"Loading UltraFeedback dataset: {DATASET_NAME}")

    # Load dataset
    dataset = load_dataset(DATASET_NAME, split=split)

    # Extract the prompt, chosen, and rejected responses
    def extract_pairs(example):
        prompt = example["prompt"]
        chosen = example["chosen"]
        rejected = example["rejected"]

        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        }

    # Process the dataset
    processed_dataset = dataset.map(
        extract_pairs,
        remove_columns=dataset.column_names,
        desc="Extracting preference pairs",
    )

    return processed_dataset


def process_dpo_dataset(tokenizer, data_path=None):
    """Process DPO dataset from JSONL file."""
    data_path = data_path or DATA_PATH
    logger.info(f"Loading DPO dataset from {data_path}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Load raw data
    raw_data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                if all(k in item for k in ["prompt", "chosen", "rejected"]):
                    raw_data.append(item)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON line, skipping")

    logger.info(f"Loaded {len(raw_data)} examples")

    # Format data for DPO
    formatted_data = []
    for item in raw_data:
        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]

        # Format prompt with USER prefix if it doesn't have one
        if not prompt.strip().startswith("USER:"):
            prompt = f"USER: {prompt}"

        # Format responses with ASSISTANT prefix if they don't have one
        if not chosen.strip().startswith("ASSISTANT:"):
            chosen = f"ASSISTANT: {chosen}"
        if not rejected.strip().startswith("ASSISTANT:"):
            rejected = f"ASSISTANT: {rejected}"

        formatted_data.append(
            {"prompt": prompt, "chosen": chosen, "rejected": rejected}
        )

    # Create dataset
    from datasets import Dataset

    dataset = Dataset.from_dict(
        {
            "prompt": [item["prompt"] for item in formatted_data],
            "chosen": [item["chosen"] for item in formatted_data],
            "rejected": [item["rejected"] for item in formatted_data],
        }
    )

    # Split into train/validation
    dataset = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    logger.info(
        f"Split into {len(train_dataset)} training and {len(eval_dataset)} validation examples"
    )

    return {"train": train_dataset, "validation": eval_dataset}


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train SmolLM with DPO")
    parser.add_argument(
        "--data_path",
        type=str,
        default=DATA_PATH,
        help=f"Path to DPO data JSONL file (default: {DATA_PATH})",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=BASE_MODEL_PATH,
        help=f"Path to the base model (default: {BASE_MODEL_PATH})",
    )
    parser.add_argument(
        "--use_moe", action="store_true", help="Whether to use Mixture of Experts"
    )
    parser.add_argument(
        "--num_experts",
        type=int,
        default=NUM_EXPERTS,
        help=f"Number of experts for MOE (default: {NUM_EXPERTS})",
    )
    parser.add_argument(
        "--num_experts_per_token",
        type=int,
        default=NUM_EXPERTS_PER_TOKEN,
        help=f"Number of experts per token (default: {NUM_EXPERTS_PER_TOKEN})",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=TOKENIZER_PATH,
        help=f"Path to tokenizer directory (default: {TOKENIZER_PATH})",
    )
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        choices=["qwen", "custom"],
        default=TOKENIZER_TYPE,
        help=f"Type of tokenizer to use (default: {TOKENIZER_TYPE})",
    )
    parser.add_argument(
        "--beta", type=float, default=BETA, help=f"DPO beta parameter (default: {BETA})"
    )
    args = parser.parse_args()

    # Update global variables with command line arguments
    global USE_MOE, NUM_EXPERTS, NUM_EXPERTS_PER_TOKEN, BASE_MODEL_PATH, BETA
    USE_MOE = args.use_moe
    NUM_EXPERTS = args.num_experts
    NUM_EXPERTS_PER_TOKEN = args.num_experts_per_token
    BASE_MODEL_PATH = args.model_path
    BETA = args.beta

    # Start logging
    logger.info("Starting DPO training...")

    # 1. Set the seed for reproducibility
    set_seed(42)

    # 2. Load tokenizer, SFT model, and reference model
    model, tokenizer = load_model_and_tokenizer(args.tokenizer_path, args.tokenizer_type)

    # 3. Process DPO dataset
    dataset = process_dpo_dataset(tokenizer, args.data_path)

    # 4. Set up DPOTrainer
    trainer = DPOTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=NUM_TRAIN_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_dir=f"{OUTPUT_DIR}/logs",
            logging_steps=LOGGING_STEPS,
            learning_rate=LEARNING_RATE,
            bf16=torch.cuda.is_available(),
            save_total_limit=3,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to="tensorboard",
            gradient_checkpointing=True,
        ),
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        beta=BETA,
        max_length=MAX_LENGTH,
        max_prompt_length=MAX_LENGTH // 2,
    )

    # 5. Train the model
    trainer.train()

    # 6. Save the model
    trainer.save_model(f"{OUTPUT_DIR}/final")

    logger.info("DPO training complete!")


if __name__ == "__main__":
    main()
