#!/usr/bin/env python
# Pretraining script for SmolLMForCausalLM model on tiny Shakespeare dataset

import os
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    get_cosine_schedule_with_warmup,
    set_seed,
    AutoTokenizer,
    PreTrainedTokenizerFast,
)
from transformers.trainer_pt_utils import get_parameter_names
from accelerate import Accelerator
from typing import Dict, List, Optional, Union, Any, Tuple
import gc
import logging
import sys
import argparse
import json

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
MODEL_NAME = "HuggingFaceTB/SmolLM-135M-Instruct"
SEQUENCE_LENGTH = 512
BATCH_SIZE = 16  # Per-device batch size
GRADIENT_ACCUMULATION_STEPS = 4  # Adjust to achieve effective batch size of 0.5M tokens
LEARNING_RATE = 6e-4
WEIGHT_DECAY = 0.1
WARMUP_STEPS = 100
NUM_TRAIN_EPOCHS = 1
LOGGING_STEPS = 10
SAVE_STEPS = 1000
USE_FLASH_ATTENTION = True  # Whether to use Flash Attention
USE_MOE = False  # Whether to use Mixture of Experts
NUM_EXPERTS = 8  # Number of experts for MOE
NUM_EXPERTS_PER_TOKEN = 2  # Number of experts to route each token to
DATA_PATH = "data/pretrain/pretrain_data.jsonl"  # Default path to pretraining data
TOKENIZER_PATH = "tokenizer"  # Path to tokenizer directory
TOKENIZER_TYPE = "qwen"  # Type of tokenizer to use ('qwen' or 'custom')


# Dataset preparation
def get_dataset(data_path=None):
    """Load dataset from JSONL file."""
    data_path = data_path or DATA_PATH
    logger.info(f"Loading dataset from {data_path}")

    # Check if file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Load data from JSONL
    train_texts = []
    val_texts = []

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            text = data.get("text", "")
            if text:
                train_texts.append(text)

    # Split into train and validation (90/10)
    val_split = int(len(train_texts) * 0.9)
    val_texts = train_texts[val_split:]
    train_texts = train_texts[:val_split]

    logger.info(
        f"Loaded {len(train_texts)} training examples and {len(val_texts)} validation examples"
    )

    # Convert to HF dataset format
    from datasets import Dataset

    train_dataset = Dataset.from_dict({"text": train_texts})
    val_dataset = Dataset.from_dict({"text": val_texts})

    return {"train": train_dataset, "validation": val_dataset}


# Model initialization with random weights
def init_model_for_training(tokenizer_path, tokenizer_type):
    """Initialize the model with random weights for pretraining, with optional Flash Attention."""
    from transformers import AutoConfig

    logger.info("Initializing SmolLM model with random weights...")

    # Load the original configuration for architecture details
    official_config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Create our custom config with Flash Attention and MOE if requested
    custom_config = SmolLMConfig(
        official_config=official_config,
        use_flashattention=USE_FLASH_ATTENTION,
        use_moe=USE_MOE,
        num_experts=NUM_EXPERTS,
        num_experts_per_token=NUM_EXPERTS_PER_TOKEN,
    )

    # Initialize our custom model
    logger.info(
        f"Flash Attention is {'enabled' if USE_FLASH_ATTENTION else 'disabled'}"
    )
    if USE_MOE:
        logger.info(
            f"MOE is enabled with {NUM_EXPERTS} experts, {NUM_EXPERTS_PER_TOKEN} experts per token"
        )
    model = SmolLMForCausalLM(custom_config)

    # Apply proper weight initialization for a transformer model
    logger.info("Applying weight initialization...")

    def _init_weights(module):
        if isinstance(module, nn.Linear):
            # Use standard initialization for linear layers
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Initialize embeddings with normal distribution
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, "padding_idx") and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)

    # Apply the initialization to all modules
    model.apply(_init_weights)

    # Apply special scaling to final layer norm
    with torch.no_grad():
        if hasattr(model.model, "norm"):
            model.model.norm.weight.fill_(1.0)

    # Tie weights if specified in config
    if model.config.tie_word_embeddings:
        model.tie_weights()
        logger.info("Tied input and output embedding weights")

    # Use torch.compile to speed up training if available
    if torch.cuda.is_available() and hasattr(torch, "compile"):
        logger.info("Applying torch.compile for faster training...")
        model = torch.compile(model)

    return model


# Custom Optimizer with AdamW (fused), weight decay exclusion, and gradient norm clipping
def get_optimizer(model, lr, weight_decay):
    """Create optimizer with proper weight decay exclusion and use fused Adam if available."""
    # Filter parameters that should not have weight decay
    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]

    # Organize parameters
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if n not in decay_parameters
            ],
            "weight_decay": 0.0,
        },
    ]

    # Use fused Adam if available (faster on CUDA)
    if torch.cuda.is_available():
        try:
            from torch.optim.adam import Adam as FusedAdam

            logger.info("Using fused Adam optimizer")
            optimizer = FusedAdam(
                optimizer_grouped_parameters,
                lr=lr,
                betas=(0.9, 0.95),
                eps=1e-8,
                fused=True,
            )
        except ImportError:
            logger.info("Fused Adam not available, using standard AdamW")
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=lr,
                betas=(0.9, 0.95),
                eps=1e-8,
            )
    else:
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=lr,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

    return optimizer


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train SmolLM with pretraining data")
    parser.add_argument(
        "--data_path",
        type=str,
        default=DATA_PATH,
        help=f"Path to pretraining data JSONL file (default: {DATA_PATH})",
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
    args = parser.parse_args()

    # Update global variables with command line arguments
    global USE_MOE, NUM_EXPERTS, NUM_EXPERTS_PER_TOKEN
    USE_MOE = args.use_moe
    NUM_EXPERTS = args.num_experts
    NUM_EXPERTS_PER_TOKEN = args.num_experts_per_token

    # 1. Initialize accelerator
    accelerator = Accelerator(
        mixed_precision="fp16", log_with="tensorboard", project_dir="./logs"
    )

    accelerator.print(f"Running on {accelerator.device}")

    # 2. Get dataset
    dataset = get_dataset(args.data_path)

    # 3. Initialize model with new tokenizer configuration
    model = init_model_for_training(
        tokenizer_path=args.tokenizer_path,
        tokenizer_type=args.tokenizer_type,
    )

    # 4. Create tokenizer from model config
    if args.tokenizer_type == "qwen":
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    else:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_path)

    # Handle missing pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")

    # 5. Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal language modeling
    )

    # 6. Prepare training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        evaluation_strategy="steps",
        eval_steps=SAVE_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        lr_scheduler_type="cosine",
        logging_dir="./logs",
        logging_steps=LOGGING_STEPS,
        report_to="tensorboard",
        fp16=True,  # Mixed precision training
        remove_unused_columns=False,
        dataloader_num_workers=4,
        gradient_checkpointing=True,  # Memory optimization
        max_grad_norm=1.0,  # Gradient norm clipping
    )

    # 7. Create optimizer and scheduler
    optimizer = get_optimizer(model, LEARNING_RATE, WEIGHT_DECAY)
    total_steps = (
        len(dataset["train"])
        // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
        * NUM_TRAIN_EPOCHS
    )
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps,
    )

    # 8. Create and start Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        optimizers=(optimizer, lr_scheduler),
    )

    # 9. Start training
    logger.info("Starting training...")
    trainer.train()

    # 10. Save final model
    trainer.save_model("./final_model")
    logger.info("Training complete! Model saved to ./final_model")


if __name__ == "__main__":
    main()
