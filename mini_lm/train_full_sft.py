#!/usr/bin/env python
# Supervised Fine-Tuning (SFT) script for SmolLMForCausalLM on the smoltalk dataset

import os
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    set_seed,
    PreTrainedTokenizerFast,
)
from transformers.trainer_pt_utils import get_parameter_names
from peft import prepare_model_for_kbit_training
from accelerate import Accelerator
import logging
import sys
import gc
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
MODEL_PATH = "./final_model"  # Path to the pretrained model
DATASET_NAME = "HuggingFaceTB/smoltalk"
TOKENIZER_PATH = "tokenizer"  # Path to tokenizer directory
TOKENIZER_TYPE = "qwen"  # Type of tokenizer to use ('qwen' or 'custom')
MAX_LENGTH = 1024
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.03
NUM_TRAIN_EPOCHS = 3
LOGGING_STEPS = 10
SAVE_STEPS = 200
OUTPUT_DIR = "./sft_output"
USE_FLASH_ATTENTION = True
USE_MOE = False  # Whether to use Mixture of Experts
NUM_EXPERTS = 8  # Number of experts for MOE
NUM_EXPERTS_PER_TOKEN = 2  # Number of experts to route each token to
DATA_PATH = "data/sft/sft_data.jsonl"  # Default path to SFT data


def load_tokenizer(tokenizer_path=TOKENIZER_PATH, tokenizer_type=TOKENIZER_TYPE):
    """Load the tokenizer for the model."""
    logger.info(f"Loading tokenizer from {tokenizer_path}")
    
    if tokenizer_type == "qwen":
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    else:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

    # Handle missing pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Setting pad_token to eos_token")

    return tokenizer


def load_model(tokenizer_path=TOKENIZER_PATH, tokenizer_type=TOKENIZER_TYPE):
    """Load the pretrained model for fine-tuning."""
    logger.info(f"Loading model from {MODEL_PATH}")

    # Load the model configuration
    config = SmolLMConfig(
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

    # Load model
    model = SmolLMForCausalLM(config)

    # Load pretrained weights if available
    try:
        # Try loading the state dict directly
        state_dict = torch.load(f"{MODEL_PATH}/pytorch_model.bin", map_location="cpu")
        model.load_state_dict(state_dict)
        logger.info("Successfully loaded pretrained weights")
    except Exception as e:
        logger.error(f"Failed to load weights: {e}")
        logger.info("Initializing with random weights")

    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.to("cuda")
        logger.info("Moved model to CUDA")

    return model


def process_dataset(tokenizer, data_path=None):
    """Process the dataset for supervised fine-tuning."""
    data_path = data_path or DATA_PATH
    logger.info(f"Loading SFT dataset from {data_path}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Load raw data
    raw_data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                if "messages" in item:
                    raw_data.append(item)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON line, skipping")

    logger.info(f"Loaded {len(raw_data)} examples")

    # Convert message format to prompt-response format
    formatted_data = []
    for item in raw_data:
        messages = item.get("messages", [])
        if not messages:
            continue

        # Build conversation with appropriate formatting
        formatted_prompt = ""
        formatted_response = ""

        for i, msg in enumerate(messages):
            role = msg.get("role", "")
            content = msg.get("content", "")

            if not content:
                continue

            if role == "user":
                # For user messages
                if formatted_prompt:
                    formatted_prompt += "\n\n"
                formatted_prompt += f"USER: {content}"
            elif role == "assistant":
                # For assistant messages
                if i == len(messages) - 1:  # If this is the last message
                    formatted_response = content
                else:
                    # Add to prompt for middle assistant messages
                    if formatted_prompt:
                        formatted_prompt += "\n\n"
                    formatted_prompt += f"ASSISTANT: {content}"

        # Only add if we have both prompt and response
        if formatted_prompt and formatted_response:
            formatted_data.append(
                {
                    "prompt": formatted_prompt,
                    "response": f"ASSISTANT: {formatted_response}",
                }
            )

    logger.info(f"Converted {len(formatted_data)} examples to prompt-response format")

    # Create dataset
    from datasets import Dataset

    dataset = Dataset.from_dict(
        {
            "prompt": [item["prompt"] for item in formatted_data],
            "response": [item["response"] for item in formatted_data],
        }
    )

    # Tokenize the dataset
    def tokenize_function(examples):
        # Combine prompt and response for training
        combined_texts = []
        for prompt, response in zip(examples["prompt"], examples["response"]):
            combined_texts.append(f"{prompt}\n\n{response}")

        # Tokenize
        tokenized = tokenizer(
            combined_texts,
            padding=False,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors=None,
        )

        # Prepare labels (same as input_ids, for causal LM)
        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized

    # Apply tokenization
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["prompt", "response"],
        desc="Tokenizing dataset",
    )

    return tokenized_dataset


def get_optimizer(model, lr, weight_decay):
    """Create optimizer with proper weight decay exclusion."""
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

    # Use AdamW optimizer
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    return optimizer


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Fine-tune SmolLM with SFT")
    parser.add_argument(
        "--data_path",
        type=str,
        default=DATA_PATH,
        help=f"Path to SFT data JSONL file (default: {DATA_PATH})",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=MODEL_PATH,
        help=f"Path to the base model (default: {MODEL_PATH})",
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
    global USE_MOE, NUM_EXPERTS, NUM_EXPERTS_PER_TOKEN, MODEL_PATH
    USE_MOE = args.use_moe
    NUM_EXPERTS = args.num_experts
    NUM_EXPERTS_PER_TOKEN = args.num_experts_per_token
    MODEL_PATH = args.model_path

    # Start logging
    logger.info("Starting supervised fine-tuning...")

    # 1. Set the seed for reproducibility
    set_seed(42)

    # 2. Load tokenizer
    tokenizer = load_tokenizer(args.tokenizer_path, args.tokenizer_type)

    # 3. Load model
    model = load_model(args.tokenizer_path, args.tokenizer_type)

    # 4. Process dataset
    tokenized_dataset = process_dataset(tokenizer, args.data_path)

    # 5. Create data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding="longest",
        return_tensors="pt",
    )

    # 6. Create optimizer and scheduler
    optimizer = get_optimizer(model, LEARNING_RATE, WEIGHT_DECAY)

    # 7. Create and start Trainer
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=OUTPUT_DIR,
            overwrite_output_dir=True,
            num_train_epochs=NUM_TRAIN_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            evaluation_strategy="steps",
            eval_steps=SAVE_STEPS,
            save_strategy="steps",
            save_steps=SAVE_STEPS,
            save_total_limit=3,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            warmup_ratio=WARMUP_RATIO,
            lr_scheduler_type="cosine",
            logging_dir=f"{OUTPUT_DIR}/logs",
            logging_steps=LOGGING_STEPS,
            report_to="tensorboard",
            fp16=True,
            gradient_checkpointing=True,
            max_grad_norm=1.0,
            dataloader_num_workers=2,
        ),
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"]
        if "validation" in tokenized_dataset
        else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        optimizers=(
            optimizer,
            get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=int(
                    NUM_TRAIN_EPOCHS
                    * (
                        len(tokenized_dataset["train"])
                        // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
                    )
                    * WARMUP_RATIO
                ),
                num_training_steps=len(tokenized_dataset["train"])
                // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
                * NUM_TRAIN_EPOCHS,
            ),
        ),
    )

    # 8. Start training
    train_result = trainer.train()

    # 9. Save final model
    logger.info("Saving final model...")
    trainer.save_model(f"{OUTPUT_DIR}/final")

    # 10. Log and save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
