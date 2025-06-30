#!/usr/bin/env python
# LoRA fine-tuning script for SmolLMForCausalLM on the smoltalk dataset

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
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)
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
MODEL_PATH = "HuggingFaceTB/SmolLM-135M-Instruct"
TOKENIZER_PATH = "tokenizer"  # Path to tokenizer directory
TOKENIZER_TYPE = "qwen"  # Type of tokenizer to use ('qwen' or 'custom')
DATASET_NAME = "HuggingFaceTB/smoltalk"
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
MAX_LENGTH = 1024
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.03
NUM_TRAIN_EPOCHS = 3
LOGGING_STEPS = 10
SAVE_STEPS = 200
OUTPUT_DIR = "./lora_outputs"
USE_FLASH_ATTENTION = True
USE_MOE = False  # Whether to use Mixture of Experts
NUM_EXPERTS = 8  # Number of experts for MOE
NUM_EXPERTS_PER_TOKEN = 2  # Number of experts to route each token to
DATA_PATH = "data/sft/sft_data.jsonl"  # Default path to SFT data for LoRA


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


def load_model_for_training(tokenizer_path=TOKENIZER_PATH, tokenizer_type=TOKENIZER_TYPE):
    """Load model and configure it for LoRA training."""
    logger.info(f"Loading model from {MODEL_PATH}")

    # Set up the custom config
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

    # Define target modules for LoRA
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    # Add router to target modules if MOE is enabled
    if USE_MOE:
        target_modules.append("router.router")
        logger.info("Added MOE router to LoRA target modules")

    # Configure model for LoRA fine-tuning
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=target_modules,
        bias="none",
        inference_mode=False,
    )

    # Prepare model for training if needed
    model = prepare_model_for_kbit_training(model)

    # Apply LoRA to model
    logger.info(f"Applying LoRA to target modules: {target_modules}")
    model = get_peft_model(model, peft_config)

    # Print trainable parameters
    model.print_trainable_parameters()

    return model


def process_dataset(tokenizer, data_path=None):
    """Process the dataset for LoRA fine-tuning."""
    data_path = data_path or DATA_PATH
    logger.info(f"Loading dataset from {data_path}")

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

    # Split into train/validation
    dataset = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    logger.info(
        f"Split into {len(train_dataset)} training and {len(eval_dataset)} validation examples"
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
    tokenized_train = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["prompt", "response"],
        desc="Tokenizing training dataset",
    )

    tokenized_eval = eval_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["prompt", "response"],
        desc="Tokenizing validation dataset",
    )

    return {"train": tokenized_train, "validation": tokenized_eval}


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Fine-tune SmolLM with LoRA")
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
    parser.add_argument(
        "--lora_r",
        type=int,
        default=LORA_R,
        help=f"LoRA r parameter (default: {LORA_R})",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=LORA_ALPHA,
        help=f"LoRA alpha parameter (default: {LORA_ALPHA})",
    )
    args = parser.parse_args()

    # Update global variables with command line arguments
    global USE_MOE, NUM_EXPERTS, NUM_EXPERTS_PER_TOKEN, MODEL_PATH, LORA_R, LORA_ALPHA
    USE_MOE = args.use_moe
    NUM_EXPERTS = args.num_experts
    NUM_EXPERTS_PER_TOKEN = args.num_experts_per_token
    MODEL_PATH = args.model_path
    LORA_R = args.lora_r
    LORA_ALPHA = args.lora_alpha

    # Start logging
    logger.info("Starting LoRA fine-tuning...")

    # 1. Set the seed for reproducibility
    set_seed(42)

    # 2. Load tokenizer and model
    tokenizer = load_tokenizer(args.tokenizer_path, args.tokenizer_type)
    model = load_model_for_training(args.tokenizer_path, args.tokenizer_type)

    # 3. Move model to GPU
    if torch.cuda.is_available():
        model = model.to("cuda")
        logger.info("Moved model to CUDA")

    # 4. Process dataset
    tokenized_dataset = process_dataset(tokenizer, args.data_path)

    # 5. Create data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding="longest",
        return_tensors="pt",
    )

    # 6. Prepare training arguments
    training_args = TrainingArguments(
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
        gradient_checkpointing=False,  # Usually not needed for LoRA
        max_grad_norm=1.0,
        dataloader_num_workers=2,
    )

    # Calculate warmup steps from ratio
    total_steps = (
        len(tokenized_dataset["train"])
        // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
        * NUM_TRAIN_EPOCHS
    )
    warmup_steps = int(total_steps * WARMUP_RATIO)

    # 7. Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"]
        if "validation" in tokenized_dataset
        else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 8. Start training
    logger.info("Starting LoRA fine-tuning...")
    train_result = trainer.train()

    # 9. Save final model
    logger.info("Saving final LoRA adapters...")
    trainer.save_model(f"{OUTPUT_DIR}/final")

    # 10. Log and save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("LoRA training complete!")


if __name__ == "__main__":
    main()
