#!/usr/bin/env python
"""
Tokenizer training and loading script.
Supports either:
1. Loading Qwen's pretrained tokenizer (recommended)
2. Training a new tokenizer with similar settings
"""

import os
import json
from typing import List, Optional
from transformers import PreTrainedTokenizer, AutoTokenizer
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from datasets import load_dataset
import argparse
import logging
from pathlib import Path
import torch

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Constants
QWEN_TOKENIZER = "Qwen/Qwen1.5-7B"  # or "Qwen/Qwen2-7B"
VOCAB_SIZE = 151936  # Qwen's vocabulary size
TARGET_TOKENIZER_DIR = "tokenizer"


def load_qwen_tokenizer(model_name: str = QWEN_TOKENIZER) -> PreTrainedTokenizer:
    """
    Load Qwen's pretrained tokenizer.
    This is the recommended approach as it's well-tested and supports both English and Chinese.
    """
    logger.info(f"Loading Qwen tokenizer from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Ensure necessary special tokens are set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Loaded tokenizer with vocabulary size: {len(tokenizer)}")
    return tokenizer


def train_new_tokenizer(
    train_data_paths: List[str],
    vocab_size: int = VOCAB_SIZE,
    save_dir: str = TARGET_TOKENIZER_DIR,
) -> Tokenizer:
    """
    Train a new tokenizer with settings similar to Qwen's.
    Only use this if you need a custom vocabulary or can't use Qwen's tokenizer.
    
    Args:
        train_data_paths: List of paths to training data files (txt or jsonl)
        vocab_size: Target vocabulary size
        save_dir: Directory to save the trained tokenizer
    """
    logger.info("Initializing new tokenizer training...")
    
    # Initialize a BPE tokenizer (similar to Qwen)
    tokenizer = Tokenizer(models.BPE())
    
    # Add normalizers
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.Strip(),
        normalizers.NFKC(),
        normalizers.Replace(pattern=" ", content=" "),  # Normalize different spaces
    ])
    
    # Add pre-tokenizers
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.ByteLevel(add_prefix_space=False),
        pre_tokenizers.Digits(individual_digits=True),
    ])
    
    # Configure the trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>", "<|pad|>", "<|sep|>", "<|bos|>", "<|eos|>"],
        show_progress=True,
    )
    
    # Prepare training data
    def data_iterator():
        for path in train_data_paths:
            if path.endswith('.jsonl'):
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        data = json.loads(line)
                        if isinstance(data, dict):
                            text = data.get('text', '')
                            if text:
                                yield text
            else:  # Assume text file
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        yield line.strip()
    
    # Train the tokenizer
    logger.info("Starting tokenizer training...")
    tokenizer.train_from_iterator(data_iterator(), trainer=trainer)
    
    # Set decoder
    tokenizer.decoder = decoders.ByteLevel()
    
    # Save the tokenizer
    os.makedirs(save_dir, exist_ok=True)
    tokenizer_path = os.path.join(save_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    logger.info(f"Saved trained tokenizer to {tokenizer_path}")
    
    # Convert to HuggingFace tokenizer
    from transformers import PreTrainedTokenizerFast
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|bos|>",
        eos_token="<|endoftext|>",
        pad_token="<|pad|>",
        sep_token="<|sep|>",
        unk_token="<|endoftext|>",  # Use endoftext as unk token like GPT
    )
    
    # Save HuggingFace tokenizer
    hf_tokenizer.save_pretrained(save_dir)
    logger.info(f"Saved HuggingFace tokenizer to {save_dir}")
    
    return hf_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Tokenizer preparation script")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["load", "train"],
        default="load",
        help="Whether to load Qwen tokenizer or train new one",
    )
    parser.add_argument(
        "--train_data",
        type=str,
        nargs="+",
        help="Paths to training data files (for training mode)",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=VOCAB_SIZE,
        help="Vocabulary size for new tokenizer",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=TARGET_TOKENIZER_DIR,
        help="Directory to save the tokenizer",
    )
    parser.add_argument(
        "--qwen_model",
        type=str,
        default=QWEN_TOKENIZER,
        help="Qwen model name to load tokenizer from",
    )
    
    args = parser.parse_args()
    
    if args.mode == "load":
        tokenizer = load_qwen_tokenizer(args.qwen_model)
        # Save the loaded tokenizer
        os.makedirs(args.save_dir, exist_ok=True)
        tokenizer.save_pretrained(args.save_dir)
        logger.info(f"Saved Qwen tokenizer to {args.save_dir}")
    else:
        if not args.train_data:
            raise ValueError("Must provide training data paths for training mode")
        tokenizer = train_new_tokenizer(
            args.train_data,
            args.vocab_size,
            args.save_dir,
        )
    
    # Test the tokenizer
    test_texts = [
        "Hello world! How are you?",
        "你好，世界！最近好吗？",
        "Mixed English and 中文 text.",
    ]
    
    logger.info("\nTesting tokenizer with sample texts:")
    for text in test_texts:
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        logger.info(f"\nOriginal : {text}")
        logger.info(f"Tokenized: {tokens}")
        logger.info(f"Decoded  : {decoded}")


if __name__ == "__main__":
    main() 