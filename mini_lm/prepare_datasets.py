#!/usr/bin/env python
# Script to prepare training datasets for pretraining, SFT, and DPO

import json
import os
import random
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import argparse

# Create output directories
os.makedirs("data/pretrain", exist_ok=True)
os.makedirs("data/sft", exist_ok=True)
os.makedirs("data/dpo", exist_ok=True)

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Config
MAX_SAMPLES_PER_DATASET = 500_000  # Max samples to take from each dataset
SAMPLE_SIZE = 1000  # Size of small sample datasets for testing
PRETRAIN_SAMPLE_SIZE = 5000  # Pretraining samples are typically larger

# Add these constants for token-based sampling
TARGET_TOKENS = 5_000_000_000  # Total target tokens for pretraining (5B tokens as in original prepare_data.py)
TARGET_TOKENS_PER_FILE = (
    2_500_000_000  # For the actual dataset files (2.5B tokens per language)
)

# Optional flag to enable smaller dataset for testing
USE_SMALL_DATASET_FOR_TESTING = (
    False  # Set to True to use smaller dataset sizes for testing the pipeline
)

# Define category mappings similar to prepare_data.py
category_map = {
    "交通运输": "transportation",
    "医学_健康_心理_中医": "medicine_health_psychology_traditional_chinese_medicine",
    "数学_统计学": "mathematics_statistics",
    "时政_政务_行政": "current_affairs_government_administration",
    "消防安全_食品安全": "fire_safety_food_safety",
    "石油化工": "petrochemical",
    "计算机_通信": "computer_communication",
    "人工智能_机器学习": "artificial_intelligence_machine_learning",
    "其他信息服务_信息安全": "other_information_services_information_security",
    "学科教育_教育": "subject_education_education",
    "文学_情感": "literature_emotion",
    "水利_海洋": "water_resources_ocean",
    "游戏": "game",
    "科技_科学研究": "technology_scientific_research",
    "采矿": "mining",
    "住宿_餐饮_酒店": "accommodation_catering_hotel",
    "其他制造": "other_manufacturing",
    "影视_娱乐": "film_entertainment",
    "新闻传媒": "news_media",
    "汽车": "automobile",
    "生物医药": "biomedicine",
    "航空航天": "aerospace",
    "金融_经济": "finance_economics",
    "体育": "sports",
    "农林牧渔": "agriculture_forestry_animal_husbandry_fishery",
    "房地产_建筑": "real_estate_construction",
    "旅游_地理": "tourism_geography",
    "法律_司法": "law_judiciary",
    "电力能源": "electric_power_energy",
    "计算机编程_代码": "computer_programming_code",
}


def estimate_tokens(text):
    """Simple token estimator"""
    if not text:
        return 0
    return len(text.split())


def save_jsonl(data, filename):
    """Save data to JSONL format"""
    with open(filename, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(data)} samples to {filename}")


def create_sample_dataset(data, filename, sample_size):
    """Create and save a small sample dataset for testing"""
    if len(data) > sample_size:
        sampled_data = random.sample(data, sample_size)
    else:
        sampled_data = data
    save_jsonl(sampled_data, filename)


def prepare_pretrain_data():
    """Prepare pretraining data using token-based sampling with category weighting"""
    print("\n=== Preparing Pretraining Data ===")

    # Adjust token targets if using small dataset for testing
    actual_target_tokens = TARGET_TOKENS
    if USE_SMALL_DATASET_FOR_TESTING:
        actual_target_tokens = 5000  # Use 5000 tokens for testing
        print("Using reduced dataset size (5000 tokens) for testing")
    else:
        print(
            f"Using full dataset size ({actual_target_tokens/1_000_000_000:.1f}B tokens)"
        )

    pretrain_data = []

    # First check if we already have processed pretraining data from prepare_data.py
    for lang in ["chinese", "english"]:
        filename = f"sampled_{lang}.jsonl"
        if os.path.exists(filename):
            print(f"Found existing pretraining data from {filename}")
            print("Will use these files for pretraining data preparation")

            with open(filename, "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f]
                pretrain_data.extend(data)

            # Create full dataset and sample
            full_filename = "data/pretrain/pretrain_data.jsonl"
            save_jsonl(pretrain_data, full_filename)

            # Create and save sample
            sample_filename = "data/pretrain/pretrain_sample.jsonl"
            create_sample_dataset(pretrain_data, sample_filename, PRETRAIN_SAMPLE_SIZE)

            return

    # If we don't have existing files, we'll sample directly
    print(
        "No existing pretraining data found. Sampling directly from IndustryCorpus2..."
    )

    # Define language list
    langs = ["chinese", "english"]

    # Define core categories (the main ones used by smolLM)
    core_categories = {
        "news_media",
        "literature_emotion",
        "subject_education_education",
        "computer_programming_code",
        "mathematics_statistics",
    }

    # All category folder names (using the English names)
    all_categories = set(category_map.values())

    # Additional categories: those not in core
    additional_categories = all_categories - core_categories

    # Allocate token targets:
    # 80% of the tokens go to core categories
    # 20% are split equally among additional categories
    core_allocation = 0.80
    additional_allocation = 0.20

    # Core category ratios
    core_ratios = {
        "news_media": 0.25,
        "literature_emotion": 0.20,
        "subject_education_education": 0.20,
        "computer_programming_code": 0.20,
        "mathematics_statistics": 0.15,
    }

    # Compute per-category token targets for each language
    target_tokens_per_category = {lang: {} for lang in langs}

    for lang in langs:
        # Core categories: distribute 80% of tokens as per the defined ratios
        for cat in core_categories:
            target_tokens_per_category[lang][cat] = int(
                actual_target_tokens * core_allocation * core_ratios[cat]
            )

        # Additional categories: split 20% equally
        additional_count = len(additional_categories)
        additional_target_each = int(
            actual_target_tokens * additional_allocation / additional_count
        )

        for cat in additional_categories:
            target_tokens_per_category[lang][cat] = additional_target_each

    # Prepare accumulators for samples and token counts per language and category
    samples = {lang: [] for lang in langs}
    token_counts = {lang: {cat: 0 for cat in all_categories} for lang in langs}

    # Function to stream data from a category subfolder
    def stream_subfolder(category, language):
        """Stream data from IndustryCorpus2 for a specific category and language"""
        try:
            pattern = f"https://huggingface.co/datasets/BAAI/IndustryCorpus2/resolve/main/{category}/{language}/high/*.parquet"
            ds = load_dataset(
                "parquet", data_files={"train": pattern}, streaming=True, split="train"
            )
            return ds
        except Exception as e:
            print(f"Error loading data for {category}/{language}: {e}")
            return None

    # Sample data from each category and language
    for lang in langs:
        for cat in all_categories:
            target = target_tokens_per_category[lang][cat]
            print(
                f"Sampling for category '{cat}' and language '{lang}' (target tokens: {target})..."
            )

            ds = stream_subfolder(cat, lang)
            if ds is None:
                continue

            try:
                for sample in ds:
                    # Assume each record has a "text" field
                    text = sample.get("text", "")
                    if not text:
                        continue

                    tokens = estimate_tokens(text)
                    if token_counts[lang][cat] < target:
                        samples[lang].append({"text": text, "category": cat})
                        token_counts[lang][cat] += tokens

                    if token_counts[lang][cat] >= target:
                        print(
                            f"Reached token target for category '{cat}' in language '{lang}'"
                        )
                        break
            except Exception as e:
                print(f"Error processing samples for {cat}/{lang}: {e}")

    # Combine samples from all languages
    for lang in langs:
        pretrain_data.extend(samples[lang])

    # Save the full dataset
    full_filename = "data/pretrain/pretrain_data.jsonl"
    save_jsonl(pretrain_data, full_filename)

    # Create and save each language to separate files
    for lang in langs:
        lang_filename = f"data/pretrain/pretrain_{lang}_data.jsonl"
        save_jsonl(samples[lang], lang_filename)

    # Create and save sample for testing
    sample_filename = "data/pretrain/pretrain_sample.jsonl"
    create_sample_dataset(pretrain_data, sample_filename, PRETRAIN_SAMPLE_SIZE)

    print(f"Pretraining data preparation complete. Total samples: {len(pretrain_data)}")

    # Also save the original files for compatibility
    for lang in langs:
        orig_filename = f"sampled_{lang}.jsonl"
        if not os.path.exists(orig_filename):
            save_jsonl(samples[lang], orig_filename)
            print(f"Saved original format file: {orig_filename}")


def prepare_sft_data():
    """Prepare SFT data by combining multiple datasets"""
    print("\n=== Preparing SFT Data ===")

    sft_data = []

    # 1. Load smol-smoltalk dataset
    print("Loading HuggingFaceTB/smol-smoltalk dataset...")
    try:
        smol_dataset = load_dataset("HuggingFaceTB/smol-smoltalk", split="train")
        print(f"Loaded {len(smol_dataset)} samples from smol-smoltalk")

        # Sample and convert to our format
        sample_count = min(len(smol_dataset), MAX_SAMPLES_PER_DATASET)
        for i in tqdm(range(sample_count)):
            example = smol_dataset[i]
            messages = example.get("messages", [])
            if messages:
                sft_data.append(
                    {
                        "id": f"smol-smoltalk-{i}",
                        "source": "smol-smoltalk",
                        "messages": messages,
                    }
                )
    except Exception as e:
        print(f"Error loading smol-smoltalk: {e}")

    # 2. Load SmolTalk for function calling examples
    print("Loading HuggingFaceTB/smoltalk dataset for function calling examples...")
    try:
        smoltalk_dataset = load_dataset("HuggingFaceTB/smoltalk", split="train")
        print(f"Loaded {len(smoltalk_dataset)} samples from smoltalk")

        # Filter for function calling examples (looking for APIGen in "source" field)
        function_call_samples = []
        for i in tqdm(range(len(smoltalk_dataset))):
            example = smoltalk_dataset[i]
            # Check if this is a function calling example
            source = example.get("source", "")
            if "apigen" in source.lower():
                messages = example.get("messages", [])
                if messages:
                    function_call_samples.append(
                        {
                            "id": f"smoltalk-function-{i}",
                            "source": "smoltalk-function",
                            "messages": messages,
                        }
                    )

                # Limit the number of function call samples
                if (
                    len(function_call_samples) >= MAX_SAMPLES_PER_DATASET // 5
                ):  # Take fewer function call examples
                    break

        print(f"Found {len(function_call_samples)} function calling examples")
        sft_data.extend(function_call_samples)
    except Exception as e:
        print(f"Error loading smoltalk for function calls: {e}")

    # 3. Load OpenO1-SFT for Chinese and English samples
    print("Loading O1-OPEN/OpenO1-SFT dataset for Chinese and English examples...")
    try:
        open_o1_dataset = load_dataset("O1-OPEN/OpenO1-SFT", split="train")
        print(f"Loaded {len(open_o1_dataset)} samples from OpenO1-SFT")

        # Filter for Chinese examples and English examples and convert to our format
        chinese_samples = []
        english_samples = []

        for i in tqdm(range(len(open_o1_dataset))):
            example = open_o1_dataset[i]
            # Check if this contains Chinese characters
            content = example.get("conversations", [{}])[0].get("value", "")
            # Simple check for Chinese characters: if any character's unicode is in the Chinese range
            has_chinese = any("\u4e00" <= char <= "\u9fff" for char in content)

            # Convert conversation format to messages format
            messages = []
            for turn in example.get("conversations", []):
                role = "user" if turn.get("role") == "human" else "assistant"
                content = turn.get("value", "")
                messages.append({"role": role, "content": content})

            if not messages:
                continue

            if has_chinese:
                # Add to Chinese samples
                chinese_samples.append(
                    {
                        "id": f"open-o1-chinese-{i}",
                        "source": "open-o1-chinese",
                        "messages": messages,
                    }
                )

                # Limit the number of Chinese samples
                if len(chinese_samples) >= MAX_SAMPLES_PER_DATASET // 5:
                    break
            else:
                # Add to English samples
                english_samples.append(
                    {
                        "id": f"open-o1-english-{i}",
                        "source": "open-o1-english",
                        "messages": messages,
                    }
                )

                # Limit the number of English samples
                if len(english_samples) >= MAX_SAMPLES_PER_DATASET // 5:
                    break

        print(
            f"Found {len(chinese_samples)} Chinese examples and {len(english_samples)} English examples from OpenO1-SFT"
        )
        sft_data.extend(chinese_samples)
        sft_data.extend(english_samples)
    except Exception as e:
        print(f"Error loading OpenO1-SFT: {e}")

    # Save full SFT dataset
    full_filename = "data/sft/sft_data.jsonl"
    save_jsonl(sft_data, full_filename)

    # Create and save sample
    sample_filename = "data/sft/sft_sample.jsonl"
    create_sample_dataset(sft_data, sample_filename, SAMPLE_SIZE)


def prepare_dpo_data():
    """Prepare DPO data from UltraFeedback"""
    print("\n=== Preparing DPO Data ===")

    dpo_data = []

    # Load UltraFeedback dataset
    print("Loading openbmb/UltraFeedback dataset...")
    try:
        ultrafeedback_dataset = load_dataset("openbmb/UltraFeedback", split="train")
        print(f"Loaded {len(ultrafeedback_dataset)} samples from UltraFeedback")

        # Sample and convert to our format
        sample_count = min(len(ultrafeedback_dataset), MAX_SAMPLES_PER_DATASET)
        for i in tqdm(range(sample_count)):
            example = ultrafeedback_dataset[i]

            prompt = example.get("prompt", "")
            chosen = example.get("chosen", "")
            rejected = example.get("rejected", "")

            if prompt and chosen and rejected:
                dpo_data.append(
                    {
                        "id": f"ultrafeedback-{i}",
                        "source": "ultrafeedback",
                        "prompt": prompt,
                        "chosen": chosen,
                        "rejected": rejected,
                    }
                )
    except Exception as e:
        print(f"Error loading UltraFeedback: {e}")

    # Save full DPO dataset
    full_filename = "data/dpo/dpo_data.jsonl"
    save_jsonl(dpo_data, full_filename)

    # Create and save sample
    sample_filename = "data/dpo/dpo_sample.jsonl"
    create_sample_dataset(dpo_data, sample_filename, SAMPLE_SIZE)


def main():
    """Main function to prepare all datasets"""
    print("Starting dataset preparation...")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Prepare datasets for SmolLM training")
    parser.add_argument(
        "--small", action="store_true", help="Use small dataset for testing"
    )
    parser.add_argument(
        "--pretrain", action="store_true", help="Prepare pretraining data"
    )
    parser.add_argument(
        "--sft", action="store_true", help="Prepare SFT data"
    )
    parser.add_argument(
        "--dpo", action="store_true", help="Prepare DPO data"
    )
    args = parser.parse_args()

    # Update global flag if small dataset requested
    global USE_SMALL_DATASET_FOR_TESTING, MAX_SAMPLES_PER_DATASET, SAMPLE_SIZE
    if args.small:
        USE_SMALL_DATASET_FOR_TESTING = True
        # Also reduce SFT and DPO dataset sizes when --small is used
        MAX_SAMPLES_PER_DATASET = 1000  # Reduce from 500k to 1k
        SAMPLE_SIZE = 100  # Reduce from 1k to 100
        print("Using small dataset sizes for testing (--small flag detected)")

    # If no specific dataset is selected, prepare all
    prepare_all = not (args.pretrain or args.sft or args.dpo)

    if prepare_all or args.pretrain:
        prepare_pretrain_data()
    if prepare_all or args.sft:
        prepare_sft_data()
    if prepare_all or args.dpo:
        prepare_dpo_data()

    print("\nDataset preparation complete!")
    if prepare_all or args.pretrain:
        print("- Pretraining data: data/pretrain/")
    if prepare_all or args.sft:
        print("- SFT data: data/sft/")
    if prepare_all or args.dpo:
        print("- DPO data: data/dpo/")
    print("Sample datasets for testing are available in each directory.")


if __name__ == "__main__":
    main()
