# SmolLM: A Lightweight Language Model with MoE

This repository contains the implementation of SmolLM, a lightweight language model from scratch.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Training Pipeline](#training-pipeline)
- [Data Preparation](#data-preparation)
- [Training Commands](#training-commands)
- [Notebooks](#notebooks)

## Overview

SmolLM is designed to be a lightweight yet powerful language model that leverages Mixture of Experts (MoE) architecture for efficient parameter usage. The model supports both standard transformer architecture and MoE variants, allowing for flexible deployment based on computational resources. The MoE implementation enables the model to achieve better performance with the same active parameter count by dynamically routing tokens to specialized experts.

Key Features:
- Flexible architecture supporting both standard and MoE variants
- Efficient parameter usage through expert specialization
- Dynamic token routing for optimized processing
- Support for both English and Chinese languages
- Complete training pipeline from tokenization to DPO

The training pipeline includes tokenizer preparation, pretraining, supervised fine-tuning (SFT), and direct preference optimization (DPO) stages.

## Installation

```bash
# Clone the repository
git clone [repository-url]
cd smollm

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
.
├── data/                      # Dataset directory
│   ├── pretrain/             # Pretraining data
│   ├── sft/                  # SFT data
│   └── dpo/                  # DPO data
├── notebooks/                 # Jupyter notebooks for experiments
│   ├── 00-intro.ipynb              # Introduction and setup
│   ├── 01-implement-MiniLM.ipynb   # MiniLM implementation details
│   ├── 02-build-tokenizer.ipynb    # Tokenizer training
│   └── 03-model-pretrain.ipynb     # Model pretraining experiments
├── accelerate_offload/       # Accelerate configuration for large models
├── prepare_datasets.py       # Dataset preparation script
├── train_tokenizer.py       # Tokenizer training and loading script
├── qwen_model.py            # Main model architecture
├── smol_model.py            # Lightweight model implementation with MoE support
├── train_pretrain.py        # Pretraining script
├── train_full_sft.py        # Full model SFT training
├── train_lora.py            # LoRA-based fine-tuning
└── train_dpo.py             # DPO training script
```

## Model Architecture

The model architecture is defined in `smol_model.py` with two main variants:

### Standard Variant
- Traditional transformer architecture
- Fixed parameter count
- Suitable for smaller-scale deployments
- SmolLM-Base: 7B parameters

### MoE Variant (Mixture of Experts)
- Dynamic routing architecture
- Sparse activation of parameters
- Better parameter efficiency
- Configuration:
  - Number of Experts: 8 (configurable)
  - Experts Per Token: 2 (configurable)
  - Capacity Factor: 1.25
  - Router Implementation: Top-k routing with auxiliary loss
  - SmolLM-MoE: 7B active parameters (30B total)

### Common Features
- Tokenizer:
  - Default: Qwen's tokenizer (151,936 tokens)
  - Support for both English and Chinese
  - Option to train custom tokenizer
- Attention Mechanism:
  - Group Query Attention (GQA)
  - Flash Attention support for faster training
- Activation: SwiGLU
- Layer Normalization: RMSNorm

### MOE-Specific Features
- Expert Design:
  - Each expert is a full MLP layer
  - Independent parameters for better specialization
  - SwiGLU activation in each expert
- Router Design:
  - Token-wise routing with learned gates
  - Load balancing through auxiliary loss
  - Configurable capacity factor for training stability
- Training Optimizations:
  - Expert parallelism for efficient training
  - Gradient checkpointing support
  - Auxiliary loss for better expert utilization

## Training Pipeline

The training process consists of four stages:

1. **Tokenizer Preparation** (`train_tokenizer.py`):
   - Option 1: Use Qwen's pretrained tokenizer (recommended)
     - Well-tested for both English and Chinese
     - Vocabulary size: 151,936 tokens
   - Option 2: Train custom tokenizer
     - BPE-based tokenization
     - Configurable vocabulary size
     - Support for multiple languages

2. **Pretraining** (`train_pretrain.py`):
   - Trains on a large corpus of text data
   - Uses efficient MoE routing for parameter updates
   - Supports both English and Chinese text
   - Distributed training support:
     - Multi-GPU data parallel training
     - Model parallel training with DeepSpeed
     - Gradient checkpointing for memory efficiency
     - Mixed precision training (FP16/BF16)

3. **Supervised Fine-Tuning** (`train_full_sft.py`, `train_lora.py`):
   - Fine-tunes on instruction-following data
   - Supports both full model and LoRA fine-tuning
   - Includes function calling capabilities
   - Distributed training support same as pretraining

4. **Direct Preference Optimization** (`train_dpo.py`):
   - Aligns model outputs with human preferences
   - Uses preference pairs for optimization
   - Improves response quality and safety
   - Distributed training support same as pretraining

## Distributed Training

The model supports various distributed training configurations through Hugging Face's Accelerate and DeepSpeed:

### Available Parallelism Strategies
1. **Data Parallel Training (DDP)**
   - Splits batches across GPUs
   - Each GPU has a full model copy
   - Automatic gradient synchronization
   - Best for smaller models or high-memory GPUs

2. **Model Parallel Training**
   - DeepSpeed ZeRO stages 1,2,3 support
   - Splits model weights across GPUs
   - Optimizes memory usage
   - Required for large models

3. **Expert Parallel Training (for MoE)**
   - Distributes experts across GPUs
   - Efficient expert routing
   - Load balancing across devices
   - Automatic expert placement

### Memory Optimizations
- Gradient checkpointing
- Mixed precision training (FP16/BF16)
- CPU offloading (with DeepSpeed)
- Activation checkpointing
- Efficient attention implementations (Flash Attention)

### Configuration
1. **Accelerate Config Setup**:
```bash
# Run once before training
accelerate config

# Or use default config
accelerate launch --multi_gpu --mixed_precision=fp16 train_script.py
```

2. **DeepSpeed Config**:
```bash
# Example config provided in accelerate_offload/ds_config.json
# Supports ZeRO stages, offloading, and expert parallelism
```

## Training Commands

1. **Tokenizer Preparation**:
```bash
# Tokenizer preparation is not distributed
python train_tokenizer.py \
  --mode load \
  --qwen_model Qwen/Qwen1.5-7B \
  --save_dir tokenizer
```

2. **Pretraining**:
```bash
# Multi-GPU training with Accelerate
accelerate launch --multi_gpu --mixed_precision=fp16 train_pretrain.py \
  --data_path data/pretrain/pretrain_data.jsonl \
  --tokenizer_path tokenizer \
  --tokenizer_type qwen \
  --model_size base \
  --num_epochs 3

# DeepSpeed training (ZeRO-3 + Offload)
deepspeed --num_gpus 8 train_pretrain.py \
  --data_path data/pretrain/pretrain_data.jsonl \
  --tokenizer_path tokenizer \
  --tokenizer_type qwen \
  --model_size base \
  --deepspeed accelerate_offload/ds_config.json
```

3. **Supervised Fine-Tuning**:
```bash
# Multi-GPU training for full model
accelerate launch --multi_gpu train_full_sft.py \
  --data_path data/sft/sft_data.jsonl \
  --base_model path/to/pretrained \
  --tokenizer_path tokenizer \
  --tokenizer_type qwen \
  --output_dir path/to/output

# Multi-GPU training for LoRA (more memory efficient)
accelerate launch --multi_gpu train_lora.py \
  --data_path data/sft/sft_data.jsonl \
  --base_model path/to/pretrained \
  --tokenizer_path tokenizer \
  --tokenizer_type qwen \
  --lora_r 8
```

4. **DPO Training**:
```bash
# Multi-GPU training
accelerate launch --multi_gpu train_dpo.py \
  --data_path data/dpo/dpo_data.jsonl \
  --model_path path/to/sft_model \
  --tokenizer_path tokenizer \
  --tokenizer_type qwen \
  --output_dir path/to/output

# DeepSpeed training
deepspeed --num_gpus 8 train_dpo.py \
  --data_path data/dpo/dpo_data.jsonl \
  --model_path path/to/sft_model \
  --deepspeed accelerate_offload/ds_config.json
```

### Environment Variables for Distributed Training
```bash
# Optional: Set these before training for better performance
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Specify GPUs to use
export NCCL_P2P_DISABLE=1  # If having GPU connection issues
export NCCL_IB_DISABLE=1   # If having InfiniBand issues
```

## Model Serving

SmolLM provides two serving options with OpenAI-compatible API endpoints:

### Option 1: vLLM Serving (Recommended for Production)

The `serve_vllm.py` script provides high-performance serving using vLLM backend with features like:
- Continuous batching for higher throughput
- PagedAttention for better memory efficiency
- Multi-GPU support via tensor parallelism
- INT8/INT4 quantization options

1. **Installation**:
```bash
pip install vllm fastapi uvicorn
```

2. **Start the Server**:
```bash
# Basic usage
python serve_vllm.py

# With custom configuration
python serve_vllm.py \
  --model_path path/to/your/model \
  --tensor_parallel_size 4 \  # Use 4 GPUs
  --gpu_memory_utilization 0.9 \
  --quantization int8  # Optional: Enable INT8 quantization
```

3. **Configuration Options**:
- `tensor_parallel_size`: Number of GPUs for tensor parallelism
- `gpu_memory_utilization`: GPU memory usage (0.0 to 1.0)
- `max_num_batched_tokens`: Maximum batch size in tokens
- `max_num_seqs`: Maximum concurrent sequences
- `quantization`: Optional quantization (int8/int4)

### Option 2: Standard Serving

The standard serving option uses PyTorch/Hugging Face for inference. It's simpler but has lower throughput:
- Easier to understand and modify
- Better for debugging
- Lower performance than vLLM
- Higher memory usage per request

### API Endpoints

Both serving options provide the same OpenAI-compatible endpoints:

1. **Chat Completions**:
```bash
POST /v1/chat/completions
{
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is photosynthesis?"}
    ],
    "temperature": 0.7,
    "max_tokens": 200
}
```

2. **Text Completions**:
```bash
POST /v1/completions
{
    "prompt": "The process of photosynthesis involves",
    "temperature": 0.7,
    "max_tokens": 200
}
```

3. **Health Check**:
```bash
GET /health
```

### Example API Usage

```python
import requests

# Chat completion
response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is photosynthesis?"}
        ],
        "temperature": 0.7
    }
)

# Text completion
response = requests.post(
    "http://localhost:8000/v1/completions",
    json={
        "prompt": "The process of photosynthesis involves",
        "temperature": 0.7
    }
)
```

### API Documentation
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Data Preparation

Use the `prepare_datasets.py` script to prepare training data:

```bash
# Prepare all datasets
python prepare_datasets.py

# Prepare specific datasets
python prepare_datasets.py --pretrain  # Only pretraining data
python prepare_datasets.py --sft       # Only SFT data
python prepare_datasets.py --dpo       # Only DPO data

# Use small datasets for testing
python prepare_datasets.py --small
```

### Data Sources and Formats

1. **Pretraining Data**:
   - Source: BAAI/IndustryCorpus2 (5B tokens)
   ```json
   {
     "text": "Sample text content...",
     "category": "news_media"
   }
   ```

2. **SFT Data**:
   - Sources: HuggingFaceTB/smol-smoltalk, HuggingFaceTB/smoltalk, O1-OPEN/OpenO1-SFT
   ```json
   {
     "id": "example-id",
     "source": "dataset-source",
     "messages": [
       {"role": "user", "content": "..."},
       {"role": "assistant", "content": "..."}
     ]
   }
   ```

3. **DPO Data**:
   - Source: openbmb/UltraFeedback
   ```json
   {
     "id": "example-id",
     "source": "ultrafeedback",
     "prompt": "Input prompt",
     "chosen": "Preferred response",
     "rejected": "Less preferred response"
   }
   ```

## Notebooks

The repository includes Jupyter notebooks for experimentation and understanding:

- `00-intro.ipynb`: Introduction to the project and setup
- `01-implement-MiniLM.ipynb`: Detailed implementation of the MiniLM architecture
- `02-build-tokenizer.ipynb`: Training and using the tokenizer
- `03-model-pretrain.ipynb`: Experiments with model pretraining

## License

[Add your license information here] 