# 🚀 LLM & AI Code Recipes 🚀

This repository serves as a collection of practical tutorials, code examples, and reference implementations for various concepts in Large Language Models (LLMs) and AI. The projects within are designed to be clear, well-documented, and reusable for educational and practical purposes.

Each folder contains a standalone project or tutorial.

---

## 📂 Projects & Tutorials

Here are the projects currently available. This collection is actively maintained and expanded.

### [minimum_language_model](./minimum_language_model/)

A from-scratch implementation of a Transformer-based language model, covering every component from the ground up. This project provides a deep dive into the fundamental building blocks of modern LLMs.

**Core Concepts Covered:**
*   **BPE Tokenizer**: Building a tokenizer with parallel processing and memory-efficient streaming.
*   **Transformer Architecture**: Implementing core components like RoPE, Multi-Head Attention, and SwiGLU.
*   **Training & Optimization**: A from-scratch AdamW optimizer, learning rate scheduling, and a full training loop with checkpointing.
*   **Text Generation**: Advanced sampling methods including temperature scaling and nucleus (top-p) sampling.

### [triton_and_distributed_training](./triton_and_distributed_training/)

A deep dive into optimizing and scaling Transformer models using advanced systems techniques. This project covers both low-level kernel optimization with Triton and high-level distributed training strategies.

**Core Concepts Covered:**
*   **Performance Engineering**: In-depth profiling, benchmarking mixed-precision (BF16) training, and analyzing attention performance.
*   **Custom Kernels with Triton**: A from-scratch implementation of a high-performance `weighted_sum` CUDA kernel.
*   **Distributed Data Parallel (DDP)**: A step-by-step implementation of four DDP strategies, from a naive baseline to an optimized, bucketed approach that overlaps communication and computation.

### [distributed_training_accelerate](./distributed_training_accelerate/)

A comprehensive guide to distributed model training in PyTorch. This project provides a series of hands-on examples that progress from basic single-GPU training to advanced, large-scale distributed setups using industry-standard tools.

**Core Concepts Covered:**
*   **Data Parallelism (DP)** vs. **Distributed Data Parallelism (DDP)**
*   Simplifying distributed training with **Hugging Face `accelerate`**
*   Optimizing large model training with **DeepSpeed**, including ZeRO stages.

The project includes a sequence of Jupyter notebooks and Python scripts to illustrate these concepts clearly.

### [mini_lm](./mini_lm/)

A project focused on building and training a "smolLM" (a small-scale Language Model) from scratch. This project will cover the entire lifecycle of creating an LLM, providing a deep dive into the underlying mechanics.

**Key Learning Objectives:**
*   Implementing the core components of a Transformer-based language model.
*   Building a data loading and preprocessing pipeline.
*   Training the model from the ground up.
*   Implementing techniques for efficient training and inference.


---

Feel free to explore the projects and adapt the code for your own use cases.
