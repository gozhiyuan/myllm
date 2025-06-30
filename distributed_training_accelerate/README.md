# Distributed Model Training in PyTorch with Accelerate

This project demonstrates various techniques for distributed model training in PyTorch, with a focus on using the Hugging Face Accelerate library. The examples progress from basic data parallelism to advanced distributed training setups using DeepSpeed.

## Core Concepts Demonstrated

*   **Data Parallelism (DP):** The simplest way to parallelize training across multiple GPUs on a single machine.
*   **Distributed Data Parallelism (DDP):** A more advanced and efficient method for multi-GPU and multi-machine training.
*   **Hugging Face Accelerate:** A library that simplifies running PyTorch training scripts on any kind of distributed setup, whether it's multiple GPUs on one machine, multiple nodes, or TPUs.
*   **DeepSpeed:** A deep learning optimization library that makes distributed training easy, efficient, and effective, integrated here with Accelerate.

## Project Structure & Files

### Jupyter Notebooks

The core of this project is a series of Jupyter notebooks that incrementally introduce more advanced concepts:

-   `1-trainer.ipynb`: Starts with a baseline single-GPU training loop.
-   `2-data-parallel.ipynb`: Introduces `torch.nn.DataParallel` for simple multi-GPU training on a single machine.
-   `3-distributed-data-parallel.ipynb`: Moves to `torch.nn.parallel.DistributedDataParallel` for more efficient single-machine, multi-GPU training.
-   `4-distributed-data-parallel-accelerate.ipynb`: Shows how to use the `accelerate` library to simplify the DDP setup.
-   `5-accelerate-advanced.ipynb`: Explores more advanced features of `accelerate`.
-   `6-accelerate-deepspeed.ipynb`: Integrates DeepSpeed with `accelerate` for powerful and efficient large-scale model training.

### Python Scripts

-   `ddp.py`, `ddp_trainer.py`, `ddp_accelerator.py`, `ddp_accelerator2.py`: Various Python scripts demonstrating DDP and `accelerate` in standalone script formats.
-   `metric_accuracy.py`, `metric_f1.py`: Example metric calculation scripts.
-   `pre_download_files.py`: A script to download necessary model weights or datasets beforehand.

### Configuration Files

-   `deepspeed_config.yaml`, `deepspeed_config2.yaml`, `deepspeed_config3.yaml`, `deepspeed_config4.yaml`: Configuration files for different DeepSpeed setups.
-   `zero_stage2_config.json`, `zero_stage3_config.json`: DeepSpeed ZeRO optimizer configuration files for Stage 2 and Stage 3.

## Dependencies

This project relies on several key libraries. You can install them using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

The main dependencies are:
- `torch`
- `transformers`
- `accelerate`
- `deepspeed`
- `scikit-learn`
- `evaluate`

## How to Use

1.  **Clone the repository.**
2.  **Install the dependencies:** `pip install -r requirements.txt`
3.  **Run the `pre_download_files.py` script** to ensure you have all the necessary models and data.
4.  **Follow the Jupyter notebooks** in numerical order (`1` to `6`) to understand the progression from basic to advanced distributed training.
5.  **Experiment with the Python scripts** and configuration files to run different distributed training jobs from the command line.

