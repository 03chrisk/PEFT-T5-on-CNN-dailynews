# T5-base Fine-tuning for Summarization with Instructional Prompts

## Introduction

This project involves fine-tuning the T5-base model on the CNN/Daily Mail dataset using LoRA (Low-Rank Adaptation) and traditional full fine-tuning. Additionally, instructional prompts like "Summarize this article:" are incorporated to enhance the model's summarization performance. The fine-tuned models are compared to the original T5-base model's summarization capabilities without fine-tuning. The aim is to observe the improvements in performance brought by fine-tuning and prompt engineering techniques.

## Table of Contents

- [Installation](#installation)
- [System Requirements](#system-requirements)
- [Dataset](#dataset)
- [Model](#model)
- [Project Overview](#project-overview)
  - [Prompt Engineering](#prompt-engineering)
  - [PEFT (Parameter-Efficient Fine-Tuning) with LoRA](#peft-parameter-efficient-fine-tuning-with-lora)
  - [Full Fine-Tuning](#full-fine-tuning)
- [Results](#results)
- [Known Issues & Limitations](#known-issues--limitations)
- [Future Work](#future-work)
- [Additional Resources](#additional-resources)
- [Contributors](#contributors)

## Installation

To set up the project environment, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo-name.git
   cd your-repo-name

2. **Set Up an Environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    
3. **Intall Requirements**
   ```bash
   pip install -r requirements.txt

## System Requirements

To run the fine-tuning and evaluation scripts efficiently, the following system requirements are recommended:

- **Python**: 3.8+
- **GPU**: NVIDIA A100 GPU with at least 40 GB of memory (recommended for faster training)
- **CUDA**: Compatible version for your GPU and PyTorch setup
- **RAM**: 16 GB or higher
- **Disk Space**: At least 50 GB free for storing datasets and model checkpoints

### Training Hardware

The models in this project were trained on an **NVIDIA A100 GPU** with 40 GB of memory, using the computing resources provided by the **University of Groningen's Habrok Computing Cluster**. This hardware setup significantly accelerated the training process and enabled efficient handling of large datasets.

## Dataset

The dataset utilized for this project is the **CNN/Daily Mail Dataset** (version 3.0.0) from the Hugging Face `datasets` library. This dataset is widely used for text summarization tasks and contains over 300,000 news articles paired with corresponding summaries. Each article is a comprehensive news piece, while the summary provides a concise overview, making it ideal for training and evaluating summarization models.

### Key Features of the Dataset:

- **Article Count**: 300,000+ news articles.
- **Source**: CNN and Daily Mail.
- **Purpose**: Benchmark for abstractive summarization tasks.
- **Version**: `v3.0.0` (latest version).

To load the dataset, the following code snippet is used:

```python
from datasets import load_dataset

# Load the CNN/Daily Mail dataset
ds = load_dataset("cnn_dailymail", "3.0.0")
```

The dataset is **pre-split** into three sets:

- **Training Set**: 287,113 articles, used to train the model.
- **Test Set**: 11,490 articles, used to evaluate the model’s performance.
- **Validation Set**: 13,368 articles, used to fine-tune and validate the model during training.

Each data entry contains:

- `article`: The full news article.
- `highlights`: The summary of the article.
- `id`: A unique identifier for each entry.

For more details, you can explore the dataset on [Hugging Face Datasets](https://huggingface.co/datasets/cnn_dailymail).


## Model

The model used for this project is the **T5-base**, a transformer-based model developed by Google. T5, or **Text-to-Text Transfer Transformer**, is designed to convert various language processing tasks into a unified text-to-text format. This approach allows T5 to handle a wide range of NLP tasks, including summarization, translation, question answering, and more, using a single framework.

### Key Features of T5-base:

- **Architecture**: Transformer with 12 layers, 768 hidden dimensions, and 12 attention heads.
- **Parameters**: Approximately 220 million.
- **Pre-trained Capabilities**: The T5-base model is pre-trained on a diverse dataset (C4) and can generalize well to various downstream tasks.
- **Text-to-Text Framework**: All tasks are framed as generating a text response based on an input, making it flexible for many NLP tasks.

For more in-depth information about the T5 architecture, you can refer to the original paper: [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683).


## Project Overview

This project focuses on improving the performance of the T5-base model for text summarization using fine-tuning techniques. It includes strategies like **Prompt Engineering**, **Parameter-Efficient Fine-Tuning (PEFT) with LoRA**, and **Full Fine-Tuning**. Each method offers unique advantages in improving model accuracy and efficiency.

### Prompt Engineering

Prompt Engineering involves crafting specific instructions or prompts to guide the model's output. In this project, an instructional prompt like "Summarize this article:" is added to the input text, aiming to improve the clarity and relevance of the generated summaries. This technique leverages the model's pre-trained capabilities, improving its performance without altering the underlying parameters.

### PEFT (Parameter-Efficient Fine-Tuning) with LoRA

PEFT (Parameter-Efficient Fine-Tuning) is a technique designed to adjust only a subset of the model's parameters, instead of all of them, to reduce computational costs. This project utilizes **LoRA (Low-Rank Adaptation)**, which fine-tunes specific low-rank matrices within the model. This approach enables efficient training with reduced memory requirements and faster convergence, making it suitable for scenarios where computational resources are limited.

### Full Fine-Tuning

Full Fine-Tuning updates all the parameters of the pre-trained model based on the target dataset—in this case, the CNN/Daily Mail dataset. This method allows the model to learn task-specific nuances in detail, often resulting in better performance for complex tasks. However, it requires significant computational resources, as it optimizes the entire model structure.
