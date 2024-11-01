# T5-base Fine-tuning for Summarization with Instructional Prompts
---
## 📖 Introduction

This project involves fine-tuning the T5-base model on the CNN/Daily Mail dataset using LoRA (Low-Rank Adaptation) and traditional full fine-tuning. Additionally, instructional prompts like "Provide a detailed and concise summary of this article: " are incorporated to enhance the model's summarization performance. The fine-tuned models are compared to the original T5-base model's summarization capabilities without fine-tuning. The aim is to observe the improvements in performance brought by fine-tuning and prompt engineering techniques.

---

## Table of Contents

- [Installation](#installation)
- [System Requirements](#system-requirements)
- [Reproducibility](#reproducibility)
- [Dataset](#dataset)
- [Model](#model)
- [Project Overview](#project-overview)
  - [Prompt Engineering](#prompt-engineering)
  - [PEFT (Parameter-Efficient Fine-Tuning) with LoRA](#peft-parameter-efficient-fine-tuning-with-lora)
  - [Full Fine-Tuning](#full-fine-tuning)
- [Results](#results)
- [Known Issues & Limitations](#known-issues--limitations)
- [Additional Resources](#additional-resources)
- [Contributors](#contributors)
  
---

## 💾 Installation

To set up the project environment, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/03chrisk/PEFT-T5-on-CNN-dailynews.git
   cd PEFT-T5-on-CNN-dailynews

2. **Set Up an Environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    
3. **Intall Requirements**
   ```bash
   pip install -r requirements.txt

---

## 🖥️ System Requirements

To run the fine-tuning and evaluation scripts efficiently, the following system requirements are recommended:

- **Python**: 3.10+
- **GPU**: NVIDIA A100 GPU with at least 40 GB of memory (recommended for faster training)
- **CUDA**: Compatible version for your GPU and PyTorch setup
- **RAM**: 16 GB or higher
- **Disk Space**: At least 50 GB free for storing datasets and model checkpoints

### Training Hardware

The models in this project were trained on an **NVIDIA A100 GPU** with 40 GB of memory, using the computing resources provided by the **University of Groningen's Habrok Computing Cluster**. This hardware setup significantly accelerated the training process and enabled efficient handling of large datasets.

---
## 🔄 Reproducibility 

Reproducing the results of this project requires careful attention to environment setup, dataset handling, and training configurations. Below are some guidelines to help ensure that you can replicate the findings as closely as possible:

### Guidelines for Reproducibility:

1. **Environment Setup**:
   - Make sure to use the same versions of key libraries, such as `transformers`, `torch`, `datasets`, and `peft`.
   - Use the provided `requirements.txt` to install the exact dependencies needed.
   - It is recommended to set up a virtual environment to isolate dependencies.

2. **Random Seeds**:
   - Random seeds have been set for data shuffling, model initialization, and other stochastic processes to increase reproducibility.
   - However, due to the nature of some operations, exact reproducibility is not always guaranteed.

3. **Hardware & Performance**:
   - The training was conducted on an **NVIDIA A100 GPU** with 40 GB of memory. Using a similar GPU will help replicate the performance metrics.
   - Differences in hardware architecture (e.g., using a different GPU or CPU) may lead to slight variations in results.

### Note on CUDA Non-Determinism:

CUDA processes can be **non-deterministic**, meaning that even with fixed seeds, the results might slightly vary when running on a GPU. This is due to the non-deterministic nature of certain CUDA operations and floating-point calculations. To minimize this effect, you can try the following:

- Set `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False`. However, be aware that this might slightly degrade training performance.
- Keep in mind that perfect reproducibility is challenging when using GPU acceleration.

### Reproducibility Checklist:

- [ ] Ensure consistent environment with `requirements.txt`.
- [ ] Set random seeds using `torch.manual_seed(seed_value)`.
- [ ] Utilize the same dataset version (`cnn_dailymail`, `v3.0.0`).
- [ ] Match the training configurations (e.g., learning rate, epochs, batch size).
- [ ] Consider hardware differences if results differ slightly.

---

## 📂 Dataset

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

> [!NOTE]
> For more details, you can explore the dataset on [Hugging Face Datasets](https://huggingface.co/datasets/cnn_dailymail).

---

## 🤖 Model

The model used for this project is the **T5-base**, a transformer-based model developed by Google. T5, or **Text-to-Text Transfer Transformer**, is designed to convert various language processing tasks into a unified text-to-text format. This approach allows T5 to handle a wide range of NLP tasks, including summarization, translation, question answering, and more, using a single framework.

### Key Features of T5-base:

- **Architecture**: Transformer with 12 layers, 768 hidden dimensions, and 12 attention heads.
- **Parameters**: Approximately 220 million.
- **Pre-trained Capabilities**: The T5-base model is pre-trained on a diverse dataset (C4) and can generalize well to various downstream tasks.
- **Text-to-Text Framework**: All tasks are framed as generating a text response based on an input, making it flexible for many NLP tasks.

> [!NOTE]
> For more in-depth information about the T5 architecture, you can refer to the original paper: [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683).

---

## 📝 Project Overview

This project focuses on improving the performance of the T5-base model for text summarization using fine-tuning techniques. It includes strategies like **Prompt Engineering**, **Parameter-Efficient Fine-Tuning (PEFT) with LoRA**, and **Full Fine-Tuning**. Each method offers unique advantages in improving model accuracy and efficiency. Below are explanations of the general intuitions behind eahc method, however for more details see the report in the repository.

---

### 🎯 Prompt Engineering

Prompt Engineering involves crafting specific instructions or prompts to guide the model's output. In this project, an instructional prompt like `"Provide a detailed and concise summary of this article:"` is added to the input text, aiming to improve the clarity and relevance of the generated summaries. This technique leverages the model's pre-trained capabilities, improving its performance without altering the underlying parameters.

---

### 🧪 PEFT (Parameter-Efficient Fine-Tuning) with LoRA

PEFT (Parameter-Efficient Fine-Tuning) is a technique designed to adjust only a subset of the model's parameters, instead of all of them, to reduce computational costs. This project utilizes **LoRA (Low-Rank Adaptation)**, which fine-tunes specific low-rank matrices within the model. This approach enables efficient training with reduced memory requirements and faster convergence, making it suitable for scenarios where computational resources are limited.

To manage the specific LoRa configuration, we use the `LoraConfig` object from the `peft` library:

**LoRA Configuration Parameters**
- **`r`**: `16` — The rank for the low-rank adaptation, controlling the number of parameters fine-tuned.
- **`lora_alpha`**: `32` — A scaling factor applied to the low-rank updates, balancing their impact on the model.
- **`lora_dropout`**: `0.1` — Dropout rate used during training to reduce overfitting.
- **`bias`**: `"none"` — No additional bias terms are added in the LoRA configuration.
- **`task_type`**: `"SEQ_2_SEQ_LM"` — Specifies the task type as sequence-to-sequence language modeling, suitable for the T5-base model.

#### Training Configuration

To manage the fine-tuning process with LoRA, the `Seq2SeqTrainingArguments` object is used. Below is a breakdown of the key hyperparameters and optimizer settings:

#### Key Training Hyperparameters

**Training Dynamics**:
- **Learning Rate**: `1e-5` — A moderately higher learning rate is used to accelerate convergence for LoRA parameters.
- **Batch Size**: `6` — A larger batch size is feasible due to the reduced parameter set.
- **Epochs**: `1` — Two passes over the dataset to balance between learning capacity and overfitting.
- **Gradient Accumulation Steps**: `2` — Accumulates gradients over 2 steps to simulate a larger effective batch size without exceeding memory limits.

**Regularization & Stability**:
- **Weight Decay**: `0.1` — Applies a small penalty to model weights to help prevent overfitting.
- **Evaluation Interval**: Every `20%` of training progress — Provides frequent checkpoints to monitor model performance.
- **Loss Function**: `Cross-Entropy Loss` — Used as the standard loss function for sequence-to-sequence tasks.


#### Optimizer & Scheduler

For LoRA fine-tuning, the `AdamW` optimizer is employed, and a linear learning rate scheduler with a warmup phase is implemented to ensure stable learning progression.

**Optimization Settings**:
- **Learning Rate**: `1e-4` — A slightly higher learning rate for faster adaptation during LoRA fine-tuning.
- **Weight Decay**: `0.1` — Regularization to mitigate overfitting.
- **Warmup Steps**: `10%` of the total training steps — A warmup phase to ease the model into the training.
- **Betas**: `(0.9, 0.999)` — Controls the decay rates for the moving averages of the gradient and its square in AdamW.
- **Epsilon**: `1e-8` — A small constant to prevent division errors in AdamW.
- **Total Training Steps**: Automatically calculated based on dataset size, batch size, and number of epochs.

---

### 🏋️ Full Fine-Tuning

Full Fine-Tuning updates all the parameters of the pre-trained model using the target dataset—in this case, the CNN/Daily Mail dataset. This approach allows the model to learn task-specific nuances in detail, often resulting in superior performance for complex tasks. However, it requires significant computational resources since the entire model is optimized during training.

#### Training Configuration

As in PEFT, to manage the fine-tuning process, the `Seq2SeqTrainingArguments` object is used.

#### Key Hyperparameters

**Training Dynamics**:
- **Learning Rate**: `5e-5` — A low learning rate ensures stable parameter updates, reducing the risk of overshooting minima.
- **Batch Size**: `8` — Small batch size due to memory constraints, compensated by gradient accumulation.
- **Epochs**: `1` — Two passes over the dataset, ensuring enough exposure without overfitting.
- **Gradient Accumulation Steps**: `2` — Accumulates gradients over 8 steps to simulate a larger effective batch size.

**Regularization & Stability**:
- **Weight Decay**: `0.001` — Regularization technique to penalize large weights, helping prevent overfitting.
- **Evaluation Interval**: Every `20%` of training progress — Provides frequent checkpoints to monitor model performance.
- **Loss Function**: `Cross-Entropy Loss`

#### Optimizer & Scheduler

To optimize the model's parameters, the `AdamW` optimizer is used, well-suited for transformer-based architectures. Additionally, a linear learning rate scheduler with a warmup phase is implemented to gradually ramp up the learning rate during the initial training phase, which helps in stabilizing the training.

**Optimization Settings**:
- **Learning Rate**: `5e-5` — A carefully chosen learning rate that balances learning speed and stability.
- **Weight Decay**: `0.001` — Adds regularization to mitigate overfitting.
- **Warmup Steps**: `10%` of the total training steps — A warmup phase to ease the model into the training.
- **Betas**: `(0.9, 0.999)` — Parameters for the AdamW optimizer controlling the momentum of the gradients.
- **Epsilon**: `1e-8` — A small constant to prevent division by zero in the optimizer calculations.
- **Total Training Steps**: Automatically calculated based on dataset size, batch size, and number of epochs.

---

## 📊 Results

The evaluation of different configurations of the T5-base model on the CNN/Daily Mail dataset showed significant improvements in performance when applying prompting, LoRA, and full fine-tuning. Key findings include:

**Prompting**: Applying prompting alone improved performance by around 4% on average (excluding BLEU), suggesting that while prompting helps guide the model, it is not sufficient for task-specific adaptation.

**LoRA + Prompting**: The combination of LoRA and prompting achieved an average performance increase of 27% across all metrics, with particularly strong improvements in ROUGE and BLEU scores. This configuration significantly enhanced the model’s ability to produce cohesive, human-like summaries.

**Full Fine-Tuning + Prompting**: Full fine-tuning with prompting resulted in the highest performance, with a further 5% average increase over LoRA. This approach demonstrated superior phrase alignment, fluency, and semantic accuracy.

While the fine-tuned models showed considerable improvements, they still fall short of state-of-the-art benchmarks, largely due to the limited parameter size of T5-base compared to larger models like T5-11b. Nonetheless, these results highlight the effectiveness of fine-tuning and prompting in improving summarization performance, and they provide a strong baseline for further research.

### Performance of the Models

| Metric       | T5-base | Prompting | LoRA  | Full FT |
|--------------|---------|-----------|-------|---------|
| **ROUGE-1**  | 30.17   | 31.30     | 34.47 | **35.68** |
| **ROUGE-2**  | 11.17   | 11.22     | 14.27 | **15.00** |
| **ROUGE-L**  | 21.19   | 22.08     | 25.92 | **26.36** |
| **BLEU**     | 9.57    | 9.57      | 13.77 | **15.15** |
| **BERT score** | 19.63 | 21.19     | 29.19 | **30.98** |

### Summaries Generated by the Models

Sample summaries were generated for an example article from the test set, which will be referenced in the appendix (end of this file)

#### T5 base
He was originally going to take his girlfriend to Eastern's prom. "She’s great... she listens and she’s easy to talk to" he said. "I’m so happy for him," Ellie’s mom said. "I’m so happy for him," Trey’s coach said.

#### T5 base + Prompting
"That’s the kind of person Trey is," says a teacher at Eastern High School. "That’s the kind of person Trey is," says a teacher. "It’s exciting to bring awareness to a good cause," says a mother.

#### T5 base + LoRA
High school basketball star Trey Moses asks Ellie Meredith to be his prom date. Ellie is a freshman with Down syndrome.

#### T5 base + Full Fine-Tuning
Trey Moses, a college basketball recruit, asked Ellie Meredith to be his prom date. Ellie is a high school freshman with Down syndrome. Trey’s mom says he’s a good son, but she says she’s numb to the moment.

---

## ⚠️ Known Issues & Limitations

One of the primary bottlenecks for this project was the limited availability of computational resources, specifically GPU cores and memory. Initially, we attempted to train on the entire dataset, which took approximately 5 hours per epoch with a batch size of 6, utilizing gradient accumulation. However, this approach proved to be unsustainable as the training process was often inconsistent and prone to divergence, likely due to factors such as model configuration, hyperparameter values or nuances in the dataset.

To address this, we adjusted our approach to training on a smaller subset of the data, around 10\% of the full dataset, which allowed lower computational time and cost while reducing the risk of divergence. Even with this subset, we had to carefully balance the batch size per core to ensure non-noisy convergence as well as remaining within the bounds of our computational resources. On average, training for one epoch with these settings took around 25 minutes.

The limited access to resources of our university's HPC cluster and the restricted availability of team members further constrained our training process, making it challenging to conduct experiments over longer periods.

--- 

## 📚 Additional Resources

For more in-depth information and further reading, check out the following resources:

- **T5 Paper**: [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) — The original paper introducing the T5 architecture, outlining its capabilities and the text-to-text framework.
- **CNN/Daily Mail Dataset**: [Hugging Face Datasets - CNN/Daily Mail](https://huggingface.co/datasets/cnn_dailymail) — The dataset used for training and evaluating the models in this project.
- **LoRA Paper**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — A detailed explanation of the Low-Rank Adaptation technique, which enables efficient fine-tuning with minimal resources.
- **Hugging Face Transformers Documentation**: [Transformers Documentation](https://huggingface.co/docs/transformers) — A comprehensive guide to using the `transformers` library for NLP tasks, including T5-based models.
- **PEFT Library Documentation**: [PEFT Library on GitHub](https://github.com/huggingface/peft) — The official documentation for the `peft` library, which facilitates parameter-efficient fine-tuning methods like LoRA.
- **ROUGE Metric**: [ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/) — Information about the ROUGE metric used for evaluating the quality of generated summaries.
- **Seq2SeqTrainer Documentation**: [Hugging Face Trainer Guide](https://huggingface.co/transformers/main_classes/trainer.html#seq2seqtrainer) — Details on using `Seq2SeqTrainer` for managing the training of sequence-to-sequence models like T5.

These resources should provide a solid foundation for understanding the methodologies and tools used in this project.

---

## 🤝 Contributors

A big thank you to everyone who contributed to this project:

- **Christian Kobriger**
- **Csenge Szoke** 
- **Lukasz Sawala**

---

## Appendix
### Original Article
(CNN) He's a blue chip college basketball recruit. She's a high school freshman with Down syndrome. At first glance Trey Moses and Ellie Meredith couldn't be more different. But all that changed Thursday when Trey asked Ellie to be his prom date. Trey -- a star on Eastern High School's basketball team in Louisville, Kentucky, who's headed to play college ball next year at Ball State -- was originally going to take his girlfriend to Eastern's prom. So why is he taking Ellie instead? "She's great... she listens and she's easy to talk to" he said. Trey made the prom-posal (yes, that's what they are calling invites to prom these days) in the gym during Ellie's P.E. class. Trina Helson, a teacher at Eastern, alerted the school's newspaper staff to the prom-posal and posted photos of Trey and Ellie on Twitter that have gone viral. She wasn't surprised by Trey's actions. "That's the kind of person Trey is," she said. To help make sure she said yes, Trey entered the gym armed with flowers and a poster that read "Let's Party Like it's 1989," a reference to the latest album by Taylor Swift, Ellie's favorite singer. Trey also got the OK from Ellie's parents the night before via text. They were thrilled. "You just feel numb to those moments raising a special needs child," said Darla Meredith, Ellie's mom. "You first feel the need to protect and then to overprotect." Darla Meredith said Ellie has struggled with friendships since elementary school, but a special program at Eastern called Best Buddies had made things easier for her. She said Best Buddies cultivates friendships between students with and without developmental disabilities and prevents students like Ellie from feeling isolated and left out of social functions. "I guess around middle school is when kids started to care about what others thought," she said, but "this school, this year has been a relief." Trey's future coach at Ball State, James Whitford, said he felt great about the prom-posal, noting that Trey, whom he's known for a long time, often works with other kids. Trey's mother, Shelly Moses, was also proud of her son. "It's exciting to bring awareness to a good cause," she said. "Trey has worked pretty hard, and he's a good son." Both Trey and Ellie have a lot of planning to do. Trey is looking to take up special education as a college major, in addition to playing basketball in the fall. As for Ellie, she can't stop thinking about prom. "Ellie can't wait to go dress shopping," her mother said. "Because I've only told about a million people!" Ellie interjected.

### Highlight
College-bound basketball star asks girl with Down syndrome to high school prom. Pictures of the two during the "prom-posal" have gone viral.

