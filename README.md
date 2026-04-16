🤖 Prompt Engineering & LLM Fine-Tuning Project
🧠 Two-Part System: Prompt-Based Reasoning + Model Fine-Tuning Pipeline



🚀 Overview
This project explores two complementary approaches to working with Large Language Models (LLMs):


🔹 Part 1: Prompt Engineering System
Prompt design for reasoning tasks
Evaluation of model responses
Hint-based prompting strategies
Repair mechanisms for improving outputs

🔹 Part 2: Fine-Tuning Pipeline
Dataset assembly for supervised training
Model fine-tuning workflow
Evaluation of fine-tuned performance on structured tasks

The project demonstrates both:

👉 Prompt-level control of LLMs
👉 Model-level adaptation via fine-tuning

🧠 Key Objectives
Understand how prompt design impacts LLM reasoning
Build structured evaluation pipelines for LLM outputs
Construct datasets for supervised fine-tuning
Train and evaluate domain-specific LLM behavior
Compare prompt-based vs fine-tuned performance

📁 Project Structure
</> Markdown

text "

📦 Project Root

├── README.md
├── requirements.txt
├── project_part1_datasets.zip

# =========================
# PART 1 - PROMPT ENGINEERING
# =========================
├── project_part1_prompts.py        # Prompt templates for reasoning tasks
├── project_part1_hint.py           # Hint-based prompt enhancement logic
├── project_part1_repair.py         # Output correction / repair strategies
├── project_part1_evaluate.py       # Evaluation of prompt outputs

# =========================
# PART 2 - FINE TUNING PIPELINE
# =========================
├── project_part2_assembeld_dataset.py     # Dataset construction pipeline
├── project_part2_dataset_training_raw.json # Training dataset (raw format)
├── project_part2_finetuning.py            # Model fine-tuning pipeline
├── project_part2_evaluate.py              # Evaluation of fine-tuned model



"



🧪 Part 1: Prompt Engineering System
This module focuses on controlling LLM behavior without modifying model weights.
Components:
Prompt Templates
Structured prompts for task completion and reasoning
Hint System
Adds intermediate guidance to improve model reasoning
Repair Module
Fixes or improves invalid/low-quality model outputs
Evaluation Module
Measures response quality and correctness

👉 Goal: Improve model performance purely through prompt design

🏗️ Part 2: Fine-Tuning Pipeline

This module focuses on adapting the model itself using training data.
Components:
Dataset Assembly
Converts raw structured data into training format
Training Dataset
JSON-based supervised fine-tuning dataset
Fine-Tuning Script
Trains model on domain-specific tasks
Evaluation Pipeline
Compares pre-trained vs fine-tuned performance

👉 Goal: Improve model behavior through parameter updates

⚙️ Tech Stack

Python 3.10+
NLP / LLM Concepts
Transformer-based models
JSON structured datasets
Prompt Engineering techniques
Fine-tuning pipelines (HuggingFace-style workflow)

🔬 Key Learnings

Difference between prompt engineering vs fine-tuning
How structured prompts influence LLM reasoning
Dataset construction for supervised learning
Evaluation strategies for generative models
Trade-offs between inference-time vs training-time optimization


📊 System Workflow
🔹 Part 1 (Prompt System)

Input → Prompt Design → LLM Response → Hint/Repair → Evaluation

🔹 Part 2 (Fine-Tuning)

Raw Data → Dataset Construction → Fine-Tuning → Evaluation → Improved Model


📈 Results

Prompt engineering significantly improves baseline LLM performance without training
Fine-tuning provides stronger task-specific consistency
Hybrid approach (prompt + fine-tuning) gives best results in structured tasks

🚀 Future Improvements
Add transformer-based evaluation metrics (BLEU, ROUGE, BERTScore)
Integrate LangChain / LangGraph for agent workflows
Expand dataset for multi-task learning
Add experiment tracking (MLflow / Weights & Biases)
Deploy inference API (FastAPI / HuggingFace Spaces)

👨‍💻 Author
Abhijeet Malik
🎓 B.Sc. Data Science & Artificial Intelligence
University of the Saarland, Germany


