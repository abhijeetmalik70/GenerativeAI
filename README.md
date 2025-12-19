RIT Project – LLM-Based Program Repair & Hint Generation
Overview
This project is part of an RIT coursework assignment focused on automatic program repair and hint generation using Large Language Models (LLMs). The system evaluates student-style programming submissions, either attempting to repair incorrect code or generate helpful hints, and then measures performance across a dataset.
The project supports running evaluations in two modes:
Repair mode – Automatically fixes incorrect programs
Hint mode – Generates instructional hints without revealing full solutions
Project Structure
.
├── project_part1_repair.py        # LLM-based program repair logic
├── project_part1_hint.py          # LLM-based hint generation logic
├── project_part1_prompts.py       # Prompt templates for LLM interaction
├── project_part1_evaluate.py      # Dataset-level evaluation pipeline
├── project_part1_datasets.zip     # Input datasets for evaluation
├── project_part1_transcripts/     # Saved LLM interaction transcripts (auto-generated)
└── project_part1_evaluation_results/ # Evaluation outputs (auto-generated)
Key Components
Repair Agent
Uses an LLM to automatically correct faulty programs based on compiler feedback and distance metrics.
Hint Agent
Generates pedagogical hints instead of full solutions, designed to guide students toward fixing errors themselves.
Evaluation Pipeline
Compiles student submissions
Applies repair or hint logic
Computes distance-based correctness metrics
Saves detailed transcripts and results for analysis
Supported Models
The project is designed to work with different LLM backends. Model selection is configurable in project_part1_evaluate.py.
Example:
model_selected = "gpt-4o-mini"
HuggingFace-hosted models (e.g. Phi-3) are also supported.
How to Run
Unzip the dataset
unzip project_part1_datasets.zip
Set the mode
In project_part1_evaluate.py, choose:
mode = "repair"  # or "hint"
Run evaluation
python project_part1_evaluate.py
View results
Evaluation metrics: project_part1_evaluation_results/
LLM transcripts: project_part1_transcripts/transcript.json
Outputs
Corrected programs or generated hints
Compiler success/failure results
Distance-based correctness metrics
Full LLM interaction logs for debugging and analysis
Educational Goals
Explore the effectiveness of LLMs in programming education
Compare repair vs. hint-based feedback strategies
Analyze automated grading and feedback systems
Notes
This project is intended for educational and research purposes
Transcripts are saved automatically and may grow large
Ensure your environment has access to the selected LLM backend
