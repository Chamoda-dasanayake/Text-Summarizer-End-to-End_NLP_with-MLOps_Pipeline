<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Transformers" />
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/MLflow-Experiment%20Tracking-0194E2?style=for-the-badge&logo=mlflow&logoColor=white" alt="MLflow" />
  <img src="https://img.shields.io/badge/DVC-Pipeline%20Orchestration-13ADC7?style=for-the-badge&logo=dvc&logoColor=white" alt="DVC" />
</p>

<h1 align="center">📝 Text Summarizer — End-to-End NLP with MLOps Pipeline</h1>

<p align="center">
  <strong>A production-grade text summarization system built with a modular MLOps architecture.</strong><br/>
  <em>Fine-tunes a T5 Seq2Seq transformer on the SAMSum dialogue dataset, evaluated with ROUGE metrics, tracked via MLflow, and served through a FastAPI backend with a glassmorphic drag-and-drop UI.</em>
</p>

<p align="center">
  <a href="#-architecture">Architecture</a> •
  <a href="#-key-features">Features</a> •
  <a href="#-tech-stack">Tech Stack</a> •
  <a href="#%EF%B8%8F-getting-started">Setup</a> •
  <a href="#-mlops-pipeline-stages">Pipeline</a>
</p>

---

## 📌 Problem Statement

Information overload is a critical bottleneck across enterprise, legal, and academic sectors. Professionals spend thousands of hours sifting through dense reports, research papers, and lengthy documents to extract key insights — a process that is slow, error-prone, and unscalable.

## 💡 Solution

This system delivers **instant, accurate text summarization** powered by a fine-tuned **T5 Seq2Seq transformer**, wrapped in a fully modular **MLOps pipeline** that separates every stage — from data ingestion to model evaluation — into independently executable, trackable components.

```
📄 Input (Text / PDF) → 🔠 Tokenization → 🧠 T5 Inference → 📊 ROUGE Evaluation → ✅ Summary
```

---

## 🧠 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  TRAINING PIPELINE (main.py — 4 sequential stages via DVC)          │
│                                                                     │
│  ┌────────────────┐    ┌────────────────┐    ┌────────────────┐     │
│  │ Stage 1        │    │ Stage 2        │    │ Stage 3        │     │
│  │ Data Ingestion │───▶│ Data Transform │───▶│ Model Trainer  │     │
│  │ (HuggingFace)  │    │ (Tokenization) │    │ (T5 Fine-Tune) │     │
│  └────────────────┘    └────────────────┘    └───────┬────────┘     │
│                                                      ▼              │
│                                              ┌────────────────┐     │
│                                              │ Stage 4        │     │
│                                              │ Model Eval     │     │
│                                              │ (ROUGE+MLflow) │     │
│                                              └────────────────┘     │
├─────────────────────────────────────────────────────────────────────┤
│  INFERENCE PIPELINE (app.py — FastAPI server)                       │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐   │
│  │ Client UI    │───▶│ FastAPI      │───▶│ PredictionPipeline   │   │
│  │ (HTML/JS)    │    │ /predict     │    │ T5 model.generate()  │   │
│  └──────────────┘    └──────────────┘    └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### Design Principles

| Principle | Implementation |
|-----------|---------------|
| **Separation of Concerns** | Each pipeline stage is an independent class — ingestion, transformation, training, and evaluation are fully decoupled |
| **Configuration-Driven** | All paths, hyperparameters, and model checkpoints are externalized to `config.yaml` and `params.yaml` |
| **Entity-Based Contracts** | Frozen `@dataclass` entities enforce strict typing between configuration and components |
| **Singleton Model Loading** | The PyTorch model loads once globally at startup to prevent OOM crashes under repeated API calls |

---

## ✨ Key Features

### NLP & Deep Learning
- **T5 Seq2Seq Fine-Tuning** — Fine-tunes the `t5-small` transformer on the [SAMSum](https://huggingface.co/datasets/knkarthick/samsum) dialogue summarization dataset using HuggingFace `Trainer` API
- **Beam Search Decoding** — Generates summaries with `num_beams=8` and `length_penalty=0.8` for high-quality, concise output
- **ROUGE Evaluation** — Automated evaluation with `rouge1`, `rouge2`, `rougeL`, and `rougeLsum` metrics
- **GPU-Accelerated** — Automatic CUDA detection with seamless CPU fallback

### MLOps & Pipeline
- **4-Stage DVC Pipeline** — Reproducible ML workflow defined in `dvc.yaml` with explicit dependency and output tracking
- **MLflow Experiment Tracking** — ROUGE metrics and evaluation artifacts are automatically logged to a local MLflow registry
- **YAML-Driven Configuration** — Hyperparameters (`params.yaml`) and paths (`config.yaml`) are fully externalized and version-controlled
- **Structured Logging** — Persistent file + console logging with timestamped output via a custom logging module

### Backend & API
- **FastAPI REST Server** — Async API with a `/predict` endpoint for summarization and a `/train` endpoint to trigger pipeline retraining in background
- **Multi-Format Ingestion** — Accepts both raw text input and PDF file uploads with automatic text extraction via `PyPDF2`
- **Memory-Safe Inference** — Global model caching prevents multi-GB model reloads; input text truncated to 5,000 chars to guard against token overflow

### Frontend UI
- **Glassmorphic Dark Theme** — Modern, polished interface built with `Outfit` font, gradient accents, and frosted-glass card design
- **Drag-and-Drop Upload** — Interactive file drop zone with visual feedback for PDF/TXT uploads
- **Async UX Flow** — Loading spinner, disabled-state button, and animated result reveal for a smooth user experience

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **NLP Model** | HuggingFace Transformers (T5-small) | Seq2Seq fine-tuning and inference for abstractive summarization |
| **Deep Learning** | PyTorch | Model backbone with CUDA acceleration support |
| **Dataset** | SAMSum (via HuggingFace `datasets`) | 16K dialogue-summary pairs for training and evaluation |
| **Evaluation** | ROUGE (via `evaluate`) | Automated scoring across rouge1, rouge2, rougeL, rougeLsum |
| **Experiment Tracking** | MLflow | Metric logging, artifact versioning, and experiment registry |
| **Pipeline Orchestration** | DVC | Reproducible 4-stage ML pipeline with dependency graphs |
| **Backend** | FastAPI + Uvicorn | Async REST API server with background task support |
| **Frontend** | HTML5 / CSS3 / JavaScript / Jinja2 | Glassmorphic UI with drag-and-drop and async form submission |
| **PDF Parsing** | PyPDF2 | Binary PDF text extraction for document upload feature |
| **Config Management** | PyYAML + python-box | YAML parsing with dot-access `ConfigBox` for clean config access |

---

## 📁 Project Structure

```
Text-Summarizer/
│
├── src/textSummarizer/                  # 🧠 Core Package
│   ├── components/
│   │   ├── data_ingestion.py                # Downloads SAMSum from HuggingFace Hub
│   │   ├── data_transformation.py           # Tokenizes dialogues & summaries via T5 tokenizer
│   │   ├── model_trainer.py                 # Fine-tunes T5 with HF Trainer API
│   │   └── model_evaluation.py              # ROUGE scoring + MLflow experiment logging
│   │
│   ├── pipeline/
│   │   ├── stage_1_data_ingestion_pipeline.py
│   │   ├── stage_2_data_transformation_pipeline.py
│   │   ├── stage_3_model_trainer.py
│   │   ├── stage_4_model_evaluation.py
│   │   └── prediction_pipeline.py           # Inference pipeline (model.generate)
│   │
│   ├── config/
│   │   └── configuration.py                 # ConfigurationManager — reads YAML, builds entities
│   │
│   ├── entity/
│   │   └── __init__.py                      # @dataclass contracts for each pipeline stage
│   │
│   ├── utils/
│   │   └── common.py                        # read_yaml(), create_directories() utilities
│   │
│   ├── constants/
│   │   └── __init__.py                      # CONFIG_FILE_PATH, PARAMS_FILE_PATH
│   │
│   └── logging/
│       └── __init__.py                      # File + console logger setup
│
├── config/
│   └── config.yaml                      # All paths, model checkpoints, dataset config
│
├── templates/
│   └── index.html                       # 🎨 Glassmorphic frontend UI
│
├── research/                            # 📓 Jupyter notebooks (EDA & experimentation)
│   ├── 1_data_ingestion.ipynb
│   ├── 2_data_transformation.ipynb
│   ├── 3_model_trainer.ipynb
│   └── 4_model_evaluation.ipynb
│
├── app.py                               # ⚡ FastAPI server entry point
├── main.py                              # 🚀 Training pipeline orchestrator (all 4 stages)
├── dvc.yaml                             # 📊 DVC pipeline definition
├── params.yaml                          # ⚙️ Training hyperparameters
├── requirements.txt                     # 📦 Python dependencies
├── template.py                          # 🏗️ Project scaffolding script
└── .github/workflows/                   # CI/CD directory (reserved)
```

---

## 📊 MLOps Pipeline Stages

Each stage is an **independently executable, trackable unit** — orchestrated sequentially via `main.py` or reproduced via `dvc repro`.

| Stage | Component | Input | Output | Key Details |
|-------|-----------|-------|--------|-------------|
| **1. Data Ingestion** | `DataIngestion` | HuggingFace Hub (`knkarthick/samsum`) | `artifacts/data_ingestion/` | Downloads and caches the SAMSum dataset locally |
| **2. Data Transformation** | `DataTransformation` | Raw dataset splits | `artifacts/data_transformation/samsum_dataset/` | Tokenizes dialogues (max 1024 tokens) and summaries (max 128 tokens) using T5 tokenizer |
| **3. Model Training** | `ModelTrainer` | Tokenized dataset | `artifacts/model_trainer/pegasus-samsum-model/` + `tokenizer/` | Fine-tunes T5-small with gradient accumulation (16 steps), warmup (500 steps), weight decay (0.01) |
| **4. Model Evaluation** | `ModelEvaluation` | Trained model + test split | `artifacts/model_evaluation/metrics.csv` + MLflow logs | Computes ROUGE scores with beam search and logs results to MLflow |

### Training Hyperparameters (`params.yaml`)

```yaml
num_train_epochs: 1
warmup_steps: 500
per_device_train_batch_size: 1
weight_decay: 0.01
gradient_accumulation_steps: 16
eval_strategy: steps
eval_steps: 500
```

---

## ⚙️ Getting Started

### Prerequisites

- **Python 3.10+**
- **GPU (optional)** — CUDA-enabled GPU recommended for training; CPU works for inference

### 1. Clone & Setup

```bash
git clone https://github.com/Chamoda-dasanayake/Text-Summarizer.git
cd Text-Summarizer
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Training Pipeline (Optional)

```bash
# Execute all 4 stages sequentially
python main.py

# OR reproduce via DVC
dvc repro
```

### 5. Start the Application

```bash
python app.py
```

Open **http://localhost:8000** → Upload a PDF or paste text → Get your summary!

---

## 🔄 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serves the glassmorphic frontend UI |
| `POST` | `/predict` | Accepts `file` (PDF/TXT) or `text_input` (raw string) — returns JSON `{ "summary": "..." }` |
| `GET` | `/train` | Triggers the full training pipeline as a background task |

---

## 🎯 Engineering Highlights

| Skill Area | Demonstrated In |
|-----------|-----------------|
| **NLP & Deep Learning** | Fine-tuning T5 Seq2Seq transformer for abstractive summarization with beam search decoding |
| **MLOps Pipeline Design** | 4-stage DVC pipeline with YAML-driven config, entity contracts, and MLflow experiment tracking |
| **Software Architecture** | Modular package structure — components, pipelines, entities, config, and utils are fully decoupled |
| **API Engineering** | FastAPI async server with background training tasks, multi-format file handling, and singleton model caching |
| **Frontend Development** | Modern glassmorphic UI with drag-and-drop, async fetch API, loading animations, and responsive design |
| **Memory Optimization** | Global model loading to prevent OOM + input truncation to guard against token overflow |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>Built with ❤️ using HuggingFace Transformers, FastAPI, MLflow & DVC</strong><br/>
  <em>Demonstrating end-to-end NLP engineering — from model fine-tuning to production API serving.</em>
</p>
