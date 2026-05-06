# 🚀 Production-Ready NLP System for Scalable Document Summarization with MLOps Pipeline

## 📌 Problem Statement
Information overload is a critical bottleneck in enterprise, legal, and academic sectors. Professionals waste thousands of aggregate hours scrolling through dense PDF dossiers, extended financial reports, and comprehensive research papers just to extract core intelligence. 

## 💡 Solution
This system addresses information overload by enabling asynchronous, scalable summarization of large documents and raw strings alike. It employs powerful Hugging Face Transformers (`pegasus-samsum`) integrated into a highly modular backend architecture to simulate a real-world machine learning deployment environment and instantly extract executive summaries.

## 🧠 Key Features
* **Multi-Format Ingestion**: Natively targets, extracts, and summarizes text from raw inputs and `.pdf` binaries seamlessly.
* **Modern Glassmorphic Interface**: A stunning, high-performance HTML/CSS frontend UI with interactive drag-and-drop support.
* **Memory-Optimized Pipeline**: Employs global proactive model loading to prevent multi-gigabyte PyTorch Out-of-Memory (OOM) failures under repeated API stress.
* **Modular MLOps Design**: Architecture separates Data Ingestion, Transformation, Model Training, and Evaluation into discrete, scalable structures.

## ⚙️ Architecture / How It Works
1. **Frontend Entry**: The client interface sends raw text or parsed PDF binaries to the backend API via `FormData`.
2. **Backend Gateway**: FastAPI processes the payload, securely parsing text using `PyPDF2` and guarding against token floods via structural truncation.
3. **NLP Engine**: The globally cached `pegasus-samsum` transformer utilizes its encoder-decoder frame to calculate contextual weight and synthesis.
4. **Resolution**: The system computes and returns an accurate executive summary to the UI asynchronously.

## 🛠️ Tech Stack

| Component | Technology Used |
| :--- | :--- |
| **Language** | Python 3.10 |
| **NLP Framework** | Hugging Face Transformers, Datasets, PyTorch |
| **Backend API** | FastAPI, Uvicorn, Python-Multipart |
| **Frontend UI** | HTML5, CSS3, Jinja2, JavaScript |
| **System Utilities**| PyPDF2, YAML Parsing, Boto3 |

## 📸 Screenshots / Demo
*(Add your stunning UI screenshot here!)*
<!-- Place an image named UI_Showcase.png in an image folder and reference it here: ![Summary UI Demo](./UI_Showcase.png) -->

## ⚙️ How to Run
**1. Clone the repository and navigate inside:**
```bash
git clone <repository-url>
cd Text-Summarizer
```
**2. Environment and Dependency Installation:**
```bash
pip install -r requirements.txt
```
**3. Boot the Uvicorn Server:**
```bash
python app.py
```
**4. Access the Application:** Navigate your browser to `http://localhost:8000/`

## 🎯 Use Cases
* **Financial Data**: Rapidly synthesizing dense earnings reports and corporate 10-K filings.
* **Legal Counsel**: Extracting immediate rulings from hundred-page case dossiers.
* **Academia**: Mass-summarizing massive literature reviews and publications.

## 📈 Future Improvements
* **Map-Reduce Architecture (LangChain)**: To chunk exceptionally large documents recursively, bypassing core transformer token limits.
* **MLflow Integration**: For native hyperparameter logging and experiment registry management.
* **Data Version Control (DVC)**: To track large scale datasets via cloud storage systems (AWS S3).

## 📄 License
This project is licensed under the MIT License.
