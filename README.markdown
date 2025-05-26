# CuisineInsight Q&A

CuisineInsight Q&A is a Retrieval-Augmented Generation (RAG) system for answering restaurant-related queries using a Yelp dataset of over 10,000 reviews. It integrates natural language processing (NLP), sentiment analysis, vector search, and a Flask web interface, optimized for GPU execution on an NVIDIA RTX 3060. The system achieves 91% sentiment accuracy and ~0.6-0.8 Retrieval Precision@5, showcasing advanced data science and machine learning capabilities.

## Features
- **Interactive Q&A:** Responds to queries like “What’s the best gluten-free pizza?” or “Which Mexican restaurants serve spicy vegetarian tacos?” using `llama3.2:3b`.
- **Sentiment Analysis:** Applies TF-IDF vectorization and logistic regression to classify review sentiments (91% accuracy).
- **Vector Search:** Utilizes Chroma with `nomic-embed-text:v1.5` embeddings for efficient review retrieval.
- **Evaluation Metrics:** Measures Retrieval Precision@5 (~0.6-0.8) for nuanced queries, ensuring robust performance.
- **Web Deployment:** Flask web app provides a browser-based query interface.
- **GPU Optimization:** Leverages PyTorch/CUDA on RTX 3060 (12 GB VRAM), using ~4-5 GB VRAM with dynamic memory management.
- **Data Processing:** Converts Yelp JSON datasets to CSV, enabling scalable review analysis.
- **Modular Design:** Separates preprocessing, vector store creation, Q&A, and evaluation for maintainability.

## Prerequisites
- **Hardware:**
  - GPU: NVIDIA RTX 3060 (12 GB VRAM recommended).
  - CPU: Multi-core processor (e.g., Intel i5 or AMD Ryzen).
  - RAM: 16 GB minimum.
- **Operating System:** Windows 10/11 (tested).
- **Software:**
  - Python 3.8 or higher.
  - `pip` for package management.
  - `ollama` for running models (`llama3.2:3b`, `nomic-embed-text:v1.5`).
  - NVIDIA CUDA Toolkit 12.6 and cuDNN (pre-installed).
- **Dataset Files:**
  - `yelp_academic_dataset_business.json`: Yelp business data.
  - `yelp_academic_dataset_review.json`: Yelp review data.
  - `yelp_reviews.csv`: Converted CSV dataset (included in repo).
- **Internet:** Required for downloading models and dependencies.

## Setup Instructions
Setting up the project in `E:\CuisineInsight Q&A` as follows:

1. **Create a Virtual Environment:**
   ```bash
   python -m venv venv
   .\venv\Scripts\Activate.ps1  # Windows
   ```
   Isolates dependencies to prevent conflicts.

2. **Install Python Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   Installs packages listed in `requirements.txt` (see Dependencies).

3. **Install PyTorch with CUDA Support:**
   
   ([Pytorch](https://pytorch.org/get-started/locally/))
   
   Installs PyTorch for CUDA 12.6, enabling GPU acceleration. Note: PyTorch is not in `requirements.txt` due to platform-specific requirements.

4. **Install Language Models:**
   ```bash
   ollama pull llama3.2:3b
   ollama pull nomic-embed-text:v1.5
   ```
   Downloads `llama3.2:3b` (~4 GB VRAM) for Q&A and `nomic-embed-text:v1.5` (~0.3 GB) for embeddings. Install `ollama` first (see [Ollama](https://ollama.ai/)).

5. **Verify Dataset Files:**
   - Ensure `yelp_academic_dataset_business.json`, `yelp_academic_dataset_review.json`, and `yelp_reviews.csv` are in the project root.
   - If `yelp_reviews.csv` is missing, run `preprocess_yelp.py` to convert JSONs to CSV (see How It Works).

6. **Verify GPU Setup:**
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```
   Should output `True`, confirming CUDA 12.6 and cuDNN functionality.

## Project Structure
```
E:\CuisineInsight Q&A\
├── app.py                          # Flask web app for browser queries
├── evaluate.py                     # Evaluates sentiment and retrieval metrics
├── main.py                         # CLI for interactive Q&A
├── Monitor_cuda.py                 # Monitors GPU VRAM usage
├── preprocess_yelp.py              # Converts JSON datasets to CSV
├── vector.py                       # Builds Chroma vector store
├── templates/
│   └── index.html                  # Flask HTML template
├── yelp_academic_dataset_business.json  # Yelp business data
├── yelp_academic_dataset_review.json   # Yelp review data
├── yelp_reviews.csv                # Converted dataset (10,000+ reviews)
├── requirements.txt                # Python dependencies (excludes PyTorch)
├── screenshots/                    # Output screenshots
│   ├── flask_precision_0.6.png
│   ├── flask_precision_0.8.png
│   ├── flask_precision_1.0.png
│   ├── ollama_list.png
│   ├── app_running.png
│   ├── vector_main_running.png
│   ├── monitor_cuda.png
│   └── nvidia_smi.png
└── README.markdown                 # Project documentation
```

- **Scripts:**
  - `app.py`: Runs Flask server for web-based Q&A.
  - `evaluate.py`: Computes `Sentiment Accuracy` (~0.91) and `Precision@5` (~0.6-1.0).
  - `main.py`: CLI for testing queries (e.g., “Best gluten-free pizza?”).
  - `Monitor_cuda.py`: Tracks VRAM usage with `pynvml`.
  - `preprocess_yelp.py`: Converts JSON datasets to `yelp_reviews.csv`.
  - `vector.py`: Creates Chroma vector store with embeddings.
- **Templates:** `index.html` renders the Flask web interface.
- **Data:**
  - `yelp_academic_dataset_business.json`: Business details (e.g., restaurant names).
  - `yelp_academic_dataset_review.json`: Raw review data.
  - `yelp_reviews.csv`: Processed reviews with `Title`, `Review`, `Rating`, `Date`.
- **Screenshots:** Visual outputs in `screenshots/` (see Output Screenshots).

## How It Works
CuisineInsight Q&A processes Yelp reviews to answer queries using a RAG pipeline. Here’s the workflow:

1. **Dataset Preprocessing:**
   - `preprocess_yelp.py` reads `yelp_academic_dataset_business.json` and `yelp_academic_dataset_review.json`.
   - Filters restaurant reviews, merges fields, and creates `yelp_reviews.csv` with columns: `Title`, `Review`, `Rating` (1-5), `Date`.
   - Labels sentiments (positive if rating ≥3, negative otherwise).
   - Removes null entries and standardizes text.

2. **Sentiment Analysis:**
   - Uses `scikit-learn`’s `TfidfVectorizer` (max 5000 features) to transform reviews.
   - Trains a logistic regression model (80-20 train-test split), achieving 91% accuracy.
   - Saves model and vectorizer as `sentiment_classifier.pkl` and `tfidf_vectorizer.pkl`.
   - Stores sentiments in review metadata.

3. **Vector Store Creation:**
   - `vector.py` embeds reviews using `nomic-embed-text:v1.5` via `langchain-ollama`.
   - Builds a Chroma database (`./chroma_db`) with batched upserts (5000 documents per batch) to avoid Chroma’s 5461 limit.
   - Stores metadata: `rating`, `date`, `sentiment`.
   - Outputs: `Sentiment Accuracy: 0.91`, processing logs (e.g., “Processed 9000 documents”).

4. **Query Retrieval:**
   - Queries are embedded with `nomic-embed-text:v1.5`.
   - Chroma retrieves top-5 relevant reviews (`k=5`, similarity search).

5. **Answer Generation:**
   - `llama3.2:3b` processes reviews and query via a `ChatPromptTemplate`.
   - Generates answers with sentiment analysis (e.g., “Slice has positive reviews for gluten-free pizza”).

6. **Evaluation:**
   - `precision_at_k` computes Precision@5 (~0.6-0.8) using query-specific keywords (e.g., `gluten-free`, `pizza`) and positive sentiment (rating ≥3.5).
   - Debug output logs relevance (e.g., `Query-Specific: True | Positive: True`).

7. **Web Interface:**
   - `app.py` runs a Flask server with `index.html`.
   - Displays answers, sentiment summary, and Precision@5.

8. **GPU Optimization:**
   - PyTorch/CUDA accelerates `llama3.2:3b` and `nomic-embed-text:v1.5`.
   - `torch.cuda.empty_cache()` and `gc.collect()` manage ~4-5 GB VRAM.
   - Pre-installed CUDA 12.6 and cuDNN ensure GPU compatibility.

## Running the Project
Ensure setup is complete before proceeding.

1. **Build the Vector Store:**
   ```bash
   python vector.py
   ```
   - Processes `yelp_reviews.csv`, creates `./chroma_db`.
   - Outputs: `Sentiment Accuracy: 0.91`, batch logs (e.g., “Upserted batch 2”).

2. **Run the CLI Interface:**
   ```bash
   python main.py
   ```
   - Prompts for queries (e.g., “What’s the best pizza with gluten-free options?”).
   - Outputs: `Retrieval Precision@5` (~0.6-0.8), answer, sentiment analysis.
   - Type `q` to exit, freeing GPU memory.

3. **Run the Flask Web App:**
   ```bash
   python app.py
   ```
   - Starts server at `http://127.0.0.1:5000`.
   - Submit queries via the web form, view results with precision and sentiment.
   - Debug mode (`debug=True`) aids development.

4. **Evaluate Performance:**
   ```bash
   python evaluate.py
   ```
   - Outputs: `Sentiment Accuracy: ~0.91`, `Retrieval Precision@5: ~0.6-1.0`.
   - Tests predefined queries.

## Monitoring GPU Usage
- **Script:** `Monitor_cuda.py` uses `pynvml` to track VRAM.
- **Run:**
  ```bash
  python Monitor_cuda.py
  ```
  - Shows real-time VRAM usage (~4-5 GB).
- **Alternative:**
  ```bash
  nvidia-smi
  ```
  - Displays GPU utilization.
- **Purpose:** Ensures efficient resource use on RTX 3060.

## Dependencies
- **Python Packages** (in `requirements.txt`):
  ```
  langchain
  langchain-ollama
  langchain-chroma
  pandas
  pynvml
  flask
  scikit-learn
  ```
  - Install with `pip install -r requirements.txt`.
- **PyTorch:**
  - Install separately: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126`.
  - Supports CUDA 12.6.
- **Language Models:**
  - `llama3.2:3b`: ~4 GB VRAM, for Q&A.
  - `nomic-embed-text:v1.5`: ~0.3 GB VRAM, for embeddings.
  - Install via `ollama pull`.
- **System Drivers:**
  - Pre-installed NVIDIA CUDA Toolkit 12.6 and cuDNN.
- **Tools:**
  - `ollama` for model management.

## Dataset
- **Files:**
  - `yelp_academic_dataset_business.json`: Business details (e.g., restaurant names, categories).
  - `yelp_academic_dataset_review.json`: Raw review data (text, ratings, dates).
  - `yelp_reviews.csv`: Processed dataset with 10,000+ reviews.
- **Source:**
  - [Yelp Open Dataset (kaggle)](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset/data).
- **Details:**
  - `yelp_reviews.csv` columns: `Title`, `Review`, `Rating` (1-5), `Date`.
  - Generated by `preprocess_yelp.py` from JSONs, filtering for restaurants.
- **Preprocessing:**
  - `preprocess_yelp.py` removes nulls, standardizes formats, labels sentiments (positive if rating ≥3).
- **Usage:**
  - All files in repo root; `vector.py` uses `yelp_reviews.csv`.

## Output Screenshots
The following screenshots demonstrate the project’s functionality in `screenshots/`.
- **Flask Web Outputs:**
  - `screenshots/flask_precision_0.6.png`: Shows query, answer, and Precision@5=0.6.
  - `screenshots/flask_precision_0.8.png`: Shows query, answer, and Precision@5=0.8.
  - `screenshots/flask_precision_1.0.png`: Shows query, answer, and Precision@5=1.0.
- **Ollama Models:**
  - `screenshots/ollama_list.png`: `ollama list` output, showing `llama3.2:3b` and `nomic-embed-text:v1.5`.
- **Flask App Running:**
  - `screenshots/app_running.png`: PowerShell (venv) output of `python app.py`, showing Flask server running.
- **Vector and Main Scripts:**
  - `screenshots/vector_main_running.png`: PowerShell (venv) output of `python vector.py` (batch upserts, accuracy) and `python main.py` (query results).
- **GPU Monitoring:**
  - `screenshots/monitor_cuda.png`: PowerShell (venv) output of `python Monitor_cuda.py`, showing VRAM.
  - `screenshots/nvidia_smi.png`: Command terminal output of `nvidia-smi`, showing GPU utilization.
