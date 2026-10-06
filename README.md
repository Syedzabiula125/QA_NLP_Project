# NLP Question Answering Project

A production-ready Question Answering application using the Hugging Face Transformers `pipeline()` API. It includes a Streamlit interface, command-line entry point, GPU/CPU detection, batch processing, CSV exports, logging, and Matplotlib analytics.

## Model compatibility

The default is `Jacaranda/YorubaLlama`. The application first runs the requested API:

```python
pipeline("question-answering", model="Jacaranda/YorubaLlama")
```

The model is access-gated on Hugging Face. To use it, request/receive access in a browser and supply a personal read/fine-grained token through the sidebar or `HF_TOKEN` in `.env`. The application automatically switches to the public `deepset/roberta-base-squad2` extractive QA model if access is unavailable, preserving confidence scores and keeping the app usable. Some Llama checkpoints are decoder-only causal-language models rather than extractive QA models. If access is approved but an extractive QA head is unavailable, the application instead switches to `pipeline("text-generation")` with a context-grounded extraction prompt. In that generative mode confidence is `N/A`, because it does not provide calibrated extractive-QA scores.

`YorubaLlama` is an 8B-parameter model, so a GPU with sufficient VRAM is recommended. For a smaller confidence-scored extractive model, set `QA_MODEL_NAME=deepset/roberta-base-squad2` in `.env` or change it in the sidebar.

## Features

- Single and batch QA with answer extraction, score, and inference time
- CUDA detection through PyTorch plus TensorFlow GPU environment reporting
- PyTorch-forced inference prevents Keras 3 / TensorFlow backend conflicts
- Text-file and editable custom contexts with token-aware length limiting
- CSV download/persistence, timestamped logs, and per-question error isolation
- Bar chart, histogram, line chart, pie chart, and box plot for scores/latency
- Save supported pipeline artifacts under `models/` and reload by local path
- Clean modular design, type hints, docstrings, PEP 8 formatting, and exceptions

## Installation

Use Python 3.10+ in Visual Studio Code.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

For NVIDIA acceleration, install the PyTorch wheel matching the installed CUDA version from the official PyTorch selector. An optional `.env` file:

```env
QA_MODEL_NAME=Jacaranda/YorubaLlama
HF_TOKEN=your_huggingface_token_if_required
QA_FALLBACK_MODEL_NAME=deepset/roberta-base-squad2
MAX_CONTEXT_LENGTH=2048
LOG_LEVEL=INFO
```

### Required Transformers version

The installed `transformers` package must be **4.57.6**. Transformers 5.x no longer exposes the extractive `question-answering` pipeline task, so it cannot satisfy this project's required Pipeline API. If an older installation already has 5.x, run:

```powershell
pip install --upgrade --force-reinstall "transformers==4.57.6"
```

Then restart Streamlit. PyPI lists 4.57.6 as the final stable 4.x release.

## Usage

Start the UI with `streamlit run app.py`; load the model in the sidebar, add context/questions, then download the result CSV. Run the CLI:

```powershell
python main.py --question "What does the pipeline API do?"
python main.py --batch-file questions.txt --context-file sample_context.txt
python main.py --save-model
```

To reload saved compatible artifacts, pass `--model models/<model-folder>` or enter that path in the Streamlit model field.

## Folder structure

```text
Question_Answering_Project/
├── app.py                 # Streamlit application
├── main.py                # CLI entry point
├── qa_pipeline.py         # Pipeline service and fallback
├── utils.py               # Logging, CSV, charts, utilities
├── config.py              # Settings and paths
├── requirements.txt
├── sample_context.txt
├── assets/                # Screenshot/static asset location
├── models/                # Saved artifacts (gitignored)
└── outputs/               # Result CSVs and qa_application.log
```

## Example output

```json
{
  "question": "What does the pipeline receive?",
  "answer": "a question and a context",
  "confidence": 0.91,
  "processing_time_seconds": 0.1832,
  "mode": "extractive question-answering"
}
```

## Screenshots

Add UI captures to `assets/dashboard.png` and `assets/visualizations.png`.

## Troubleshooting

- **Download/authentication error:** check internet access, permissions, and `HF_TOKEN`.
- **CUDA memory error:** use CPU, lower `MAX_CONTEXT_LENGTH`, or choose a smaller model.
- **No GPU:** CPU inference remains supported; device details appear in the sidebar.
- **Invalid QA head:** the app logs the cause and tries generative extraction; select an extractive QA model if scores are mandatory.
- **File missing:** check the context/batch path and UTF-8 text content.

## Future enhancements

- Retrieval-augmented QA over document collections
- Asynchronous batches and persistent results storage
- Evaluation datasets, automated tests, quality dashboards, and secure deployment
