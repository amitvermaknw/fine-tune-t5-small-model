# Fine-Tune T5-Small for Article Summarization 📝

A personal project to fine-tune Google's T5-Small model for automatic article summarization using the CNN/DailyMail dataset. Training is performed on **Google Vertex AI** due to the compute requirements of fine-tuning transformer models.

---

## Overview

This project fine-tunes the `t5-small` model to summarize news articles. Given a full article as input, the model generates a concise summary (highlights). The project uses Hugging Face `transformers`, and `datasets`.

---

## Project Structure

```
fine-tune-t5-small-model/
├── src/
│   ├── const.py              # Constants (model name, max lengths etc.)
│   ├── load_dataset.py       # Dataset loading utility
│   ├── tokenizer.py          # Tokenization and preprocessing
│   ├── train.py              # Training logic
│   └── main.py           
├── requirements.txt
└── .gitignore
```

---

## How It Works

```
Raw Article (CNN/DailyMail)
        ↓
Add prefix: "summarize: <article>"
        ↓
Tokenize input + labels (highlights)
        ↓
Fine-tune T5-Small
        ↓
Evaluate with ROUGE score (pending)
        ↓
Summarized Output
```

---

## Dataset

Uses the **CNN/DailyMail** dataset from Hugging Face:
- `article` → input to the model
- `highlights` → target summary (labels)
- Split into `train` and `validation` sets

---

## Model

- **Base Model:** `google-t5/t5-small`
- **Task:** Text Summarization (Seq2Seq)
- **Framework:** Hugging Face Transformers
- **Prefix:** `"summarize: "` prepended to each article (T5 task format)

---

## Key Parameters

| Parameter | Value |
|---|---|
| Model | t5-small |
| Max Input Length | defined in `Constant.MAX_INPUT_LEN` |
| Max Output Length | defined in `Constant.MAX_OUTPUT_LEN` |
| Padding | max_length |
| Truncation | True |
| Evaluation Metric | ROUGE |

---

## Tech Stack

- **Model:** Hugging Face Transformers (`t5-small`)
- **Dataset:** `datasets` (CNN/DailyMail)
- **Evaluation:** `evaluate`, `rouge_score`, `nltk` (Penidng work)
- **Training:** `accelerate`, PyTorch
- **Platform:** Google Vertex AI (cloud training)
- **Other:** `sentencepiece`, `numpy`, `pandas`

---

## Installation

```bash
# Clone the repo
git clone https://github.com/amitvermaknw/fine-tune-t5-small-model.git
cd fine-tune-t5-small-model

# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Training on Google Vertex AI

Local training is not feasible for transformer fine-tuning due to compute requirements. This project is trained on **Google Vertex AI**.

```bash
# Submit training job to Vertex AI
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=t5-summarization \
  --python-package-uris=gs://your-bucket/trainer.tar.gz \
  --python-module=src.train
```

---

## Usage

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

model = T5ForConditionalGeneration.from_pretrained("path/to/fine-tuned-model")
tokenizer = T5Tokenizer.from_pretrained("path/to/fine-tuned-model")

article = "Your long news article here..."
input_text = f"summarize: {article}"

inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
outputs = model.generate(**inputs, max_length=150)

summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(summary)
```

---

## Evaluation

Model is evaluated using **ROUGE scores**:
- `ROUGE-1` — unigram overlap
- `ROUGE-2` — bigram overlap
- `ROUGE-L` — longest common subsequence

---

## Why Vertex AI?

Fine-tuning transformer models requires significant GPU compute. Local Mac/CPU training is extremely slow for models like T5. Google Vertex AI provides:
- On-demand GPU/TPU instances
- Scalable training jobs
- No local hardware limitations

---

## Future Improvements

- [ ] Try larger model (`t5-base`, `t5-large`)
- [ ] Experiment with different datasets
- [ ] Deploy fine-tuned model as API endpoint on Vertex AI
- [ ] Add beam search for better summary quality
- [ ] Compare with other summarization models (BART, Pegasus)

---