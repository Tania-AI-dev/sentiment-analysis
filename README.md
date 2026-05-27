# Sarcasm-Aware Sentiment Analysis with Multi-Task BERT

A two-phase NLP system for sentiment classification that goes beyond standard approaches by jointly detecting **sarcasm and irony**. Built on a custom `MultiTaskBERT` architecture with contrastive loss, trained on ~230,000 samples across three datasets.

> Developed as part of an MSc dissertation in Artificial Intelligence at Dublin Business School, supervised by Dr. David Williams.

---

## Problem Statement

Standard sentiment classifiers fail on pragmatic language. A model that has never encountered sarcasm will confidently label *"I just love being ignored"* as **Positive** — because the words are positive, even though the meaning is not.

This project addresses that gap through a two-phase approach:

- **Phase 1** establishes a strong baseline by fine-tuning `bert-base-cased` on 100,000 tweets, achieving 84% validation accuracy — and exposing its weakness on sarcastic and ironic text.
- **Phase 2** introduces a `MultiTaskBERT` model that jointly predicts sentiment and detects sarcasm using a shared BERT encoder, two task-specific classification heads, and a contrastive loss term that pushes sarcastic and sincere embeddings apart in representation space.

---

## Model Architecture

### Phase 1 — Baseline Sentiment Model

- Pretrained model: `bert-base-cased`
- Task: Binary sentiment classification (Positive / Negative)
- Dataset: Sentiment140 (100,000 tweets sampled from 1.6M)
- Training: 3 epochs, AdamW (lr=2e-5), batch size 16
- Loss: CrossEntropyLoss
- Validation accuracy: **84%**
- Training loss: 0.40 → 0.29 → 0.18

**Limitation identified:** The model misclassifies sarcastic text — e.g. *"Best day ever... not"* and *"I just love being ignored"* are both predicted Positive.

---

### Phase 2 — Sarcasm-Aware Multi-Task Model

A custom `MultiTaskBERT` architecture with:

- **Shared encoder:** `bert-base-cased` (frozen subword tokenisation, [CLS] token representation)
- **Sentiment head:** Linear classifier → Positive / Negative
- **Sarcasm head:** Linear classifier → Sarcastic / Not Sarcastic
- **Dropout:** 0.3 applied to [CLS] output before both heads

**Combined training loss:**

```
total_loss = sentiment_loss + sarcasm_loss + 0.5 × contrastive_loss
```

The contrastive loss encourages the model to cluster sarcastic examples together and push them away from sincere examples in embedding space, using cosine similarity between the two heads' logit vectors.

**Training data:** ~230,000 rows (after upsampling sarcastic class to balance classes)

---

## Datasets

| Dataset | Source | Size | Labels |
|---|---|---|---|
| Sentiment140 | Stanford / HuggingFace | 100,000 tweets (sampled) | Positive, Negative |
| Sarcasm News Headlines | `raquiba/Sarcasm_News_Headline` (HuggingFace) | ~26,000 headlines | Sarcastic, Not Sarcastic |
| SemEval-2018 Twitter Irony | `cardiffnlp/tweet_eval` (irony split) | ~3,800 tweets | Ironic, Not Ironic |
| **Combined (balanced)** | — | **~230,000 rows** | Sentiment + Sarcasm |

The sarcastic class was upsampled using `sklearn.utils.resample` to match the size of the non-sarcastic class before training.

---

## Tech Stack

- Python
- PyTorch
- HuggingFace Transformers
- HuggingFace Datasets
- Scikit-learn
- Pandas
- Google Colab (GPU runtime)

---

## Results

### Phase 1 — Baseline

| Metric | Value |
|---|---|
| Validation Accuracy | **84%** |
| Training Loss (Epoch 1) | 0.40 |
| Training Loss (Epoch 2) | 0.29 |
| Training Loss (Epoch 3) | 0.18 |

### Phase 2 — Multi-Task Model

*(Replace with your actual evaluation output from Section 2.7)*

**Sentiment Head:**

| Metric | Negative | Positive |
|---|---|---|
| Precision | XX | XX |
| Recall | XX | XX |
| F1 Score | XX | XX |

**Sarcasm Head:**

| Metric | Not Sarcastic | Sarcastic |
|---|---|---|
| Precision | XX | XX |
| Recall | XX | XX |
| F1 Score | XX | XX |

---

## Sample Predictions

### Phase 1 — Baseline Model

```
Input:     "I just love being ignored."
Sentiment: Positive  ← incorrect (sarcasm missed)

Input:     "Best day ever... not."
Sentiment: Positive  ← incorrect (negation missed)
```

### Phase 2 — Multi-Task Model

```
Input:     "I just love being ignored."
Sentiment: Negative
Sarcasm:   Sarcastic  ✓

Input:     "I absolutely love waiting in traffic."
Sentiment: Negative
Sarcasm:   Sarcastic  ✓

Input:     "This is the best day of my life."
Sentiment: Positive
Sarcasm:   Not Sarcastic  ✓
```

---

## How to Run

This project runs as a Jupyter notebook on Google Colab with a GPU runtime.

1. Open the notebook in [Google Colab](https://colab.research.google.com/)
2. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU)
3. Run all cells sequentially
4. Phase 1 and Phase 2 are self-contained — you can run either independently

> **Note:** The full training pipeline (~230,000 rows, 3 epochs) takes approximately 2–3 hours on a T4 GPU. To test quickly, reduce `balanced_df` to a smaller sample before building the DataLoaders.

---

## Limitations

- **Cross-domain sarcasm:** The sarcasm head was trained on news headlines and tweets. Performance on conversational sarcasm from other domains (e.g. Reddit, product reviews) is likely to degrade.
- **Binary sentiment only:** The model classifies Positive/Negative and does not handle neutral or fine-grained emotion.
- **Label proxy for sarcasm→sentiment:** Sarcastic examples were assigned an inverted sentiment label as a proxy (sarcastic = negative sentiment). This is a simplification and may introduce label noise.
- **No cross-lingual support:** The model is English-only.

---

## Future Improvements

- Deploy as a REST API using FastAPI or Streamlit
- Experiment with `roberta-base` or `distilbert-base-cased` for speed/accuracy tradeoffs
- Fine-tune on Reddit or product review sarcasm data for better cross-domain generalisation
- Add neutral sentiment class
- Convert notebook into a modular ML pipeline with separate `train.py`, `evaluate.py`, and `predict.py` scripts

---

## Reproducibility

All experiments were run on Google Colab with a T4 GPU.

**To reproduce:**

1. Open the notebook in Colab and set runtime to GPU
2. Install dependencies:
```bash
   pip install transformers torch scikit-learn pandas datasets
```
3. Run all cells sequentially — Phase 1 and Phase 2 are self-contained

**Seeds used:**
- `random_state=42` in all `train_test_split` and `resample` calls (data splitting and upsampling)
- `df.sample(100_000, random_state=42)` for Sentiment140 sampling

**Note:** `torch.manual_seed()` is not set in this project, so minor variation in training loss across GPU runs is expected.

notebooks/
src/
models/
data/

## Author

**Tania Marietta Anto**  
MSc in Artificial Intelligence — Dublin Business School  
BA in Computer Applications  

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/tania-marietta-277344301/)  
📧 taniamarietta16@gmail.com
