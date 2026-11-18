# Sarcasm-Aware Sentiment Analysis Using BERT and Multi-Task Learning

A two-phase NLP project exploring how contextual language models and multi-task learning can improve sentiment classification when dealing with sarcasm, irony and ambiguous language.

## Project Overview

Traditional sentiment analysis models can struggle with expressions where the literal meaning differs from the intended sentiment. Sarcasm, irony and contemporary slang can make positive and negative language difficult to distinguish.

This project investigates these limitations through two approaches:

- **Phase 1:** A baseline sentiment classifier fine-tuned using `bert-base-cased`.
- **Phase 2:** A sarcasm-aware multi-task BERT model that jointly learns sentiment and sarcasm classification, incorporating contrastive loss to improve the learned representation space.

The project also includes an ablation experiment to examine the contribution of contrastive learning.

## Objectives

- Develop a BERT-based binary sentiment classification model.
- Investigate the limitations of conventional sentiment classification on pragmatic language.
- Extend the baseline using multi-task learning for sentiment and sarcasm detection.
- Apply contrastive loss to the shared BERT representation.
- Evaluate the contribution of contrastive learning through an ablation study.

## Model Architecture

### Phase 1 – Baseline BERT Sentiment Classifier

A pretrained `bert-base-cased` model is fine-tuned on 100,000 tweets from the Sentiment140 dataset.

**Classification categories:**
- Positive
- Negative

The baseline achieved 82% accuracy on both validation and held-out test data.

Qualitative testing highlighted difficulties with contemporary slang and expressions whose sentiment depends on context.

### Phase 2 – Sarcasm-Aware Multi-Task BERT

The baseline is extended into a multi-task architecture with:

- A shared BERT encoder.
- A sentiment classification head.
- A sarcasm classification head.
- A contrastive loss component operating on the shared `[CLS]` embeddings.

Both classification tasks share the contextual representation produced by BERT, while their separate output heads learn task-specific predictions.

**Combined training objective:**

`Total Loss = Sentiment Loss + Sarcasm Loss + 0.5 × Contrastive Loss`

The sentiment loss ignores samples without reliable sentiment annotations using `ignore_index=-100`.

### Contrastive Learning

A batch-level pairwise contrastive loss is applied to the shared BERT embeddings.

- Samples with the same sarcasm label are encouraged to have similar representations.
- Samples with different sarcasm labels are penalised when their cosine similarity exceeds a margin of 0.3.

This encourages the shared representation space to capture distinctions between sarcastic and non-sarcastic language.

## Datasets

The multi-task model combines three datasets:

| Dataset | Purpose |
|---|---|
| Sentiment140 | Sentiment classification |
| Sarcasm News Headlines | Sarcasm detection |
| SemEval-2018 Twitter Irony (TweetEval) | Twitter irony detection |

**Combined dataset:** 131,481 rows.

An important distinction in the data pipeline is that the sarcasm and irony datasets do not receive fabricated sentiment labels. Their sentiment labels are assigned a placeholder value of `-100` and excluded from sentiment loss and evaluation.

### Data Preparation

- Text tokenisation using the BERT tokenizer.
- Maximum sequence length of 128 tokens.
- Train, validation and test split.
- Stratification using sarcasm labels.
- Sarcastic-class upsampling applied only to the training split.
- Validation and test sets retain their natural class distributions.

The split is performed before upsampling to prevent duplicated samples from leaking into validation or test data.

## Experimental Results

### Test Performance

| Model | Sentiment Accuracy | Sarcasm Accuracy |
|---|---:|---:|
| Baseline BERT | 82% | — |
| Multi-task BERT with contrastive loss | 83% | 97% |
| Multi-task BERT without contrastive loss | 81% | 97% |

### Ablation Study

An ablation experiment was conducted by removing the contrastive loss term while keeping the architecture, datasets and training hyperparameters otherwise unchanged.

Key observations:

- Sentiment accuracy increased from 81% to 83% with contrastive loss.
- Sarcastic-class recall increased from 0.89 to 0.92.
- Sarcastic-class precision decreased from 0.87 to 0.84.
- Sarcastic-class F1-score remained at 0.88.

The results suggest that contrastive learning influenced the model's classification behaviour, particularly its recall for sarcastic samples.

**Important:** These results are from a single-seed experiment. The observed two-percentage-point sentiment accuracy difference should not be interpreted as statistically conclusive without repeated runs and uncertainty estimates.

## Technology Stack

- **Programming Language:** Python
- **Deep Learning:** PyTorch
- **NLP / Transformers:** Hugging Face Transformers
- **Pretrained Model:** BERT (`bert-base-cased`)
- **Dataset Handling:** Hugging Face Datasets, Pandas, NumPy
- **Machine Learning Utilities:** Scikit-learn
- **Environment:** Kaggle Notebooks
- **Hardware:** NVIDIA GPU (Tesla T4 in the recorded run)

## Repository Structure

```text
bert-sarcasm-aware-sentiment-analysis/
│
├── README.md
├── sentiment-analysis-bert-multitask.ipynb
├── requirements.txt
├── .gitignore
│
└── screenshots/
    ├── baseline-results.png
    └── multitask-results.png
```

## Installation and Execution

The notebook was developed and executed in Kaggle.

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/bert-sarcasm-aware-sentiment-analysis.git

cd bert-sarcasm-aware-sentiment-analysis
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Open the notebook

Import `sentiment-analysis-bert-multitask.ipynb` into Kaggle Notebooks.

### 4. Configure the dataset

Attach the Sentiment140 dataset used in the notebook:

`krishbaisoya/tweets-sentiment-analysis`

The notebook currently references a Kaggle input path. If your attached dataset uses a different path, update `train_path` accordingly.

The Sarcasm News Headlines and TweetEval datasets are loaded through Hugging Face Datasets.

### 5. Enable GPU acceleration

In Kaggle Notebook settings, select a GPU accelerator.

### 6. Execute the notebook

Run the cells sequentially to train and evaluate the baseline and multi-task models.

## Limitations and Future Improvements

- Evaluate performance across multiple random seeds.
- Introduce confidence intervals and statistical significance testing.
- Investigate cross-domain generalisation between news headlines and conversational text.
- Explore additional datasets containing contemporary slang, sarcasm and irony.
- Experiment with alternative transformer architectures.
- Investigate more advanced contrastive learning strategies.

## Author

**Tania Marietta Anto**  
M.Sc. Artificial Intelligence  
Dublin Business School

---

This project was completed as part of my MSc Artificial Intelligence dissertation.

**Supervisor:** Dr. David Williams
