# Sentiment Analysis of Public Opinion on Social Media Platform X Regarding the Free Nutritious Meal (MBG) Program Using BERT–CNN

> **Analisis Sentimen Opini Publik di Platform Media Sosial X Mengenai Program Makan Bergizi Gratis (MBG) Menggunakan BERT–CNN**

This repository contains the full research pipeline, datasets, and experimental
results for an undergraduate thesis (_skripsi_) that classifies Indonesian public
sentiment toward the government's **Free Nutritious Meal Program**
(_Program Makan Bergizi Gratis_, **MBG**) on **X (formerly Twitter)**.
The proposed model is a **dual-path IndoBERT + 1D-CNN** architecture that fuses the
global `[CLS]` sentence representation with local n-gram features.

- **Author:** Khoeru Roziqin
- **Affiliation:** Universitas Diponegoro (UNDIP)
- **Task:** 3-class sentiment classification — `positive` / `negative` / `neutral`
- **Language:** Indonesian (`lang:id`)
- **Proposed model:** IndoBERT (`indobenchmark/indobert-base-p2`) + Dual-Path CNN 1D

---

## Table of Contents

1. [Key Results](#key-results)
2. [Research Pipeline](#research-pipeline)
3. [Repository Structure](#repository-structure)
4. [Dataset](#dataset)
5. [Preprocessing & Labeling](#preprocessing--labeling)
6. [Proposed Model — IndoBERT + CNN (Dual-Path)](#proposed-model--indobert--cnn-dual-path)
7. [Experimental Setup](#experimental-setup)
8. [Results](#results)
9. [Reproducibility](#reproducibility)
10. [Ethics & Data Statement](#ethics--data-statement)
11. [Citation](#citation)
12. [License](#license)

---

## Key Results

Final evaluation on a **fixed, held-out test set of 1,329 tweets** (20% of the
labeled corpus), evaluated **once** with the proposed dual-path model
(condition **S2 — random undersampling**):

| Metric            | Score      |
| ----------------- | ---------- |
| **Accuracy**      | **85.70%** |
| **F1-macro**      | **0.8547** |
| F1-weighted       | 0.8587     |
| Precision (macro) | 0.8551     |
| Recall (macro)    | 0.8585     |

**Per-class F1:** `positive` = 0.889 · `negative` = 0.872 · `neutral` = 0.803

5-fold cross-validation on train+val (S2): **F1-macro = 0.846 ± 0.016**.

---

## Research Pipeline

```
                    ┌─────────────────────────────────────────────┐
  00_crawling  ───▶ │ Crawl X via tweet-harvest 2.7.1             │
                    │ keywords: "makan bergizi gratis" + "mbg"    │
                    │ window: 6 Jan 2025 – 6 Jan 2026 · lang:id   │
                    └─────────────────────────────────────────────┘
                                       │  172,009 raw tweets
                                       ▼
                    ┌─────────────────────────────────────────────┐
  01_data_prep ───▶ │ Merge · timestamp normalization ·           │
                    │ layered deduplication · relevance filter ·  │
                    │ stratified sampling (570 × 12 periods)      │
                    └─────────────────────────────────────────────┘
                                       │  ~6,840 sampled
                                       ▼
                    ┌─────────────────────────────────────────────┐
  02_preproc   ───▶ │ BERT-oriented preprocessing (4 steps) +     │
     & labeling     │ IndoRoBERTa pseudo-labeling + manual        │
                    │ annotation (ground truth) + Cohen's κ (IAA) │
                    └─────────────────────────────────────────────┘
                                       │  6,642 labeled tweets
                                       ▼
                    ┌─────────────────────────────────────────────┐
  03_training  ───▶ │ IndoBERT + Dual-Path CNN · 3-phase grid     │
                    │ search · 5-fold CV · final test evaluation  │
                    └─────────────────────────────────────────────┘
                                       │
                                       ▼
                            Accuracy 85.70% · F1-macro 0.855
```

Each stage is a self-contained Jupyter notebook under [`research/pipeline/`](research/pipeline/),
with figures and metrics written to the matching folder under [`research/results/`](research/results/).

---

## Repository Structure

```
mbg_bert_cnn/
├── dataset/
│   ├── raw/                     # raw crawl output (git-ignored; 70 + 40 CSV batches)
│   │   ├── makan_bergizi_gratis/
│   │   └── mbg/
│   ├── merge/                   # merged & validated corpora (git-ignored)
│   └── final/
│       ├── raw_sampling_mbg.csv       # sampled tweets (pre-label)
│       ├── raw_sample_labeled_mbg.csv # manual annotation working file
│       ├── final_validated_mbg.csv    # validated labels + metadata
│       └── final_mbg_labeled.csv      # ★ final labeled corpus (6,642 rows)
│
├── preprocessing/kamus/         # custom Indonesian lexicons (dictionaries)
│   ├── kamus_alay_mbg.csv             # slang/colloquial normalization (7,742)
│   ├── demoji_code_mbg.csv            # emoji → Indonesian phrase (285)
│   ├── akun_x_mbg.csv                 # @mention → real name mapping (90)
│   ├── whitelist_hashtag_mbg.csv      # relevant hashtags to keep (329)
│   └── additional_stopwords_mbg.csv   # loaded for completeness (358, unused in BERT path)
│
├── research/
│   ├── pipeline/                # ★ the four ordered pipeline notebooks
│   │   ├── 00_crawling_data_x.ipynb
│   │   ├── 01_data_preparation.ipynb
│   │   ├── 02_preprocessing_and_labeling.ipynb
│   │   └── 03_indobert_cnn_training.ipynb
│   ├── helper/                  # auxiliary EDA / annotation-check notebooks
│   └── results/
│       ├── 00_exploration/            # keyword, sentiment & token distributions
│       ├── 01_data_preparation/       # temporal trends & sampling distributions
│       ├── 02_preprocessing_labeling/ # confidence distribution & IAA figure
│       └── 03_model_results/          # phase logs, CV metrics, final model & CM
│
├── LICENSE                      # The Unlicense (public domain)
└── README.md
```

> **Note.** Raw/merged CSVs and model checkpoints (`*.pt`) are excluded from version
> control via [`.gitignore`](.gitignore) due to size and platform terms of service.
> The final labeled corpus and all result artifacts (metrics, logs, figures) are tracked.

---

## Dataset

### Collection

Tweets were harvested with **[tweet-harvest](https://github.com/helmisatria/tweet-harvest) v2.7.1**
(Helmi Satria) using two complementary Indonesian keywords, restricted to
`lang:id` and the **LATEST** tab, over a full year of the program's rollout.

| Keyword                | Purpose                                              |
| ---------------------- | ---------------------------------------------------- |
| `makan bergizi gratis` | full program name (formal references)                |
| `mbg`                  | official acronym (informal / high-volume references) |

- **Collection window:** 6 January 2025 – 6 January 2026
- **Batches:** 70 CSV files (`makan bergizi gratis`) + 40 CSV files (`mbg`)

### Data funnel

| Stage                            |      Tweets | Notes                                                    |
| -------------------------------- | ----------: | -------------------------------------------------------- |
| Raw crawl                        | **172,009** | 70,075 (`makan bergizi gratis`) + 101,934 (`mbg`)        |
| Merge & validation               | **118,340** | after schema alignment & timestamp normalization         |
| Deduplication & relevance filter | **~53,669** | layered dedup + null/short/irrelevant removal            |
| Stratified sampling              |   **6,840** | 570 tweets × 12 monthly periods (500 target + 70 buffer) |
| **Final labeled corpus**         |   **6,642** | after preprocessing post-filter & annotation             |

### Sampling strategy

An **equal-allocation stratified sampling** design partitions the year into
**12 monthly periods** (anchored to the 6th of each month) and draws **570 tweets per
period** — **500** for statistical adequacy (margin of error ≤ 4.4% at α = 0.05;
Cochran, 1977) plus a **70-tweet (14%) curation buffer** for tweets eliminated during
annotation (Babbie, 2010; Pustejovsky & Stubbs, 2012). Low-volume periods use partial
census. This guarantees temporal balance so that no single month dominates the corpus.

### Label distribution (final corpus, n = 6,642)

| Label    | Count | Share |
| -------- | ----: | ----: |
| positive | 2,625 | 39.5% |
| negative | 2,085 | 31.4% |
| neutral  | 1,932 | 29.1% |

The corpus is split **80/20** into train+val (**5,313**) and a **fixed test set
(1,329)** with stratification (`seed = 42`); the test set is touched exactly once.

---

## Preprocessing & Labeling

### BERT-oriented preprocessing (`02_preprocessing_and_labeling.ipynb`)

A **single-path** pipeline produces `text_bert`, the input for every model in the study:

| Step | Stage         | Operations                                                                                          |
| ---- | ------------- | --------------------------------------------------------------------------------------------------- |
| 1    | Case Folding  | lowercasing + HTML-entity unescaping                                                                |
| 2    | Cleaning      | strip URLs, mentions, currency; emoji → phrase; hashtag whitelist handling; character normalization |
| 3    | Tokenization  | whitespace split (parsing helper)                                                                   |
| 4    | Normalization | slang/`alay` dictionary, negation handling, phrase deduplication → **output `text_bert`**           |

> **No stemming or stopword removal.** IndoBERT uses a WordPiece subword tokenizer
> trained on natural Indonesian text; stemming produces OOV surface forms and stopword
> removal deletes syntactic cues (e.g. removing the negation _"tidak"_ can flip
> sentiment), creating a distribution shift from pre-training. Aggressive normalization
> is therefore deliberately avoided.

### Two-annotator labeling with IAA

Ground-truth `label` is assigned by **manual human annotation** by the researcher.
A second, machine annotator — **IndoRoBERTa**
(`w11wo/indonesian-roberta-base-sentiment-classifier`) — provides `label_roberta`
and a `confidence_roberta` score, serving as a _pseudo-annotator_ to quantify
labeling reliability.

**Inter-Annotator Agreement — Cohen's κ = 0.644** (observed agreement 76.3%), which
falls in the **substantial agreement** band (0.61–0.80; Landis & Koch, 1977), meeting
the study's minimum reliability target. Low-confidence predictions
(`confidence < 0.60`, 364 tweets) were prioritized for manual review.

> IndoRoBERTa labels are **never** used as ground truth — only for the κ computation.

---

## Proposed Model — IndoBERT + CNN (Dual-Path)

The proposed classifier fuses two complementary views of IndoBERT's
`last_hidden_state` `[batch, 128, 768]`:

```
IndoBERT → last_hidden_state [B, 128, 768]
                 │
     ┌───────────┴───────────┐
     ▼                       ▼
  Path 1: [CLS] token     Path 2: CNN 1D multi-kernel
  [B, 768]                Conv1D(k=1,2,3) → GlobalMaxPool
  global context          [B, filter × n_kernels]
  Dropout(0.1)            local n-gram patterns
     │                       │
     └────── Concatenate ────┘
             [B, 768 + filter × n_kernels]
                     ▼
        Dropout → Dense(256) → Dropout → Dense(3)
```

- **Path 1 — `[CLS]` token** captures the global sentence context aggregated across
  12 self-attention layers (Devlin et al., 2019).
- **Path 2 — CNN 1D multi-kernel** captures local n-gram sentiment cues
  (e.g. _"tidak bagus"_, _"sangat buruk"_) in parallel across kernel widths (Kim, 2014).
- **Concatenation** lets the CNN _complement_ rather than replace `[CLS]` (Chen et al., 2020).

Key training techniques: **differential learning rates** (small LR for the pre-trained
encoder to avoid catastrophic forgetting, larger LR for the from-scratch CNN + head;
Howard & Ruder, 2018) and **weight-decay exclusion** for `bias` / `LayerNorm`
parameters (Loshchilov & Hutter, 2019).

---

## Experimental Setup

Training (`03_indobert_cnn_training.ipynb`) runs a **three-phase grid search** with
**5-fold stratified cross-validation**, optimizing **F1-macro**. Base config:
`max_len = 128`, `max_epochs = 10`, `patience = 3`, `warmup_ratio = 0.1`, `seed = 42`.

| Phase                        | Search space                                                               | Runs               |
| ---------------------------- | -------------------------------------------------------------------------- | ------------------ |
| **1 — Differential LR**      | `lr_bert ∈ {1e-5, 2e-5, 3e-5}` · `lr_cnn ∈ {1e-4, 1e-3}` · `bs ∈ {16, 32}` | 12 × 5 = 60        |
| **2A — CNN, n-gram [1,2,3]** | `filter ∈ {128,256}` · `dropout ∈ {0.3,0.5}` · `act ∈ {relu,gelu,elu}`     | 12 × 5 = 60        |
| **2B — CNN, n-gram [2,3,4]** | same grid, wider kernels                                                   | 12 × 5 = 60        |
| **3 — Data condition**       | S1 (imbalanced + class weighting) vs. S2 (random undersampling)            | 2 × 5 = 10 + final |

**Selected best configuration:**

| Hyperparameter            | Value                                   |
| ------------------------- | --------------------------------------- |
| Encoder                   | `indobenchmark/indobert-base-p2`        |
| `lr_bert` / `lr_cnn`      | `1e-5` / `1e-4`                         |
| batch size / weight decay | `32` / `0.01` (non-bias/LayerNorm only) |
| CNN n-grams / filters     | `[1, 2, 3]` / `256`                     |
| CNN dropout / activation  | `0.5` / `elu`                           |
| Data condition            | **S2 — random undersampling**           |

---

## Results

### Phase 3 — data condition (5-fold CV)

| Condition                         |   Accuracy |           F1-macro | F1-weighted |
| --------------------------------- | ---------: | -----------------: | ----------: |
| S1 — Imbalanced + class weighting |     0.8423 |     0.8394 ± 0.014 |      0.8432 |
| **S2 — Random undersampling**     | **0.8487** | **0.8461 ± 0.016** |  **0.8494** |

### Final held-out test set (n = 1,329, evaluated once)

| Class         | Precision | Recall |        F1 | Support |
| ------------- | --------: | -----: | --------: | ------: |
| positive      |      0.94 |   0.84 | **0.889** |     525 |
| negative      |      0.86 |   0.88 | **0.872** |     417 |
| neutral       |      0.76 |   0.86 | **0.803** |     387 |
| **Accuracy**  |           |        | **0.857** |   1,329 |
| **Macro avg** |     0.855 |  0.859 | **0.855** |   1,329 |

`neutral` is the hardest class (lowest F1), consistent with its greater semantic
ambiguity in social-media text. Confusion matrices, learning curves, LR-sensitivity
plots, and K-fold stability figures are in
[`research/results/03_model_results/`](research/results/03_model_results/).

Selected result figures:

| Figure                 | Path                                                                                                                                  |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| Sentiment distribution | [`00_exploration/chart_distribusi_sentimen.png`](research/results/00_exploration/chart_distribusi_sentimen.png)                       |
| Sampling per period    | [`01_data_preparation/distribusi_sampling_per_periode.png`](research/results/01_data_preparation/distribusi_sampling_per_periode.png) |
| IAA (Cohen's κ)        | [`02_preprocessing_labeling/iaa_result.png`](research/results/02_preprocessing_labeling/iaa_result.png)                               |
| S1 vs S2 comparison    | [`03_model_results/model_results/phase3_s1_vs_s2.png`](research/results/03_model_results/model_results/phase3_s1_vs_s2.png)           |
| Final confusion matrix | [`03_model_results/model_results/proposed_S2_cm_pair.png`](research/results/03_model_results/model_results/proposed_S2_cm_pair.png)   |

---

## Reproducibility

### Environment

| Notebook                         | Platform            | Why                                                                                                                    |
| -------------------------------- | ------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `00_crawling_data_x.ipynb`       | **Google Colab**    | `tweet-harvest` (Node.js + Playwright) is set up for the Colab runtime; needs an X auth token.                         |
| `01`–`03` (data prep → training) | **Kaggle Notebook** | uniform `/kaggle/input` (read-only) → `/kaggle/working` (output) file management; GPU **Tesla T4** (15.6 GB) for `03`. |

The three Kaggle notebooks follow one **consistent file convention**: inputs are read
from `/kaggle/input/<dataset-slug>/` and every artifact is written to
`/kaggle/working/`. Because Kaggle resets `/kaggle/working/` each session, each stage's
output is published as a Kaggle Dataset and attached as the next stage's input. Full
training spans ~4 sessions (~27–28 GPU-hours across the three search phases).

**Core dependencies:** `torch`, `transformers`, `scikit-learn`, `pandas`, `numpy`,
`matplotlib`, `seaborn`, `tqdm` (Python 3.10+; preinstalled on the Kaggle image).
Crawling additionally requires Node.js 20 and `tweet-harvest@2.7.1` (Colab).

**Run order & dataset chaining:**

| #   | Notebook (platform)                      | Input dataset(s)                      | Key output → next dataset                                         |
| --- | ---------------------------------------- | ------------------------------------- | ----------------------------------------------------------------- |
| 0   | `00_crawling_data_x` (Colab)             | X auth token                          | raw CSV batches → `mbg-raw-tweets`                                |
| 1   | `01_data_preparation` (Kaggle)           | `mbg-raw-tweets`                      | `mbg_sampled.csv` → _(manual annotation)_ → `mbg-sampled`         |
| 2   | `02_preprocessing_and_labeling` (Kaggle) | `mbg-sampled`, `mbg-kamus`            | `final_mbg_labeled.csv` → _(manual `label` fill)_ → `mbg-labeled` |
| 3   | `03_indobert_cnn_training` (Kaggle)      | `mbg-labeled`, `mbg-training-outputs` | `saved_models/indobert_cnn_dualpath_S2.pt`                        |

> The `mbg-kamus` dataset holds the five lexicons in [`preprocessing/kamus/`](preprocessing/kamus/).
> Enable **Internet: On** in Kaggle Settings so `02`/`03` can download the IndoRoBERTa /
> IndoBERT weights and the Nasal colloquial lexicon.

Determinism is enforced with a fixed `seed = 42` across `random`, `numpy`, `torch`,
CUDA, and the stratified train/test split; the test-set indices are persisted
(`test_set_indices.npy`) so the held-out set is stable across sessions.

> The auth token shown in the crawling notebook is a **redacted placeholder** — supply
> your own and never commit real credentials.

---

## Ethics & Data Statement

- Data originate from **publicly available** posts on X, collected solely for
  **academic, non-commercial research** in accordance with the platform's terms.
- Analysis is performed **in aggregate**; no attempt is made to profile or identify
  individual users. Users referenced in this README should treat usernames/handles as
  incidental metadata, not study subjects.
- Automatic (IndoRoBERTa) labels are advisory only; the **human annotations are the
  ground truth**. Known limitations include domain shift (IndoRoBERTa is not
  MBG-specific) and difficulty with sarcasm/irony in informal Indonesian text.
- Redistribution of full raw tweet content may be subject to platform policy; hence raw
  corpora are git-ignored and only derived, labeled artifacts are shared here.

---

## Citation

If you use this repository, please cite the thesis:

```bibtex
@thesis{roziqin2026mbgbertcnn,
  author = {Khoeru Roziqin},
  title  = {Analisis Sentimen Opini Publik di Platform Media Sosial X
            Mengenai Program Makan Bergizi Gratis Menggunakan BERT-CNN},
  school = {Universitas Diponegoro},
  year   = {2026},
  type   = {Undergraduate Thesis (Skripsi)}
}
```

**Key references**

- Devlin et al. (2019). _BERT: Pre-training of Deep Bidirectional Transformers._ NAACL.
- Kim (2014). _Convolutional Neural Networks for Sentence Classification._ EMNLP.
- Chen et al. (2020). _Combining BERT with CNN for text classification._
- Howard & Ruder (2018). _Universal Language Model Fine-tuning (ULMFiT)._ ACL.
- Loshchilov & Hutter (2019). _Decoupled Weight Decay Regularization (AdamW)._ ICLR.
- Liu et al. (2019). _RoBERTa: A Robustly Optimized BERT Pretraining Approach._ arXiv:1907.11692.
- Cohen (1960); Landis & Koch (1977). _Inter-annotator agreement / Cohen's Kappa._
- Cochran (1977); Babbie (2010); Pustejovsky & Stubbs (2012). _Sampling & annotation methodology._

Models used: [`indobenchmark/indobert-base-p2`](https://huggingface.co/indobenchmark/indobert-base-p2),
[`w11wo/indonesian-roberta-base-sentiment-classifier`](https://huggingface.co/w11wo/indonesian-roberta-base-sentiment-classifier).

---

## License

Released into the public domain under **[The Unlicense](LICENSE)**. Note that the
underlying tweet data and third-party pre-trained models are governed by their
respective platform terms and model licenses.
