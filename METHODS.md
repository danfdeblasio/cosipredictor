# Methods

## Overview

We developed a supervised text classifier that assigns a scientific submission to one of 21 Communities of Special Interest (COSIs) of the ISMB conference, given any combination of its title, keywords and abstract. The classifier outputs a probability for every COSI, so that a submission can be ranked against all candidate tracks rather than assigned a single label. Documents are represented with field-specific TF-IDF (term frequency–inverse document frequency) features, and COSI membership is modeled with multinomial logistic regression. We evaluated the model prospectively on the most recent conference year, compared it against dense scientific-document embeddings, and deployed it as a web service that also accepts English Wikipedia articles as input.

## Data

### Source

We used the records of accepted submissions to eight editions of ISMB, 2019–2026, including the joint ISMB/ECCB meetings in odd years. Each record contained the submission title, author list, track name, keywords, review decision and abstract. Records from other ISCB-affiliated meetings (regional conferences and partner meetings) were also available. Their track names did not correspond to any COSI, so they were removed by the label filter described below and did not contribute to training or evaluation.

### Label harmonization

Track names were recorded inconsistently across years. We mapped them to a fixed set of 21 canonical COSI labels: 3DSIG, BioInfo-Core, BioVis, BOKR, BOSC, CAMDA, CompMS, CSI, Education, EvolCompGen, Function, HiTSeq, iRNA, MICROBIOME, MLCSB, NetBio, RegSys, SysMod, TextMining, TransMed and VarI. Each raw track name was normalized before matching:

1. Remove a trailing "COSI" suffix (e.g. "MLCSB COSI" → "MLCSB").
2. Convert to lowercase and remove all non-alphanumeric characters (e.g. "HitSeq" and "HiTSeq" → "hitseq"; "Text Mining" → "textmining").

We also merged COSIs that had been renamed or absorbed into successor COSIs:

| Historical track name | Canonical COSI |
|---|---|
| RNA COSI (2019) | iRNA |
| Function/CAFA 4 (2020) | Function |
| Bio-Ontologies | BOKR (Bio-Ontologies and Knowledge Representation) |

Records whose normalized track name matched no canonical COSI were excluded. These included general-track, special-session, keynote, tutorial, panel and poster-only tracks, as well as all tracks of non-ISMB meetings. Special sessions were excluded even when they shared a COSI's topic, for example the 2019 "Text Mining (Special Session)".

### Filtering and deduplication

Starting from 14,678 records in 21 files:

1. We removed 273 records with a reject decision (these appeared only in the 2020 export). All other decision values were retained, including talk, poster and "probably accept".
2. We removed 5,069 records whose track did not map to a COSI.
3. Of the remaining 9,336 records, we removed 233 duplicates that shared both title and COSI label. These were typically the same submission listed under more than one presentation type.

The final dataset contained **9,103 labeled submissions** (Table 1).

**Table 1.** Labeled submissions by conference year.

| Year | Meeting | Submissions |
|---|---|---|
| 2019 | ISMB/ECCB | 1,134 |
| 2020 | ISMB | 875 |
| 2021 | ISMB/ECCB | 769 |
| 2022 | ISMB | 926 |
| 2023 | ISMB/ECCB | 1,439 |
| 2024 | ISMB | 1,091 |
| 2025 | ISMB/ECCB | 1,589 |
| 2026 | ISMB | 1,280 |
| **Total** | | **9,103** |

The classes were imbalanced. MLCSB was largest (1,483 submissions, 16.3%) and BioInfo-Core smallest (70, 0.8%); the full distribution is given in Table S1. Titles had a median length of 11 words and abstracts a median of 199 words. Keywords were missing for 12.9% of submissions, almost all of them from 2019 (84.1% missing) and part of the 2020 export. Missing fields were treated as empty strings.

## Text representation

Each submission was represented by three separate TF-IDF vectors, one each for the title, the keywords and the abstract. These were concatenated into a single feature vector. Keeping the fields separate lets the classifier give different weights to the same term depending on where it appears.

### Preprocessing

For each field, text was:

1. converted to lowercase;
2. stripped of diacritics, using Unicode compatibility decomposition (NFKD) followed by removal of combining marks;
3. split into tokens with the regular expression `\b\w\w+\b`, i.e. runs of two or more Unicode word characters, so punctuation and single characters were discarded;
4. filtered to remove the 318 words of the scikit-learn English stop-word list.

Unigrams and bigrams were then formed from the remaining tokens. Multi-line keyword lists were joined into a single string before tokenization.

### Vocabulary and weighting

A term was retained if it occurred in at least two training documents within its field. For abstracts, terms occurring in more than 50% of training documents were also discarded. For term $t$ in document $d$, the weight was the sublinear (log-scaled) term frequency multiplied by the smoothed inverse document frequency:

$$
w_{t,d} = \left(1 + \ln \mathrm{tf}_{t,d}\right) \cdot \left( \ln \frac{1 + n}{1 + \mathrm{df}_t} + 1 \right),
$$

where $n$ is the number of training documents and $\mathrm{df}_t$ is the number of those documents containing $t$. Each field vector was then scaled to unit $\ell_2$ norm, so that field length did not dominate the representation. The final model's vocabulary contained 172,809 features: 12,503 from titles, 11,391 from keywords and 148,915 from abstracts.

## Classification model

We modeled the probability of COSI $k$ given feature vector $\mathbf{x}$ with multinomial (softmax) logistic regression:

$$
P(k \mid \mathbf{x}) = \frac{\exp(\boldsymbol{\beta}_k^\top \mathbf{x} + b_k)}{\sum_{j=1}^{K} \exp(\boldsymbol{\beta}_j^\top \mathbf{x} + b_j)}, \qquad K = 21.
$$

Parameters were fit by minimizing the $\ell_2$-regularized, class-weighted cross-entropy

$$
\tfrac{1}{2}\lVert \mathbf{B} \rVert_F^2 \;-\; C \sum_{i=1}^{n} s_{y_i} \log P(y_i \mid \mathbf{x}_i),
$$

with $C = 10$. The class weights $s_k = n / (K\, n_k)$, where $n_k$ is the number of training submissions in COSI $k$, offset the class imbalance so that small COSIs were not overwhelmed by large ones. Optimization used L-BFGS for up to 2,000 iterations. The feature settings and $C$ were fixed before evaluation and were not tuned.

### Restricting the candidate set

The set of active COSIs varies from year to year. At prediction time, the model can be restricted to a user-specified subset $A$ of COSIs by renormalizing:

$$
P_A(k \mid \mathbf{x}) = \frac{P(k \mid \mathbf{x})}{\sum_{j \in A} P(j \mid \mathbf{x})}, \qquad k \in A.
$$

This is equivalent to applying the softmax over the logits of the COSIs in $A$ only.

## Evaluation

### Protocol

To estimate performance on future submissions, we used a temporal hold-out. The model was trained on all submissions from 2019–2025 (7,823 submissions) and evaluated on the 1,280 submissions from ISMB 2026. The vocabulary, IDF weights and class weights were all computed from the training years only. Predictions were restricted to the COSIs present at ISMB 2026, which were all 21.

We report:
- **top-1 accuracy:** the fraction of submissions whose highest-probability COSI is correct;
- **top-3 and top-5 accuracy:** whether the correct COSI is among the three or five highest-ranked;
- **macro-averaged F1:** the unweighted mean of per-COSI F1 scores.

95% confidence intervals were obtained with a nonparametric bootstrap over test submissions (1,000 resamples, percentile method).

As reference points, we computed two baselines:
- **Majority class:** always predicting the most frequent training COSI (MLCSB).
- **Frequency ranking:** ranking COSIs by their training frequency and checking whether the correct COSI is among the top three.

To check for leakage between years, we counted test titles that also appeared in the training data, comparing titles case-insensitively with punctuation removed.

### Comparison with neural document embeddings

We compared the TF-IDF representation with SPECTER2, a transformer encoder pretrained on scientific literature. We used the `allenai/specter2_base` model with each of two task adapters, `allenai/specter2` (proximity) and `allenai/specter2_classification`.

Each submission was encoded as its title, a separator token, its keywords and its abstract, truncated to 512 tokens. The final-layer [CLS] vector (768 dimensions) was used as the document embedding. The embeddings were standardized and classified with the same class-weighted multinomial logistic regression, with $C \in \{0.001, 0.003, 0.01, 0.03, 0.1, 1\}$. We also evaluated late fusion: the average of the TF-IDF and embedding models' predicted probabilities.

For the embedding models, $C$ was chosen by performance on the hold-out set. Their reported results are therefore optimistic relative to the untuned TF-IDF model.

### Final model

After evaluation, the TF-IDF model was retrained with identical settings on all 9,103 submissions from 2019–2026. This final model was used for deployment.

## Results

### Hold-out performance

On ISMB 2026, the TF-IDF model ranked the correct COSI first for 65.1% of submissions. The correct COSI was within its top three for 91.2% and within its top five for 95.7% (Table 2). Both baselines were far lower: majority-class prediction reached 20.5% top-1 accuracy, and the frequency ranking placed the correct COSI in its top three for 35.4% of submissions.

Only four test titles also appeared in the training data. Excluding them did not change top-1 accuracy (65.2%, n = 1,276).

**Table 2.** Performance on the ISMB 2026 hold-out set (n = 1,280). Brackets show 95% bootstrap confidence intervals.

| Metric | TF-IDF + logistic regression | Majority class | Frequency ranking |
|---|---|---|---|
| Top-1 accuracy | 0.651 [0.626, 0.677] | 0.205 | 0.205 |
| Top-3 accuracy | 0.912 [0.898, 0.927] | — | 0.354 |
| Top-5 accuracy | 0.957 [0.946, 0.967] | — | — |
| Macro F1 | 0.630 [0.602, 0.655] | — | — |

Per-COSI F1 ranged from 0.89 (BioVis, CompMS) to 0.00 (BioInfo-Core, n = 7 test submissions); see Table S2. COSIs with a distinctive technical vocabulary were recognized most reliably, including CompMS, BioVis, Education, MICROBIOME and EvolCompGen. The most frequent errors were between methodologically overlapping COSIs, above all MLCSB and TransMed (32 errors in both directions combined), MLCSB and RegSys, and MLCSB and 3DSIG. CAMDA had high precision (1.00) but low recall (0.21). Its submissions were most often assigned to MLCSB, consistent with CAMDA being organized around data-analysis challenges rather than a methodological or biological domain.

### Calibration

Predicted probabilities were reasonably well calibrated (Table 3). When the top prediction had probability 0.8 or higher, it was correct 80.5% of the time. When the top probability was between 0.2 and 0.4, accuracy was 38.6%.

**Table 3.** Accuracy of the top prediction, binned by its predicted probability (ISMB 2026 hold-out set).

| Top-1 probability | Submissions | Top-1 accuracy |
|---|---|---|
| < 0.2 | 15 | 0.200 |
| 0.2–0.4 | 189 | 0.386 |
| 0.4–0.6 | 311 | 0.576 |
| 0.6–0.8 | 298 | 0.678 |
| ≥ 0.8 | 467 | 0.805 |

### Comparison with SPECTER2 embeddings

The SPECTER2 embeddings performed worse than TF-IDF, even with their regularization chosen on the test set (Table 4). Late fusion also failed to improve on TF-IDF alone. We attribute this to COSI boundaries being defined largely by specific technical terms, such as instrument, data-type and method names. Sparse lexical features keep these terms, whereas a single dense vector does not. In addition, SPECTER2 reads only the first 512 tokens of each submission, so the end of a long abstract is ignored. The TF-IDF model was therefore retained.

**Table 4.** Comparison of document representations on the ISMB 2026 hold-out set.

| Representation | Top-1 | Top-3 | Top-5 | Macro F1 |
|---|---|---|---|---|
| **TF-IDF (title, keywords, abstract)** | **0.651** | **0.912** | **0.957** | **0.630** |
| TF-IDF + SPECTER2 proximity, probabilities averaged | 0.627 | 0.905 | 0.954 | 0.630 |
| SPECTER2 proximity adapter (C = 0.01)ᵃ | 0.591 | 0.868 | 0.945 | 0.612 |
| SPECTER2 classification adapter (C = 0.03)ᵃ | 0.549 | 0.842 | 0.925 | 0.558 |

ᵃ Best of six values of $C$, selected on the hold-out set, so these figures are optimistic. The fusion row averages the TF-IDF model with the proximity model at C = 0.1.

## Application to Wikipedia articles

To place general topics in the COSI landscape, the tool also accepts English Wikipedia articles. Articles are retrieved through the MediaWiki Action API (`action=query`), with redirects resolved automatically. Each article is mapped onto the model's three input fields:

| Model field | Wikipedia source |
|---|---|
| Title | Article title |
| Keywords | The article's short description plus its visible (non-hidden) categories, with the "Category:" prefix removed |
| Abstract | The plain-text lead section (TextExtracts extension, `exintro`) by default, or optionally the whole article |

Hidden maintenance categories are excluded. Disambiguation pages, identified by the `disambiguation` page property, are rejected, as are non-existent pages and articles from non-English editions. These articles are out of domain relative to the training data, and no labeled Wikipedia evaluation set exists. The resulting probabilities should therefore be read as indicative only.

## Implementation and deployment

The model was trained and evaluated in Python 3.9.6 with scikit-learn 1.6.1, pandas 2.2.2 and NumPy 1.26.4. SPECTER2 embeddings were computed with transformers 4.57.6, adapters 1.3.0 and PyTorch 2.8.0.

### Web service

For deployment on standard PHP web hosting, we reimplemented the inference pipeline in PHP 8.3 with no machine-learning dependencies:

1. **Export.** The vocabulary, IDF weights and logistic-regression coefficients of the trained model were exported to an SQLite database. It contains one row per (field, term) pair, holding that term's IDF and its 21 coefficients as single-precision floats.
2. **Scoring.** At query time, only the terms present in the input are looked up. The TF-IDF transformation, $\ell_2$ normalization, linear scoring and softmax are then computed as described above.
3. **Diacritic removal.** Diacritics are removed with a lookup table of 4,554 characters generated from Python's Unicode database. It covers every Basic Multilingual Plane character that NFKD-based removal changes, except precomposed Hangul syllables, and does not require PHP's `intl` extension.

On 303 test documents, the PHP and Python probabilities differed by at most $6.0 \times 10^{-8}$, the rounding error of single-precision coefficient storage, and the top-ranked COSI was identical for every document. Scoring takes approximately 1 ms per document.

The service provides three interfaces:
- **Single submission:** a form for a title, keywords and abstract, returning all COSIs ranked by probability.
- **Wikipedia article:** a lookup of a single article by title or URL.
- **Bulk CSV:** batch processing of an uploaded CSV of abstracts or of Wikipedia URLs. The file is parsed in the browser and scored in batches of 25 abstracts or 5 articles per request. The output is the input table with a probability column appended for every COSI.

Equivalent command-line tools in Python produce identical predictions for CSV files and lists of Wikipedia articles. Submitted text is not stored.

## Limitations

1. **Label noise.** Labels reflect the track under which each submission was accepted. Authors choose a track, and a submission may fit several COSIs, so some labels are arguably ambiguous. The top-3 accuracy (91.2%) may therefore describe practical usefulness better than top-1 accuracy.
2. **Small COSIs.** COSIs with few examples, such as BioInfo-Core (70 training submissions), were predicted poorly.
3. **Changing scope.** COSIs change over time. Merged COSIs (e.g. Bio-Ontologies into BOKR) were harmonized to the current name, but COSIs created recently (e.g. CSI) have fewer training examples.
4. **Leakage.** Resubmitted or closely related work across years could inflate hold-out accuracy. Exact title overlap between training and test years was negligible (4 of 1,280), but near-duplicates were not assessed.
5. **Wikipedia input.** Predictions for Wikipedia articles fall outside the training distribution and have not been validated.

## Supplementary tables

**Table S1.** Number of labeled submissions per COSI, 2019–2026 (n = 9,103).

| COSI | n | COSI | n | COSI | n |
|---|---|---|---|---|---|
| MLCSB | 1,483 | iRNA | 482 | BioVis | 206 |
| 3DSIG | 770 | MICROBIOME | 475 | CAMDA | 168 |
| HiTSeq | 673 | BOSC | 449 | Education | 164 |
| TransMed | 648 | SysMod | 417 | BOKR | 163 |
| RegSys | 629 | VarI | 369 | TextMining | 157 |
| NetBio | 528 | Function | 336 | CSI | 153 |
| EvolCompGen | 514 | CompMS | 249 | BioInfo-Core | 70 |

**Table S2.** Per-COSI performance on the ISMB 2026 hold-out set.

| COSI | Precision | Recall | F1 | Test n |
|---|---|---|---|---|
| 3DSIG | 0.74 | 0.74 | 0.74 | 105 |
| BioInfo-Core | 0.00 | 0.00 | 0.00 | 7 |
| BioVis | 0.94 | 0.84 | 0.89 | 19 |
| BOKR | 0.72 | 0.55 | 0.62 | 33 |
| BOSC | 0.65 | 0.69 | 0.67 | 65 |
| CAMDA | 1.00 | 0.21 | 0.34 | 29 |
| CompMS | 0.89 | 0.89 | 0.89 | 19 |
| CSI | 0.67 | 0.43 | 0.52 | 51 |
| Education | 0.82 | 0.95 | 0.88 | 19 |
| EvolCompGen | 0.77 | 0.84 | 0.80 | 61 |
| Function | 0.63 | 0.59 | 0.61 | 41 |
| HiTSeq | 0.64 | 0.58 | 0.61 | 85 |
| iRNA | 0.56 | 0.76 | 0.64 | 51 |
| MICROBIOME | 0.77 | 0.85 | 0.81 | 68 |
| MLCSB | 0.64 | 0.66 | 0.65 | 263 |
| NetBio | 0.53 | 0.45 | 0.49 | 55 |
| RegSys | 0.55 | 0.62 | 0.58 | 77 |
| SysMod | 0.59 | 0.48 | 0.53 | 60 |
| TextMining | 0.61 | 0.92 | 0.73 | 24 |
| TransMed | 0.56 | 0.63 | 0.59 | 101 |
| VarI | 0.61 | 0.64 | 0.62 | 47 |

## References

- Cohan A, Feldman S, Beltagy I, Downey D, Weld DS. SPECTER: Document-level representation learning using citation-informed transformers. *Proceedings of ACL* 2020:2270–2282.
- Efron B, Tibshirani RJ. *An Introduction to the Bootstrap.* Chapman & Hall; 1993.
- Liu DC, Nocedal J. On the limited memory BFGS method for large scale optimization. *Mathematical Programming* 1989;45:503–528.
- Pedregosa F, et al. Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research* 2011;12:2825–2830.
- Poth C, et al. Adapters: A unified library for parameter-efficient and modular transfer learning. *Proceedings of EMNLP: System Demonstrations* 2023.
- Salton G, Buckley C. Term-weighting approaches in automatic text retrieval. *Information Processing & Management* 1988;24(5):513–523.
- Singh A, D'Arcy M, Cohan A, Downey D, Feldman S. SciRepEval: A multi-format benchmark for scientific document representations. *Proceedings of EMNLP* 2023.

## Code and data availability

[To be completed: repository URL for the training, evaluation and web-service code, and the web service URL, https://cosipredictor.dandeblasio.com.]
