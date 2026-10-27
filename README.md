# Which Words Give Away an AI Model?
### Text classification of Claude-generated passages with TF-IDF

Can we tell which AI model wrote a paragraph from its text alone? This project uses text mining and machine learning to identify which of **five Claude models** wrote a passage, and finds the words that give each model away.

---

## Research questions

1. Can the model that wrote a passage be identified from its text alone?
2. Which words give away each AI model?

## Dataset

**[Name That LLM – Extra API Passages](https://www.kaggle.com/datasets/seyitkaangunes/ntl-extra-api-passages)** (Kaggle, CC BY 4.0)

- **4,998** single-paragraph passages
- **5 balanced classes:** `claude-fable-5`, `claude-opus-4-6`, `claude-opus-4-8`, `claude-opus-5`, `claude-sonnet-5`
- Every model answered the **same 1,000 prompts**, so the differences between passages come from writing style, not topic.

The CSV is not included in this repository. Download `ntl_extra_passages_claude.csv` from Kaggle and place it in the project folder.

## Methodology

1. **Exploratory data analysis:** class balance, text length, domains, most frequent words, example passages.
2. **Text preprocessing:** missing and duplicate check, lower-casing, removal of URLs and numbers. Stop words are **kept**, because function words carry style information.
3. **Feature extraction:** TF-IDF with single words and two-word phrases (20,000 features).
4. **Grouped train/test split:** `GroupShuffleSplit` by `prompt_id` (80/20). All five answers to a prompt stay in the same set, so the models are tested on unseen topics and must learn **style**, not topic.
5. **Models:** Multinomial Naive Bayes, Logistic Regression and Linear SVM.
6. **Evaluation:** accuracy, precision, recall, F1-score and confusion matrices. Also `GroupKFold` cross-validation with TF-IDF inside a `Pipeline` (to avoid data leakage) and `GridSearchCV` tuning.
7. **Interpretation:** Logistic Regression coefficients (top words per model) and the most confused model pairs.

## Results

| Model | CV accuracy | Test accuracy | Test macro F1 |
|---|---|---|---|
| **Logistic Regression** (final) | **42.9%** | 44.9% | **0.445** |
| Linear SVM | 42.4% | **45.4%** | 0.441 |
| Linear SVM (tuned) | – | 45.0% | 0.437 |
| Multinomial Naive Bayes | 37.8% | 42.6% | 0.397 |
| *Random baseline* | *20%* | *20%* | *0.20* |

- All models are **more than twice as good as random guessing**.
- **claude-opus-5** is the easiest model to recognise; **claude-fable-5** and **claude-opus-4-8** are the hardest.
- Tuning confirmed that **keeping stop words** and **using two-word phrases** give the best results.

### Which words give each model away?

| Model | Typical words | Style |
|---|---|---|
| claude-opus-5 | *so, so the, and the, nothing, nobody, went* | Plain, story-like |
| claude-opus-4-6 | *approximately, however, such as, known as* | Formal, explanatory |
| claude-opus-4-8 | *thus, this, these, here, called, you* | Academic, addresses the reader |
| claude-sonnet-5 | *since, just, even, but, doesn't, isn't* | Conversational |
| claude-fable-5 | *while, remarkably, strangely, cannot* | Literary |

**Key finding:** each model's "fingerprint" lies in small **connector and function words**, not in topic words.

## Repository structure

```
├── llm_text_classification.ipynb   # full analysis notebook
├── llm_project.pdf                 # written report
├── README.md
└── ntl_extra_passages_claude.csv   # download from Kaggle (not included)
```

## How to run

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install pandas numpy matplotlib seaborn scikit-learn nltk
jupyter notebook llm_text_classification.ipynb
```

Run the cells from top to bottom. Results can differ slightly between library versions.

## Limitations and future work

- More than half of the passages are still misclassified: models from the same family write very similarly.
- TF-IDF ignores punctuation and sentence structure, although punctuation differs clearly between the models.
- Possible improvements: punctuation and sentence-length features, character n-grams, transformer models (e.g. BERT), and adding other AI providers or human-written texts.

## References

- Güneş, S. K. (2026). *Name That LLM – Extra API Passages* [Dataset]. Kaggle. CC BY 4.0.
- Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.

## Author

**Magamed Imamguliev** – Data Mining and Analytics project
