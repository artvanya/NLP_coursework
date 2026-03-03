# PCL detection (NLP coursework)

Coursework for the SemEval 2022 Task 4 (subtask 1): binary classification of patronising and condescending language (PCL) in news text. Baseline was RoBERTa-base at 0.48 dev / 0.49 test F1.

## What's in this repo

| What | Where |
|------|--------|
| **Best model + predictions** | [**BestModel/**](BestModel/) — [final_model_roberta.pt](BestModel/final_model_roberta.pt) (weights), [dev.txt](BestModel/dev.txt), [test.txt](BestModel/test.txt) |
| **Training code** | [pcl_roberta_improved.ipynb](pcl_roberta_improved.ipynb) — full pipeline (data load, training, threshold tuning, saving predictions) |
| **Report** | [doc.md](doc.md) — LaTeX source for the coursework report (compile to PDF for submission) |
| **EDA** | [eda.py](eda.py) — script that generates the two EDA figures; outputs go into [figures/](figures/) |
| **Data** | [**data/**](data/) — `dontpatronizeme_pcl.tsv`, `train_semeval_parids-labels.csv`, `dev_semeval_parids-labels.csv`, `task4_test.tsv` (train/dev split IDs and main data; test set has no labels) |

## What I changed vs the baseline

I kept RoBERTa-base and added: (1) **class-weighted BCE** (pos_weight 9.55) to handle the 1:9.5 PCL/No-PCL imbalance; (2) **auxiliary loss** on the 7 PCL categories (weight 0.3) so the model learns framing, not just keywords; (3) **threshold search** on dev in [0.30, 0.70] and **two-stage training** (train on train only → pick best epoch and threshold → retrain on train+dev, then predict). Dev F1 after tuning was 0.615 in my run.

## Replicating the results

1. **Data:** The TSV and CSV files in the repo are the same as the task’s train/dev split and unlabelled test set. They live in the [data/](data/) folder; the notebook and eda.py read from there.
2. **Environment:** Python 3, PyTorch, `transformers`, pandas, scikit-learn, tqdm. Install with e.g. `pip install torch transformers pandas scikit-learn tqdm`.
3. **Training:** Open [pcl_roberta_improved.ipynb](pcl_roberta_improved.ipynb) and run all cells. It will create `BestModel/` (best checkpoint from stage 1, then final model from stage 2), write `dev.txt` and `test.txt`, and print the chosen threshold and dev F1.
4. **Using the saved model only:** Load `final_model_roberta.pt` into a `RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=8)` and apply the same tokenization and threshold (e.g. 0.69) as in the notebook. The notebook has the exact prediction loop.

EDA figures for the report: run `python eda.py` from the repo root; it writes into [figures/](figures/) only (no graphs in the repo root). Compile `doc.md` with the `figures/` subfolder next to it.
