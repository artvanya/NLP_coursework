"""
RoBERTa-base fine-tuned for PCL binary classification.
Approach: class-weighted cross-entropy loss + dev-set threshold tuning.

Expected files in the parent directory (../):
  dontpatronizeme_pcl.tsv          - full labelled dataset
  train_semeval_parids-labels.csv  - official train split par_ids
  dev_semeval_parids-labels.csv    - official dev split par_ids
  task4_test.tsv                   - official test set (no labels), same TSV format

Outputs written to this directory:
  dev.txt   - one prediction (0/1) per line for the dev set
  test.txt  - one prediction (0/1) per line for the test set
"""

import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import f1_score, classification_report

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Config ───────────────────────────────────────────────────────────────────
DATA_DIR    = os.path.join(os.path.dirname(__file__), '..')
MODEL_NAME  = 'roberta-base'
MAX_LENGTH  = 128
BATCH_SIZE  = 16
EPOCHS      = 4
LR          = 2e-5
WARMUP_FRAC = 0.1       # fraction of total steps used for LR warmup
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device: {DEVICE}")


# ── Data loading ─────────────────────────────────────────────────────────────
def load_tsv(path):
    """Parse dontpatronizeme_pcl.tsv → {par_id: (text, label_raw)}."""
    records = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t', 5)
            if len(parts) == 6 and parts[0].strip().isdigit():
                pid  = int(parts[0])
                text = parts[4].strip()
                lbl  = int(parts[5])
                records[pid] = (text, lbl)
    return records


def load_test_tsv(path):
    """Parse the test TSV (no label column, IDs like t_0, t_1) → [(par_id, text)]."""
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t', 5)
            if len(parts) >= 5 and parts[0].strip().startswith('t_'):
                records.append((parts[0].strip(), parts[4].strip()))
    return records


def binarise(label_raw):
    """Official binarisation for Subtask 1: >=2 annotators → PCL."""
    return 1 if label_raw >= 2 else 0


def build_split(par_ids, records):
    texts, labels = [], []
    for pid in par_ids:
        text, lbl_raw = records[pid]
        texts.append(text)
        labels.append(binarise(lbl_raw))
    return texts, labels


print("Loading data...")
records   = load_tsv(os.path.join(DATA_DIR, 'dontpatronizeme_pcl.tsv'))
train_ids = list(pd.read_csv(os.path.join(DATA_DIR, 'train_semeval_parids-labels.csv'))['par_id'])
dev_ids   = list(pd.read_csv(os.path.join(DATA_DIR, 'dev_semeval_parids-labels.csv'))['par_id'])

train_texts, train_labels = build_split(train_ids, records)
dev_texts,   dev_labels   = build_split(dev_ids,   records)

print(f"  Train: {len(train_texts)} samples  |  PCL={sum(train_labels)} ({sum(train_labels)/len(train_labels)*100:.1f}%)")
print(f"  Dev  : {len(dev_texts)}  samples  |  PCL={sum(dev_labels)}  ({sum(dev_labels)/len(dev_labels)*100:.1f}%)")


# ── Dataset / DataLoader ─────────────────────────────────────────────────────
tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)


class PCLDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=MAX_LENGTH,
            return_tensors='pt',
        )
        self.labels = labels

    def __len__(self):
        return self.encodings['input_ids'].shape[0]

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


train_dataset = PCLDataset(train_texts, train_labels)
dev_dataset   = PCLDataset(dev_texts,   dev_labels)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
dev_loader   = DataLoader(dev_dataset,   batch_size=BATCH_SIZE)


# ── Class weights ─────────────────────────────────────────────────────────────
n_total = len(train_labels)
n_pcl   = sum(train_labels)
n_npcl  = n_total - n_pcl
w0 = n_total / (2 * n_npcl)
w1 = n_total / (2 * n_pcl)
class_weights = torch.tensor([w0, w1], dtype=torch.float).to(DEVICE)
print(f"\nClass weights  →  No-PCL: {w0:.4f}  |  PCL: {w1:.4f}")


# ── Model ─────────────────────────────────────────────────────────────────────
model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(DEVICE)

loss_fn   = nn.CrossEntropyLoss(weight=class_weights)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)

total_steps   = len(train_loader) * EPOCHS
warmup_steps  = int(total_steps * WARMUP_FRAC)
scheduler     = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)


# ── Training loop ─────────────────────────────────────────────────────────────
def evaluate(loader, threshold=0.5):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop('labels').to(DEVICE)
            batch  = {k: v.to(DEVICE) for k, v in batch.items()}
            logits = model(**batch).logits
            probs  = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())
    preds = (np.array(all_probs) >= threshold).astype(int)
    f1 = f1_score(all_labels, preds, pos_label=1, zero_division=0)
    return f1, np.array(all_probs), np.array(all_labels)


print("\nTraining...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for batch in train_loader:
        labels = batch.pop('labels').to(DEVICE)
        batch  = {k: v.to(DEVICE) for k, v in batch.items()}
        logits = model(**batch).logits
        loss   = loss_fn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    f1_dev, _, _ = evaluate(dev_loader)
    print(f"  Epoch {epoch}/{EPOCHS}  |  loss={avg_loss:.4f}  |  dev F1 (t=0.5)={f1_dev:.4f}")


# ── Threshold tuning on dev set ───────────────────────────────────────────────
print("\nTuning decision threshold on dev set...")
_, dev_probs, dev_true = evaluate(dev_loader)

best_t, best_f1 = 0.5, 0.0
for t in np.arange(0.1, 0.91, 0.01):
    preds = (dev_probs >= t).astype(int)
    f1 = f1_score(dev_true, preds, pos_label=1, zero_division=0)
    if f1 > best_f1:
        best_f1, best_t = f1, t

print(f"  Best threshold: {best_t:.2f}  |  Dev F1: {best_f1:.4f}")

dev_preds = (dev_probs >= best_t).astype(int)
print("\nDev set classification report:")
print(classification_report(dev_true, dev_preds, target_names=['No-PCL', 'PCL'], digits=4))


# ── Write dev.txt ─────────────────────────────────────────────────────────────
out_dir = os.path.dirname(__file__)
with open(os.path.join(out_dir, 'dev.txt'), 'w') as f:
    for pred in dev_preds:
        f.write(f"{pred}\n")
print(f"Saved dev.txt  ({len(dev_preds)} predictions)")


# ── Test set predictions ──────────────────────────────────────────────────────
test_tsv_path = os.path.join(DATA_DIR, 'task4_test.tsv')
if os.path.exists(test_tsv_path):
    print("\nGenerating test predictions...")
    test_records = load_test_tsv(test_tsv_path)
    test_texts_raw = [t for _, t in test_records]
    test_dataset = PCLDataset(test_texts_raw)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model.eval()
    test_probs = []
    with torch.no_grad():
        for batch in test_loader:
            batch  = {k: v.to(DEVICE) for k, v in batch.items()}
            logits = model(**batch).logits
            probs  = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            test_probs.extend(probs)

    test_preds = (np.array(test_probs) >= best_t).astype(int)
    with open(os.path.join(out_dir, 'test.txt'), 'w') as f:
        for pred in test_preds:
            f.write(f"{pred}\n")
    print(f"Saved test.txt  ({len(test_preds)} predictions)")
else:
    print(f"\nTest TSV not found at {test_tsv_path} — skipping test.txt generation.")
    print("Download the official test set, save it as task4_test.tsv in the data directory, and re-run.")
