"""
RoBERTa-base fine-tuned for PCL binary classification (SemEval 2022 Task 4.1).

Proposed approach (for CW Exercise 3):
  - Oversample PCL in training so the model sees more positive examples each epoch.
  - Freeze RoBERTa for the first 2 epochs and train only the classification head;
    then unfreeze with a lower backbone LR (1e-5) for full fine-tuning.
  - Mild class weight for PCL; higher LR for the head; early stopping and best
    checkpoint; then tune decision threshold on the dev set to maximise PCL F1.

Expected files in the parent directory (../):
  dontpatronizeme_pcl.tsv          - full labelled dataset
  train_semeval_parids-labels.csv  - official train split par_ids
  dev_semeval_parids-labels.csv    - official dev split par_ids
  task4_test.tsv                   - official test set (no labels), same TSV format

Outputs written to this directory:
  best_model.pt  - best model state (by dev F1 at t=0.5)
  dev.txt        - one prediction (0/1) per line for the dev set
  test.txt       - one prediction (0/1) per line for the test set
"""

import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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
BATCH_SIZE  = 8
EPOCHS      = 15
LR          = 2e-5
WARMUP_FRAC = 0.1       # fraction of total steps used for LR warmup
EARLY_STOPPING_PATIENCE = 5   # stop if tuned dev F1 does not improve for this many epochs
USE_CLASS_WEIGHTS       = True   # mild weight for PCL
MILD_PCL_WEIGHT         = 2.5    # weight for PCL class (slightly higher to push toward 0.48)
FREEZE_BACKBONE_EPOCHS  = 2      # train only head for this many epochs, then unfreeze
LR_HEAD                 = 2e-4   # higher LR for classifier head
PCL_OVERSAMPLE_FACTOR   = 5      # oversample PCL so model sees more positives (1=no oversample)
BEST_CKPT_PATH          = os.path.join(os.path.dirname(__file__), 'best_model.pt')
# GPU: use CUDA if available (install PyTorch with CUDA: https://pytorch.org/get-started/locally/)
_cuda_available = torch.cuda.is_available()
DEVICE = torch.device('cuda' if _cuda_available else 'cpu')
if _cuda_available:
    print(f"Using GPU: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})")
else:
    print("WARNING: CUDA not available. Using CPU. Install PyTorch with CUDA to use your GPU:")
    print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
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

# Oversample PCL so the model sees more positive examples each epoch
train_labels_np = np.array(train_labels)
sample_weights = np.where(train_labels_np == 1, PCL_OVERSAMPLE_FACTOR, 1.0).astype(np.float64)
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(train_dataset),
    replacement=True,
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
dev_loader   = DataLoader(dev_dataset,   batch_size=BATCH_SIZE)


# ── Class weights ─────────────────────────────────────────────────────────────
n_total = len(train_labels)
n_pcl   = sum(train_labels)
n_npcl  = n_total - n_pcl
if USE_CLASS_WEIGHTS:
    # Mild fixed weight so the model doesn't collapse to "all No-PCL"
    w0, w1 = 1.0, MILD_PCL_WEIGHT
    class_weights = torch.tensor([w0, w1], dtype=torch.float).to(DEVICE)
    print(f"\nClass weights  →  No-PCL: {w0:.1f}  |  PCL: {w1:.1f}  (mild)")
else:
    class_weights = None
    print("\nClass weights: none")


# ── Model ─────────────────────────────────────────────────────────────────────
model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(DEVICE)

# Freeze backbone for first epoch(s) so the head learns on frozen features
for p in model.roberta.parameters():
    p.requires_grad = False

loss_fn = nn.CrossEntropyLoss(weight=class_weights)
# Head-only first; we'll add backbone params to optimizer after unfreezing
optimizer = AdamW(model.classifier.parameters(), lr=LR_HEAD, weight_decay=0.01)

# Scheduler will be recreated after unfreezing (different param count)
total_steps_full = len(train_loader) * EPOCHS
warmup_steps_full = int(total_steps_full * WARMUP_FRAC)
scheduler = None  # set after unfreeze


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


def tuned_f1(probs, labels):
    """Best PCL F1 over threshold in [0.2, 0.8]."""
    best_f1, _ = 0.0, 0.5
    for t in np.arange(0.2, 0.81, 0.01):
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
    return best_f1


print("\nTraining...")
best_tuned_f1 = 0.0
epochs_no_improve = 0
for epoch in range(1, EPOCHS + 1):
    # After freeze period: unfreeze backbone and switch to full fine-tuning
    if epoch == FREEZE_BACKBONE_EPOCHS + 1:
        for p in model.roberta.parameters():
            p.requires_grad = True
        optimizer = AdamW([
            {"params": model.roberta.parameters(), "lr": 1e-5},   # lower to avoid forgetting
            {"params": model.classifier.parameters(), "lr": LR_HEAD},
        ], weight_decay=0.01)
        steps_remaining = len(train_loader) * (EPOCHS - FREEZE_BACKBONE_EPOCHS)
        scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps_remaining), steps_remaining)
        print(f"  → Unfroze backbone; full fine-tuning for remaining {EPOCHS - FREEZE_BACKBONE_EPOCHS} epochs.")

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
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    f1_at_05, dev_probs_epoch, dev_true_epoch = evaluate(dev_loader)
    tuned_f1_epoch = tuned_f1(dev_probs_epoch, dev_true_epoch)
    pos_probs = dev_probs_epoch[dev_true_epoch == 1]
    neg_probs = dev_probs_epoch[dev_true_epoch == 0]
    mean_p_pos = float(np.mean(pos_probs)) if len(pos_probs) else 0.0
    mean_p_neg = float(np.mean(neg_probs)) if len(neg_probs) else 0.0
    print(f"  Epoch {epoch}/{EPOCHS}  |  loss={avg_loss:.4f}  |  dev F1 (t=0.5)={f1_at_05:.4f}  |  tuned F1={tuned_f1_epoch:.4f}  |  mean P(PCL): pos={mean_p_pos:.3f} neg={mean_p_neg:.3f}")

    if tuned_f1_epoch >= best_tuned_f1:
        best_tuned_f1 = tuned_f1_epoch
        epochs_no_improve = 0
        torch.save(model.state_dict(), BEST_CKPT_PATH)
        print(f"             → saved best checkpoint (tuned F1={tuned_f1_epoch:.4f})")
    else:
        epochs_no_improve += 1
    if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
        print(f"  Early stopping after {epoch} epochs (no improvement for {EARLY_STOPPING_PATIENCE} epochs).")
        break

# Load best model for threshold tuning and predictions
if os.path.isfile(BEST_CKPT_PATH):
    print(f"\nLoading best checkpoint (best tuned F1 = {best_tuned_f1:.4f})...")
    model.load_state_dict(torch.load(BEST_CKPT_PATH, map_location=DEVICE))
else:
    print("\nNo checkpoint saved; using final epoch weights.")


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
if best_t <= 0.15 or best_t >= 0.85:
    print("  (Warning: extreme threshold may indicate poor calibration; consider more epochs or different class weights.)")

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
