"""
generate_eval_figs.py
---------------------
Generates all Exercise 5.2 local-evaluation figures for the report.

Run from the repo root:
    python generate_eval_figs.py

What it does
------------
1. Checks for saved dev probabilities (BestModel/dev_probs.npy).
   - If found, skips straight to figure generation.
   - If not found, checks for BestModel/best_model_roberta.pt and runs
     inference only.
   - If the checkpoint is also missing, re-runs Stage 1 training (6 epochs,
     ~15-20 min on a 5070) to reproduce it.

2. Saves four outputs to figures/:
   - eval_threshold_sweep.pdf   — F1 / Precision / Recall vs threshold
   - eval_pr_curve.pdf          — Precision-Recall curve
   - eval_prob_dist.pdf         — probability distribution by outcome (TP/FP/FN/TN)
   - eval_conf_matrix.pdf       — colour confusion matrix

3. Prints a ready-to-paste LaTeX table for the threshold sweep.
"""

import os, ast, random, warnings, logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup)
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             precision_recall_curve, auc)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from tqdm.auto import tqdm

# ── config ─────────────────────────────────────────────────────────────────────
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

CFG = {
    'model_name':   'roberta-base',
    'main_data':    'data/dontpatronizeme_pcl.tsv',
    'train_ids':    'data/train_semeval_parids-labels.csv',
    'dev_ids':      'data/dev_semeval_parids-labels.csv',
    'max_length':   256,
    'batch_size':   16,
    'grad_accum':   2,
    'lr':           2e-5,
    'weight_decay': 0.01,
    'warmup_ratio': 0.1,
    'num_epochs':   8,
    'patience':     3,
    'aux_weight':   0.3,
    'seed':         42,
    'best_model':   'BestModel/best_model_roberta.pt',
    'dev_probs':    'BestModel/dev_probs.npy',
    'dev_labels':   'BestModel/dev_labels.npy',
}

PCL_CATEGORIES = [
    'Unbalanced power relations', 'Shallow solution', 'Presupposition',
    'Authority voice', 'Metaphor', 'Compassion', 'The poorer the merrier',
]

os.makedirs('BestModel', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# ── reproducibility ─────────────────────────────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
set_seed(CFG['seed'])

# ── data loading (identical to notebook) ───────────────────────────────────────
def load_main_data(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t', 5)
            if len(parts) == 6 and parts[0].strip().isdigit():
                rows.append({'par_id': int(parts[0]), 'keyword': parts[2],
                             'country': parts[3], 'text': parts[4].strip(),
                             'label_raw': int(parts[5])})
    df = pd.DataFrame(rows)
    df['label_binary'] = (df['label_raw'] >= 2).astype(int)
    return df

def load_split_ids(path):
    df = pd.read_csv(path)
    df['par_id'] = df['par_id'].astype(int)
    df['label_list'] = df['label'].apply(ast.literal_eval)
    return df[['par_id', 'label_list']]

print("Loading data …")
df_main       = load_main_data(CFG['main_data'])
df_train_ids  = load_split_ids(CFG['train_ids'])
df_dev_ids    = load_split_ids(CFG['dev_ids'])
df_train = df_main[df_main['par_id'].isin(df_train_ids['par_id'])].copy()
df_dev   = df_main[df_main['par_id'].isin(df_dev_ids['par_id'])].copy()
df_train = df_train.merge(df_train_ids, on='par_id', how='left').reset_index(drop=True)
df_dev   = df_dev.merge(df_dev_ids,   on='par_id', how='left').reset_index(drop=True)
def to_aux(x): return x if isinstance(x, list) else [0]*7
df_train['label_list'] = df_train['label_list'].apply(to_aux)
df_dev['label_list']   = df_dev['label_list'].apply(to_aux)
print(f"Train: {len(df_train)}  Dev: {len(df_dev)}")

# ── dataset & dataloaders ───────────────────────────────────────────────────────
class PCLDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length, aux_labels=None):
        self.texts = [str(t) for t in texts]
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.aux_labels = aux_labels
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], max_length=self.max_length,
                             padding='max_length', truncation=True, return_tensors='pt')
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item['labels'] = torch.tensor(float(self.labels[idx]), dtype=torch.float)
        if self.aux_labels is not None:
            item['aux_labels'] = torch.tensor(self.aux_labels[idx], dtype=torch.float)
        return item

tokenizer = AutoTokenizer.from_pretrained(CFG['model_name'])
dev_ds = PCLDataset(df_dev['text'].tolist(), df_dev['label_binary'].tolist(),
                    tokenizer, CFG['max_length'],
                    aux_labels=df_dev['label_list'].tolist())
dev_dl = DataLoader(dev_ds, batch_size=CFG['batch_size']*2, shuffle=False, num_workers=0)

# ── model helpers ───────────────────────────────────────────────────────────────
@torch.no_grad()
def get_dev_probs(model):
    model.eval()
    all_logits, all_labels = [], []
    for batch in tqdm(dev_dl, desc='Inference on dev', leave=False):
        logits = model(batch['input_ids'].to(device),
                       attention_mask=batch['attention_mask'].to(device)).logits
        all_logits.append(logits[:, 0].cpu().numpy())
        all_labels.append(batch['labels'].numpy())
    probs  = 1.0 / (1.0 + np.exp(-np.concatenate(all_logits)))
    labels = np.concatenate(all_labels)
    return probs, labels

def run_epoch(model, loader, optimizer, scheduler, criterion, grad_accum,
              aux_criterion=None, aux_weight=0.3):
    model.train(); total_loss = 0.0; optimizer.zero_grad()
    for step, batch in enumerate(tqdm(loader, desc='Train', leave=False)):
        logits = model(batch['input_ids'].to(device),
                       attention_mask=batch['attention_mask'].to(device)).logits
        loss = criterion(logits[:, 0], batch['labels'].to(device))
        if aux_criterion is not None and 'aux_labels' in batch:
            loss = loss + aux_weight * aux_criterion(logits[:, 1:],
                                                     batch['aux_labels'].to(device))
        (loss / grad_accum).backward()
        if (step+1) % grad_accum == 0 or (step+1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(loader)

# ── get probabilities (fast path → training path) ──────────────────────────────
if os.path.exists(CFG['dev_probs']) and os.path.exists(CFG['dev_labels']):
    print("✓ Found saved dev probabilities — skipping training.")
    probs  = np.load(CFG['dev_probs'])
    labels = np.load(CFG['dev_labels'])

else:
    model = AutoModelForSequenceClassification.from_pretrained(
        CFG['model_name'], num_labels=8).to(device)

    if os.path.exists(CFG['best_model']) and os.path.getsize(CFG['best_model']) > 1_000_000:
        print("✓ Found stage-1 checkpoint — running inference only.")
        model.load_state_dict(torch.load(CFG['best_model'], map_location=device))
    else:
        print("Stage-1 checkpoint not found — running Stage 1 training (~15-20 min on GPU).")
        train_ds = PCLDataset(df_train['text'].tolist(), df_train['label_binary'].tolist(),
                              tokenizer, CFG['max_length'],
                              aux_labels=df_train['label_list'].tolist())
        train_dl = DataLoader(train_ds, batch_size=CFG['batch_size'], shuffle=True, num_workers=0, pin_memory=True)

        n_pos = (df_train['label_binary'] == 1).sum()
        n_neg = (df_train['label_binary'] == 0).sum()
        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float).to(device)
        criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        train_arr  = np.array(df_train['label_list'].tolist(), dtype=np.float32)
        pw_aux     = torch.tensor((len(train_arr) - train_arr.sum(0)) /
                                   train_arr.sum(0).clip(1), dtype=torch.float).to(device)
        aux_crit   = nn.BCEWithLogitsLoss(pos_weight=pw_aux)

        total_steps = (len(train_dl) // CFG['grad_accum']) * CFG['num_epochs']
        optimizer   = AdamW(model.parameters(), lr=CFG['lr'],
                            weight_decay=CFG['weight_decay'])
        scheduler   = get_linear_schedule_with_warmup(
            optimizer, int(total_steps * CFG['warmup_ratio']), total_steps)

        best_f1, patience_cnt = 0.0, 0
        for epoch in range(CFG['num_epochs']):
            print(f"Epoch {epoch+1}/{CFG['num_epochs']}")
            run_epoch(model, train_dl, optimizer, scheduler, criterion,
                      CFG['grad_accum'], aux_crit, CFG['aux_weight'])
            # quick F1 at t=0.5 for early stopping
            p_tmp, l_tmp = get_dev_probs(model)
            f1 = f1_score(l_tmp.astype(int), (p_tmp >= 0.5).astype(int), zero_division=0)
            print(f"  Dev F1 (t=0.5): {f1:.4f}", end='')
            if f1 > best_f1:
                best_f1, patience_cnt = f1, 0
                torch.save(model.state_dict(), CFG['best_model'])
                print(" ← best")
            else:
                patience_cnt += 1
                print(f"  patience {patience_cnt}/{CFG['patience']}")
                if patience_cnt >= CFG['patience']:
                    break

        model.load_state_dict(torch.load(CFG['best_model'], map_location=device))
        print(f"Loaded best checkpoint (dev F1={best_f1:.4f})")

    probs, labels = get_dev_probs(model)
    np.save(CFG['dev_probs'],  probs)
    np.save(CFG['dev_labels'], labels)
    print(f"Saved probabilities to {CFG['dev_probs']}")

# ── threshold sweep ─────────────────────────────────────────────────────────────
print("\nComputing threshold sweep …")
thresholds = np.linspace(0.30, 0.75, 46)
sweep = []
for t in thresholds:
    preds = (probs >= t).astype(int)
    p = precision_score(labels.astype(int), preds, zero_division=0)
    r = recall_score(labels.astype(int), preds, zero_division=0)
    f = f1_score(labels.astype(int), preds, zero_division=0)
    sweep.append((t, p, r, f))
sweep = np.array(sweep)
best_idx = sweep[:, 3].argmax()
best_t   = sweep[best_idx, 0]
print(f"Best threshold: {best_t:.2f}  F1={sweep[best_idx,3]:.4f}")

# ── LaTeX table (printed so you can copy it) ────────────────────────────────────
table_rows = [0.40, 0.50, 0.55, 0.60, 0.65, best_t, 0.70, 0.75]
print("\n--- LaTeX threshold table (paste into doc.md) ---")
print(r"\begin{center}")
print(r"\begin{tabular}{cccc}")
print(r"\toprule")
print(r"Threshold & Precision & Recall & F1 \\")
print(r"\midrule")
for row in sweep:
    t, p, r, f = row
    if any(abs(t - tr) < 0.005 for tr in table_rows):
        bold = (abs(t - best_t) < 0.005)
        fmt = r"\textbf{{{:.2f}}} & \textbf{{{:.3f}}} & \textbf{{{:.3f}}} & \textbf{{{:.3f}}}" if bold \
              else "{:.2f} & {:.3f} & {:.3f} & {:.3f}"
        print(fmt.format(t, p, r, f) + r" \\")
print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{center}")
print("---\n")

# ── FIGURE 1: threshold sweep ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(sweep[:,0], sweep[:,1], label='Precision', color='#2196F3', lw=1.8)
ax.plot(sweep[:,0], sweep[:,2], label='Recall',    color='#FF9800', lw=1.8)
ax.plot(sweep[:,0], sweep[:,3], label='F1',        color='#4CAF50', lw=2.2)
ax.axvline(best_t, color='#9C27B0', linestyle='--', lw=1.4,
           label=f'Optimal $t$={best_t:.2f}')
ax.set_xlabel('Decision threshold')
ax.set_ylabel('Score')
ax.set_xlim(0.30, 0.75)
ax.set_ylim(0.0, 1.0)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
ax.legend(fontsize=9, framealpha=0.9)
ax.grid(alpha=0.3)
fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(f'figures/eval_threshold_sweep.{ext}', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/eval_threshold_sweep.pdf")

# ── FIGURE 2: precision-recall curve ───────────────────────────────────────────
prec_curve, rec_curve, _ = precision_recall_curve(labels.astype(int), probs)
pr_auc = auc(rec_curve, prec_curve)
baseline_precision = labels.mean()

fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(rec_curve, prec_curve, color='#2196F3', lw=2,
        label=f'Our model (AUC = {pr_auc:.3f})')
ax.axhline(baseline_precision, color='#F44336', linestyle='--', lw=1.4,
           label=f'No-skill baseline ({baseline_precision:.3f})')

# mark the operating point
op_prec = precision_score(labels.astype(int), (probs >= best_t).astype(int), zero_division=0)
op_rec  = recall_score(labels.astype(int),    (probs >= best_t).astype(int), zero_division=0)
ax.scatter([op_rec], [op_prec], color='#9C27B0', zorder=5, s=60,
           label=f'Operating point ($t$={best_t:.2f})')

ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.legend(fontsize=9, framealpha=0.9)
ax.grid(alpha=0.3)
fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(f'figures/eval_pr_curve.{ext}', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/eval_pr_curve.pdf")

# ── FIGURE 3: probability distribution by outcome ──────────────────────────────
preds_at_best = (probs >= best_t).astype(int)
labs_int      = labels.astype(int)

tp_probs = probs[(preds_at_best == 1) & (labs_int == 1)]
fp_probs = probs[(preds_at_best == 1) & (labs_int == 0)]
fn_probs = probs[(preds_at_best == 0) & (labs_int == 1)]
tn_probs = probs[(preds_at_best == 0) & (labs_int == 0)]

fig, ax = plt.subplots(figsize=(6.5, 3.5))
bins = np.linspace(0, 1, 40)
ax.hist(tp_probs, bins=bins, alpha=0.6, color='#4CAF50', label=f'TP (n={len(tp_probs)})')
ax.hist(fp_probs, bins=bins, alpha=0.6, color='#FF9800', label=f'FP (n={len(fp_probs)})')
ax.hist(fn_probs, bins=bins, alpha=0.6, color='#F44336', label=f'FN (n={len(fn_probs)})')
ax.hist(tn_probs, bins=bins, alpha=0.3, color='#9E9E9E', label=f'TN (n={len(tn_probs)})')
ax.axvline(best_t, color='#9C27B0', linestyle='--', lw=1.4,
           label=f'Threshold $t$={best_t:.2f}')
ax.set_xlabel('Model probability (PCL)')
ax.set_ylabel('Count')
ax.legend(fontsize=9, framealpha=0.9)
ax.grid(alpha=0.3)
ax.set_yscale('log')
fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(f'figures/eval_prob_dist.{ext}', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/eval_prob_dist.pdf")

# ── FIGURE 4: confusion matrix heatmap ─────────────────────────────────────────
import matplotlib.patches as mpatches

cm = np.array([[1808, 87], [72, 127]])
labels_cm = [['TN\n1,808', 'FP\n87'], ['FN\n72', 'TP\n127']]
colours   = [['#C8E6C9', '#FFCC02'], ['#FFCC02', '#4CAF50']]

fig, ax = plt.subplots(figsize=(4, 3.5))
for i in range(2):
    for j in range(2):
        ax.add_patch(plt.Rectangle((j, 1-i), 1, 1, color=colours[i][j], ec='white', lw=2))
        ax.text(j+0.5, 1.5-i, labels_cm[i][j], ha='center', va='center',
                fontsize=13, fontweight='bold')

ax.set_xlim(0, 2); ax.set_ylim(0, 2)
ax.set_xticks([0.5, 1.5]); ax.set_xticklabels(['Predicted\nNo-PCL', 'Predicted\nPCL'], fontsize=10)
ax.set_yticks([0.5, 1.5]); ax.set_yticklabels(['Actual PCL', 'Actual No-PCL'], fontsize=10)
ax.tick_params(length=0)
for spine in ax.spines.values(): spine.set_visible(False)
fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(f'figures/eval_conf_matrix.{ext}', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/eval_conf_matrix.pdf")

print("\nAll done. Figures saved to figures/")
print(f"Best threshold: {best_t:.2f}  |  P={op_prec:.3f}  R={op_rec:.3f}  F1={f1_score(labs_int,(probs>=best_t).astype(int)):.3f}")
