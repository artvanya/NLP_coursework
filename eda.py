import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
import re
import os

FIG_DIR = 'figures'
os.makedirs(FIG_DIR, exist_ok=True)

# ---------- Load and merge data ----------
rows = []
with open('data/dontpatronizeme_pcl.tsv', 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.rstrip('\n').split('\t', 5)
        if len(parts) == 6 and parts[0].strip().isdigit():
            rows.append({
                'par_id': int(parts[0]),
                'keyword': parts[2],
                'text': parts[4].strip(),
                'label_raw': int(parts[5]),
            })

df = pd.DataFrame(rows)

# Binary label: >= 2 annotators marked as PCL → PCL=1
df['label'] = (df['label_raw'] >= 2).astype(int)

# Restrict to train split
train_ids = set(pd.read_csv('data/train_semeval_parids-labels.csv')['par_id'])
df = df[df['par_id'].isin(train_ids)].reset_index(drop=True)

df['word_count'] = df['text'].apply(lambda t: len(t.split()))

pcl = df[df['label'] == 1]
npcl = df[df['label'] == 0]

print(f"Train split  |  Total: {len(df)}  |  PCL: {len(pcl)} ({len(pcl)/len(df)*100:.1f}%)  |  No-PCL: {len(npcl)}")


# =====================================================================
# EDA 1: Class distribution + word-count distributions
# =====================================================================
pcl_wc  = pcl['word_count']
npcl_wc = npcl['word_count']

print(f"\n--- Word count stats ---")
print(f"Overall:  mean={df['word_count'].mean():.1f}, median={df['word_count'].median():.0f}, "
      f"min={df['word_count'].min()}, max={df['word_count'].max()}")
print(f"PCL:      mean={pcl_wc.mean():.1f}, median={pcl_wc.median():.0f}")
print(f"No-PCL:   mean={npcl_wc.mean():.1f}, median={npcl_wc.median():.0f}")
print(f"% samples <= 128 words: {(df['word_count'] <= 128).mean()*100:.1f}%")
print(f"% samples <= 256 words: {(df['word_count'] <= 256).mean()*100:.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
fig.subplots_adjust(wspace=0.35)

# Panel A – class counts
counts = [len(npcl), len(pcl)]
bars = axes[0].bar(['No-PCL (0)', 'PCL (1)'], counts,
                    color=['#4C72B0', '#DD8452'], width=0.5, edgecolor='white', linewidth=0.8)
for bar, count in zip(bars, counts):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 80,
                 f'{count:,}\n({count/len(df)*100:.1f}%)',
                 ha='center', va='bottom', fontsize=10)
axes[0].set_title('(a) Class Distribution', fontsize=12, pad=8)
axes[0].set_ylabel('Number of samples', fontsize=10)
axes[0].set_ylim(0, max(counts) * 1.18)
axes[0].spines[['top', 'right']].set_visible(False)

# Panel B – word-count histograms (capped at 300 for readability)
cap = 300
axes[1].hist(npcl_wc.clip(upper=cap), bins=40, range=(0, cap),
             alpha=0.55, color='#4C72B0', label='No-PCL', density=True)
axes[1].hist(pcl_wc.clip(upper=cap),  bins=40, range=(0, cap),
             alpha=0.55, color='#DD8452', label='PCL',    density=True)
axes[1].axvline(npcl_wc.median(), color='#4C72B0', linestyle='--', linewidth=1.4,
                label=f'No-PCL median ({npcl_wc.median():.0f})')
axes[1].axvline(pcl_wc.median(),  color='#DD8452', linestyle='--', linewidth=1.4,
                label=f'PCL median ({pcl_wc.median():.0f})')
axes[1].set_title('(b) Word-Count Distribution by Class', fontsize=12, pad=8)
axes[1].set_xlabel('Word count (capped at 300)', fontsize=10)
axes[1].set_ylabel('Density', fontsize=10)
axes[1].legend(fontsize=8.5, framealpha=0.7)
axes[1].spines[['top', 'right']].set_visible(False)

fig.suptitle('EDA 1 – Class Imbalance and Text Length', fontsize=13, y=1.01, fontweight='bold')
fig.savefig(os.path.join(FIG_DIR, 'eda1_class_length.pdf'), bbox_inches='tight', dpi=150)
fig.savefig(os.path.join(FIG_DIR, 'eda1_class_length.png'), bbox_inches='tight', dpi=150)
print(f"\nSaved {FIG_DIR}/eda1_class_length.pdf and .png")
plt.close()


# =====================================================================
# EDA 2: Discriminative bigrams – log-odds ratio
# =====================================================================
STOPWORDS = {
    'the','a','an','is','it','in','of','to','and','for','on','at','by','with',
    'was','be','as','are','that','this','its','he','she','they','we','his',
    'her','their','our','have','has','had','not','but','from','or','been',
    'were','which','who','also','about','will','would','could','should',
    'more','there','than','then','up','out','said','says','all','just',
    'one','two','can','do','did','i','you','my','your','so','if','what',
    'after','before','when','how','any','some','no','new','been','into',
    'over','other','these','those','such','very','much'
}

def get_bigrams(texts, stopwords=STOPWORDS):
    bg = Counter()
    for text in texts:
        tokens = re.findall(r"[a-z']+", text.lower())
        tokens = [t for t in tokens if t not in stopwords and len(t) > 2]
        for i in range(len(tokens) - 1):
            bg[(tokens[i], tokens[i+1])] += 1
    return bg

pcl_bg  = get_bigrams(pcl['text'])
npcl_bg = get_bigrams(npcl['text'])

pcl_total  = sum(pcl_bg.values())
npcl_total = sum(npcl_bg.values())

# Log-odds ratio (with +1 smoothing to avoid log(0))
all_bigrams = set(pcl_bg.keys()) | set(npcl_bg.keys())
log_odds = {}
for bg in all_bigrams:
    p_pcl  = (pcl_bg[bg]  + 1) / (pcl_total  + len(all_bigrams))
    p_npcl = (npcl_bg[bg] + 1) / (npcl_total + len(all_bigrams))
    # Require minimum absolute count to avoid noise
    if pcl_bg[bg] + npcl_bg[bg] < 5:
        continue
    log_odds[bg] = np.log(p_pcl / p_npcl)

sorted_lo = sorted(log_odds.items(), key=lambda x: x[1], reverse=True)
top_pcl   = sorted_lo[:15]
top_npcl  = sorted_lo[-15:][::-1]

print("\n--- Top 15 bigrams over-represented in PCL ---")
for bg, lo in top_pcl:
    print(f"  {' '.join(bg):30s}  log-odds={lo:+.3f}  (PCL count={pcl_bg[bg]}, No-PCL count={npcl_bg[bg]})")

print("\n--- Top 15 bigrams over-represented in No-PCL ---")
for bg, lo in top_npcl:
    print(f"  {' '.join(bg):30s}  log-odds={lo:+.3f}  (PCL count={pcl_bg[bg]}, No-PCL count={npcl_bg[bg]})")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.subplots_adjust(wspace=0.55)

def make_bar(ax, items, color, title):
    labels = [' '.join(bg) for bg, _ in items]
    values = [abs(lo) for _, lo in items]
    y = range(len(labels))
    bars = ax.barh(list(y), values, color=color, edgecolor='white', linewidth=0.6, height=0.65)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel('|log-odds ratio|', fontsize=10)
    ax.set_title(title, fontsize=11, pad=8)
    ax.spines[['top', 'right']].set_visible(False)

make_bar(axes[0], top_pcl,  '#DD8452', '(a) Most PCL-associated bigrams')
make_bar(axes[1], top_npcl, '#4C72B0', '(b) Most No-PCL-associated bigrams')

fig.suptitle('EDA 2 – Discriminative Bigrams per Class (Log-Odds Ratio)', fontsize=13, y=1.01, fontweight='bold')
fig.savefig(os.path.join(FIG_DIR, 'eda2_bigrams.pdf'), bbox_inches='tight', dpi=150)
fig.savefig(os.path.join(FIG_DIR, 'eda2_bigrams.png'), bbox_inches='tight', dpi=150)
print(f"\nSaved {FIG_DIR}/eda2_bigrams.pdf and .png")
plt.close()
