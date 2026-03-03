"""
Generate EDA figures for the report: eda1_class_length.pdf, eda2_bigrams.pdf.
Run from project root. Uses same data as the notebook (train split, binary label).
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def load_main(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t', 5)
            if len(parts) == 6 and parts[0].strip().isdigit():
                rows.append({'par_id': int(parts[0]), 'text': parts[4].strip(), 'label_raw': int(parts[5])})
    df = pd.DataFrame(rows)
    df['label_binary'] = (df['label_raw'] >= 2).astype(int)
    return df

def load_train_ids(path):
    df = pd.read_csv(path)
    df['par_id'] = df['par_id'].astype(int)
    return df[['par_id']]

def tokenize(text):
    return text.lower().split()

def bigrams(tokens):
    return [tuple(tokens[i:i+2]) for i in range(len(tokens)-1)] if len(tokens) >= 2 else []

def main():
    df_main = load_main('dontpatronizeme_pcl.tsv')
    train_ids = load_train_ids('train_semeval_parids-labels.csv')['par_id'].tolist()
    df = df_main[df_main['par_id'].isin(train_ids)].copy()
    df['nwords'] = df['text'].apply(lambda t: len(tokenize(t)))

    # --- EDA 1: class distribution + word count density ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    counts = df['label_binary'].value_counts().sort_index()
    ax1.bar(['No-PCL (0)', 'PCL (1)'], [counts.get(0, 0), counts.get(1, 0)], color=['#4dabf7', '#f783ac'])
    ax1.set_ylabel('Count')
    ax1.set_title('Class distribution')
    for c in [0, 1]:
        subset = df[df['label_binary'] == c]['nwords']
        ax2.hist(subset, bins=40, alpha=0.6, density=True, label='PCL' if c else 'No-PCL', range=(0, 250))
    ax2.set_xlabel('Word count')
    ax2.set_ylabel('Density')
    ax2.set_title('Word count by class')
    ax2.legend()
    ax2.set_ylim(0, None)
    plt.tight_layout()
    plt.savefig('eda1_class_length.pdf', bbox_inches='tight')
    plt.close()
    print('Saved eda1_class_length.pdf')

    # --- EDA 2: bigram log-odds ---
    c0 = Counter()
    c1 = Counter()
    for _, row in df.iterrows():
        toks = tokenize(row['text'])
        for bg in bigrams(toks):
            if row['label_binary'] == 0:
                c0[bg] += 1
            else:
                c1[bg] += 1
    n0, n1 = sum(c0.values()), sum(c1.values())
    all_bigrams = set(c0) | set(c1)
    delta = {}
    for bg in all_bigrams:
        p0 = (c0[bg] + 0.5) / (n0 + 0.5)
        p1 = (c1[bg] + 0.5) / (n1 + 0.5)
        delta[bg] = np.log(p1) - np.log(p0)
    top_pcl = sorted([(bg, d) for bg, d in delta.items() if d > 0], key=lambda x: -x[1])[:15]
    top_nopcl = sorted([(bg, d) for bg, d in delta.items() if d < 0], key=lambda x: x[1])[:15]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    labels_pcl = [ ' '.join(bg) for bg, _ in top_pcl ]
    ax1.barh(range(len(labels_pcl)), [d for _, d in top_pcl], color='#f783ac', alpha=0.8)
    ax1.set_yticks(range(len(labels_pcl)))
    ax1.set_yticklabels(labels_pcl, fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel('Log-odds (PCL vs No-PCL)')
    ax1.set_title('Top 15 bigrams more in PCL')
    labels_nopcl = [ ' '.join(bg) for bg, _ in top_nopcl ]
    ax2.barh(range(len(labels_nopcl)), [d for _, d in top_nopcl], color='#4dabf7', alpha=0.8)
    ax2.set_yticks(range(len(labels_nopcl)))
    ax2.set_yticklabels(labels_nopcl, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel('Log-odds (PCL vs No-PCL)')
    ax2.set_title('Top 15 bigrams more in No-PCL')
    plt.tight_layout()
    plt.savefig('eda2_bigrams.pdf', bbox_inches='tight')
    plt.close()
    print('Saved eda2_bigrams.pdf')

if __name__ == '__main__':
    main()
