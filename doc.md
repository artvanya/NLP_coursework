\documentclass{coursework}
\usepackage{graphicx} % Required for inserting images

\title{Natural_Language_Processing}
\author{Ivan Artiukhov}
\date{February 2026}

\begin{document}

\subsection*{Exercise 1: Critical Review of \textit{Don't Patronize Me!}}

This section reviews the paper by Perez-Almendros et al. (COLING 2020), which introduces the \textit{Don't Patronize Me!} dataset for detecting patronizing and condescending language (PCL) in news text.

\subsubsection*{Q1. Primary contributions of this work (2 marks)}

The main contribution of this paper is the creation of a new NLP dataset for detecting patronizing and condescending language. The authors build a dataset of 10,637 news paragraphs taken from articles across 20 English-speaking countries, focusing on text about vulnerable communities (for example, refugees, homeless people, and poor families). This is an important contribution because PCL is much more subtle than hate speech or offensive language, and there were very few resources for studying it.

A second major contribution is the annotation framework. The paper does not only label whether a paragraph contains PCL or not, but also provides a finer-grained category scheme for different types of PCL. This makes the dataset useful for both binary classification and more detailed analysis of how patronizing language is expressed.

The paper also contributes baseline experiments for the task. The authors compare traditional machine learning methods and neural models (including transformer-based models), which helps show that the task is possible but still challenging. This is useful for future work because it gives a clear starting point for comparison.

\subsubsection*{Q2. Technical strengths that justify publication (2 marks)}

One clear strength of the paper is that the problem is well motivated. The authors explain clearly why PCL matters and why it is different from other harmful language tasks. In particular, they show that PCL is often implicit and can sound positive on the surface, which makes it harder to detect and worth studying on its own.

Another strength is the annotation design. The authors use expert annotators and include a referee annotator when there are major disagreements. They also use a two-stage process (first detecting PCL, then annotating spans and categories), which is a sensible way to handle a difficult and subjective task. I also think it is a strength that they openly discuss ambiguity instead of pretending the labels are always clear.

The taxonomy is also a strong point. By defining different types of patronizing language and applying those labels in the dataset, the paper gives future researchers more than just a binary benchmark. This increases the long-term value of the resource.

Finally, the paper includes a reasonable set of baseline experiments and some qualitative analysis. The comparison across different model types, plus examples of model errors, makes the paper more useful and easier to build on.

\subsubsection*{Q3. Key weaknesses / areas with insufficient evidence (2 marks)}

The biggest weakness is the level of subjectivity in the annotations. The authors are honest about this, which is good, but the agreement scores show that the task is difficult even for humans. This does not make the dataset unusable, but it does mean that stronger validation or a deeper discussion of annotation consistency would have made the paper stronger.

Another weakness is possible sampling bias in how the data was collected. The dataset is built using pre-selected keywords linked to vulnerable groups. This is practical, but it may introduce lexical patterns that models can exploit. The paper does not fully test whether models are learning real PCL patterns or just relying on keyword-related shortcuts.

The experimental section is useful, but still fairly limited. The baselines are a good start, but the paper does not include many diagnostic experiments or ablations. For example, it would have been helpful to test class imbalance strategies, threshold effects, or cross-domain generalization more directly.

Some of the explanations in the analysis are reasonable, but not fully proven. For instance, the authors suggest that poor performance on certain categories is related to the need for world knowledge or common sense. That may be true, but the paper does not provide direct evidence for this claim, so it remains a plausible interpretation rather than a demonstrated result.

Overall, I think the paper is still a strong and publishable contribution. The dataset and taxonomy are valuable, and the paper opens up an important NLP task. Its main limitations are mostly about annotation subjectivity and limited experimental depth, rather than a lack of usefulness.

\subsection*{Exercise 2: Exploratory Data Analysis}

All analysis in this section is performed on the official training split (8,375 samples) using the binary label derived from the raw annotation scores in \texttt{dontpatronizeme\_pcl.tsv}: a paragraph is labelled PCL (1) if at least two annotators flagged it, and No-PCL (0) otherwise.

\subsubsection*{EDA Technique 1: Class Distribution and Text-Length Analysis}

\begin{figure}[h]
    \centering
    \includegraphics[width=\linewidth]{eda1_class_length.pdf}
    \caption{Class distribution (left) and word-count density per class (right) in the training split.}
    \label{fig:eda1}
\end{figure}

\noindent
\textbf{Summary statistics:}
\begin{center}
\begin{tabular}{lrrrr}
\hline
Class & Count & Proportion & Mean words & Median words \\
\hline
No-PCL (0) & 7,581 & 90.5\% & 48.2 & 42 \\
PCL (1)    &   794 &  9.5\% & 53.5 & 47 \\
\hline
\end{tabular}
\end{center}

\noindent
\textbf{Analysis.}
The dataset is heavily skewed: roughly 9 out of every 10 paragraphs carry no patronizing language, giving a positive-to-negative ratio of about 1:9.5. The word-count histograms show that both classes have broadly similar length profiles, with a slight tendency for PCL paragraphs to be a little longer (median 47 vs.\ 42 words). The vast majority of samples — 98.3\% — contain 128 words or fewer, and 99.9\% fit within 256 words.

\noindent
\textbf{Impact on approach.}
The class imbalance is significant enough that a model trained with a standard cross-entropy loss will tend to collapse towards predicting No-PCL, achieving around 90\% accuracy while completely missing the positive class. This makes the F1 score on the positive class a far more informative metric than accuracy, which is exactly what the shared-task evaluation uses. Practically, this means training must use class-weighted loss or oversampling of the PCL class. The length analysis also tells us that \texttt{max\_length=128} tokens is a safe default for the tokenizer — it covers nearly the entire dataset without padding-induced waste.

\subsubsection*{EDA Technique 2: Discriminative Bigram Analysis}

\begin{figure}[h]
    \centering
    \includegraphics[width=\linewidth]{eda2_bigrams.pdf}
    \caption{Top 15 bigrams by log-odds ratio for each class. Bars show how much more likely a bigram is in one class relative to the other.}
    \label{fig:eda2}
\end{figure}

\noindent
\textbf{Analysis.}
The log-odds ratio surfaces a striking contrast between the two classes. PCL-associated bigrams — ``less fortunate'', ``homeless person'', ``giving back'', ``better lives'', ``below poverty'' — are the kind of soft, sympathetic-sounding phrases that characterise patronising framing: the writer positions themselves as a benefactor and the subject as an object of pity. The No-PCL side is dominated by factual and political bigrams: ``illegal immigrants'', ``anti immigrant'', ``climate change'', ``trump administration'', ``refugee camp''. These are direct, neutral references to events and policies rather than evaluative statements about people.

This asymmetry reveals something non-obvious: it is not the topic that makes language patronising (both classes talk about refugees, the homeless, and poverty) but the \emph{framing}. Highly charged political terms are, paradoxically, more associated with neutral reporting than with PCL.

\noindent
\textbf{Impact on approach.}
Because the discriminative signal lies in framing and subtle phrasing rather than in topic keywords, surface-level features will not generalise well. A bag-of-words or TF-IDF classifier would be partly misled — it might learn to associate topic words with No-PCL and softer vocabulary with PCL, but the same words can appear in both classes depending on context. This is a strong argument for using a pre-trained transformer (such as RoBERTa) that encodes the broader sentence context, rather than treating the paragraph as a bag of independent tokens. It also suggests that data augmentation or keyword-based filtering should be used with care, as removing topic-related words could strip useful contextual signal.

\subsection*{Exercise 3: Proposed Approach}

\subsubsection*{Proposed approach}

The baseline is a fine-tuned RoBERTa-base model trained with standard cross-entropy loss, achieving an F1 score of 0.48 on the official dev set. My approach keeps the same model architecture but makes two targeted changes motivated directly by the EDA findings.

\textbf{Class-weighted loss.} The training data contains 794 PCL examples against 7,581 No-PCL examples — a ratio of roughly 1:9.5. A standard cross-entropy loss treats both classes equally, which in practice pushes the model towards always predicting No-PCL (the majority class), since that path minimises the loss most easily. To counteract this, I weight the loss inversely proportional to class frequency. The weight for the PCL class is set to $w_1 = N / (2 \cdot N_1)$ and for No-PCL to $w_0 = N / (2 \cdot N_0)$, where $N$ is the total training size and $N_0, N_1$ are the per-class counts. This forces the model to pay equal aggregate attention to each class during training.

\textbf{Threshold tuning.} A classification model outputs a probability score, but the default decision boundary of 0.5 is rarely optimal on an imbalanced dataset. After training, I search over candidate thresholds in the range $[0.1, 0.9]$ on the official dev set and select the value that maximises the F1 score on the positive (PCL) class. This is a low-cost post-processing step that can shift meaningful gains without any re-training.

Everything else follows the standard fine-tuning setup: \texttt{max\_length=128} tokens (covering 98.3\% of samples without truncation, as shown in EDA Technique 1), AdamW optimiser, learning rate of $2\times10^{-5}$, and 4 training epochs.

\subsubsection*{Rationale and expected outcome}

The EDA provides clear motivation for both modifications. The severe class imbalance identified in Technique 1 is the most obvious single reason a straightforward fine-tune underperforms. Simply re-weighting the loss is a well-established fix for this problem, and the magnitude of the imbalance here (1:9.5) is large enough that the effect should be substantial.

The bigram analysis in Technique 2 showed that PCL is characterised by subtle framing rather than surface keywords, which confirms that the underlying RoBERTa-base encoder is the right tool — it can model context and nuance in a way that simpler approaches cannot. What the baseline likely lacks is the ability to \emph{commit} to predicting PCL given the training signal, since the loss rarely penalises missing the minority class.

Threshold tuning addresses the complementary problem: even after class-weighted training, the model's probability calibration may not be well-aligned with a 0.5 decision boundary on the positive class. Searching the threshold on the dev set is a direct way to maximise the evaluation metric that matters.

Together, I expect these two changes to produce a noticeable improvement over the 0.48 baseline F1, particularly in the recall of the positive class, which tends to be the bottleneck in imbalanced binary classification tasks.
\subsection*{Exercise 4: Critical Review of \textit{Don't Patronize Me!}}
\subsection*{Exercise 5: Critical Review of \textit{Don't Patronize Me!}}
\subsection*{Exercise 6: Critical Review of \textit{Don't Patronize Me!}}

\end{document}
