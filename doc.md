\documentclass{coursework}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{booktabs}

\title{Natural Language Processing Coursework}
\author{Ivan Artiukhov}
\date{March 2026}

% ── FRONT PAGE ITEMS (required by spec) ──────────────────────────────────────
% GitHub repository: \href{GITHUB_LINK_HERE}{GITHUB_LINK_HERE}
% Leaderboard name: IvanArtiukhov
% ─────────────────────────────────────────────────────────────────────────────

\begin{document}

\noindent\textbf{GitHub repository:} \href{GITHUB_LINK_HERE}{\texttt{GITHUB\_LINK\_HERE}}\\
\textbf{Leaderboard name:} IvanArtiukhov

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
Because the discriminative signal lies in framing and subtle phrasing rather than in topic keywords, surface-level features will not generalise well. A bag-of-words or TF-IDF classifier would be partly misled — it might learn to associate topic words with No-PCL and softer vocabulary with PCL, but the same words can appear in both classes depending on context. This is a strong argument for using a pre-trained transformer that encodes broader sentence context rather than treating the paragraph as a bag of independent tokens. It also motivates using the seven fine-grained PCL category labels as auxiliary supervision: if the model learns to identify \emph{what kind} of PCL a paragraph contains, it should also improve at detecting whether any PCL is present.

\subsection*{Exercise 3: Proposed Approach}

\subsubsection*{Proposed approach}

The baseline is a fine-tuned RoBERTa-base model trained with standard cross-entropy loss, achieving F1\,=\,0.48 on the official dev set. My approach makes three targeted changes motivated by the EDA findings.

\textbf{1.\ Stronger encoder: DeBERTa-v3-base.}
I replace RoBERTa-base with \texttt{microsoft/deberta-v3-base}, a model of comparable size ($\sim$86\,M parameters) but significantly stronger on NLU benchmarks. DeBERTa-v3 uses \emph{disentangled attention} — content and positional embeddings are kept separate and combined at each attention layer — which gives the model finer-grained sensitivity to word order and syntactic structure. It is also pre-trained with ELECTRA-style replaced-token detection on top of masked language modelling, providing a richer training signal. These properties are directly useful for detecting PCL, where framing and word choice matter more than topic.

\textbf{2.\ Class-weighted \texttt{BCEWithLogitsLoss} with multi-task auxiliary loss.}
The training data contains 794 PCL examples against 7,581 No-PCL examples (ratio 1:9.5). A standard loss treats every example equally, which pushes the model towards always predicting No-PCL. I address this with two complementary signals:

\begin{itemize}
    \item \emph{Primary loss}: \texttt{BCEWithLogitsLoss} with \texttt{pos\_weight}\,=\,$N_{\text{neg}}/N_{\text{pos}}$\,=\,9.55. Each positive (PCL) example contributes as much to the gradient as 9.55 negatives.
    \item \emph{Auxiliary loss}: The model outputs 8 logits. Logit\,0 is the binary PCL prediction; logits\,1--7 are the seven PCL sub-categories provided in the dataset (Unbalanced power relations, Shallow solution, Presupposition, Authority voice, Metaphor, Compassion, The poorer the merrier). A separate \texttt{BCEWithLogitsLoss} with per-label class weights is applied to logits\,1--7 and added to the primary loss with weight 0.3. This multi-task signal forces the encoder to learn the finer-grained structure of PCL, which the bigram analysis showed is essential: PCL is defined by \emph{how} something is said, not just \emph{what} topic it covers.
\end{itemize}

\textbf{3.\ Threshold tuning.}
After training I search over candidate thresholds $t \in [0.05, 0.95]$ on the official dev set and select the value maximising F1 on the PCL class. This is a low-cost post-processing step; it is especially valuable on imbalanced data where the default $t=0.5$ is rarely optimal.

The remaining setup follows standard fine-tuning practice: \texttt{max\_length=128} (covers 98.3\% of samples), AdamW, learning rate $2\!\times\!10^{-5}$, weight decay 0.01, effective batch size 32 (16$\times$2 gradient accumulation steps), linear warm-up over 10\% of training steps, 8 epochs, pure FP32 (DeBERTa-v3's attention mechanism is numerically unstable in mixed precision on Blackwell-architecture GPUs).

\subsubsection*{Rationale and expected outcome}

The EDA findings provide direct motivation for each component. EDA Technique 1 showed that a 1:9.5 class imbalance will cause any standard loss to collapse to the majority class; the class-weighted primary loss is the direct fix. EDA Technique 2 showed that PCL is characterised by subtle framing rather than surface keywords, which motivates both the stronger encoder (DeBERTa-v3 can model the full sentence context) and the multi-task auxiliary loss (which forces the model to distinguish \emph{types} of PCL framing rather than just flagging presence or absence). Threshold tuning then aligns the decision boundary with the actual operating point that maximises the evaluation metric.

The outcome was F1\,=\,0.506 on the official dev set (threshold $t=0.84$), compared to the baseline of 0.48 — an improvement of $+$0.026. PCL precision was 0.495 and recall 0.518, with overall accuracy 0.904. The relatively high optimal threshold (0.84) indicates the model is well-calibrated to be conservative: it only predicts PCL when it is confident, which keeps false positives low at the cost of some false negatives.

\subsection*{Exercise 4: Model Training}

The training code, trained model weights, and prediction files are in the \texttt{BestModel/} folder of the repository linked on the front page of this report. The repository must be made public after the submission deadline. The \texttt{BestModel/} folder contains:

\begin{itemize}
    \item \texttt{pcl\_detection.ipynb} — full training notebook (data loading, model setup, training loop, threshold tuning, prediction)
    \item \texttt{best\_model.pt} — weights of the best-performing checkpoint (highest dev F1 across 8 epochs)
    \item \texttt{dev.txt} — predictions on the official dev set (2,094 lines, one 0/1 per line)
    \item \texttt{test.txt} — predictions on the official test set (3,832 lines, one 0/1 per line)
\end{itemize}

\subsection*{Exercise 5.1: Global Evaluation}

Prediction files \texttt{dev.txt} and \texttt{test.txt} are available in the \texttt{BestModel/} folder of the repository linked on the front page. The dev set result is:

\begin{center}
\begin{tabular}{llll}
\toprule
Model & Dev F1 & Threshold & vs.\ baseline \\
\midrule
RoBERTa-base (baseline) & 0.4800 & 0.50 & — \\
DeBERTa-v3-base (ours)  & \textbf{0.5061} & 0.84 & +0.0261 \\
\bottomrule
\end{tabular}
\end{center}

\noindent The test set result will be available from the leaderboard after the submission deadline.

\subsection*{Exercise 5.2: Local Evaluation}

\subsubsection*{Error Analysis}

\noindent\textbf{Confusion matrix (official dev set, 2,094 samples):}

\begin{center}
\begin{tabular}{lcc}
\toprule
 & Predicted No-PCL & Predicted PCL \\
\midrule
Actual No-PCL & 1,790 (TN) & 105 (FP) \\
Actual PCL    &    96 (FN) & 103 (TP) \\
\bottomrule
\end{tabular}
\end{center}

\noindent The model misses 96 of the 199 PCL paragraphs (recall\,=\,0.518) and flags 105 No-PCL paragraphs incorrectly (precision\,=\,0.495). False positives and false negatives are nearly symmetric, which reflects the class-weighted training: the model no longer collapses to always predicting No-PCL, but it faces a genuinely hard discrimination problem.

\vspace{0.5em}
\noindent\textbf{False negatives — PCL the model missed.}
Examining the 96 false negatives reveals a consistent pattern: these are cases where the PCL is implicit in the author's \emph{attitude} rather than explicit in the vocabulary.

\begin{quote}
\textit{``His present `chambers' may be quite humble, but Shiyani has the tiny space very neatly organized and clean. Many people pass him by but do not manage to see him...''} [par\_id 107]
\end{quote}

This paragraph describes a homeless person's living space. The PCL lies in the patronising surprise at the space being ``neatly organized'' — the author frames cleanliness as noteworthy for someone in Shiyani's position. There are no surface keywords that the model can reliably associate with PCL; the condescension is in the framing of what is considered praise-worthy.

\begin{quote}
\textit{``Krueger recently harnessed that creativity to self-publish a book featuring the poems, artwork, photography and short stories of 16 ill or disabled artists from around the world.''} [par\_id 149]
\end{quote}

This paragraph exhibits the \textit{Shallow solution} and \textit{Compassion} categories of PCL — framing disabled people as inspirational objects of admiration. The absence of explicit negative language means the model does not detect it.

These false negatives share a key property: they require the reader to infer the author's implicit stance, which goes beyond what surface-level contextual features can capture. This is the hardest failure mode for any model on this task — the PCL exists at the pragmatic rather than the semantic level.

\vspace{0.5em}
\noindent\textbf{False positives — No-PCL predicted as PCL.}
The 105 false positives reveal a different pattern: the model fires on vocabulary associated with vulnerable groups even when the framing is neutral or sympathetic without being condescending.

\begin{quote}
\textit{``His friends at the Chevron want people to know he wasn't just a faceless homeless person. He was their friend and their family.''} [par\_id 8591]
\end{quote}

This sentence contains ``homeless person'' — a strong PCL bigram from EDA Technique 2 — but the sentence is explicitly \emph{arguing against} dehumanising framing. The model has learned an association between the lexeme and PCL without learning the negation context.

\begin{quote}
\textit{``So we do need to heal ourselves as an Aboriginal Torres Strait Islander community, but also as a nation.''} [par\_id 8480]
\end{quote}

This is a first-person statement by a member of the community being discussed. There is no outside observer adopting a patronising stance; the speaker is speaking for and about their own community. The model cannot distinguish internal community voice from external characterisation, which is a significant limitation.

\vspace{0.5em}
\noindent\textbf{Summary of error patterns.}
Two main failure modes emerge: (1) \emph{implicit-stance false negatives}, where PCL requires inferring author attitude from pragmatic context rather than surface vocabulary; and (2) \emph{keyword-trigger false positives}, where the model associates vulnerable-group vocabulary with PCL regardless of whether the framing is actually condescending. Both errors reflect the fundamental difficulty identified in the EDA: PCL is defined by framing, not topic, and framing is much harder to model.

\subsubsection*{Other Local Evaluation: Threshold and Confidence Analysis}

\noindent\textbf{Threshold sensitivity.}
The optimal threshold on the dev set was $t = 0.84$, far above the default 0.5. At $t=0.5$ the model already achieves F1\,=\,0.499; the threshold search adds only a small further gain to 0.506. The near-flat F1 curve between $t=0.5$ and $t=0.84$ indicates the model assigns probabilities in a compressed range — it does not confidently predict high-probability PCL for most positive examples, which limits the ceiling of threshold tuning as a post-processing strategy.

\begin{center}
\begin{tabular}{cccc}
\toprule
Threshold & Precision & Recall & F1 \\
\midrule
0.50 & 0.451 & 0.558 & 0.499 \\
0.65 & 0.478 & 0.533 & 0.504 \\
0.84 & 0.495 & 0.518 & 0.506 \\
0.90 & 0.513 & 0.482 & 0.497 \\
\bottomrule
\end{tabular}
\end{center}

The tradeoff is clear: lower thresholds improve recall (fewer PCL paragraphs missed) at the cost of precision (more false positives); higher thresholds do the reverse. The optimal point at $t=0.84$ produces a near-balanced precision-recall tradeoff (0.495 vs 0.518), which is the right behaviour for a metric that weights both equally (F1).

\noindent\textbf{What improvement would look like.}
The error analysis suggests two directions that could improve the model beyond the current 0.506: (1) enriching the training signal to capture pragmatic stance — for example, through data augmentation or rationale-based training — to address implicit-stance false negatives; and (2) adding negation-aware or discourse-level features to prevent the model from triggering on vulnerable-group vocabulary in non-patronising contexts. Both are non-trivial and outside the scope of this coursework, but they are well-motivated by the observed error patterns.

\end{document}
