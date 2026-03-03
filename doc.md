\documentclass{coursework}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{url}

\title{Natural Language Processing Coursework}
\author{Ivan Artiukhov}
\date{March 2026}

% ── FRONT PAGE (spec: link to repo + leaderboard name) ─────────────────────────

\begin{document}

\noindent\textbf{GitHub repository:} \url{https://github.com/artvanya/NLP_coursework}\\
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

The baseline is fine-tuned RoBERTa-base with standard cross-entropy, achieving F1\,=\,0.48 on dev and 0.49 on test. I keep RoBERTa-base and add three changes motivated by the EDA.

\textbf{1. Class-weighted \texttt{BCEWithLogitsLoss}.}
The training set has 794 PCL and 7,581 No-PCL (ratio 1:9.5). I use \texttt{BCEWithLogitsLoss} with \texttt{pos\_weight}\,=\,9.55 so each PCL example counts as much as 9.55 No-PCL in the gradient. I do not use an oversampling sampler so the loss is the only correction for imbalance.

\textbf{2. Multi-task auxiliary loss.}
The model has 8 outputs: logit\,0 for binary PCL, logits\,1--7 for the seven PCL categories in the dataset (Unbalanced power relations, Shallow solution, Presupposition, Authority voice, Metaphor, Compassion, The poorer the merrier). I train the binary head with the weighted BCE above and add a second BCE loss on the 7 category logits with per-category \texttt{pos\_weight}, scaled by 0.3. This pushes the encoder to learn the kind of PCL, not only whether it is present, which fits the EDA finding that framing matters more than topic.

\textbf{3. Threshold tuning and two-stage training.}
After stage 1 (train on train only, early stop on dev F1), I search the threshold in $[0.35, 0.65]$ on dev and pick the one that maximises PCL F1. Then in stage 2 I retrain from scratch on train+dev for the same number of epochs as the best epoch, save that model, and use it with the chosen threshold to produce \texttt{dev.txt} and \texttt{test.txt}.

Rest of the setup: \texttt{max\_length}=256, AdamW, lr $2\!\times\!10^{-5}$, weight decay 0.01, batch size 16 with gradient accumulation 2 (effective 32), linear warmup 10\%, up to 8 epochs with patience 3.

\begin{figure}[h]
\centering
\fbox{\parbox{0.85\linewidth}{\centering
\textbf{Stage 1:} Train on train $\rightarrow$ track dev F1 $\rightarrow$ save best checkpoint $\rightarrow$ tune threshold on dev.\\
\vspace{0.3em}
\textbf{Stage 2:} Retrain on train+dev for best\_epoch epochs $\rightarrow$ save \texttt{final\_model\_roberta.pt} $\rightarrow$ predict dev and test with best threshold.
}}
\caption{Two-stage training and prediction pipeline.}
\label{fig:pipeline}
\end{figure}

\subsubsection*{Rationale and expected outcome}

EDA showed strong class imbalance (weighted loss) and that discriminative signal is in framing rather than keywords (auxiliary loss on categories helps). Threshold tuning and retraining on train+dev are standard ways to use the dev set without leaking it into the model choice in stage 1. Dev F1 after threshold tuning (e.g. around 0.61 in a typical run) improves over the baseline 0.48; test performance is reported on the leaderboard.

\subsection*{Exercise 4: Model Training}

The repository linked on the front page contains the training notebook and the \texttt{BestModel/} folder. The folder includes the submitted model and prediction files:

\begin{itemize}
    \item \texttt{final\_model\_roberta.pt} — weights of the model used for \texttt{dev.txt} and \texttt{test.txt} (stage 2: trained on train+dev)
    \item \texttt{dev.txt} — one prediction per line for the official dev set (2,094 lines; 0 = No PCL, 1 = PCL)
    \item \texttt{test.txt} — one prediction per line for the official test set (3,832 lines; 0 = No PCL, 1 = PCL)
\end{itemize}

The full training pipeline (data loaders, model, two-stage training, threshold tuning, saving predictions) is in \texttt{pcl\_roberta\_improved.ipynb} in the repository root. The repository must be made public after the submission deadline.

\subsection*{Exercise 5.1: Global Evaluation}

\texttt{dev.txt} and \texttt{test.txt} are in \texttt{BestModel/}. Format: one prediction (0 or 1) per line, matching the order of the official dev and test inputs. Dev set result (from a full run of the notebook):

\begin{center}
\begin{tabular}{llll}
\toprule
Model & Dev F1 & Threshold & vs.\ baseline \\
\midrule
RoBERTa-base (baseline) & 0.48 & 0.50 & — \\
Ours (RoBERTa + weighted BCE + auxiliary + threshold) & \textbf{0.614} & 0.54 & +0.13 \\
\bottomrule
\end{tabular}
\end{center}

Test set results will appear on the leaderboard after the deadline.

\subsection*{Exercise 5.2: Local Evaluation}

\subsubsection*{Error Analysis}

\noindent\textbf{Confusion matrix (official dev set, 2,094 samples, threshold\,=\,0.54):}

\begin{center}
\begin{tabular}{lcc}
\toprule
 & Predicted No-PCL & Predicted PCL \\
\midrule
Actual No-PCL & 1,803 (TN) & 92 (FP) \\
Actual PCL    & 70 (FN) & 129 (TP) \\
\bottomrule
\end{tabular}
\end{center}

\noindent So 70 of 199 PCL paragraphs are missed (recall on PCL $\approx$ 0.65) and 92 No-PCL are predicted as PCL (precision on PCL $\approx$ 0.58). The model no longer collapses to No-PCL thanks to the weighted loss, but many errors remain where the boundary is ambiguous.

\vspace{0.5em}
\noindent\textbf{False negatives — PCL the model missed.}
Among the false negatives a consistent pattern: these are cases where the PCL is implicit in the author's \emph{attitude} rather than explicit in the vocabulary.

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
Among the false positives a different pattern: the model fires on vocabulary associated with vulnerable groups even when the framing is neutral or sympathetic without being condescending.

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
The threshold search is restricted to $[0.35, 0.65]$ so the chosen value generalises better to test. In a typical run the best dev F1 is reached around $t \approx 0.54$. Lower $t$ increases recall and lowers precision; higher $t$ does the opposite. The chosen threshold balances the two for F1.

\noindent\textbf{What would help next.}
The errors point to two gaps: (1) implicit or pragmatic PCL that is not signalled by clear keywords, and (2) over-reaction to phrases about vulnerable groups even when the sentence is not patronising. Addressing (1) would need extra signal (e.g. rationale or stance); addressing (2) would need better use of negation and context. Both are beyond this coursework.

\end{document}
