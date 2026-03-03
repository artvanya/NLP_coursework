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

I reviewed the paper by Perez-Almendros et al. (COLING 2020) that introduces the Don't Patronize Me! dataset for detecting patronizing and condescending language (PCL) in news.

\subsubsection*{Q1. Primary contributions of this work (2 marks)}

The main contribution is a new dataset for PCL. They collect 10,637 news paragraphs from 20 English-speaking countries, focusing on text about vulnerable communities (refugees, homeless, poor families, etc.). That matters because PCL is subtler than hate speech and there wasn't much data for it before.

A second major contribution is the annotation framework. The paper does not only label whether a paragraph contains PCL or not, but also provides a finer-grained category scheme for different types of PCL. This makes the dataset useful for both binary classification and more detailed analysis of how patronizing language is expressed.

The paper also contributes baseline experiments for the task. The authors compare traditional machine learning methods and neural models (including transformer-based models), which helps show that the task is possible but still challenging. This is useful for future work because it gives a clear starting point for comparison.

\subsubsection*{Q2. Technical strengths that justify publication (2 marks)}

One clear strength of the paper is that the problem is well motivated. The authors explain clearly why PCL matters and why it is different from other harmful language tasks. In particular, they show that PCL is often implicit and can sound positive on the surface, which makes it harder to detect and worth studying on its own.

Another strength is the annotation design. The authors use expert annotators and include a referee annotator when there are major disagreements. They also use a two-stage process (first detecting PCL, then annotating spans and categories), which is a sensible way to handle a difficult and subjective task. They're also upfront about ambiguity in the labels, which I think is a plus.

The taxonomy is also a strong point. By defining different types of patronizing language and applying those labels in the dataset, the paper gives future researchers more than just a binary benchmark. This increases the long-term value of the resource.

Finally, the paper includes a reasonable set of baseline experiments and some qualitative analysis. The comparison across different model types, plus examples of model errors, makes the paper more useful and easier to build on.

\subsubsection*{Q3. Key weaknesses / areas with insufficient evidence (2 marks)}

The biggest weakness is the level of subjectivity in the annotations. The authors are honest about this, which is good, but the agreement scores show that the task is difficult even for humans. This does not make the dataset unusable, but it does mean that stronger validation or a deeper discussion of annotation consistency would have made the paper stronger.

Another weakness is possible sampling bias in how the data was collected. The dataset is built using pre-selected keywords linked to vulnerable groups. This is practical, but it may introduce lexical patterns that models can exploit. The paper does not fully test whether models are learning real PCL patterns or just relying on keyword-related shortcuts.

The experimental section is useful, but still fairly limited. The baselines are a good start, but the paper does not include many diagnostic experiments or ablations. For example, it would have been helpful to test class imbalance strategies, threshold effects, or cross-domain generalization more directly.

Some of the explanations in the analysis are reasonable, but not fully proven. For instance, the authors suggest that poor performance on certain categories is related to the need for world knowledge or common sense. That may be true, but the paper does not provide direct evidence for this claim, so it remains a plausible interpretation rather than a demonstrated result.

Overall the paper is a solid contribution — the dataset and taxonomy are useful and the task is important. The weaknesses are mostly about how subjective the labels are and that the experiments could have gone deeper, not that the work isn't useful.

\subsection*{Exercise 2: Exploratory Data Analysis}

All analysis in this section is performed on the official training split (8,375 samples) using the binary label derived from the raw annotation scores in \texttt{dontpatronizeme\_pcl.tsv}: a paragraph is labelled PCL (1) if at least two annotators flagged it, and No-PCL (0) otherwise.

\subsubsection*{EDA Technique 1: Class Distribution and Text-Length Analysis}

\begin{figure}[h]
    \centering
    \includegraphics[width=\linewidth]{figures/eda1_class_length.pdf}
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
The data is very imbalanced: about 9 in 10 paragraphs are No-PCL, so the ratio is roughly 1:9.5. Length is similar in both classes; PCL paragraphs are slightly longer on average (median 47 vs 42 words). Almost all samples are under 256 words.

\noindent
\textbf{Impact on approach.}
With a standard loss the model would just predict No-PCL and get ~90\% accuracy while missing PCL entirely. So F1 on the positive class is the right metric (and that's what the task uses). I used a class-weighted loss and \texttt{max\_length}=256 so most paragraphs aren't truncated.

\subsubsection*{EDA Technique 2: Discriminative Bigram Analysis}

\begin{figure}[h]
    \centering
    \includegraphics[width=\linewidth]{figures/eda2_bigrams.pdf}
    \caption{Top 15 bigrams by log-odds ratio for each class. Bars show how much more likely a bigram is in one class relative to the other.}
    \label{fig:eda2}
\end{figure}

\noindent
\textbf{Analysis.}
The log-odds ratio separates the two classes clearly. On the PCL side you get phrases like ``less fortunate'', ``homeless person'', ``giving back'', ``better lives'' — the kind of soft, sympathetic wording that can still be patronising. On the No-PCL side you get more factual/political bigrams: ``illegal immigrants'', ``climate change'', ``trump administration'', ``refugee camp''. So it's not the topic that makes something PCL (both sides talk about similar issues) but the \emph{framing}. Political or charged terms often show up in neutral reporting; the patronising stuff is in how people are described.

\noindent
\textbf{Impact on approach.}
Because the signal is in framing rather than keywords, a bag-of-words style model would get confused — the same words can be PCL or not depending on context. So I used a transformer (RoBERTa) and added auxiliary supervision on the seven PCL categories so the model has to learn the \emph{type} of PCL, not just presence. That fits the idea that framing is what matters.

\subsection*{Exercise 3: Proposed Approach}

\subsubsection*{Proposed approach}

The baseline is fine-tuned RoBERTa-base with standard cross-entropy, achieving F1\,=\,0.48 on dev and 0.49 on test. I keep RoBERTa-base and add three changes motivated by the EDA.

\textbf{1. Class-weighted \texttt{BCEWithLogitsLoss}.}
The training set has 794 PCL and 7,581 No-PCL (ratio 1:9.5). I use \texttt{BCEWithLogitsLoss} with \texttt{pos\_weight}\,=\,9.55 so each PCL example counts as much as 9.55 No-PCL in the gradient. I do not use an oversampling sampler so the loss is the only correction for imbalance.

\textbf{2. Multi-task auxiliary loss.}
The model has 8 outputs: logit\,0 for binary PCL, logits\,1--7 for the seven PCL categories in the dataset (Unbalanced power relations, Shallow solution, Presupposition, Authority voice, Metaphor, Compassion, The poorer the merrier). I train the binary head with the weighted BCE above and add a second BCE loss on the 7 category logits with per-category \texttt{pos\_weight}, scaled by 0.3. This pushes the encoder to learn the kind of PCL, not only whether it is present, which fits the EDA finding that framing matters more than topic.

\textbf{3. Threshold tuning and two-stage training.}
After stage 1 (train on train only, early stop on dev F1), I search the threshold in $[0.30, 0.70]$ on dev and pick the one that maximises PCL F1. Then in stage 2 I retrain from scratch on train+dev for the same number of epochs as the best epoch, save that model, and use it with the chosen threshold to produce \texttt{dev.txt} and \texttt{test.txt}.

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

EDA showed strong class imbalance (weighted loss) and that discriminative signal is in framing rather than keywords (auxiliary loss on categories helps). Threshold tuning and retraining on train+dev are standard ways to use the dev set without leaking it into the model choice in stage 1. In my run, best epoch was 6, best threshold 0.69, dev F1 0.615 after tuning. Test performance will be on the leaderboard.

\subsection*{Exercise 4: Model Training}

The repo (link on the front page) has the notebook and a \texttt{BestModel/} folder. In \texttt{BestModel/} you'll find:

\begin{itemize}
    \item \texttt{final\_model\_roberta.pt} — the model used to generate the predictions (trained on train+dev in stage 2)
    \item \texttt{dev.txt} — one prediction per line for the official dev set (2,094 lines; 0 = No PCL, 1 = PCL)
    \item \texttt{test.txt} — one prediction per line for the official test set (3,832 lines; 0 = No PCL, 1 = PCL)
\end{itemize}

The notebook \texttt{pcl\_roberta\_improved.ipynb} in the repo root has the full pipeline (data load, model, two-stage training, threshold tuning, writing the prediction files). Repo needs to be public after the deadline.

\subsection*{Exercise 5.1: Global Evaluation}

\texttt{dev.txt} and \texttt{test.txt} live in \texttt{BestModel/}. One prediction per line (0 or 1), same order as the official dev/test. Dev result from my run:

\begin{center}
\begin{tabular}{llll}
\toprule
Model & Dev F1 & Threshold & vs.\ baseline \\
\midrule
RoBERTa-base (baseline) & 0.48 & 0.50 & — \\
Ours (RoBERTa + weighted BCE + auxiliary + threshold) & \textbf{0.615} & 0.69 & +0.14 \\
\bottomrule
\end{tabular}
\end{center}

Test set results will appear on the leaderboard after the deadline.

\subsection*{Exercise 5.2: Local Evaluation}

\subsubsection*{Error Analysis}

\noindent\textbf{Confusion matrix (official dev set, 2,094 samples, threshold\,=\,0.69):}

\begin{center}
\begin{tabular}{lcc}
\toprule
 & Predicted No-PCL & Predicted PCL \\
\midrule
Actual No-PCL & 1,808 (TN) & 87 (FP) \\
Actual PCL    & 72 (FN) & 127 (TP) \\
\bottomrule
\end{tabular}
\end{center}

\noindent So 72 of 199 PCL paragraphs are missed (recall on PCL 0.64) and 87 No-PCL are predicted as PCL (precision on PCL 0.59). The weighted loss stops the model collapsing to No-PCL; the rest of the errors are mostly borderline cases.

\vspace{0.5em}
\noindent\textbf{False negatives — PCL the model missed.}
Among the false negatives there's a consistent pattern: the PCL is in the author's \emph{attitude} rather than in the words themselves.

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
The false positives are different: the model often fires on vocabulary about vulnerable groups even when the sentence isn't actually condescending.

\begin{quote}
\textit{``His friends at the Chevron want people to know he wasn't just a faceless homeless person. He was their friend and their family.''} [par\_id 8591]
\end{quote}

This sentence contains ``homeless person'' — a strong PCL bigram from EDA Technique 2 — but the sentence is explicitly \emph{arguing against} dehumanising framing. The model has learned an association between the lexeme and PCL without learning the negation context.

\begin{quote}
\textit{``So we do need to heal ourselves as an Aboriginal Torres Strait Islander community, but also as a nation.''} [par\_id 8480]
\end{quote}

This is a first-person statement by a member of the community being discussed. There is no outside observer adopting a patronising stance; the speaker is speaking for and about their own community. The model cannot distinguish internal community voice from external characterisation, which is a significant limitation.

\vspace{0.5em}
\noindent\textbf{Summary.}
So two kinds of mistakes: missing PCL when it's only in the attitude (no clear keywords), and flagging PCL when the wording is about vulnerable groups but the sentence isn't actually patronising. Both come back to the same thing — the EDA showed that framing matters more than topic, and that's the hard part to get right.

\subsubsection*{Other Local Evaluation: Threshold and Confidence Analysis}

\noindent\textbf{Threshold sensitivity.}
I search over $[0.30, 0.70]$ so the optimum isn't at the extremes. In my run the best was $t = 0.69$, which gave dev F1 0.615. Lower $t$ pushes recall up and precision down; higher $t$ the reverse. So 0.69 is a bit on the conservative side (fewer PCL predictions, higher precision on PCL).

\noindent\textbf{What would help next.}
To fix the missed PCL you'd want something that picks up on stance or pragmatics, not just wording. To cut the false positives you'd need the model to use negation and context better (e.g. ``wasn't just a faceless homeless person''). I didn't try that here.

\end{document}
