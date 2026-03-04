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

The main contribution is a new dataset for PCL. The authors collect exactly 10,637 news paragraphs drawn from the News on Web (NoW) corpus, covering 10 vulnerability keywords (e.g.\ \textit{homeless}, \textit{refugee}, \textit{immigrant}) across 20 English-speaking countries, annotated by three expert annotators (two primary, one referee). That matters because PCL is subtler than hate speech and there was no dedicated English-language paragraph-level dataset for it before this work.

A second major contribution is the annotation framework and taxonomy. The paper not only labels PCL presence at paragraph level, but also identifies 3,554 PCL text spans and assigns each to one of seven fine-grained categories (e.g.\ \textit{Unbalanced power relations}, \textit{Compassion}, \textit{Metaphor}) grouped under three higher-level types (\textit{The Saviour}, \textit{The Expert}, \textit{The Poet}). This makes the resource useful for both binary classification (Task 1) and multi-label category recognition (Task 2).

The paper also contributes a suite of baseline experiments across SVM, BiLSTM, and four BERT-family models (BERT-base, BERT-large, RoBERTa, DistilBERT), evaluated via 10-fold cross-validation. RoBERTa achieves F1\,=\,70.63 on Task 1, setting a clear performance ceiling for future work to beat.

\subsubsection*{Q2. Technical strengths that justify publication (2 marks)}

One clear strength of the paper is that the problem is well motivated. The authors explain clearly why PCL matters and why it is different from other harmful language tasks. In particular, they show that PCL is often implicit and can sound positive on the surface, which makes it harder to detect and worth studying on its own.

Another strength is the annotation design. Three expert annotators are used, with ann1 and ann2 annotating the full dataset independently and ann3 acting as a referee only for the 590 total disagreements (label~0 vs label~2). The two-step process — Step 1 determines PCL presence using a 3-point scale (0/1/2), then Step 2 annotates spans and assigns category labels via the BRAT tool — is a sensible way to handle a difficult and subjective task. The authors are also upfront about ambiguity, reporting a paragraph-level Cohen's $\kappa$ of 41\% overall, rising to a substantial 61\% once borderline cases are excluded (Landis and Koch, 1977).

The taxonomy is also a strong point. By defining different types of patronizing language and applying those labels in the dataset, the paper gives future researchers more than just a binary benchmark. This increases the long-term value of the resource.

Finally, the paper includes a reasonable set of baseline experiments and some qualitative analysis. The comparison across different model types, plus examples of model errors, makes the paper more useful and easier to build on.

\subsubsection*{Q3. Key weaknesses / areas with insufficient evidence (2 marks)}

The biggest weakness is the level of subjectivity in the annotations. The paragraph-level $\kappa$ is only 41\%, with 590 outright contradictions (label~0 vs label~2) out of 10,637 paragraphs. Category-level span agreement ranges from 48.34\% (Authority voice) to 66.72\% (The poorer, the merrier). The paper reports these numbers honestly, which is commendable, but it does not discuss how this annotation noise affects the reliability of downstream model comparisons. For example, a 3-point F1 difference between models may not be meaningful if the gold labels themselves are uncertain for a fifth of the data.

Another weakness is possible sampling bias. The dataset is built using 10 pre-selected keywords linked to vulnerable groups. This is practical, but it likely introduces lexical patterns that models can exploit as shortcuts. The experimental results are consistent with this: RoBERTa achieves F1\,=\,89.4 on \textit{Unbalanced power relations} — whose markers (``us'', ``they'', ``help'') are lexically predictable — but only F1\,=\,43.4 on \textit{Metaphor} and F1\,=\,20.5 on \textit{The poorer, the merrier}. The paper attributes the poor performance on difficult categories to the need for world knowledge, but provides no direct evidence (no probing experiments, no keyword removal ablations) to support that claim.

A further reproducibility concern is the anomalous BERT-large result. BERT-large achieves F1\,=\,53.91, which is lower than the much simpler BiLSTM (F1\,=\,57.75). The paper notes that BERT-large may overfit given the small positive class (995 PCL paragraphs), but offers no ablation to confirm this, leaving an unexplained result for readers to build on.

\textbf{Recommendation: Weak Accept.} The dataset and taxonomy are a genuine community contribution and the task is important. I recommend the authors (1) add a keyword-removal ablation to test whether models are exploiting lexical shortcuts rather than learning genuine PCL patterns, and (2) report per-category F1 alongside overall F1 as the primary Task~1 metric, so that future work can track progress on the harder categories (Metaphor, Authority voice) independently of the easier ones (Unbalanced power relations).

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

Each of the three changes addresses a specific problem identified in the EDA.

The weighted loss directly attacks the class imbalance. Without it, a model trained with standard cross-entropy quickly learns that predicting No-PCL on everything gives ~90\% accuracy, which is technically correct but completely useless for the task. By setting \texttt{pos\_weight}\,=\,9.55 I force the gradient to treat each PCL example as heavily as the majority class, so the model cannot ignore the minority class. I expected this change alone to substantially improve recall on PCL, even if it trades some precision.

The auxiliary category loss addresses the finding from EDA Technique 2: the discriminative signal is in how a community is \emph{framed}, not which community is mentioned. A model that only sees a binary PCL/No-PCL signal can still learn surface patterns like ``less fortunate'' without understanding why they are patronising. By also training on the seven category logits, I push the encoder to represent the type of patronising stance being expressed — the saviour dynamic, the compassion framing, the presupposition — rather than just whether certain words appear. This should improve precision on the harder, more framing-dependent cases that a purely binary signal would miss.

Threshold tuning is motivated by the asymmetry of the task. With a 1:9.5 class ratio, the default threshold of 0.5 is not optimal: it was calibrated for balanced classes. Searching over $[0.30, 0.70]$ lets me find the point where the model's precision–recall trade-off actually maximises F1 on the positive class. Retraining on train+dev in stage 2 then makes full use of all labelled data without having leaked the dev set into the model selection decision in stage 1.

In practice, best epoch was 6, best threshold 0.69, dev F1 0.615 after tuning — an improvement of $+0.135$ over the 0.48 baseline. The higher threshold (0.69 rather than 0.50) confirms the model needed to be pushed toward precision: with the weighted loss pulling toward recall, the threshold compensates to find a better balance. Test performance will appear on the leaderboard.

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

\noindent\textbf{Confusion matrix (official dev set, 2,094 samples, threshold\,=\,0.69).}

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

\begin{figure}[h]
    \centering
    \includegraphics[width=0.45\linewidth]{figures/eval_conf_matrix.pdf}
    \caption{Confusion matrix for the PCL class on the official dev set at threshold $t=0.69$.}
    \label{fig:conf_matrix}
\end{figure}

\noindent Taken together, Table~\ref{tab:threshold} and Figure~\ref{fig:conf_matrix} show that 72 of 199 PCL paragraphs are missed (recall on PCL 0.64) and 87 No-PCL are predicted as PCL (precision on PCL 0.59). The weighted loss stops the model collapsing to No-PCL; the rest of the errors are mostly borderline cases.

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

\subsubsection*{Other Local Evaluation: Component Analysis and Precision--Recall Trade-off}

\noindent\textbf{Indirect ablation via error breakdown.}
A full ablation would require retraining the model with each component removed in turn. As a proxy, I use the confusion matrix and the error analysis above to reason about what each change contributed.

\begin{center}
\begin{tabular}{lcc}
\toprule
Metric & PCL (positive) & No-PCL (negative) \\
\midrule
Precision     & 0.59 & 0.96 \\
Recall        & 0.64 & 0.95 \\
False rate    & FNR\,=\,36\% & FPR\,=\,4.6\% \\
\bottomrule
\end{tabular}
\end{center}

\noindent The model predicts PCL actively rather than collapsing to the majority class (FPR only 4.6\%), which is consistent with the weighted loss working as intended. Without it, the RoBERTa baseline (F1\,=\,0.48) was near majority-class collapse. The false negative rate of 36\% is the remaining problem: one in three PCL paragraphs is still missed.

The nature of the false negatives points to what the auxiliary category loss did and did not fix. Both missed examples (par\_id 107, par\_id 149) involve no surface PCL vocabulary — the condescension is purely attitudinal. The auxiliary loss was designed to push the encoder to learn the \emph{type} of patronising stance (compassion framing, saviour dynamic), not just the presence of PCL-adjacent words. The fact that these hard pragmatic cases are still missed suggests the category supervision helps at the macro level but cannot fully bridge the gap between semantic and pragmatic understanding.

The false positives (negation context, internal community voice) are a separate failure mode that the auxiliary loss also cannot resolve, because the category labels are assigned at the instance level and do not encode whether the voice is internal or external to the community.

\vspace{0.5em}
\noindent\textbf{Threshold sweep.}
Table~\ref{tab:threshold} and Figure~\ref{fig:threshold} summarise how performance changes as the decision threshold moves across the search range $[0.30, 0.75]$; the chosen threshold is $t = 0.69$.

\begin{table}[h]
\begin{center}
\begin{tabular}{cccc}
\toprule
Threshold & Precision & Recall & F1 \\
\midrule
0.40 & — & — & — \\
0.50 & — & — & — \\
0.55 & — & — & — \\
0.60 & — & — & — \\
0.65 & — & — & — \\
\textbf{0.69} & \textbf{0.593} & \textbf{0.638} & \textbf{0.615} \\
0.70 & — & — & — \\
0.75 & — & — & — \\
\bottomrule
\end{tabular}
\end{center}
\caption{Precision, recall, and F1 (PCL class) at selected thresholds on the dev set. Bold = chosen threshold (from confusion matrix).}
\label{tab:threshold}
\end{table}

\begin{figure}[h]
    \centering
    \includegraphics[width=0.92\linewidth]{figures/eval_threshold_sweep.pdf}
    \caption{Precision, recall, and F1 on the dev set as the decision threshold varies. The dashed line marks the chosen threshold $t=0.69$.}
    \label{fig:threshold}
\end{figure}

The optimal threshold is at the upper end of the search range. The F1 curve is fairly flat around 0.69, but lowering the threshold much further would add false alarms faster than it recovers true positives — consistent with the error analysis showing the remaining false negatives are hard implicit cases that the model assigns genuinely low probability to.

\vspace{0.5em}
\noindent\textbf{Precision--recall trade-off and confidence.}

\begin{figure}[h]
    \centering
    \includegraphics[width=0.48\linewidth]{figures/eval_pr_curve.pdf}\hfill
    \includegraphics[width=0.48\linewidth]{figures/eval_prob_dist.pdf}
    \caption{Left: Precision--Recall curve for the PCL class. The dot marks the chosen operating point ($t=0.69$); the dashed line shows a no-skill baseline. Right: distribution of model probabilities split by outcome (TP/FP/FN/TN, log scale).}
    \label{fig:pr_and_dist}
\end{figure}

The metrics at $t = 0.69$ (Table~\ref{tab:threshold}) give precision 0.59 and recall 0.64 on the PCL class. The PR curve in Figure~\ref{fig:pr_and_dist} (left) shows that the model has room above the no-skill baseline across most recall values, but precision degrades steadily as recall increases; the chosen operating point balances the two. Pushing recall toward 0.80+ would require accepting precision well below 0.50.

The probability distribution in Figure~\ref{fig:pr_and_dist} (right) is consistent with the qualitative error analysis. False negatives (red) cluster near zero probability — the model is genuinely uncertain about the hard pragmatic cases rather than confidently wrong. False positives (orange) are concentrated at high probabilities, driven by strong lexical cues about vulnerable groups even when the sentence is not patronising (e.g.\ negation or first-person community voice). True positives (green) and true negatives (grey) sit at the expected ends of the spectrum. Overall, the threshold $t = 0.69$ separates the classes as well as we can reasonably hope for on this task.

\end{document}
