\documentclass{coursework}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{url}
\renewcommand{\thefigure}{\arabic{figure}}
\renewcommand{\thetable}{\arabic{table}}

\title{Natural Language Processing Coursework}
\author{Ivan Artiukhov}
\date{March 2026}

% ── FRONT PAGE (spec: link to repo + leaderboard name) ─────────────────────────

\begin{document}

\noindent\textbf{GitHub repository:} \url{https://github.com/artvanya/NLP_coursework}\\
\textbf{Leaderboard name:} IvanArtiukhov

\subsection*{Exercise 1: Critical Review of \textit{Don't Patronize Me!}}

\subsubsection*{Q1. Primary contributions of this work (2 marks)}

The main contribution is the dataset itself. The authors collected 10,637 news paragraphs from the News on Web corpus, using 10 vulnerability keywords (e.g.\ \textit{homeless}, \textit{refugee}, \textit{poor families}) across 20 English-speaking countries, and had three expert annotators label them. Before this, there was no English paragraph-level dataset focused on PCL, which is harder to pin down than hate speech because it often sounds positive on the surface.

Alongside the data, the paper introduces a taxonomy of seven PCL categories (e.g.\ \textit{Unbalanced power relations}, \textit{Compassion}, \textit{Metaphor}) grouped under three types, with 3,554 labelled spans. This makes the resource useful beyond binary classification. The baselines — SVM, BiLSTM, and four transformer variants via 10-fold cross-validation — give a clear starting point; RoBERTa reaches F1\,=\,70.63, the best of the bunch.

\subsubsection*{Q2. Technical strengths that justify publication (2 marks)}

The motivation is strong. The paper explains clearly why PCL is different from other harmful language tasks — it's usually unintentional, can sound supportive, and was barely studied in NLP before this. That makes a convincing case for why the dataset is needed.

The annotation process is well thought out. Ann1 and ann2 labelled the whole dataset independently; ann3 only stepped in for the 590 outright contradictions. Reporting a paragraph-level $\kappa$ of 41\% (61\% once borderline cases are excluded) is honest, and the two-step design — paragraph label first, then span-level category — is a sensible way to manage a difficult, subjective task.

The baselines round it off nicely. Six models are compared, which shows the task is feasible while also leaving plenty of room for improvement. Including error analysis examples makes the paper easier to build on.

\subsubsection*{Q3. Key weaknesses / areas with insufficient evidence (2 marks)}

The biggest issue is what the low agreement actually means for the experiments. $\kappa$\,=\,41\% with 590 outright contradictions out of 10,637 paragraphs is low, and category-level agreement ranges from 48.34\% (Authority voice) to 66.72\% (The poorer, the merrier). The paper flags these numbers honestly but doesn't discuss the implications — a 3-point F1 gap between two models doesn't mean much if the labels themselves are uncertain for a large chunk of the data.

Sampling bias is also a problem that goes unexamined. Because the data was collected with 10 fixed keywords, models can learn vocabulary shortcuts rather than real PCL. The results hint at this: RoBERTa scores F1\,=\,89.4 on \textit{Unbalanced power relations} — where words like ``us'', ``they'', and ``help'' are strong signals — but only 43.4 on \textit{Metaphor} and 20.5 on \textit{The poorer, the merrier}. The paper says the harder categories require world knowledge, but there's no ablation or probing experiment to back that up.

There's also an unexplained result: BERT-large (F1\,=\,53.91) does worse than the BiLSTM (57.75). Overfitting on a small positive class (995 examples) is a reasonable guess, but the authors don't verify it.

\textbf{Recommendation: Weak Accept.} The dataset and taxonomy are a useful contribution to the field. My main suggestions are: (1) add a keyword-removal ablation to check whether models are learning real PCL patterns or just vocabulary shortcuts, and (2) report per-category F1 as a primary metric so future work can track progress on the harder categories separately from the easier ones.

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
With a standard loss the model would just predict No-PCL and get ~90\% accuracy while missing PCL entirely. So F1 on the positive class is the right metric. I used a class-weighted loss and \texttt{max\_length}=256 so most paragraphs aren't truncated.

\subsubsection*{EDA Technique 2: Discriminative Bigram Analysis}

\begin{figure}[h]
    \centering
    \includegraphics[width=\linewidth]{figures/eda2_bigrams.pdf}
    \caption{Top 15 bigrams by log-odds ratio for each class. Bars show how much more likely a bigram is in one class relative to the other.}
    \label{fig:eda2}
\end{figure}

\noindent
\textbf{Analysis.}
The log-odds ratio separates the two classes clearly. On the PCL side you get phrases like ``less fortunate'', ``homeless person'', ``giving back'', ``better lives'' — the kind of soft, sympathetic wording that can still be patronising. On the No-PCL side you get more factual/political bigrams: ``illegal immigrants'', ``climate change'', ``trump administration'', ``refugee camp''. So it's not the topic that makes something PCL (both sides talk about similar issues) but the framing of it. Political or charged terms often show up in neutral reporting; the patronising stuff is in how people are described.

\noindent
\textbf{Impact on approach.}
Because the signal is in framing rather than keywords, a bag-of-words style model would get confused — the same words can be PCL or not depending on context. So I used a transformer (RoBERTa) and added auxiliary supervision on the seven PCL categories so the model has to learn the type of PCL, not just presence.

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

The auxiliary category loss addresses the finding from EDA Technique 2: the discriminative signal is in how a community is framed, not which community is mentioned. A model that only sees a binary PCL/No-PCL signal can still learn surface patterns like ``less fortunate'' without understanding why they are patronising. By also training on the seven category logits, I push the encoder to represent what kind of patronising language is being used — the saviour framing, the compassion angle, the presupposition — rather than just whether certain words appear. This should improve precision on the harder, more framing-dependent cases that a purely binary signal would miss.

Threshold tuning is motivated by the asymmetry of the task. With a 1:9.5 class ratio, the default threshold of 0.5 is not optimal: it was calibrated for balanced classes. Searching over $[0.30, 0.70]$ lets me find the point where the model's precision–recall trade-off actually maximises F1 on the positive class. Retraining on train+dev in stage 2 then makes full use of all labelled data without having leaked the dev set into the model selection decision in stage 1.

In practice, best epoch was 6, best threshold 0.69, dev F1 0.615 after tuning — an improvement of $+0.135$ over the 0.48 baseline. The higher threshold (0.69 rather than 0.50) confirms the model needed to be pushed toward precision: with the weighted loss pulling toward recall, the threshold compensates to find a better balance. Test performance will appear on the leaderboard.

\subsection*{Exercise 4: Model Training}

The repo (link on the front page) contains the notebook \texttt{pcl\_roberta\_improved.ipynb} with the full pipeline (data loading, model, two-stage training, threshold tuning, prediction files) and a \texttt{BestModel/} folder. In \texttt{BestModel/} you'll find:

\begin{itemize}
    \item \texttt{final\_model\_roberta.pt} — the model used to generate the predictions (trained on train+dev in stage 2)
    \item \texttt{dev.txt} — one prediction per line for the official dev set (2,094 lines; 0 = No PCL, 1 = PCL)
    \item \texttt{test.txt} — one prediction per line for the official test set (3,832 lines; 0 = No PCL, 1 = PCL)
\end{itemize}

\subsection*{Exercise 5.1: Global Evaluation}

\texttt{dev.txt} and \texttt{test.txt} live in \texttt{BestModel/}. One prediction per line (0 or 1). Dev result from my run:

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

\noindent Test set results will appear on the leaderboard after the deadline.

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

\noindent So 72 of 199 PCL paragraphs are missed (recall 0.64) and 87 No-PCL are flagged as PCL (precision 0.59). The weighted loss stops the model collapsing to No-PCL; the remaining errors are mostly borderline cases.

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

What these false negatives have in common is that you need to read between the lines to spot the condescension — there's nothing in the words themselves that flags it. That's the hardest failure mode for any model on this task.

\vspace{0.5em}
\noindent\textbf{False positives — No-PCL predicted as PCL.}
The false positives are different: the model often fires on vocabulary about vulnerable groups even when the sentence isn't actually condescending.

\begin{quote}
\textit{``His friends at the Chevron want people to know he wasn't just a faceless homeless person. He was their friend and their family.''} [par\_id 8591]
\end{quote}

This sentence contains ``homeless person'' — a strong PCL bigram from EDA Technique 2 — but the sentence is arguing against dehumanising framing. The model has learned an association between the lexeme and PCL without learning the negation context.

\begin{quote}
\textit{``So we do need to heal ourselves as an Aboriginal Torres Strait Islander community, but also as a nation.''} [par\_id 8480]
\end{quote}

This is a first-person statement by a member of the community being discussed. There's no outside observer being condescending — the speaker is talking about their own community. The model can't tell the difference between someone speaking from within a group and someone talking down to it from outside, which is a real limitation.

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

The nature of the false negatives points to what the auxiliary category loss did and did not fix. Both missed examples (par\_id 107, par\_id 149) have no surface PCL vocabulary — the condescension is entirely in the attitude, not the words. The auxiliary loss was meant to push the encoder toward recognising what \emph{type} of patronising language is being used (compassion framing, saviour angle), not just whether PCL words are present. The fact that these cases are still missed suggests the category supervision helps at a broad level, but can't close the gap when the PCL is purely implied.

The false positives (negation context, internal community voice) are a separate failure mode that the auxiliary loss also cannot resolve, because the category labels are assigned at the instance level and do not encode whether the voice is internal or external to the community.

\vspace{0.5em}
\noindent\textbf{Threshold sweep.}
Figure~\ref{fig:threshold} shows how precision, recall, and F1 change across the full search range $[0.30, 0.75]$.

\begin{figure}[h]
    \centering
    \includegraphics[width=0.92\linewidth]{figures/eval_threshold_sweep.pdf}
    \caption{Precision, recall, and F1 on the dev set as the decision threshold varies. The dashed line marks the chosen threshold $t=0.69$.}
    \label{fig:threshold}
\end{figure}

The optimal threshold lands near the top of the search range. The F1 curve is fairly flat around 0.69, but going much lower adds false alarms faster than it picks up true positives — which makes sense given the error analysis: the remaining false negatives are hard implicit cases the model assigns low probability to regardless of threshold.

\vspace{0.5em}
\noindent\textbf{Precision--recall trade-off and confidence.}

\begin{figure}[h]
    \centering
    \includegraphics[width=0.48\linewidth]{figures/eval_pr_curve.pdf}\hfill
    \includegraphics[width=0.48\linewidth]{figures/eval_prob_dist.pdf}
    \caption{Left: Precision--Recall curve for the PCL class. The dot marks the chosen operating point ($t=0.69$); the dashed line shows a no-skill baseline. Right: distribution of model probabilities split by outcome (TP/FP/FN/TN, log scale).}
    \label{fig:pr_and_dist}
\end{figure}

At $t = 0.69$, precision is 0.59 and recall 0.64 on the PCL class. The PR curve in Figure~\ref{fig:pr_and_dist} (left) shows that the model has room above the no-skill baseline across most recall values, but precision degrades steadily as recall increases; the chosen operating point balances the two. Pushing recall toward 0.80+ would require accepting precision well below 0.50.

The probability distribution in Figure~\ref{fig:pr_and_dist} (right) is consistent with the qualitative error analysis. False negatives (red) cluster near zero probability — the model is genuinely uncertain about the hard pragmatic cases rather than confidently wrong. False positives (orange) are concentrated at high probabilities, driven by strong lexical cues about vulnerable groups even when the sentence is not patronising (e.g.\ negation or first-person community voice). True positives (green) and true negatives (grey) sit at the expected ends of the spectrum. Overall, the threshold $t = 0.69$ separates the classes as well as we can reasonably hope for on this task.

\end{document}
