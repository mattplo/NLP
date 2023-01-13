# NLP

## 1

### To Balkanisation

* Low level: POS (Part Of Speech) tagging, morphological analysis, language modelling, lemmatisation
* Mid low: Named entity recognition, co-reference resolution, constituency parsing, dependency parsing, word sense disambiguation, multi-word expressions
* Mid high: sentiment analysis, text classification, anaphora resolution (winograd schemes: the cup doesn't fit in the suitcase because it is too big/small)
* high level: translation, summarizing, paraphrasing, natural language understanding/inference, generation, question answering..

### Voc

* Lemma (quotation form): single form that groups conjugated words..
* Morpheme: smallest meaningful linguistic unit (tion)
* Type, token: // class, instance
* Word:
* Character:

## 2

### Structured prediction

Linguistic units are structured objects that are interpreted in each other's context.
ex: Characters, morphology

### Hidden Markov Chain and Viterbi Algorithm

$$\hat{t} = argmax_t P(s|t)P(t) = argmax_t \prod_i P(s_i|t_i)P(t_i)\prod_i P(t_i|t_{i-1})$$
$$P(t_1) \prod_{i=1}^n P(s_i|t_i) \prod_{i=2}^n P(t_i|t_{i-1})
= \left[ P(t) \prod_{i=1}^{n-1} P(s_i|t_i) \prod_{i=2}^{n-1} P(t_i|t_{i-1})\right] \left[ P(s_n|t_n) P(t_n|t_{n-1})\right]
$$

We need to know

* $P(t)$ start probas for each tag (POS)
* $T$ transition probability between pairs of tags (POS x POS)
* $E$ emission probability between tags and words (POS x VOC)

Set $V_{1t}= I_t E_{ts_1}$

$V \in \mathbb{R}^{n \times |T|}$

$P \in T^{n \times |T|}$

$\forall i \in [2, n]$

$V_{it} = \max_{t'} T_{t't} V_{(i-1) t'} E_{t s_i}$

$T_{it} = argmax_{t'} E_{t s_i} T_{t' t} V_{(i-1) t'}$

Use sum instead of max -> $\sum_t P(s|t)P(t) = P(s)$

## 4

Language model

Assign probability to a given sentence

$\mathcal{V}$ vocabulary

$\mathcal{P}(\mathcal{V}^+)$

$f:\mathcal{V}^+ \rightarrow [0, 1]$

$S \in f:\mathcal{V}^+$

$\mathcal{L} \subset \mathcal{P}(S)=0 \text{ iff } S \notin \mathcal{L}$

$\mathcal{C}$ corpus: set of sentences

$w \in \mathcal{V}, \qquad \pi(w)=\frac{freq(w)}{length(\mathcal{C})}$

$\lvert S \rvert = n$

$f_1(S)= \prod_{i=1}^n \pi(w_i)$

$f_2(S)= \pi(w_1) \prod_{i=1}^n \pi(w_{i+1} \mid w_i)$

$f_k(S)= \pi(w_1 \dots w_{k-1}) \prod_{i=k}^n \pi(w_{i} \mid w_{i-(k+1)} \dots w_{i-1})$

Space for $f_1$: O(n)

Space for $f_2$: n2 + n

Space for $f_k$: n^k

Pseudo-counts

For pos tagging: f2 np

pick paper from acl anthology in 2 weeks (less than 5 years)
(must read paper in nlp)
