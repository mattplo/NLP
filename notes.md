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
