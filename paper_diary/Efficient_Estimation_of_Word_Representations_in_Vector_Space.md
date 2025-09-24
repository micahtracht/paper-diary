# Efficient Estimation of Word Representations in Vector Space - Mikolov, et al., 2013

## Problem
Training classic neural LMs (NNLM/RNNLM) to learn high-quality word vectors is computationally expensive, largely because the output softmax scales with vocabulary size V. The paper's goal is to learn high quality vectors from huge datasets (billions of words, or tokens) and with million+ word vocabularies at a far lower cost.

## Methods
 - More efficient softmax. In the standard training complexity for a neural network language model (NNLM) is N x D + N x D x H + H x V, where H x V dominates since V is so large. This term comes from computing the output softmax, which requires logits for all V words. In standard softmax, to do this you need to compute the logit for every word, then use that to compute the total sum. Doing this over all words V is where the V part of the H x V term comes from. However, by using hierarchical softmax over a Huffman binary tree, this can be reduced to log_2(unigram perplexity(V)), far lower.
 - New log-linear architectures:
    - CBOW. Take a symmetric context window (the C previous and C future tokens), and use those to predict the middle token. It does this through embedding each token with a shared projection matrix, and averaging them. Then, put this averaged vector into a log-linear classifier that predicts the current token. Since order doesn't matter in this set up, it's called a "bag of words" approach.
    - Skip-gram. Use the current word as input, and predict the C past and future words (the inverse of CBOW). For each center token, pick R in [1, C] (where C is your max distance from the center), and predict the R past and future words. Each prediction is just a hierarchical softmax computation (part of why optimizing softmax is so important!).

## Key idea/claim
Two simple, no-hidden-layer models (CBOW and Skip-gram) learn vectors that preserve linear regularities (analogy structure) while being orders of magnitude cheaper to train; hierarchical softmax with a Huffman vocabulary tree makes training practical at million-word scale.

## Results

Summary. On the smaller 320M-word setup with 640-dimensional vectors (paper Table 3), Skip-gram achieves the highest semantic analogy accuracy, while CBOW achieves the highest syntactic accuracy. On large-scale distributed training (6B words, 1000-dimensional vectors; paper Table 6), Skip-gram yields the best overall and semantic accuracy, while CBOW remains best on syntactic.  
Speed: In distributed training, CBOW and Skip-gram complete in ~2–2.5 days × O(10²) cores versus NNLM’s ~14 days × ~180 cores; CBOW is the fastest in wall-clock days.

Key points
- Accuracy (smaller setup): Skip-gram tops semantic; CBOW tops syntactic.  
- Accuracy (large-scale): Skip-gram tops overall and semantic; CBOW tops syntactic.  
- Speed: CBOW/Skip-gram are much faster than NNLM; CBOW is fastest in wall-clock days.

Table 1 — Accuracy on the Semantic–Syntactic Word Relationship benchmark (640-d vectors, 320M words; paper Table 3).

| Model Architecture | Semantic Accuracy (%) | Syntactic Accuracy (%) | MSR Word Relatedness (%) |
|---|---:|---:|---:|
| RNNLM            | 9   | 36  | 35 |
| NNLM             | 23  | 53  | 47 |
| CBOW             | 24  | 64 | 61 |
| Skip-gram        | 55 | 59  | 56 |

Table 2 — Large-scale distributed training: accuracy vs. training time (6B words; 1000-d vectors; paper Table 6).

| Model Architecture      | Semantic Accuracy (%) | Syntactic Accuracy (%) | Total Accuracy (%) | Training Time (days × CPU cores) |
|---|---:|---:|---:|---:|
| NNLM (100-d)           | 34.2 | 64.5 | 50.8 | 14 × 180 |
| CBOW (1000-d)          | 57.3 | 68.9 | 63.7 | 2 × 140 |
| Skip-gram (1000-d)     | 66.1 | 65.1 | 65.6 | 2.5 × 125 |


## Implications
 - Scalability: Hierarchical softmax + simple log linear models can easily train on far larger corpora, potentially >1 trillion tokens. As we all know, scale turned out to be *very* important.
 - NLP progress. NLP is about processing natural language, or, if you prefer, words. So making models that can generate semantically rich representations of words is key if we ever want machines to understand them. Modern GPT models use the same trick of projecting words into vectors, showing how fundamental the concept is.

## Limitations
 - No multi-word tokens. This makes it extra hard to ask models "questions" (fill in the blanks) where the answer is multiple words. The authors use the example of the state of New York: semantically one thing, but two words.
 - Word order is not used. It's a "bag of words" model, meaning the positions of the words, and the meaning that encodes, are lost.

## Takeaways
 - You can perform arithmetic on vector representations of words, and get reasonable results back (eg king - man + woman = queen).
 - Optimizations are often not where you expect them. Softmax is never something I'd have thought could be optimized, but it was, and it was a very important step.
 - You can do a surprising amount from just piles of vectors. Without even factoring in word order, you can still predict words correctly in a sentence a shocking amount of the time.

 [Paper Link (arXiv)](https://arxiv.org/pdf/1301.3781)