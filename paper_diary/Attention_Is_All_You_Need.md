# Attention Is All You Need - Ashish Vaswani et al., 2017

## Problem
This paper addresses the lack of parallelizability in recurrent neural networks, which at the time, were SOTA for language - language translation tasks. However, due to their dependence on hidden states, training could not easily be parallelized, and thus they were slower to train on GPUs. By introducing a new architecture - the transformer - based entirely on attention mechanisms, the sequential costs of recurrent neural networks could be avoided, allowing the model to train far faster on GPUs. After all, breadth is free, but depth is costly.

## Method
 - **Core Mechanism:** The attention mechanism allows the model to consider all tokens and their relationships to each other, by using key, value, & query vectors. This is parallelizable and allows for every token to transfer meaning to every other token, regardless of how far away they are in the input.
 - **Key equation: (attention equation)**
  $$ 
  \text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^\top}{\sqrt{d_k}} \right)V 
  $$
 - **Equation Explanation:** This is the equation for attention. The inputs \( Q, K, V \) represent the queries, keys, and values, which are learned projections of the input tokens. The queries can be thought of as the piece of information being sought. The keys can be thought of as indicators for what information is present in each token. And the values can be thought of as the meaning that token carries. The term \( QK^\top \), then, is the model checking which tokens have the information that the attention head is currently seeking. This could, for instance, be sentence structure, in which case tokens representing punctuation (like . , and ;), and conjunctions (and, or, but, etc) would likely have high attention values (the corresponding entry of QK^T would be large), while other tokens would not (the corresponding entry of \( QK^\top \) would be small). Dividing this by the \( \sqrt{d_k} \) ensures that the variances don't grow dramatically, as this can push softmax into areas where it has vanishing gradients, killing learning. Softmax then takes the scaled output of \( QK^\top \) and 'smooths' it, creating the final weighted sum that the values are multiplied by, which can be thought of as extracting the information the query was seeking.
 - **Novelty vs. prior work:** Though attention mechanisms had been used in machine translation tasks before, this is the first time that they had been the sole base for a model. Usage of multiple heads of attention also allowed the model to split its focus on different semantic parts of the input, something that average prohibited before when using only a single head. Also, fixed positional encodings with sin and cos allow this new attention-only architecture to learn positional relationships (this was never needed in the past due to the presence of recurrence).
## Results
|      Metric / Task    | Baseline | This paper | Δ        |
|-----------------------|----------|------------|----------|
| BLEU (WMT-2014 EN-DE) | 24.6     | **28.4**   | +3.8     |
| FLOPS (training)      | 9.6e18   | **3.3e18** | x2.9 less|

## Takeaways
 - The attention mechanism is useful for three main reasons: One, it is far more parallelizable during training, as teacher forcing allows the model to predict every token at once (with the auto-regressive property being enforced using masked attention). Two, it is more efficient, especially for smaller input sizes. Three, tokens can easily communicate meaning to every other token, resulting in a max path length of O(1), far better than any other model.
 - Attention does not innately encode positions, and so positional encodings should be used. Learned & fixed encodings perform around the same, and good fixed encodings allow for constant offsets to be expressed as linear transformations of the original position, making it easier for the model to learn.
 - Dropout & label smoothing improve model generalizability and prevent overfitting.

## Remaining questions / topics for further reading
 - How does the architecture for modern generative LLMs differ from this?
 - How is RL applied to the weights in this architecture? It is not yet clear to me how a ranking of responses could be used to update model weights.
 - How was the leap from translation to text generation made?
 - What other contexts can the transformer architecture be used in?
 - How was the attention mechanism originally derived?

[Paper Link (arXiv)](https://arxiv.org/abs/1706.03762)