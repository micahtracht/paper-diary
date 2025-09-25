# From Orthography to Semantics: a Study of Morphological Processing through Deep Learning Neural Networks - Davila, Morris (2018).

## Problem
Can a neural network map letters to word meaning by leveraging morpho-orthographic structure (roots & affixes visible in the word's spelling)? This paper attempts to do this by making use of dual-route orthographic processing, featuring two routes: a diagnosticity route (most informative letter evidence, but quick), and a chunking route (contiguous sublexical units), to predict words. It does so by constructing a neural network topology that goes from characters -> semantic representations (and passes through the two channels), then -> word identity.

## Methods
 - Letters to embeddings. The paper maps 28 letters into a 5-dim learned distributed space, then represents each word as a sequence of 28 of those letters (the length of the longest word was 28 letters), zero-padded.
 - Dual-route convolutions. The paper featured two channels for convolution, that take inspiration from a model of human linguistic processing that suggests we use diagnosticity (quick, informative letter pairings that give us information about a word's identity) and chunking (finer grained analysis that gives us a better idea of what the finer meaning of the word is) to turn letters into semantically useful chunks. The paper mimicks this by passing the outputs of the letter embeddings (described above) into two different convolutional neural nets, one which mimicks diagnosticity (width = 140 for full word, no stride), and another which mimicks chunking  (width = 2 letters, stride = 1).

## Key Idea
Fusing two complementary, character level signals (precise chunks, and position tolerant, diagnostic letter evidence), combined with a letter embedding, can lead to superior performance in extracting meaning from words.

## Results

Task: 50k-class word identity recovery from characters, with an auxiliary 128-d semantic prediction head.

Top-1 accuracy (avg across runs):

| Topology                         | Accuracy |
|----------------------------------|---------:|
| Dual Route + Letter Embeddings   | 0.4309 |
| Dual Route + One-hot Letters     | 0.4173 |
| Chunking Only                    | 0.4197 |
| Diagnosticity Only               | 0.3469 |
| NDL + Letter Embeddings          | 0.4004 |
| NDL + One-hot Letters            | 0.1379 |

Optimization / learning dynamics
- Embeddings speed learning: models with letter embeddings stabilize far earlier (≈ 20k epochs) and avoid very high NLL plateaus seen in non-embedding variants.
- Ablations confirm dual-route advantage: chunking alone > diagnosticity alone, but both together (dual route) are best; adding embeddings yields the strongest overall performance.
- NDL baseline is substantially weaker and requires many more epochs to reach comparable NLL.

Sanity checks on learned representations
- Letter embeddings capture regularities: vowels cluster; ‘i’ and ‘y’ emerge as nearest neighbors, indicating shared usage patterns.

## Implications
 - Chunking letter-level inputs can lead to superior performance in extracting meaning, by allowing the model to learn common attributes of language (like 'er' or 'ing'). Ends up being useful for tokenization in LLMs.
 
## Takeaways
 - Humans can be a valuable source of inspiration for ML. This approach came from a theory in linguistics, and as it turns out, the same approach works for neural nets.
 - Chunking is the most important part of this process, even though it's limited by only being bigrams.
## Remaining Questions
 - This model has two channels. Do the separate attention heads in a transformer act similarly?
 - Does tokenization basically do the same thing in LLMs, except that it allows for different sized chunks and learns what features to look at?

[Paper Link (IEEE)](https://ieeexplore.ieee.org/document/8489686)