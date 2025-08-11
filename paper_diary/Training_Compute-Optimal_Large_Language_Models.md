# Training Compute-Optimal Large Language Models - Hoffmann, et al., (2022)

## Problem
Given a fixed compute budget, how should it be allocated between model size (parameters, $N$) and data (training tokens, $D$) to minimize pre-training loss? Prior practice, largely guided by Kaplan et al. (2020), favored scaling model parameters while not significantly increasing data. This paper argues that this left models comparatively **undertrained** for their size.
> FLOPs model used throughout the paper: **$F \approx 6ND$** (ignoring smaller terms). This linear product constraint is what makes the compute-optimal $N$ and $D$ scale with the **same exponent**.

## Methods
 - Minimum over training curves (fix $N$, vary $D$). The paper authors trained families of models at fixed parameter counts, and swept the number of training tokens used. For each model size, they smoothed and interpolated the loss curves, before taking the envelope of lowest loss at each FLOP budget to map compute, $C$, to the loss-minimizing $(N\_opt, D\_opt)$. Finally, fitting power laws to these envelope points gave exponents $a$, $b$ with $N\_opt ∝ C^a, D\_opt ∝ C^b$, yielding $a ≈ 0.5$, and $b ≈ 0.5$.
 - IsoFLOP profiles (fixed $C$, vary $N$). Here, the authors picked several target compute budgets and, for each one, trained multiple models, and plotted the final (smoothed) training loss vs parameter curves. Each IsoFLOP curve had a clear, distinctive valley, suggesting an optimal model size for that compute budget. They then fit a parabola around the valley to estimate the minimizing $N$ (and corresponding $D = C/6N$), then again fit power laws against the data, finding $a ≈ 0.49$ and $b ≈ 0.51$.
 - Parametric loss fit (global model, $L^{(N, D)}$). Finally, they pooled the results gathered in approaches 1 and 2, and fit a decomposed loss model  $L^{(N,D)}=E + A/N^{alpha} + B/D^{beta}$, interpreting terms as irreducible data entropy, $E$, finite capacity error, $A/n^{alpha}$, and finite-data error, $B/D^{beta}$. They estimated $(E, A, B, alpha, beta)$ via Huber-loss regression, then solved the constrained optimization problem minimize L^ subject to FLOPs $= 6ND$ to trace the efficient frontier and scaling exponents, yielding $a ≈ 0.46$, $b ≈ 0.54$.

## Key Claim
Compute optimal training allocates compute equally to parameters and data. As the compute, $C$, grows, the optimal $N$ and $D$ both scale like sqrt($C$). In essence, what this means is: if you double parameters, you should double your training data, too. This also revealed that many recent LLMs were far too large for their data budgets.

## Results
 - Across 3 estimates, the optimal $a$ and $b$ were around $0.5$ each, in contrast to Kaplan's results, which showed larger $a$ than $b$.
 - Chinchilla vs. Gopher (same training FLOPs, Chinchilla trained according to this paper's results, Gopher not):
     - MMLU (5-shot): Chinchilla ~67.6%, +7–8 points over Gopher (SOTA at publication).
     - BIG-bench: Chinchilla scores +~11 points average over Gopher.
     - Language modeling: Chinchilla achieves lower perplexity/BPC on Wikitext-103 and Pile subsets.
     - Reading/commonsense QA: Chinchilla broadly wins on LAMBADA, RACE-m/h, PIQA, HellaSwag, Winogrande, BoolQ, etc.

## Implications
 - Shifting focus towards data. As a result of this paper, focus shifted towards curating large, high quality datasets, over merely scaling up the number of parameters in a model.
 - Easier finetuning and lower downstream costs. The balance shifting to favor smaller models made finetuning easier and reduced inference costs, as smaller models are much easier to run and finetune.
 - A new rule of thumb. Target $D ≈ KN$, or in other words, constant tokens per parameter.

## Limitations
 - Difficulties in extrapolation. Due to compute constraints, it is not practical to obtain large amounts of empirical data in the 10B+ parameter regime. Only two very large datapoints, Chinchilla and Gopher, anchor near the top, and modern models are often trained with 4 OOMs more FLOPs than these. Though the results appear robust,  extrapolations over multiple OOMs are always difficult.
 - Sub-epoch assumptions. Most experiments in this paper are sub-epoch. But, in the multiple-epoch regime, it is hard to tell if these results will hold, or if data reuse changes the balance.
 - Architecture scope. While this result does show a strong relationship for dense decoder-only LMs, MoE models, retrieval techniques, and different optimizers could shift the balance.
 - Data distribution and quality. This result treats all tokens as created equal. In reality, however, some data is better than other data. So varying data quality could also produce shifts in the optimal balance.

## My Takeaways
 - Yet again, simpler is often correct. This paper proposed a far simpler result (that you should scale compute and data equally), and it turned out to be correct, over the more complicated ones.
 - The modern bottleneck is now more often data than parameters. Sourcing high quality, deduplicated, PII scrubbed data is hard.
 - Smaller can be good. This seems like the first demonstration of small models really packing a punch and beating larger ones. Now, we have models like Gemma 3 27B, which pack a ridiculous amount of intelligence into a very small model. So there are clearly more gains to be made in regard to performance at a fixed model size.
 - Data is most important. If you can only scale parameters or scale data, scale data. It'll make the model easier to work with downstream, and will likely lead to similar performance.

## Further questions / topics for further reading
 - Just how much intelligence *can* you pack into a tiny model? While this is certainly a very difficult research question that humanity will be trying to answer for a while, I still want to read papers to help get an idea of what other techniques have been tried to this end.
 - How do optimizers, long context scaling, MoE architectures, width-depth trade-offs, and sequence packing affect the compute-optimal laws?
 - Data quality vs quantity. How does deduplication, contamination control, and curriculum affect the optimal amount of data to use?
 - How does reasoning affect this frontier? Now that models are optimized for reasoning in internal CoTs, smaller models often perform better in the real world per unit of compute because they can reason for longer. Does that mean even more emphasis should be placed on data over model size now?