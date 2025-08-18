# Decision Transformer: Reinforcement Learning via Sequence Modeling - Chen, et al., (2021)

## Problem
Offline RL from fixed datasets is hard. TD-learning methods can be unstable (bootstrapping, discounting, “deadly triad”), struggle with long-term credit assignment, sparse/delayed rewards, and value overestimation. This paper proposes a paradigm shift: train a Transformer to model trajectories directly, bypassing dynamic programming and leveraging scalable sequence modeling instead. By using transformers, the attention mechanism provides a natural method of credit assignment, and not needing to optimize value functions removes the need for regularization or conservatism.

## Methods
 - Frame RL as conditional sequence modeling. Represent each trajectory as an interleaved token sequence of return-to-go (cumulative return from the current time onwards), state, and action. Train a causally masked (GPT-style) Transformer to autoregressively predict the next action token, conditioned on the desired return-to-go rather than past rewards, and you get a pretty good way of predicting what actions will lead to what total rewards.
 - Supervised training on offline trajectories. Sample length-K windows (where K is the context length, because context starts with a K) from the dataset and minimize action-prediction loss (XE for discrete, MSE for continuous).
 - Evaluation as prompting. At test time, provide the target return and current state, execute the predicted action, then decrement the return token by the observed reward and repeat.

## Results
| Game     | DT (Ours)    | CQL   | QR-DQN | REM  | BC           |
|----------|---------------|-------|--------|------|--------------|
| Breakout | 267.5± 97.5   | 211.1 | 17.1   | 8.9  | 138.9± 61.7  |
| Qbert    | 15.4± 11.4    | 104.2 | 0.0    | 0.0  | 17.3± 14.7   |
| Pong     | 106.1± 8.1    | 111.9 | 18.0   | 0.5  | 85.2± 20.0   |
| Seaquest | 2.5± 0.4      | 1.7   | 0.4    | 0.7  | 2.1± 0.3     |

## Takeaways
 - The transformer is a very dynamic and generally applicable architecture. It's remarkable that this architecture, originally devised for translation tasks, can do so well at so many tasks. This paper is proof of the architecture's efficacy in yet another domain: offline RL.
 - Value functions aren't always needed. Probably the neatest thing about this paper is the complete dodging of any value function, and this has amazing effects! It feels like it shouldn't work: why should this perform well when it can't decide the value of a certain state? Yet, through changing the framing, it works.
 - Frame changes can be important. To me at least, the fundamental idea that made this paper possible was changing the perception of RL as learning to maximize reward to a sequence modeling problem. This was what made transformers possible, and resulted in such a breakthrough paper.
 - Attention is very useful. Attention is what allows the transformers to perform proper credit assignment: if I paid a lot of attention to that feature, it probably should get a lot of the credit for whatever happened. It's intuitive, and it works.
 - Data is still key. Even transformers can't make garbage into gold: poor datasets still kill performance. That being said, they are still able to obtain remarkable feats, like learning the optimal path in a graph off of random walks, despite random walks not being the best data.

## Remaining questions / topics for further reading
 - Can we turn this from an offline RL solution to an online one? What if we let transformers interact with the environment and learn that way? How well would they perform?
 - How well does the extrapolation compare? If I give it a prompt outside the dataset, does it generalize better, as transformers are often known to do, or not?
 - Can this be used for interpretability? Already attention is used to explain the strong credit assignment these models exhibit. So can we use it, in a similar vein, to explore how the model makes its decisions?
 - Can the transformer architecture be finetuned to do better at offline RL? A standard GPT architecture was used, but what about adding back in encoders? Or any of the other modifications to transformers?

[Paper Link (arXiv)](https://arxiv.org/pdf/2106.01345)