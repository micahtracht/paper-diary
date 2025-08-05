# DeepSeekMath: Pushing The Limits of Mathematical Reasoning in Open Language Models - Zhihong Shao el at., 2024

Note: This diary entry will primarily focus on Group Relative Policy Optimization (GRPO), and will neglect most other parts of the paper.

## One Liner
GRPO improves on PPO for LLMs by estimating token advantages relative to group-average rewards, removing the need for an expensive, inefficient critic model, boosting reasoning accuracy, and increasing reasoning stability.

## Problems With PPO
 - Expensive critic model in PPO. PPO trains a special value model (critic model) thats the same size as the policy, which doubles compute and memory costs during training.
 - Sparse and coarse rewards. For math tasks, the central focus of the paper, only the final token is scored. So, from just one scalar, the critic model has to guess token-level rewards for every other token. This leads to issues like incorrect reasoning steps still getting rewarded, or the key tokens that actually led to the solution being rewarded as much as junk filler. This makes training noisy and often unstable.
 - Weird entangled KL divergence. PPO puts the KL penalty directly into the per-token rewards, which entangles advantage estimation with regularization and makes things more complicated.

## Method
 - **Key Challenge:** Eliminate the need for an expensive value model while reducing noise and neatening up the KL penalty.
 - **Key Solution:** Why train some big expensive model to estimate the values for advantage calculation when we can just run the model a bunch of times and get a (less noisy) estimate for cheaper? Sample some candidate completions under your current policy (model), score them using your reward model, and then normalize them (this step is key!) to obtain a relative reward. Add an explicit KL penalty to ensure it doesn't diverge too much, and voila, your normalized rewards work *perfectly* as your advantages. Simply substitute them in and skip all the messy value estimation stuff. As is a trend I'm noticing in the literature, simple solutions are often better.
 - **Main Equation & Explanation:**
$$
\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{q \sim P(Q),\, \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(O \mid q)} \left[
\frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left\{
\min \left(
\frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})} \hat{A}_{i,t},
\operatorname{clip} \left(
\frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})},
1 - \varepsilon, 1 + \varepsilon
\right) \hat{A}_{i,t}
\right) - \beta \mathbb{D}_{\text{KL}} \left[ \pi_\theta \, \| \, \pi_{\text{ref}} \right] \right\}
\right]
$$. I'll admit, this is scary looking. But it's actually not that bad. The first term is just PPO with group-relative advantages, the second term is just the explicit KL penalty, and pi_ref is just some frozen snapshot of the model that gets updated every outer iteration. When taken together as such, it reads like this: Do PPO using advantages calculated from the group, and penalize it if it strays too far from our reference model.
 - **Exploration Strategy:** Two main strategies are used. One, temperature or nucleus sampling is used so the G candidates differ. This helps us ensure our rewards have some useful variance. Then, an iterative RLHF loop. After each outer iteration, make the reference policy the current policy, and train the reward-model on newly collected pairwise preferences with a 10% replay buffer.
 - **Actual step taking process:** 1. Collect a batch (of size G, paper uses G=64), and compute the raw rewards. 2. Compute the advantages (lucky for you, GRPO makes this easy! Just center and scale the rewards, and you have your advantages). 3. Update the policy. Run several minibatch gradient steps on the clipped loss. 4. Regularize. Measure the empirical KL divergence, and adjust the lr or beta if need be. 5. Reset your reference model & reward. Each outer loop, set your reference model as your current model, and fine tune the reward model on the latest comparisons. 6. Repeat until your reward plateaus.

## Results
| Model                                    | GSM8K (CoT) | MATH (CoT) | CMATH      | Notes                           |
|------------------------------------------|-------------|------------|------------|---------------------------------|
| DeepSeekMath-Instruct 7B (after SFT)     | **82.9 %**  | **46.8 %** | 84.6 %     | PPO-style critic removed        |
| **DeepSeekMath-RL 7B (GRPO)**            | **88.2 %**  | **51.7 %** | **88.8 %** | +5.3 pp GSM8K, +4.9 pp MATH     |
| PPO (ablation, same data & reward model) | 86.0 %      | 48.1 %     | 86.9 %     | Requires separate value network |

## Takeaways
 - GRPO is much more compute friendly. Not needing the critic model halves your memory needs, so it's a much easier method to run.
 - Simplicity beats complexity. TRPO won out because it stopped trying to come up with weird arbitrary rates to fix step size and said "OK, we want the model not to change too much each iteration. So let's measure how much it changes each iteration, and put a cap on that." PPO beat TRPO because PPO said "Instead of worrying about the KL divergence, let's look at what we really want. The models are similar if their probabilities of taking the same actions in the same states are similar. So cap how much we're going to reward the model for changing these ratios." GRPO beat PPO because it said "Rather than doing complex advantage estimation to know what worked using an actor critic method, why not just run it a bunch of times and use that as a baseline?"
 - An explicit, separate KL term is much simpler (and, as they so often are correlated, better) than a per-token penalty that gets blended in with the rewards.
 - GRPO isn't too different from PPO. It just replaces the advantage estimation with relative advantages and explicit KL. That makes it easy to port to any pre-existing RLHF based setup.
 - Division by 1/length of output is used to normalize the loss contribution per sequence. That way, longer sequences don't have artificially inflated losses (a 1000 tok sequence should not contribute 10x as much as a 100 tok sequence). Think of it as averaging the loss. For the same reason, in GRPO, the loss is divided by 1/number of elements in the group.

## Remaining Questions
 - How important is picking the right group size, G? Do larger models require larger G because they need more data, or smaller G because they have less variance? Or no change?
 - We're still calling a response good and then updating all the tokens in it. Is there a way to allocate what tokens were helpful and what weren't *without* step-wise analysis? An idea that comes to mind for me is freezing the response after n tokens, and then sampling trajectories from those points. If the work preceding that point was correct, it'd have higher average rewards than baseline, and if it were incorrect, it'd have lower average rewards than baseline.
 - Does it work beyond math and other verifiable domains? Especially in domains where response quality difference is lower.
 - TRPO had guaranteed monotonic improvement (under certain assumptions). PPO, under more assumptions, did as well. Does the same hold for GRPO? And if so, how would you even go about proving that?