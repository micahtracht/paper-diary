# Proximal Policy Optimization - John Schulman, et al., (2017)

## One Liner
PPO ensures that the new policy is not too different from the existing policy by clipping the probability ratios of the new policy versus the old, removing the gradient incentive for larger changes and making training more stable, resulting in an algorithm that runs quickly, only uses first order differentiation, and achieves SOTA performance on a range of tasks.

## Problem
Often, RL algorithms will create policy changes that are too large, resulting in sharp, unexpected performance declines, or unstable training. Previous results in the literature, most notably Trust Region Policy Optimization (TRPO) from 2015, had addressed this by using their own bounds. TRPO, the previous SOTA, relied on a bound on the KL divergence, but this required an expensive and difficult second order optimization technique, and couldn't be used with techniques like dropout, hindering the algorithm's performance. PPO aims to keep the benefits of small, contained policy updates without the difficulty of implementation or slow training speeds of TRPO by exchanging the expensive and difficult to calculate KL divergence bound with a much simpler bound on the ratio of probabilities for each action. This performs just as well at ensuring policy updates are not too large, but runs much faster with wall time (as it is more parallelizable) and tends to achieve equal or superior performance.

## Method
 - **Key Challenge:** Multiple stochastic gradient descent (SGD) passes on a single batch of data could often push the new policy far away from the old, invalidating the advantage estimates used.
 - **Key Solution:** Clip the probability ratios of the new policy using a clipped surrogate objective function that is identical to the TRPO approximation in first order when the policy is near, but provides a pessimistic bound once that ratio crosses the clip wall.
 - **Main Equations & Explanations:**
$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[ \min \left( r_t(\theta) \hat{A}_t,\ \operatorname{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon)\ \hat{A}_t \right) \right]
$$
 - This equation is the equation for the clipped surrogate objective. It is very similar to the usual surrogate objective, except with the presence of `min` and `clip`. In words, what the `min` and `clip` does is take the surrogate objective from asking "What's the absolute best policy I think I can come up with?" to asking "What's the absolute best policy I think I can come up with that's still close to my original policy?" The way the math does this can be thought of as follows: If the advantage is nonzero (positive or negative), the algorithm would think the 'best' policy is whatever makes $r_t(\theta) \cdot \hat{A}_t$ as large as possible, so it picks a policy that makes the ratio either very large (if $\hat{A}_t > 0$) or very small (if $\hat{A}_t < 0$). But, by selecting the `min` of that choice and the clipped objective, which keeps the ratio inside $1 - \epsilon$ and $1 + \epsilon$, the function gets no better for policy choices that make $r_t(\theta)$ go outside of that range. Thus, the gradient will usually end up pushing $r_t(\theta)$ to around that bound, but not past it. Make it a decent bit better, where we can be confident it's a real improvement and still understand the policy landscape, rather than gambling on a massive change.

$$
L(\theta) = L^{\text{CLIP}}(\theta)\ -\ c_1 \left( V_\theta(s_t) - V_t^{\text{targ}} \right)^2\ +\ c_2\ \mathcal{S}\left[ \pi_\theta \right] (s_t)
$$
 - This is the full objective function used in the paper. It, in addition to the $L^{\text{CLIP}}$ described above, adds a penalty for value function estimates that differ from the empirically observed values (calculated using generalized advantage estimation and adding that to the policy's value) and an entropy bonus. The penalty for inaccurate value function estimates ensures that the advantage function estimates remain reliable, and helps keep the policy accurate. The entropy bonus can be thought of as analogous to $\epsilon$-greedy exploration in Deep Q Learning: Let's explore a little, so we can find potentially new, better ways of doing things, and avoid getting stuck in local minima. It's how much exploration vs exploitation we want to do. It also serves to counteract premature determinism, helping the policy avoid getting stuck in the same actions over and over.
 - **Using Multiple Epochs:** PPO, unlike some other RL algorithms, uses multiple epochs on each minibatch to achieve better sample efficiency. This is only possible since PPO ensures updates never get too large, so the advantage estimates stay roughly correct, and further steps can be taken without issue. This helps make PPO far more sample efficient, as it can learn more from each sample it takes.
 - **How PPO Takes Steps:**
 Step 1: Collect N actors with T timesteps each trajectories (N*T total).
 Step 2: Compute advantages, using GAE (often with γ = 0.99 and λ = 0.95).
 Step 3: Optimize the surrogate objective with K epochs using minibatch SGD/Adam (often with 3-10 epochs, 64-4096 batch size).
 Step 4: Update policy parameters, and do it all again!
 Note that because every update keeps things close *by design*, no difficult line search or fancy matrix inversion is needed. Simplicity wins.

## Results (highlights)
 - **Continuous Control (MuJoCo, 1M steps):** PPO with ε = 0.2 beat or tied TRPO, CEM, vanilla PG, and A2C on 6/7 locomotion tasks. Also achieved the top average normalized score (0.82 vs 0.76 for second best).
 - **Atari (40M frames):** PPO tied or exceeded ACER & A2C when judged by average reward over all episodes, and still did well even when only judged by final performance.
 - **3D Humanoid Showcase:** PPO learns running, steering, and get up skills (even while being pelted by cubes!) in 50-100M steps, without any special case engineering.

## My implementation
I made a very simple PPO for Cartpole implementation. It's not much, but I wanted to get experience coding an algorithm from nothing more than the paper and the PPO docs on OpenAI's website. I was genuinely shocked to see it achieve 500 reward in < 10 episodes, given how long it took my DQN approach to converge. Great learning experience for me, showed me just how powerful some of these algorithms are.
[My Cartpole with PPO by hand](https://github.com/micahtracht/CartPole-PPO)

## Takeaways
 - Simplicity often wins. PPO is decidedly simpler than TRPO (and a much easier read), yet runs faster and achieves better performance. Complicated methods are rarely better than simpler ones.
 - Clipping basically does the same thing as trust region. This doesn't just help me understand clipping, but also makes the KL divergence bound of TRPO make more sense. It's a more formal way of doing what clipping does, which is say the policy shouldn't differ too much.
 - Multiple epochs / batch are actually useful. I thought that it would just lead to overfitting or policy updates that are too large, but turns out, with the right algorithm, it does very well.
 - It is still first order friendly. PPO allows for other techniques like dropout to be used that TRPO doesn't allow because it is first order friendly. This makes it a much more generally useful algorithm.

## Remaining questions / topics for further reading
 - How does GRPO build on PPO and how does it differ from it?
 - How should I decide what hyperparameters (ε, c1, c2, γ, λ) to pick?
 - Can adaptive clipping (changing ε) outperform fixed ε?
 - PPO strayed fairly far from the theory. What guarantees could be proven about PPO? Why or why not?

[Paper Link (arXiv)](https://arxiv.org/pdf/1707.06347)