# Trust Region Policy Optimization - John Schulman, et al., (2015)

## One Liner
TRPO improves upon previous policy optimization methods in RL by constraining the size of updates made to the policy to a region where all approximations hold up (the "trust region" in Trust Region Policy Optimization).


## Problem
Existing RL techniques at the time suffered from a variety of issues. Black-box methods, like CEM and CMA, struggle with complex or high dimensional policies, and often require hand crafted policies for each task, which hurts generalizability. Policy iteration methods, like Exact $\pi$ or DQN variants, struggle with stability and having high memory & compute costs, while requiring accurate, low variance returns. Policy gradient methods (of which TRPO is a member), like REINFORCE and A2C, struggle with high variance and poor sample efficiency. Thus, an algorithm with the ability to learn and optimize complex policies, like policy gradient methods do, without the drawbacks of poor sample efficiency, would be a major improvement. It is this gap that TRPO fulfills.

## Method
 - **Key Challenge:** Optimizing the actual expected return, $\eta$, is hard. But by using a surrogate objective that behaves like $\eta$ in the region around the current policy, we can indirectly optimize it. However, this runs into a challenge: What if our step takes us out of the area where our surrogate objective is a good approximation?
 - **Key Solution:** By constraining policy updates to be within a fixed distance\* of the previous policy, TRPO can ensure that each policy is better than the previous one. This is because TRPO allows us to be sure that each update takes place in an area where our surrogate objective gives a good approximation of the true one. To constrain the distance between two policies, TRPO ensures the KL divergence between the old and new policies is bounded to within small constant, $\delta$. Thus, our new task is to find the policy that gives the best localized approximation, while having a KL divergence with respect to the old policy less than $\delta$.  
   \*KL divergence is not a measure of distance, or even a metric at all. This is because it is not symmetric ($D_{KL}(p || q)$ is not always equal to $D_{KL}(q || p)$) and does not satisfy the triangle inequality.
 - **Main equations & explanations:**  
   - $L_{\pi}(\tilde{\pi}) = \eta(\pi) + \sum_{s} \rho_{\pi}(s) \sum_{a} \tilde{\pi}(a|s)A_{\pi}(s, a)$. In this equation, $L_{\pi}(\tilde{\pi})$ represents the localized approximation of the total expected return of $\tilde{\pi}$ ($\eta(\tilde{\pi})$) given by the policy $\pi$. $\eta(\pi)$ represents the total expected return of $\pi$. $\rho_{\pi}(s)$ is the (unnormalized) visitation frequency of state $s$ under policy $\pi$. $A_{\pi}(s, a)$ is the advantage of taking action $a$ in state $s$ under policy $\pi$.
   - $\eta(\tilde{\pi}) \geq L_{\pi}(\tilde{\pi}) - C D^{\max}_{KL}(\pi, \tilde{\pi})$, where $C = \frac{4 \epsilon \gamma}{(1-\gamma)^2}$, and $\epsilon = \max_{s, a} |A_{\pi}(s, a)|$. This equation provides a lower bound for the total expected returns of a new policy, $\tilde{\pi}$, and is one of the core theoretical motivators for TRPO. $L_{\pi}(\tilde{\pi})$ is given in the equation above. $\epsilon$ can be thought of as the biggest advantage given any state $s$ or action $a$ present in the policy $\pi$. $D^{\max}_{KL}$ is the maximum KL divergence over any state of the two policies, $\pi$ and $\tilde{\pi}$. Think of this as the max amount the two policies $\pi$ and $\tilde{\pi}$ ever disagree.
   - $\max_{\theta} L_{\theta_{old}}(\theta)$ subject to $D^{\rho_{\theta_{old}}}_{KL}(\theta_{old}, \theta) \leq \delta$. $L_{\theta_{old}}(\theta)$ is the local approximation of the total return for the policy parameterized by $\theta$ given by the policy parameterized by $\theta_{old}$. $D^{\rho_{\theta_{old}}}_{KL}(\theta_{old}, \theta)$ represents the average KL divergence over the states, weighted by visitation frequency of the previous policy $\pi_{old}$, of the policy parameterized by $\theta_{old}$ and the policy parameterized by $\theta$. Note that $\pi_{old}$ *is* the policy parameterized by $\theta_{old}$. This equation is the key practical equation used in TRPO to optimize policies, as it lets us rely on values we can estimate using monte carlo simulation.
 **Exploration methods:**
 - Single path method: In TRPO, the single path method involves using the old policy to generate a trajectory, whose data is then used to solve the constrained optimization problem listed above, resulting in a new policy. Repetition of this process allows for even hard problems like walking gaits to be solved. The key advantage of this method, however, is that it can be implemented in a physical system, and does not require being able to reset the system to arbitrary states.
 - Vine method: The vine method involves first constructing a rollout set, sampling actions to be taken in that rollout set, and creating a short trajectory according to those states and actions. This lowers variance and increases sample efficiency by allowing you to treat observed $Q$ values as paired observations, reducing Monte Carlo noise. However, it requires being able to reset the system to arbitrary states, which is not possible in a physical environment. This can make it impossible to use for real world environments.
 **How TRPO actually takes steps:**
 - To actually make a step, TRPO approximates the policy improvement step as a quadratic: maximize the surrogate objective’s gradient, while keeping the KL divergence under a fixed threshold ($\delta$). This yields a natural gradient direction $F^{-1} g$, where $F$ is the Fisher Information Matrix (FIM) and $g$ is the policy gradient. Since inverting $F$ is unfeasible in practice, TRPO uses the conjugate gradient method to efficiently solve for the entire term, $F^{-1} g$, using Hessian-vector products. The resulting step is then scaled to ensure it satisfies the quadratic KL estimate, and a backtracking line search ensures that the actual empirical KL divergence remains within bounds and that the surrogate objective can guarantee improvements for. The result is stable and monotonic policy improvement.

## Results
The authors tested TRPO in two distinct domains: continuous robotic locomotion (swimmer, hopper, and walker), and high dim, vision based Atari games. On continuous robotic locomotion tasks, TRPO clearly performs best, ahead of other techniques like CEM or CMA. Even on Atari games, despite not being designed for it, TRPO achieves reasonable scores (though not SOTA). This demonstrates just how general TRPO is.

### TRPO Atari Game Scores  (500 training iterations)

| Game            | Random | Human* | Deep Q-Learning | UCC-I   | TRPO (Single Path) | TRPO (Vine) |
|-----------------|--------|--------|-----------------|---------|--------------------|-------------|
| Beam Rider      |  354   | 7,456  | 4,092           | 5,702   | 1,425.2 | 859.5                  |  
| Breakout        |  1.2   | 31.0   | 168.0           | 380     | 10.8    | 34.2                   |
| Enduro          |  0     |  368   | 470             | 741     | 534.6   | 430.8                  |
| Pong            | –20.4  | –3.0   | 20.0            | 21      | 20.9    | 20.9                   |
| Q*bert          |  157   | 18,900 | 1,952           | 20,025  | 1,973.5 | 7,732.5                |
| Seaquest        |  110   | 28,010 | 1,705           | 2,995   | 1,908.6 | 788.4                  |
| Space Invaders  |  179   | 3,690  | 581             | 692     | 568.4   | 450.2                  |

*Human scores reproduced from Mnih et al. (2013).

Source: Table 1 in Schulman et al., 2015

## Takeaways
 - KL divergence is a measure of the difference (not distance, as it is not a metric. It is not symmetric and does not satisfy the triangle inequality) between two probability distributions, $P$ and $Q$. In the discrete case, it is given by: $\sum_x P(x) \log \frac{P(x)}{Q(x)}$. In the continuous case, it is given by: $\int_{-\infty}^{\infty} P(x) \log \frac{P(x)}{Q(x)} dx$.
 - The Hessian of the KL divergence is the Fisher Information Matrix (FIM), evaluated at the paramters $\theta$. It gives an idea of the curvature of the policy space, allowing better informed steps to be taken (if you know there's a cliff right in front of you, don't walk off it). Formally, it represents the information gained about the policy by sampling some action from it.
 - TRPO essentially works by regulating the size of the change of the policy, rather than just the size of the change of the parameters. This is important, because changing policy parameters by the same amount can result in wildly different magnitudes of changes in the policy. This is what KL divergence is used to measure.
 - Another way to think of TRPO is that it consistently pushes up the lower bound of the current policy by making constrained but well informed steps. In other words, it minorizes the true return function, $\eta(\pi)$.
 - The KL divergence penalty is replaced with a KL divergence restriction to promote large steps. Despite theoretical motivation, the penalty simply results in updates too small for the model to effectively learn. Using a bound on the KL divergence keeps most of the benefits while greatly accelerating model learning.

## Remaining questions / topics for further reading
 - How does PPO improve on TRPO?
 - How were the results in TRPO proven? Some time later, I'll return to the appendices.
 - Could a dynamically changing $\delta$ be used to promote large updates at the start, and slower updates as the model progresses, in a way analogous to learning rate decay?
 - Why do DQN techniques beat it on tasks like Atari games?
 - Why did previous methods (even gradient methods) struggle with high dimensional policies?

[Paper Link (arXiv)](https://arxiv.org/pdf/1502.05477)