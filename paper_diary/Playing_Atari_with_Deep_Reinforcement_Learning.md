# Playing Atari with Deep Reinforcement Learning - Mnih, et al., (2013)

## Problem
How can agents learn effective control policies directly from raw, unfiltered, high-dimensional pixel data while dealing with sparse and/or delayed rewards, correlated samples, and shifting data distributions caused by policy changes? Prior work used hand crafted features, special pre-processing, or linear approximators, but these are less general, don't learn from unfiltered input, and struggle to obtain strong performance. This paper presents Deep Q-Networks (DQNs), which tackle all 3 problems at once: A neural network that can learn directly from pixels, that can exhibit stable learning, and can be generalized to different games. This allowed it to become the first end-to-end deep RL method from raw pixels, a major step up over previous progress in RL.

## Key Methods
 - Experience replay. A primary issue with using and training neural networks is that subsequent samples are highly correlated. This leads to vastly increased variance in training, which can lead to large, often uninformative updates. To get around this, all samples are recorded and added to a buffer. To train, samples are uniformly randomly selected from this buffer, in a process called "experience replay." Since distinct samples are selected each time, the sample selections are less correlated, reducing variance in training and making it more likely the update improves the model.
 - Deep Q-Learning. After sampling minibatches from the replay buffer, use off-policy Q learning to update the model's parameters based on the sampled data.
 - Frame stacking. One frame does not encode all the information about the game state. For example, one frame is not enough to deduce velocity, or acceleration, key data that the model would need to play the game well. To get around this, the last 4 frames are stacked together, letting the model learn this more valuable data without some expensive or difficult technique like recurrence.
 - Frame processing. Each frame is gray-scaled, down-sampled from 210x160 to 110x84, and cropped to reduce the dimensionality from 210x160x3 to 84x84 per frame (inputs stack 4 frames, so the input is actually 84x84x4). This helps the model have an easier time learning the data, since it has a lower dimension, while keeping meaning intact. (*Note: The final cropping is only used to make the inputs square so that the authors could use a specific GPU implementation of convolutions. It was not done for the purpose of reducing dimensionality.*)
 - Frame skipping. Taking an action each frame is extremely computationally expensive, and does not meaningfully aid performance. So, instead, the model chooses an action every fourth frame*, and that action is executed until the next time the model picks an action. Since emulator steps are far cheaper than running the model, this results in an around 4x speedup in simulation time.
 *Space invaders skipped every 3rd frame since skipping every 4th frame would make the lasers invisible due to the period at which they blinked.
 - One forward pass to predict Q values. The intuitive approach would be to use a network that takes as input the game state and an action, and outputs the predicted Q-value. However, this would require one forward pass per action. Instead, the paper authors opt for a model that takes as input only the game state, and produces as output the predicted Q-value for each action, allowing the model to run in one forward pass only. This makes it far more compute efficient, and ends up not harming performance in any notable way.
 - Reward clipping. The different games have rewards of different magnitudes. As such, to ensure the method can be moved from game to game without issue, all rewards are clipped: Positive rewards become 1, neutral rewards become 0, and negative rewards become -1. Though this has the downside of preventing the model from learning to pursue greater rewards over smaller ones, the model still obtains high performance, and it allows the model to easily generalize between games.
 - Epsilon-greedy exploration. To help the model explore and break out of local minima, the model sometimes is forced to take a random exploratory action with probability epsilon. This probability is annealed throughout the training run, decayed from 1 to 0.1 over the first 1 million frames. After that, epsilon remains fixed at 0.1.
 - Bellman update (Q-learning target). The paper trains the Q-network by minimizing \(L(\theta)=\mathbb{E}_{(s,a,r,s')\sim\mathcal D}\!\left[\big(r+\gamma\max_{a'}Q(s',a';\theta_{\text{prev}})-Q(s,a;\theta)\big)^2\right]\). Here \(s\) is the current state, \(a\) is the action taken, \(r\) is the immediate reward, \(s'\) is the next state sampled from the replay buffer \(\mathcal D\); \(\gamma\in[0,1)\) discounts future rewards; \(\theta\) are the current network weights/parameters, and \(\theta_{\text{prev}}\) (in the 2013 paper, the previous iteration’s weights. In 2015, a target network would be introduced.) supplies the bootstrapped target. Intuitively, the target says “the value of doing \(a\) in \(s\) should equal immediate reward plus the best attainable value thereafter”; bringing \(Q(s,a;\theta)\) to this self-consistent fixed point propagates future rewards backward to the actions that caused them, steadily improving the policy.

## Implementation & Training Details

### Optimizer & Training Budget

 - **Optimizer / batch:** RMSProp with minibatches of 32.
 - **Replay & horizon:** Replay memory of the **most recent 1M** transitions; total training budget = 10M frames.
 - **Epoch definition (for plots):** 1 epoch = 50,000 minibatch updates (~ 30 min of training time).

### Model Architecture
 - Conv1: 16 filters, 8×8, stride 4, ReLU
 - Conv2: 32 filters, 4×4, stride 2, ReLU
 - FC: 256 units, ReLU
 - Output: |A| linear heads (one per action; 4–18, depending on the game).
 - Action-value head: The network outputs Q(s, ·) in a **single forward pass** (state-only input; separate output unit per action), avoiding the cost of a pass per action.

## Results
### Main evaluation (average score; ε-greedy with ε = 0.05)

| Agent              | Beam Rider | Breakout | Enduro | Pong  | Q*bert | Seaquest | Space Invaders |
|--------------------|-----------:|---------:|-------:|------:|-------:|---------:|---------------:|
| Random             |        354 |      1.2 |      0 | -20.4 |    157 |     110  |            179 |
| Sarsa              |        996 |      5.2 |    129 |  -19  |    614 |     665  |            271 |
| Contingency        |       1743 |      6   |    159 |  -17  |    960 |     723  |            268 |
| **DQN (2013)**     |   **4092** |  **168** | **470** |**20**|**1952**| **1705** |        **581** |
| Human (expert)     |       7456 |     31   |    368 |   -3  |  18900 |   28010  |          3690  |

---

### Single best episode (reported for comparison)

| Agent        | Beam Rider | Breakout | Enduro | Pong | Q*bert | Seaquest | Space Invaders |
|--------------|-----------:|---------:|-------:|-----:|-------:|---------:|---------------:|
| HNeat Best   |       3616 |      52  |    106 |  19  |   1800 |     920  |        **1720**|
| HNeat Pixel  |       1332 |       4  |     91 | -16  |   1325 |     800  |           1145 |
| DQN Best     |   **5184** |   **225**| **661**|**21**|**4500**|  **1740**|           1075 |

As shown, DQN achieves the best average performance of any method on all 7 games, and achieves the best top episode performance for 6 of 7 games. It also, for the first time, surpassed human performance on Breakout, Enduro, and Pong, while achieving near human performance on Beam Rider. Though, due to difficulties with long time horizons, it considerably underperformed humans on Q*bert, Seaquest, and Space Invaders.

## Takeaways
 - Replay buffers are what make it stable. This key advance significantly reduces correlation between samples and variance, smoothes the behavior distribution, breaks feedback loops caused by on policy training, and helps prevent divergence. Reading this paper helped me understand just why replay buffers are so essential, and why they are so often used.
 - The more you can give to a neural net, the better. DQN performs so well because it lets the neural net learn on its own, so it can learn features with more potent meaning than hand crafted features could have. So often in the literature, it seems that the breakthrough is just finding techniques that allow the task to be completed end-to-end with a neural net (AlexNet is another great example of this).
 - Long time horizons are still hard. As impressive as DQN is, it still struggles to learn when time horizons are too lengthy, and it lacks the ability to plan long term. This is probably still a key area of research.
 - Throughput matters over quality. Frame skipping was used because the added experience in the games outweighed the tiny benefit from letting the model take more precise actions. So potential improvements to model performance could simply be optimizations that let it train longer.

## Remaining questions / topics for further reading
 - How much does reward clipping actually harm performance? Is there a way to get around reward clipping while preserving generalizability?
 - Why, exactly, does DQN underperform HNeat Best in the single top episode case in space invaders specifically?
 - How much does frame skipping actually impact performance, assuming you hold number of episodes trained constant? How much would varying the amount of frames you skip each time impact performance?
 - Would training with a more sophisticated optimizer, like Adam, help performance in any meaningful way? Or is SGD good enough?
 - Why does it lose on top episode performance for space invaders specifically? Is it a weakness of DQN, or a strength of HNeat Best?

[Paper link (arXiv)](https://arxiv.org/abs/1312.5602)