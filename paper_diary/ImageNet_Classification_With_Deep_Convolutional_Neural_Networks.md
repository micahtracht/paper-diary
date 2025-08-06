# ImageNet Classification with Deep Convolutional Neural Networks - Krizhevsky, Sutskever, Hinton., 2012

## Problem
How do you make a computer-vision model that can effectively leverage large amounts of data and compute to make accurate classifications across many categories without overfitting or using far too much compute? This paper improves upon the classic shallow classifiers with hand crafted features by using a much deeper, learned feature hierarchy, trained on GPUs, while dodging overfitting by using data augmentation & dropout.
## Methods
 - Using a deeper neural network with learned feature extraction. By adding depth to the neural network, and just letting it learn on the data, the model learns its own useful feature representations that are often superior to the hand crafted ones. This makes for a simpler, more robust, and more effective model.
 - Using ReLU activations. Standard activations tended to be either tanh or sigmoid, which had the problem of being "saturating", meaning that at very small or very large inputs, they barely changed at all, which led to vanishing gradients. ReLUs (Rectified Linear Units) are, by contrast, non-saturating, meaning that (as long as the value is positive), they'll have a gradient. This helps the model avoid the vanishing gradient problem and learn the data far faster, as shown by a model (predating AlexNet) using ReLUs reaching 25% error rate on CIFAR-10 6x faster than one using tanh activations.
 - Dropout. By randomly setting the output of half of the neurons to 0, the model is forced to learn more robust feature classifications, reducing overfitting. Fragile, complex neuron structures cannot survive dropout because neurons in those structures can no longer rely on the other neurons in the structure. This forces the network to learn simpler & more robust relationships, which are much less likely to lead to overfitting. As the saying goes, simplicity is intelligence, and so by forcing the model to learn simple structures, it learns simpler relationships in the data, and so makes 'smarter' predictions.
 - Data Augmentation. An image of a car is still an image of a car if it's desaturated, flipped, or if the colors are slightly off (in psychology, this is analogous to color & shape constancy). So we apply the same principle in training: By creating variants of each image, we force the model to learn what actually matters about the image, and teach it to tolerate minor, inconsequential variations, like humans do. The paper authors accomplish this by generating translations/reflections, and by generating color shifted images (using PCA based color jitter, which uses eigenvalue-scaled noise, to mimic natural image variance).
 - Overlapping pooling. By letting some data overlap, it gets viewed by the model multiple times in different contexts, which aids in feature capture. Also, overlapping pooling means small shifts in the data cause smaller changes in the pooled outputs, and the added redundancy that overlapping pools cause makes it harder for the model to memorize precise pixel arrangements. This makes it harder for the model to overfit, improving generalizability.
 - Training on multiple GPUs. To obtain a larger model, the authors find a way to train it over two GPUs, circumventing the GPU memory constraints of the time. It's worth noting that nowadays, this technique is behind every large scale training run of any LLM.

## Results
| Metric (ILSVRC) | Baseline 2011 (best conventional pipeline) | **AlexNet 2012** | Δ (absolute) |
|-----------------|--------------------------------------------|------------------|--------------|
| **Top-5 error** | 26.2 %                                     | **15.3 %**       | **−10.9 pp** |
| **Top-1 error** | ~45 %                                      | **37.5 %**       | **−7.5 pp**  |

## Takeaways
 - I had always thought of dropout as only eliminating fragile neural network structures. The perspective of dropout as a way of sampling many different architectures for cheap, achieving a similar feat as just using many different neural nets, was new to me, and changed how I see dropout.
 - Scaling data & compute will almost always beat techniques that don't scale (see: bitter lesson).
 - Never knew about data augmentation. It makes a lot of sense, but seems very interesting.
 - The amount of information in each training example is related to how many potential labels you have (specifically, info = log_2(# examples)).
 - The more you can leave up to the model, the better. Ex: learned feature representations outperform hand crafted ones.

## Remaining questions / topics for further reading
 - How does this scale? We saw 1.2 million images. What about 12 million? 120 million? And same for GPUs.
 - How hard is it to recreate AlexNet precisely, or something with AlexNet performance, on a home computer now? It'd be interesting to see how far the field has come.
 - What improvements were made on the architecture? Can more gains be made just by stacking more convolutional layers?
 - Do more sophisticated optimizers (e.g. Adam, AdamW) improve performance appreciably, or is SGD w/ momentum good enough?
 - Would dropping out less or more neurons change the results considerably? Can you do better than dropout rate = 0.5?

[Paper Link (Neurips)](https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)