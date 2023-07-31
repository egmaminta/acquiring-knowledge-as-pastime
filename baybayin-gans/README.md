**Title**
-----------------------------------
Using Generative Adversarial Networks (GANs) in generating synthetic _Baybayin_ characters.

**Setup**
-----------------------------------
- Generate only one (1) character (i.e., the first syllable "_a_").
- Only the final losses (i.e., generator and discriminator losses after the last training epoch) are reported.

**Current observation**
-----------------------------------
- For the `Vanilla GAN`, generating all characters led to poor performance. Especially when initializing the weights and biases of each layer following a normal distribution (see `init_weights` function).
- Learning rate scheduler used is `CosineAnnealingLR`.

**Summary of Results**
-----------------------------------
| Model | Parameters | Epochs | Learning rate | Noise dimension | Batch size | Loss |
| ----- | ---------- | ------ | ------------- | --------------- | ---------- | ---- |
| `Vanilla GAN` | 515K | 500 | 3e-4 | 784 | 128 | `disc_loss=0.229`, `gen_loss=3.67` |
