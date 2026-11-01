---
exercise: vae
ai_use: "Claude assisted with the code implementation and the drafting of this report; all results and analyses were reviewed and validated by me."
---

# VAE — Variational Autoencoders

A variational autoencoder trained on MNIST twice: first with a 16-dimensional latent space, logging the two terms of its loss separately, then with a 2-dimensional one that can be drawn, to see what the prior buys and what it costs. The model is built and trained with PyTorch and autograd, as the statement allows; every VAE-specific part is written out in `code/vae.py`: the reparameterization, the reconstruction term (the binary cross-entropy summed over the 784 pixels), the closed-form KL to the prior, and the sampling. torchvision only loads MNIST, and SciPy only supplies Φ⁻¹ (`scipy.stats.norm.ppf`). Both loss terms are reported in nats per image: summed over the pixels (or the latent dimensions), averaged over the images.

**Approach.** The code is split into files under `code/`: `vae.py` (the model, the reparameterization, both loss terms, the sampling and the training loop), `common.py` (settings, the MNIST split and plotting helpers), `exercise1.py` and `exercise2.py` (the experiments, the figures and the measurements behind each analysis), `loss_check.py` (checks of the hand-written terms against independent computations) and `main.py`, which runs everything and writes `results/results.json`; `check_report.py` then confirms that the numbers of this report match that file. Every number and figure in this report is reproduced by installing `requirements.txt` and running, from the repository root:

```
python docs/exercises/vae/code/main.py
python docs/exercises/vae/code/check_report.py
```

The first run downloads MNIST into `data/MNIST` (not committed). The whole script takes about 80 seconds on a laptop CPU; the results below were produced with torch 2.15.0 and torchvision 0.29.2, and two runs give identical numbers.

**Seeds.** `torch.manual_seed(42)` is called right before building each of the two models, so both start from the same seed; the mini-batch shuffling and the ε of training come from that same generator. The split uses `torch.Generator().manual_seed(42)`, exactly as given, and the prior samples of Figure 3 are drawn after `torch.manual_seed(0)`. Evaluation draws its ε from generators of its own (seed 1 for the validation set, seed 2 for the test set), so measuring a model never changes its training, and every epoch is evaluated on the validation set with the same ε.

**Training.** Both models use the recipe of the statement: Adam with learning rate 10⁻³, batches of 128, 20 epochs, β = 1. After each epoch both terms are logged on the training set (the average over the epoch's batches) and on the validation set (measured at the end of the epoch). No decision is taken during training: the budget is fixed by the statement, with no early stopping and no model selection, so the validation set is only monitored. The test set is used once training is over, for the final numbers and for the figures the statement draws from it.

**Challenges.**

- *Hand-written terms that can be trusted.* `loss_check.py` compares each term with an independent computation. The reconstruction term equals `F.binary_cross_entropy(..., reduction="sum")` divided by the batch size, the call the statement names, to a relative 9.2 × 10⁻⁸ on all 10 000 validation images; its gradient agrees to 7.7 × 10⁻⁸ wherever the built-in function does not floor x̂(1 − x̂) at 10⁻¹² in its backward pass, which happens on 3.49% of the pixels, all but 1 of them background pixels. The closed-form KL matches a Monte Carlo estimate with 200 000 samples to within 2.04 standard errors (0.06%), and the reparameterization passes the gradients ∂z/∂μ = 1 and ∂z/∂log σ² = σε/2 exactly.
- *The reduction of the reconstruction term.* Summing over the pixels and averaging over the images keeps the two terms on the same scale. `reduction="mean"` would divide the reconstruction term by 784.00 (checked in `loss_check.py`), the same as training with β ≈ 784.
- *An answer to every analysis question, with a number.* Each claim in items 1E and 2E rests on a measurement printed by the code: the density of the cloud of training codes at the samples of Figure 3, the share of grey pixels, the information a KL can carry, the spread of the reconstructions, the ink of each interpolated image.

``` python title="code/main.py"
--8<-- "docs/exercises/vae/code/main.py"
```

## Exercise 1

*A VAE on MNIST.*

### A — Load and split the data

MNIST is loaded with `torchvision.datasets.MNIST(..., download=True, transform=transforms.ToTensor())`, which puts the pixels in [0, 1] (the range of the training images is [0.0, 1.0]), and each 28 × 28 image is flattened into a 784-vector. The 60 000 training images are split with `random_split(train_full, [50_000, 10_000], generator=torch.Generator().manual_seed(42))`, exactly as in the statement: 50 000 images for training and 10 000 for validation. The 10 000 official test images are the test set. The only change to the statement's lines is the root folder: `data/` at the repository root, written as an absolute path so the script runs from any directory (`load_mnist` in `code/common.py`).

### B — Build the model

| Part | Layers |
|---|---|
| Encoder | 784 → 512 → 256, ReLU after each |
| Heads | two linear layers 256 → d: one for μ, one for log σ² |
| Decoder | d → 256 → 512 → 784, ReLU after the hidden layers, sigmoid at the output |

With d = 16 the model has 1 079 600 parameters. The loss of one image, minimized on the average over the batch, is the negative ELBO:

$$
L = \underbrace{-\sum_{j=1}^{784} \Big[x_j \log \hat{x}_j + (1 - x_j) \log (1 - \hat{x}_j)\Big]}_{\text{reconstruction (BCE)}} \;+\; \beta \cdot \underbrace{\tfrac{1}{2}\sum_{i=1}^{d} \left(\mu_i^2 + \sigma_i^2 - 1 - \log \sigma_i^2\right)}_{\text{KL}}, \qquad \beta = 1 .
$$

The four VAE-specific parts are the functions of `code/vae.py`, all written out:

- `reparameterize`: z = μ + σ ⊙ ε with σ = exp(½ log σ²) and ε ~ N(0, I). The noise enters as an input, so the gradient of the reconstruction reaches both heads.
- `reconstruction_term`: −log p(x | z) for a decoder that outputs, per pixel, the probability x̂ⱼ that it is "on" (a Bernoulli likelihood). It is the formula above, summed over the 784 pixels and averaged over the images. As in `F.binary_cross_entropy`, each log is floored at −100, so a saturated sigmoid would cost 100 nats rather than an infinite loss.
- `kl_term`: the KL from N(μ, σ²) to N(0, I) in closed form, summed over the d dimensions and averaged over the images; `torch.distributions` is not used anywhere.
- `sample`: decodes z ~ N(0, I).

``` python title="code/vae.py"
--8<-- "docs/exercises/vae/code/vae.py"
```

### C — Train

The untrained model has a validation reconstruction of 544.72 nats and a KL of 0.0291 nats. Both terms, logged after each epoch:

| Epoch | Training reconstruction | Training KL | Validation reconstruction | Validation KL | Validation total |
|---|---|---|---|---|---|
| 1 | 177.28 | 6.78 | 129.82 | 13.37 | 143.19 |
| 2 | 117.18 | 14.62 | 108.16 | 15.78 | 123.93 |
| 5 | 92.54 | 18.13 | 91.90 | 17.76 | 109.67 |
| 10 | 85.54 | 19.01 | 86.19 | 19.13 | 105.33 |
| 15 | 82.72 | 19.39 | 83.89 | 19.39 | 103.29 |
| 20 | 81.03 | 19.60 | 82.79 | 19.44 | 102.24 |

On the test set, after training:

| d = 16, test set | nats per image |
|---|---|
| Reconstruction | **82.30** |
| KL | **19.48** |
| Negative ELBO (their sum) | **101.78** |

The statement's reference values are about 81 and 20. The reconstruction term cannot reach zero: for a grey pixel the BCE is smallest at x̂ⱼ = xⱼ, where it equals the pixel's entropy, so even a perfect decoder would pay 45.53 nats per test image (the summed pixel entropies). The 82.30 nats are 36.77 above that floor.

![Figure 1](figures/fig1.png)
/// caption
Figure 1 — The reconstruction term (left) and the KL term (right) per epoch, on the training set (average over the epoch) and on the validation set (end of the epoch), d = 16.
///

In epoch 1 the training average is far above the validation value (177.28 against 129.82) because it averages over the whole epoch, starting from the untrained model, while the validation is measured after the epoch. From then on the two curves stay close; after epoch 20 the validation reconstruction is 1.77 nats above the training average.

### D — Reconstructions and samples

Figure 2 uses the first test image of each digit, at test indices 3, 2, 1, 18, 4, 8, 11, 0, 61 and 7 (digits 0 to 9), and reconstructs each by decoding μ, without sampling z. Figure 3 decodes 64 codes z ~ N(0, I) drawn after `torch.manual_seed(0)`. Both figures show the decoder's output x̂, the probability of each pixel.

![Figure 2](figures/fig2.png)
/// caption
Figure 2 — Top: the first test image of each digit (test index below the digit). Bottom: its reconstruction, the decoder applied to μ.
///

![Figure 3](figures/fig3.png)
/// caption
Figure 3 — 64 images decoded from z ~ N(0, I), drawn after `torch.manual_seed(0)`, in an 8 × 8 grid.
///

### E — Analysis

**Why the optimizer pays more KL.** The KL is the price of a code that says something about the image. Averaged over the data it splits into the information the code carries about x and the mismatch between the cloud of all codes and the prior (the latent-space lecture's split, with q(z) the aggregate posterior):

$$
\mathrm{E}_x\big[\mathrm{KL}(q(z \mid x)\,\|\,p(z))\big] = I_q(x; z) + \mathrm{KL}(q(z)\,\|\,p(z)) .
$$

The untrained encoder returns almost the prior for every image (KL 0.0291 nats), a code that tells the decoder almost nothing about which image it came from. With no information in the code the decoder can only output one image for every input, and the best such image for the test set, the mean test image, still costs 205.65 nats of reconstruction. The optimizer minimizes the sum of the two terms, so it buys information while a nat of KL saves more than a nat of reconstruction. Between epoch 1 and epoch 20 the validation KL rose by 6.07 nats (13.37 to 19.44) while the reconstruction fell by 47.03 (129.82 to 82.79), and the total fell by 40.95. It pays only where it gains: on the test set, 11 of the 16 dimensions carry between 1.195 and 2.449 nats each, with a mean σ between 0.09 and 0.32, while the other 5 stay on the prior, with at most 0.0016 nats and a mean σ between 0.98 and 1.00. A dimension costs KL as soon as it carries anything, and the optimizer leaves these 5 unused.

**Why Figure 3 looks worse than Figure 2: two reasons.**

1. *Where the prior samples land.* The decoder is trained only on codes drawn from q(z | x) for training images, and the KL pulls each of those Gaussians toward N(0, I) without making their union fill N(0, I). That mismatch is the second term of the split above, and it is far from zero. With 50 000 training images the code can carry at most log 50 000 = 10.82 nats about which image it came from, so of the mean training KL of 19.51 nats at least 8.69 nats are KL(q(z) ‖ p(z)). Measured directly: at the codes of the test images, the density of the cloud of training codes has a median log q(z) of −31.28 and a 1st percentile of −73.02. At the 64 codes of Figure 3 the median is −51.32, 20.04 nats lower, and 10 of the 64 codes fall below that 1st percentile, where 1% of 64 would be 0.64. Those samples come from regions where the decoder was hardly trained ("off the manifold, nothing was trained"), and that is where shapes that are no digit at all can come from.
2. *What the decoder outputs for a given z.* The decoder does not output an image, it outputs the probability of each pixel, which is the mean of p(x | z). If several images are consistent with a code, the loss-minimizing output is their average, and averages of digits whose strokes fall in slightly different places are grey where real digits are black or white (the lecture's section on blur: maximum likelihood is mode-covering and returns the conditional mean). The share of grey pixels, strictly between 0.1 and 0.9, shows it: 8.59% in the test images, 14.53% in their reconstructions from μ and 16.53% in the samples of Figure 3, the blurriest of the three.

**Why μ in Figure 2.** μ is the centre of q(z | x), the single most likely code for the image. A sampled z = μ + σε moves the code by the encoder's uncertainty, a mean σ between 0.09 and 0.32 on the active dimensions, toward codes of other images. The reconstruction would then change from one draw to the next and show that noise on top of what the code keeps. Decoding μ shows the code itself, the same image every time, and it is also the better reconstruction: 77.02 nats on the test set against 82.30 with a sampled z.

### Code

``` python title="code/exercise1.py"
--8<-- "docs/exercises/vae/code/exercise1.py"
```

``` python title="code/common.py"
--8<-- "docs/exercises/vae/code/common.py"
```

``` python title="code/loss_check.py"
--8<-- "docs/exercises/vae/code/loss_check.py"
```

## Exercise 2

*A latent space you can see.*

### A — Train a 2-D VAE

The model, data, seed, β and training budget are those of Exercise 1; only d changes, from 16 to 2 (`torch.manual_seed(42)` and then `VAE(2)`). After epoch 20 its validation reconstruction is 134.58 nats and its KL 6.76 nats. On the test set, next to the d = 16 model:

| Model | Test reconstruction | Test KL | Negative ELBO |
|---|---|---|---|
| d = 16 (Exercise 1) | 82.30 | 19.48 | 101.78 |
| d = 2 | **133.88** | **6.78** | 140.66 |

The statement expects roughly 135 and 7 nats. With two dimensions the reconstruction is 51.58 nats worse and the KL 12.70 nats smaller.

### B — The encoded test set

![Figure 4](figures/fig4.png)
/// caption
Figure 4 — μ of all 10 000 test images under the d = 2 model, colored by digit. The dashed square is the region decoded in Figure 5 (|z| ≤ 1.64 on both axes).
///

### C — The decoded plane

Figure 5 decodes a 15 × 15 grid of z = (z₁, z₂) with each coordinate z = Φ⁻¹(q), for the 15 evenly spaced q from 0.05 to 0.95. The grid values run from −1.64 to 1.64, close together near 0 and far apart toward the edges. **Because the prior N(0, I) is the product of two independent N(0, 1) and Φ is their CDF, a coordinate falls between two consecutive grid values with probability equal to the step in q, 0.0643, for every step, so every cell between neighbouring grid lines holds the same prior mass, 0.0643² = 0.0041.** A linear grid would give the cells near the edges much less mass than those at the centre.

![Figure 5](figures/fig5.png)
/// caption
Figure 5 — The decoder of the d = 2 model on a 15 × 15 grid, z = Φ⁻¹(q) with q = 0.05, …, 0.95 on each axis, tiled in the same layout as the plane: z₁ grows to the right, z₂ upward. The tick labels give the z of each column and row.
///

### D — Interpolation

The two images are the first two of the test set, **test index 0 (a 7) and test index 1 (a 2)**, a rule fixed before looking at any interpolation. Figure 6 uses t = 0, 1/9, …, 1. The top row mixes the pixels, (1 − t) x_a + t x_b. The bottom row decodes (1 − t) μ_a + t μ_b with the d = 16 model of Exercise 1, so its two ends are the reconstructions of the two images, not the images themselves.

![Figure 6](figures/fig6.png)
/// caption
Figure 6 — From test image 0 (a 7) to test image 1 (a 2), for t from 0 to 1. Top: interpolation in pixel space. Bottom: interpolation in the latent space of the d = 16 model, decoded.
///

### E — Analysis

**Which digits overlap in Figure 4, and the same transition in Figure 5.** To measure the overlap, each test code is given the majority digit of its 10 nearest training codes, and the test codes that land among another digit's codes are counted per pair. In both directions: 4 and 9, 641; 5 and 8, 269; 3 and 5, 228; 7 and 9, 118; 3 and 8, 108. The largest overlap is between **4 and 9**, the purple and cyan clouds right of the centre of Figure 4. Their mixed codes are centred at μ = (0.96, −0.11), and a second, smaller group mixes 3, 5 and 8 left of the centre.

The same transition is in the right half of Figure 5. The cells whose nearest training codes switch between 4 and 9 form a band from row 1, column 10 down to row 14, column 15 (rows counted from the top, columns from the left). In rows 4 to 7 the switch happens between columns 11 and 12, from z₁ = 0.50 to 0.70, and the mixed codes of Figure 4 fall around row 9, column 13. Along those rows the open 4 closes. The images in between are 4s whose two upper strokes meet in a squarish loop, which is also a 9 with an angular loop. One cell to the right the loop is round and the stem straight, and the image is a 9. Where the 4s meet the 5s and the 6s (column 9) the in-between images are faint and blurred. In the 3–5–8 group, in the centre-left, the in-between images are 8s with one side of a loop left open, which can be read as a 3 or a 5.

**What two numbers per image force the model to throw away.** The d = 2 model reconstructs 51.58 nats worse because its code carries less. By the split of item 1E, the KL is an upper bound on the information the code holds: 6.78 nats, that is 9.8 bits, against 28.1 bits with d = 16. Naming the digit alone takes log₂ 10 = 3.3 bits. What goes is the detail of each image. The reconstructions of all test images of one digit vary only 34% as much as the images themselves (summed pixel variance 14.31 against 41.67), against 73% (30.33) with d = 16. Part of the digit's identity goes too: the majority digit of the 10 nearest training codes matches the label for 80.84% of the test images in 2-D and 96.36% in 16-D. Figure 5 is the evidence. The plane is spent on which digit it is: every digit gets its own patch, between 18 and 29 of the 225 cells by the nearest training codes. Inside a patch the images change smoothly along one direction only, so the 1s along the top differ only in slant and the 0s along the bottom only in width and slant. Every image in Figure 5 is a smooth, typical digit or a blend of two neighbouring ones; what two numbers cannot hold are the quirks of each handwriting, which is what the 34% measures.

**The two rows of Figure 6.** The **bottom (latent) row** passes through images that look like digits; the top (pixel) row does not.

- The pixel row is two digits at once. Between t = 0.22 and 0.56 each image has 209 to 212 ink pixels (above 0.1), close to the 222 that the 7 and the 2 cover together, while the 7 has 105, the 2 has 145 and a test image has 137.7 on average. About a quarter of its pixels are grey (26.0% to 26.1% at t = 0.22 to 0.44, against 8.4% at both ends): the strokes of both digits at partial intensity. Digit images do not form a convex set, so the average of two digits is not a digit.
- The latent row is decoded, and the decoder only produces the kind of image it was trained on. The straight line from μ_a to μ_b stays inside the cloud of training codes: its lowest log q(z), −40.25, is far above the 1st percentile of the test codes, −73.02. The KL term makes the posteriors overlap, and the lecture credits that overlap with filling the space between training examples. Along the row the 7 bends into a 3-like shape (t = 0.33 to 0.56; at t = 0.56 the majority digit of the 10 nearest training images is a 3) and then into the 2, one stroke at a time: the ink stays between 104 and 175 pixels, one digit's worth. Like every output of the decoder, these images are blurrier than real digits (10.1% to 21.4% grey pixels), but each is a single digit.

### Code

``` python title="code/exercise2.py"
--8<-- "docs/exercises/vae/code/exercise2.py"
```

``` python title="code/check_report.py"
--8<-- "docs/exercises/vae/code/check_report.py"
```

## Results summary

| # | Quantity | Value |
|---|---|---|
| 1 | Ex. 1 — test reconstruction, KL and negative ELBO (d = 16) | reconstruction 82.30, KL 19.48, negative ELBO 101.78 nats per image |
| 2 | Ex. 1 — validation KL at the first and at the last epoch | 13.37 nats after epoch 1, 19.44 nats after epoch 20 |
| 3 | Ex. 2 — test reconstruction and KL (d = 2) | reconstruction 133.88, KL 6.78 nats per image |
| 4 | Ex. 2 — the two test indices interpolated | 0 (a 7) and 1 (a 2) |
