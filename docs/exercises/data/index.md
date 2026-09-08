---
exercise: data
ai_use: "Claude assisted with the code implementation and the drafting of this report; all results and analyses were reviewed and validated by me."
---

# Data — Preparation and Analysis for Neural Networks

Data preparation and analysis for neural networks. Every script in this activity fixes the random seed with `rng = np.random.default_rng(42)` and consumes the generator in a fixed, documented order, so every number and figure below is exactly reproducible.

## Exercise 1

### A — Generate the clouds

I generated 400 samples divided equally among 4 classes (100 samples each), drawing each class from a 2D Gaussian with the parameters given in the statement:

| Class | Mean | Standard deviation |
|---|---|---|
| 0 | [2, 3] | [0.8, 2.5] |
| 1 | [5, 6] | [1.2, 1.9] |
| 2 | [8, 1] | [0.9, 0.9] |
| 3 | [15, 4] | [0.5, 2.0] |

The generator is created once with seed 42 and consumed in a fixed order (the four scaled datasets of item B, from s = 0.5 to s = 4.0), so the s = 1.0 dataset shown in Figure 1 is always the same.

![Figure 1 — Four Gaussian clouds (s = 1.0), class means marked with X](figures/fig1.png)
/// caption
Figure 1 — The four point clouds at s = 1.0, one color per class, with each class mean marked with a black X.
///

### B — More or less spread out

The same 4 classes were generated four times over, multiplying all standard deviations by the scale factor s ∈ {0.5, 1.0, 2.0, 4.0} — the means never change, only the spread. Figure 2 shows the four datasets with shared axis limits, so the growth of the overlap is directly comparable.

![Figure 2 — The same four classes under the four scale factors](figures/fig2.png)
/// caption
Figure 2 — The same four classes under s ∈ {0.5, 1.0, 2.0, 4.0}, with shared axis limits.
///

**Separation ratio (s = 1.0).** For each pair of classes, r_ij = ‖μ_i − μ_j‖ / (σ̄_i + σ̄_j), with σ̄_k the average of the two axis standard deviations of class k (σ̄_0 = 1.65, σ̄_1 = 1.55, σ̄_2 = 0.90, σ̄_3 = 1.25):

| Pair (i, j) | ‖μ_i − μ_j‖ | σ̄_i + σ̄_j | r_ij |
|---|---|---|---|
| (0, 1) | 4.243 | 3.20 | **1.326** |
| (0, 2) | 6.325 | 2.55 | 2.480 |
| (0, 3) | 13.038 | 2.90 | 4.496 |
| (1, 2) | 5.831 | 2.45 | 2.380 |
| (1, 3) | 10.198 | 2.80 | 3.642 |
| (2, 3) | 7.616 | 2.15 | 3.542 |

The smallest ratio is **r_01 = 1.326** — classes 0 and 1 are the closest pair relative to their spreads, which matches the visible contact between the blue and orange clouds in Figure 1. Since the means do not change with s, r_ij scales with 1/s: at s = 2 the smallest ratio becomes **r_01 = 1.326 / 2 = 0.663**, without generating anything new.

**Mixing rate.** For each s, the fraction of points whose nearest class center (among the four means) is not the one of their own class — a purely geometric measure, nothing is trained:

| s | Mixing rate |
|---|---|
| 0.5 | 0.25% |
| 1.0 | 7.25% |
| 2.0 | 19.25% |
| 4.0 | 48.25% |

![Figure 3 — Mixing rate as a function of the scale factor s](figures/fig3.png)
/// caption
Figure 3 — Mixing rate as a function of s. With 4 balanced classes, random assignment would give 75%.
///

**From which scale factor can the clouds no longer be separated by straight lines?** From **s = 2.0** on. At s = 1.0 the mixing rate is still 7.25% — a set of straight lines separates the classes with only a thin contaminated strip between classes 0 and 1. At s = 2.0 the mixing rate jumps to 19.25%: roughly one point in five already sits closer to another class's center than to its own, so no arrangement of straight lines can keep the classes apart — errors are no longer confined to a thin boundary strip. Exactly at that point the smallest separation ratio drops to r_01 = 0.663 < 1, meaning the distance between the two closest centers is smaller than the sum of their average spreads: the clouds interpenetrate rather than merely touch.

### C — Analysis

**Overlap at s = 1.0.** Classes 2 (green) and 3 (red) are essentially isolated: their smallest ratios towards any other class are ≥ 2.38, and Figure 1 shows clear gaps around them. The overlap is concentrated in the pair 0–1 (r_01 = 1.326): class 0 is very elongated vertically (σ_y = 2.5) and its upper tail invades the region of class 1, producing the 7.25% mixing measured above.

**Could a single linear boundary separate all classes?** No — this is impossible regardless of the data: one straight line divides the plane into only two half-planes, and we have four classes. **A set of linear boundaries?** Yes, to a good approximation at s = 1.0: the classes are arranged so that piecewise-linear frontiers (equivalently, one-vs-rest linear separators) isolate each cloud, misclassifying only the thin 0–1 contact strip.

**Sketched boundaries.** Figure 1 (annotated) below sketches the boundaries a trained network could plausibly learn, drawn as the nearest-center (Voronoi) partition of the plane — piecewise-linear frontiers, consistent with the fact that errors concentrate on the 0–1 edge:

![Figure 1 (annotated) — sketched decision boundaries](figures/fig1_boundaries.png)
/// caption
Figure 1 (annotated) — Sketched piecewise-linear decision boundaries (nearest-center regions) over the s = 1.0 data.
///

**Spread × inevitable-error region.** The sketched boundaries pass through the low-density valleys between clouds. As s grows, each cloud's tails cross the boundary into its neighbors' regions, and the "band of confusion" around every frontier widens — the mixing rate quantifies exactly this growth (0.25% → 7.25% → 19.25% → 48.25%). Since class-conditional distributions overlap, no decision boundary — linear or not — can avoid errors in the overlap region: a network necessarily makes mistakes there, and the size of that region grows with the spread. At s = 4.0 nearly half the points (48.25%) are already on the "wrong side" geometrically, approaching the 75% of pure chance for 4 balanced classes.

### Code

``` python
--8<-- "docs/exercises/data/code/exercise1.py"
```

## Exercise 2

### A — Dataset I: shifted Gaussians

I generated 500 samples for Class A and 500 for Class B with `rng.multivariate_normal`, using exactly the mean vectors and covariance matrices of the statement: μ_A = [0, 0, 0, 0, 0] with Σ_A (unit variances, +0.8 correlation between the first two features), and μ_B = [1.5, 1.5, 1.5, 1.5, 1.5] with Σ_B (variances 1.5, −0.7 correlation between the first two features). Before sampling, the script checks that both matrices are valid covariances: symmetric, with smallest eigenvalues 0.158 (Σ_A) and 0.498 (Σ_B), both positive.

Sanity check of the generation (sample statistics over 500 points): the sample means are [0.026, 0.085, 0.037, 0.019, 0.035] for Class A and [1.524, 1.452, 1.473, 1.519, 1.530] for Class B, and the largest absolute deviation between a sample covariance entry and its target is 0.086 (A) and 0.137 (B) — the expected sampling noise for n = 500.

### B — Dataset II: concentric shells

The second dataset also has 500 samples per class in 5 dimensions, but with radial structure. For each point I draw a direction v ~ N(0, I₅) and normalize it, u = v / ‖v‖ (uniform on the unit sphere of ℝ⁵), then draw a radius ρ ~ N(2.0, 0.4) for Class C (core) or ρ ~ N(5.0, 0.4) for Class D (shell), and set x = ρ · u. I read N(μ, 0.4) as mean μ and standard deviation 0.4, the same convention as Exercise 1. The script verifies that every direction has unit norm before scaling. The generator is consumed in the order Class A, Class B, Class C (directions, then radii), Class D (directions, then radii).

### C — Visualize and compare

**Figure 4 — PCA projection.** PCA (scikit-learn, `n_components=2`) was fitted on each complete dataset (both classes together, unsupervised) and the two projections are plotted side by side with shared axis limits and equal aspect, so the scatter plots are directly comparable.

![Figure 4 — PCA projection to 2D of both datasets](figures/fig4.png)
/// caption
Figure 4 — PCA projection to 2D of Dataset I (left) and Dataset II (right), colored by class, with shared axes.
///

**Explained variance of the first two components:**

| Dataset | PC1 | PC2 | PC1 + PC2 |
|---|---|---|---|
| I — shifted Gaussians | 51.27% | 15.77% | **67.04%** |
| II — concentric shells | 21.59% | 21.32% | **42.91%** |

**In which dataset does the 2D projection better preserve the information relevant for classification?** In **Dataset I**. There, the shift of the means along (1, 1, 1, 1, 1) is the largest source of variance in the pooled data, so PC1 aligns with the direction that separates the classes and captures 51.27% of the variance on its own: in Figure 4 (left) the two classes appear as two displaced blobs. In Dataset II the distribution is isotropic — every direction carries about 1/5 = 20% of the variance (21.59% ≈ 21.32%), so PCA has no preferred direction to find and the projection is essentially arbitrary. Worse, the only information that separates C from D is the radius, and a projection to 2 coordinates discards the contribution of the other 3 coordinates to ‖x‖²: the projected shell fills a disk instead of a ring, and the core sits inside it. Quantitatively, 38.80% of the Class D points project inside the disk (radius 2.717 in the PCA plane) that contains every Class C point.

**Distance between the class centers, computed in 5D** from the sample means:

| Dataset | ‖μ₁ − μ₂‖ (sample) | Theoretical |
|---|---|---|
| I — shifted Gaussians | **3.264** | 1.5 · √5 = 3.354 |
| II — concentric shells | **0.266** | 0 (both classes centered at the origin) |

**Figure 5 — Radius histograms.** The radius ‖x‖ of every point was computed in 5D and plotted per class, both classes overlaid on the same axis, one panel per dataset:

![Figure 5 — Histogram of the radius of each point](figures/fig5.png)
/// caption
Figure 5 — Histogram of ‖x‖ (computed in 5D) for each class, overlaid, in Dataset I (left) and Dataset II (right).
///

Per class, the radius is 2.109 ± 0.787 (A) and 4.175 ± 1.160 (B) in Dataset I — overlapping distributions — and 1.972 ± 0.396 (C) versus 5.005 ± 0.409 (D) in Dataset II — two disjoint peaks, matching the radii 2.0 and 5.0 with standard deviation 0.4 that generated them. The two datasets are mirror images of each other: in Dataset I the classes differ by their **centers** (distance 3.264) and overlap in radius; in Dataset II the centers **coincide** (distance 0.266 ≈ 0) and the classes differ only by their radius.

### D — Analysis

**Coincident centers × separated radii → no hyperplane can separate the classes.** A hyperplane w·x + b = 0 splits ℝ⁵ into two half-spaces. Because both classes are centered at the same point (the origin) and spread symmetrically in every direction, the projection w·x of either class onto any direction w is symmetric around the same value, w·0 = 0: whatever w is, about half of Class C and half of Class D fall on each side of a hyperplane through the center. Moving the hyperplane away from the center does not help either: to leave the whole shell (radius ≈ 5) on one side, its distance from the origin |b| / ‖w‖ would have to exceed 5, but then the entire core (radius ≈ 2 < 5) lies on that same side. A linear boundary can only exploit a difference between the class **centers**, and here that difference is essentially zero (0.266), even though the radius histograms in Figure 5 (right) do not even touch. The classes are perfectly separable — just not by a hyperplane.

**Why more data cannot fix it.** This is a property of the geometry of the two populations, not of the sample: the core is (approximately) the sphere of radius 2 and the shell the sphere of radius 5 around the same center, and every half-space either contains parts of both spheres or contains both entirely. Collecting more points only fills the two spheres more densely; it never creates a direction along which the classes are displaced. In Dataset I, by contrast, the classes are displaced (distance 3.264), and a single hyperplane — the one equidistant from the two centers, i.e. the nearest-center rule — already separates 88.20% of the points in 5D (nearest-center mixing rate of 11.80%, a purely geometric measure). That is exactly the kind of boundary a Perceptron can learn; Dataset II is exactly the kind it cannot.

**Does a mixed 2D projection prove inseparability?** No. PCA is a linear transformation: it can only discard information, never create it, so overlap in the projection is a property of the projection, not of the data. My own results show this directly: in the PCA plane, 38.80% of the Class D points land inside the disk containing all of Class C (Figure 4, right), yet in the original 5D space the classes are separated with 100.00% accuracy (1000/1000 points) by the simple function below. The converse does hold: separation in a linear projection *does* prove separability in the original space, because the projection is itself a linear function of the inputs — this is what Figure 4 (left) shows for Dataset I.

**A simple function of the inputs that separates Dataset II.** Following the hint, use the squared radius:

f(x) = ‖x‖² − 3.5² = x₁² + x₂² + x₃² + x₄² + x₅² − 12.25,  predict **Class D if f(x) > 0**, **Class C otherwise**.

The threshold 3.5 is the midpoint between the two mean radii (2.0 and 5.0), 3.75 standard deviations (0.4) away from each; on the generated data this rule classifies 1000 of 1000 points correctly (100.00% accuracy). It is non-linear in x (quadratic), which is why a linear model cannot represent it — but a network with one hidden layer of non-linear units can build it, or equivalently a Perceptron becomes sufficient if the inputs are first expanded with the squared features x_i², in which f is linear.

### Code

``` python
--8<-- "docs/exercises/data/code/exercise2.py"
```

## Exercise 3

### A — Get to know the data

**Source.** Spaceship Titanic (Kaggle), file `train.csv` — the only labeled file. It is committed under `data/spaceship-titanic/train.csv` so the script runs from a clean checkout. The file has **8 693 passengers × 14 columns**.

**Goal and target.** The dataset comes from a fictional accident: the Spaceship Titanic collided with a spacetime anomaly and part of the passengers were "transported to an alternate dimension". The column `Transported` is the boolean target — `True` if the passenger was transported — and the goal is to predict it from the passenger records (a binary classification problem). The classes are almost perfectly **balanced**: 4 378 `True` (**50.36%**) and 4 315 `False` (49.64%), so no class-imbalance treatment is needed and 50% is the baseline of a trivial classifier.

**Features by type.**

| Type | Columns |
|---|---|
| Numerical (6) | `Age`, `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck` (the last five are amounts spent on board) |
| Categorical (5) | `HomePlanet` (Earth / Europa / Mars), `CryoSleep` (True / False), `Destination` (TRAPPIST-1e / 55 Cancri e / PSO J318.5-22), `VIP` (True / False), `Cabin` (a deck/number/side code with 6 560 distinct values) |
| Identifiers (2) | `PassengerId`, `Name` |
| Target | `Transported` (boolean) |

**Missing values per column** (absolute count and percentage of 8 693 rows):

| Column | Missing | % |
|---|---|---|
| PassengerId | 0 | 0.00% |
| HomePlanet | 201 | 2.31% |
| CryoSleep | 217 | 2.50% |
| Cabin | 199 | 2.29% |
| Destination | 182 | 2.09% |
| Age | 179 | 2.06% |
| VIP | 203 | 2.34% |
| RoomService | 181 | 2.08% |
| FoodCourt | 183 | 2.11% |
| ShoppingMall | 208 | 2.39% |
| Spa | 183 | 2.11% |
| VRDeck | 188 | 2.16% |
| Name | 200 | 2.30% |
| Transported | 0 | 0.00% |

Every feature has about 2% of missing entries; the identifier and the target are complete. Since the gaps are small and spread over all columns, imputation is preferable to dropping rows (dropping any row with a missing value would discard far more than 2% of the data).

**Spending columns** (mean, median and maximum on the full `train.csv`; the skewness is added to support the reading):

| Column | Mean | Median | Maximum | Skewness |
|---|---|---|---|---|
| RoomService | 224.69 | 0.0 | 14 327 | 6.33 |
| FoodCourt | 458.08 | 0.0 | 29 813 | 7.10 |
| ShoppingMall | 173.73 | 0.0 | 23 492 | 12.63 |
| Spa | 311.14 | 0.0 | 22 408 | 7.64 |
| VRDeck | 304.85 | 0.0 | 24 133 | 7.82 |

**Mean × median.** In all five columns the median is **0** while the mean is in the hundreds: more than half of the passengers spent nothing, and the mean is pulled up by a minority of big spenders (maxima between 14 327 and 29 813 credits, i.e. 30 to 130 times the mean). A mean far above the median is the signature of a strongly **right-skewed** distribution with a **heavy tail** (skewness between 6.3 and 12.6 — a Gaussian has 0): the spread is enormous relative to the typical value, and the mean is not a representative "typical passenger". These are exactly the features that need the log transformation in item C.

### B — Split before you transform

The data was split **80/20, stratified by `Transported`, with the fixed seed** — using the seeded generator `rng` to shuffle the rows of each class separately and send 20% of each class to the test set. Result: **6 954 training rows (80.0%) and 1 739 test rows (20.0%)**, with a positive share of 0.5036 in the training set and 0.5037 in the test set, against 0.5036 in the full file — the stratification preserves the class balance in both parts. No statistic was computed before this point (item A only described the raw file).

**Why the split comes before imputation and scaling.** Every value used by the transformations — the medians and modes for imputation, the categories of the encoder, the minimum and maximum of the scaler — is a statistic estimated from data. If they were estimated on the full dataset, information from the test rows (their averages, their extreme values, their categories) would leak into the transformed training data, and the test set would no longer be an honest sample of unseen data: the performance measured on it would be optimistically biased. Fitting every transformation on the training split alone and only *applying* it to the test split reproduces the real situation of deployment, where new data arrives after the preprocessing has been fixed.

On the training set, before any transformation, `FoodCourt` has **mean 465.35, median 0.00 and maximum 29 813** (Results summary, item 11).

### C — Preprocess

All five steps below are **fitted on the training set only** and then applied, unchanged, to the test set. The tanh activation outputs values in [−1, 1], so the goal is a feature matrix on a compatible, bounded scale.

**1. Missing data.** Numerical columns are imputed with the **training median**: Age → 27.0; RoomService, FoodCourt, ShoppingMall, Spa, VRDeck → 0.0. The median is robust to the heavy tails just described (the mean would fill the gaps of a majority-zero column with a value of several hundred, inventing spenders that do not exist) and, for the spending columns, it coincides with the typical passenger. Categorical columns are imputed with the **training mode**: HomePlanet → Earth, CryoSleep → False, Destination → TRAPPIST-1e, VIP → False. The mode keeps every imputed entry inside the set of valid categories and, with ~2% of gaps per column, its bias towards the majority class is negligible. The learned medians and modes are stored and reused on the test set.

**2. Categorical features → numbers.** `HomePlanet`, `CryoSleep`, `Destination` and `VIP` are **one-hot encoded** with scikit-learn's `OneHotEncoder`, fitted on the training categories, producing 10 binary columns: `HomePlanet_{Earth, Europa, Mars}`, `CryoSleep_{False, True}`, `Destination_{55 Cancri e, PSO J318.5-22, TRAPPIST-1e}`, `VIP_{False, True}`. One-hot is the right choice because these categories have no order (encoding Earth = 0, Europa = 1, Mars = 2 would invent one). **Unseen categories:** the encoder is created with `handle_unknown="ignore"`, so a category that appears in the test set but not in the training set produces an all-zero row for that feature instead of raising an error — the network simply sees "none of the known categories". In this split no such category occurs (the script checks: the sets of test categories minus training categories are all empty), but the mechanism is in place for genuinely new data. The 0/1 columns already lie inside [−1, 1] and need no further scaling.

**3. Feature engineering.** `TotalSpend` = `RoomService + FoodCourt + ShoppingMall + Spa + VRDeck`, computed after imputation so it never contains NaN. `Cabin`, `Name` and `PassengerId` are dropped: the last two are identifiers with no predictive content, and `Cabin` is a free-form code with 6 560 distinct values — one-hot encoding it would create thousands of nearly-empty columns (it could be decomposed into deck/side features, but that is beyond this exercise).

**4. Heavy tails → log(1 + x).** The transformation is applied to the five spending columns and to `TotalSpend` (which inherits their tail). On the training set, `FoodCourt` goes from mean 465.35 / median 0 / max 29 813 (skewness 7.26) to mean 1.910 / median 0 / max 10.303 (skewness 1.14): the tail that spanned five orders of magnitude is compressed to a range of about 10 units, while zeros stay at zero (log(1 + 0) = 0). **Why this helps a tanh network:** tanh saturates — its derivative is close to zero for inputs beyond roughly ±2–3, so units driven by huge inputs stop learning. Without the log, scaling to [−1, 1] would be dictated by the 29 813 maximum, squashing the 0-to-500 range where most passengers live into a sliver of width ≈ 0.03 next to −1: the network would be unable to distinguish "spent nothing" from "spent 500", while a handful of outliers would dominate the weights. After the log, the bulk of the distribution occupies the available range and the gradient flows for the typical passenger, not only for the outliers (Figure 6).

**5. Scaling → normalization to [−1, 1].** The seven numerical columns (`Age`, the five log-spending columns and `log(1 + TotalSpend)`) are scaled with min–max normalization to **[−1, 1]** (`MinMaxScaler(feature_range=(-1, 1))`), fitted on the training set. I chose normalization over standardization because it maps the features exactly onto the output range of tanh and yields a bounded, predictable scale; standardization would leave values roughly in [−2, 4] with no guaranteed bounds. Min–max is normally vulnerable to outliers, but step 4 has already removed the extreme tails, so the range is no longer dictated by a single passenger. **Resulting minimum and maximum:** training set **−1.0000 / 1.0000** (by construction), test set **−1.0000 / 1.1383**. The single value above 1 comes from `ShoppingMall`: one test passenger spent more than any training passenger, and since the scaler knows only the training range, its log-value maps slightly beyond 1. This is the expected — and correct — behavior of a leakage-free pipeline; the exceedance is mild (14% on one feature, after the log) and harmless for tanh, whose inputs are weighted sums anyway. Clipping the test set to [−1, 1] would be an acceptable alternative; I preferred to report the honest value.

### D — Verify and visualize

**Figure 6** shows `FoodCourt` on the training set before and after preprocessing, split by class: raw values (left; the y-axis is logarithmic so the tail up to 29 813 credits is visible), after log(1 + x) (center) and after log(1 + x) followed by scaling to [−1, 1] (right).

![Figure 6 — FoodCourt before and after preprocessing](figures/fig6.png)
/// caption
Figure 6 — Histogram of FoodCourt on the training set: raw (left, log-scale counts), after log(1 + x) (center) and after log(1 + x) + min–max scaling to [−1, 1] (right), colored by Transported.
///

The raw histogram is a spike at zero followed by a tail thousands of credits long; after the log the same data occupies a compact range of about 10 units, with the zero spike (4 513 of the 6 954 training passengers, 64.9%, spent nothing at the FoodCourt) and a broad hump between 5 and 8 corresponding to the passengers who did spend. The scaled panel is the same shape mapped onto [−1, 1]. The colors also reveal a strong signal: the zero-spend bar is dominated by `Transported = True`, while the spenders lean `False`.

**Final checks (explicitly reported).**

- **No remaining NaN:** 0 in the training matrix, 0 in the test matrix.
- **Final shape of the feature matrix:** training **(6 954, 17)**, test **(1 739, 17)** — 7 scaled numerical columns (`Age`, `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`, `TotalSpend`) + 10 one-hot columns.
- **Value range compatible with tanh:** training set exactly in [−1.0000, 1.0000]; test set in [−1.0000, 1.1383] (one feature, `ShoppingMall`, slightly above 1 as explained in step 5); the one-hot columns are in {0, 1}. All inputs are within or very near the [−1, 1] range in which tanh operates without saturating.

**Reflection.** The decision I expect to matter most for training is the **log(1 + x) transformation of the spending columns** — and it matters precisely because of what Figure 6 shows: the spending features carry a strong class signal (passengers who spent nothing are mostly transported, big spenders mostly not), but in their raw form that signal is unusable by a tanh network. Any bounded scaling of the raw values would compress the informative 0-to-1 000 range into a few hundredths next to −1, so the first layer would see almost identical inputs for the vast majority of passengers and receive gradients dominated by a few dozen outliers with saturated units; with the log, the differences that distinguish "no spend", "moderate spend" and "big spend" are spread over the whole input range, the activations stay in tanh's sensitive region and the optimization is well conditioned. The leakage-free split (item B) is the decision that most affects how *trustworthy* the measured performance is, but it does not change what the network learns; imputation and encoding are necessary but low-impact here (2% gaps, few categories). The log is the choice that changes whether the most informative features can be learned at all.

### Code

``` python
--8<-- "docs/exercises/data/code/exercise3.py"
```

## Results summary

| # | Item | Your value |
|---|---|---|
| 1 | Mixing rate at s = 0.5 | 0.25% |
| 2 | Mixing rate at s = 1.0 | 7.25% |
| 3 | Mixing rate at s = 2.0 | 19.25% |
| 4 | Mixing rate at s = 4.0 | 48.25% |
| 5 | Smallest r_ij at s = 1.0, and which pair | r_01 = 1.326, pair (0, 1) |
| 6 | Distance between centers — Dataset I | 3.264 (sample means, 5D; theoretical 1.5·√5 = 3.354) |
| 7 | Distance between centers — Dataset II | 0.266 (sample means, 5D; theoretical 0) |
| 8 | Explained variance PC1 + PC2 — Dataset I | 67.04% (51.27% + 15.77%) |
| 9 | Explained variance PC1 + PC2 — Dataset II | 42.91% (21.59% + 21.32%) |
| 10 | Share of the positive class in Transported | 50.36% (4 378 of 8 693 are True) |
| 11 | Mean and median of FoodCourt (training set, before transforming) | mean = 465.35, median = 0.00 |
| 12 | Final shape of the training feature matrix | (6 954, 17) — test: (1 739, 17) |
| 13 | Min and max of the training and test sets after scaling | training: −1.0000 / 1.0000; test: −1.0000 / 1.1383 |
