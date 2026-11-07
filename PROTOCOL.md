# PROTOCOL: PBC generative augmentation study

This file freezes experimental decisions **before** the experiments they govern are run.
Its git history is the timestamp: a decision committed before a result cannot have been
chosen because of that result. Sections are added as each step is reached; a section
marked *not yet frozen* has no binding content.

Changes after freezing are allowed only as dated amendments at the bottom, each with
its reason. Nothing above an amendment is edited.

---

## 1. Data (frozen 6 November 2026)

Evidence for every decision below: `00_data_audit.py`.

**Dataset.** PBC (Acevedo et al. 2020), Barcelona release, pre-split into
`train / valid / test` folders: 14,441 / 1,605 / 845 images, 8 classes, 360×363 RGB JPEG.
Origin of the split: provided by the project supervisor; the procedure used to create it
is not documented. The audit shows it is stratified by class and was made image by image.

**D1. Evaluation sets are never modified.** Valid and test are used exactly as received,
in every condition, never subsampled, never cleaned, never augmented.

**D2. Duplicate rule.** Exact duplicates (identical md5) are handled by kind:
- label conflict, or any copy outside train → every train copy removed;
- same-class duplicates entirely inside train → first copy (alphabetical) kept, rest removed;
- groups with no train copy → kept, recorded as a known flaw.

Result: 17 training images excluded, listed in `excluded.txt`; train = 14,424.
Known flaw: one `ig` (`pmy`) image appears in both valid and test.

**D3. Rare classes.** Rule: the two smallest classes in the cleaned training set.
Result: **basophil** (1035) and **lymphocyte** (1038). Both are single-subtype.

**D4. Resolution.** Images are centre-cropped to 360×360 and resized to **128×128**
(antialiased). Every image used by any classifier, real or synthetic, passes through
128×128, so real and synthetic images are on equal footing.

Reason: visual inspection (audit, section 7) shows that at 64×64 the granule texture
that identifies basophils, one of the two rare classes, is largely lost; at 128×128 it
is still visible.

Fallback: if, after its full training budget, a generator's rare-class samples
are classified as their intended class less than 50% of the time by the baseline
classifier, all arms move to 64×64 and the switch is reported.

The cost of the bottleneck is measured by training the baseline classifier at native
resolution and at 128×128; both numbers are reported.

**D5. Test set usage.** Each final model is evaluated on test once, at the end.
Model selection, early stopping and hyperparameter choices use valid only. No decision
is made after seeing a test score.

**D6. Statistical precision.** Rare classes have 61 test images each, so the standard
error of a single recall measurement is about ±4 points. No conclusion is drawn from a
single seed.

**Prior use of the test set.** Before this protocol was written, an exploratory
ImageNet-pretrained ResNet-18 was trained on this data (224×224, selection on valid)
and evaluated once on the test set. No decision in this protocol was derived from that
evaluation, and that model is not part of the study.

---

## 2. Induced imbalance (frozen 6 November 2026)

**Levels.** Both rare classes (D3) are cut to the same N ∈ {50, 150, 400}. The other six
classes keep all their training images. The uncut training set is a reference condition.
Against the largest class (neutrophil, 2,802) the ratios are about 56:1, 19:1 and 7:1.

**Procedure.** For replicate r, each rare class is randomly permuted with seed r and its
first N images are kept (`data.subsample`). The subsets are therefore nested: at a given
seed, the 50 images are contained in the 150, which are contained in the 400. Changing N
changes only the number of images, never which images are present.

**Rare classes in code.** Computed from the cleaned training labels by the D3 rule
(`data.rare_classes`), never hardcoded.

**Validation is not subsampled** (D1). The validation set is the same in every condition
and contains about 115 images of each rare class. This makes model selection reliable and
identical across arms, but it is unrealistic: a practitioner with only 50 basophils for
training would not have 115 more to validate on. Every arm has this advantage, and it is
reported as a limitation.

## 3. Seeds and replicates (frozen 6 November 2026)

**Replicate.** One complete repetition of the experiment with independent randomness.
Replicate r ∈ {0, 1, 2, 3, 4} uses subset seed r, generator seed r and classifier seed r.
The subset (which images are kept) is treated as part of the randomness being measured,
not fixed once.

**Pairing.** Within a replicate, every arm sees the same subset. Arms are compared through
per-replicate differences (arm minus baseline), which removes subset-to-subset variation
from the comparison.

**Reference runs** on the uncut training set use classifier seeds 0–4.

**Seeds are never selected.** Every listed seed is run and reported; no seed is replaced.

**GPU nondeterminism.** cuDNN nondeterministic kernels are allowed (speed). The same seed
may give results differing in the last decimals. Uncertainty is measured across
replicates, which is much larger than this effect.

## 4. Downstream classifier (frozen 6 November 2026)

Identical in every arm and every condition. It is a control variable, not tuned during
the study. Implementation: `train_classifier.py`.

**Architecture.** ResNet-18 with ImageNet-pretrained weights (`IMAGENET1K_V1`); final
layer replaced by a linear layer with 8 outputs.

**Input.** 360×360 centre crop → 128×128 (D4) → 224×224, ImageNet normalisation.

**Augmentation (training only).** Random horizontal and vertical flips; rotation by a
random multiple of 90°; colour jitter (brightness 0.1, contrast 0.1, saturation 0.1,
hue 0.02).

**Training.** Cross-entropy loss; AdamW, learning rate 3×10⁻⁴, weight decay 10⁻⁴;
cosine schedule over 30 epochs; batch size 64.

**Model selection.** The epoch with the highest validation macro-F1 is kept.

**Logged every epoch.** Learning rate; training loss and accuracy (measured on augmented
images); validation loss, accuracy, macro-F1, balanced accuracy, MCC, and recall for each
class.

### 4b. Resolution-cost experiment

Measures the cost of the 128×128 bottleneck (D4). Two conditions, uncut training set,
classifier seeds 0–4 each:

- **native:** 360 → 224
- **128:** 360 → 128 → 224

Everything else as above. Both conditions feed the network 224×224 inputs, so the only
difference is the detail lost at 128×128. Reported: mean and standard deviation over
seeds of validation macro-F1 and per-class recall, for both conditions and their
difference; test scores are computed at the end of the study (D5). No decision in the
study depends on the outcome.

## 5. Baseline arm (frozen 7 November 2026)

The best treatment of imbalance without a generative model, on the classifier of section 4,
which already includes its augmentation. A synthetic arm counts as a success only if it
beats this arm: against a weak baseline, any generator would look useful.

**Configurations (3).**
- **none:** plain cross-entropy;
- **weighted:** cross-entropy with class weights proportional to 1 / (number of training
  images of the class), normalised to mean 1, so that a mistake on a rare image costs more;
- **oversampling:** each training image is drawn with probability proportional to
  1 / (number of training images of its class), so every class is drawn equally often. An
  epoch has as many draws as there are training images.

**Selection.** For each replicate and each N, the configuration with the highest validation
macro-F1 is kept. Taking the best of 3 makes the winner's validation score slightly
optimistic. Its test score is not affected, since the test set plays no part in the choice.

**Equal tuning.** Every arm, baseline or synthetic, has exactly 3 configurations, selected
by this rule. An arm with more configurations would have more chances to win by luck.

## 6. Generative arms (frozen 7 November 2026)

**Models.** A class-conditional VAE (cVAE) and a class-conditional GAN (cGAN). Each takes a
class label and produces a 128×128 RGB image of that class. Implementation:
`train_generator.py`, `models.py`.

**Architecture.** Latent dimension 128; the label enters as a one-hot vector. Batch size 64;
one iteration is one batch.

- *Decoder / generator.* The cVAE decoder and the cGAN generator are the same network: a
  linear layer to a 512×4×4 grid, then five transposed convolutions (kernel 4, stride 2)
  with batch normalisation and ReLU, of widths 512, 256, 128, 64, 32
  (4 → 8 → 16 → 32 → 64 → 128), then a 3×3 convolution to the three colour channels,
  followed by a sigmoid (cVAE) or a tanh (cGAN). Images are generated with the network in
  evaluation mode.
- *cVAE encoder.* A linear layer turns the label into an extra 128×128 image channel. Then
  five stride-2 convolutions of widths 32, 64, 128, 256, 512, with batch normalisation
  (except the first) and LeakyReLU 0.2, and linear layers to the mean and log-variance.
- *cVAE loss* for one image: 50 × the sum of squared pixel errors, plus the KL divergence.
  The factor 50 is 1/(2σ²) with σ = 0.1, a Gaussian likelihood. Adam, learning rate 5×10⁻⁴.
- *cGAN discriminator.* Five stride-2 convolutions of widths 32 to 512 with LeakyReLU 0.2,
  summed over positions. The output is a linear score plus the inner product of these
  features with a learned vector for the label (projection discriminator, Miyato and
  Koyama 2018). Spectral normalisation on every layer, which limits how fast the
  discriminator can change. No batch normalisation: with it, the score of one image would
  depend on the other images of its batch, and the attack of 8.3 needs a score per image.
- *cGAN training.* Hinge loss. Adam, β = (0, 0.999), learning rate 10⁻⁴ for the generator
  and 4×10⁻⁴ for the discriminator, one discriminator step per generator step. DiffAugment
  (Zhao et al. 2020): a random change of colour, a random translation and a random cutout
  are applied to every image, real or generated, before the discriminator. This stops the
  discriminator from memorising the few rare images, the usual way GANs fail on small
  datasets.

**Training data.** The replicate's training subset: all 8 classes, rare classes cut to N.
Removed images are never seen by a generator or a classifier; they are the non-members of
section 8.

Batches are class-balanced, as in the oversampling configuration of section 5. With natural
sampling at N = 50, each rare class would be 50 / 12,451 ≈ 0.4% of the draws and the
generator would barely learn it. With balanced sampling, each rare image is drawn about 31
times per epoch at N = 50. Repeated exposure is also how small training sets get memorised,
which section 8 measures.

Each real image is transformed by one of the 8 rotations and flips of the square, the
symmetries the classifier already uses (section 4). A rotated cell is still a valid cell.
Since the generators can then produce rotated copies of training images, every distance in
section 8 is minimised over these 8 transforms.

**Pilot.** One run per model on the uncut training set: 40,000 iterations, 20 checkpoints
(every 2,000). Its purpose is to find bugs and fix the hyperparameters and the budget; it
may change learning rates, widths, latent dimension and loss weights. It is judged only on
training losses, samples looked at by eye, judge accuracy on rare-class samples (500 per
class), and the Fréchet distance to the validation images.

The Fréchet distance fits a Gaussian to the features (section 8) of each of two image sets
and measures the distance between the two Gaussians; lower means the generated images are
distributed more like the real ones. It is computed between the validation images and one
generated image per validation image, with the same label and fixed noise, so every
checkpoint is compared on the same draws.

Classifier scores and the test set are never used, so the pilot cannot tune a generator
towards the result being measured. Every pilot run is kept.

**Budget.** Counted in iterations and identical at every N; an epoch would have a different
length at each N. The budget is the earliest pilot checkpoint whose Fréchet distance is
within 5% of the pilot's minimum: training longer barely improves quality, and memorisation
tends to grow with training time. The hyperparameters finally used and the budget are
written in a dated amendment before the first subsampled run.

**Checkpoint.** The final one. No checkpoint is chosen per run.

**Interruptions.** A run stopped by a technical failure resumes from its last saved state,
same seed, and the resumption is recorded in its log. The result can differ slightly from
an uninterrupted run (section 3, GPU nondeterminism).

**Synthetic sets.** For replicate r, 2000 images per rare class are drawn once with seed r
and saved as 128×128 PNG, the format of the real images after resizing. They enter the
classifier pipeline of section 4 at the 128×128 step. No image is removed: filtering would
be a choice made after seeing the output. The saved set is also the released set attacked
in section 8.

**Configurations (3).** S ∈ {500, 1000, 2000} synthetic images added per rare class: the
first S of the saved set, so the sets are nested as in section 2. Plain cross-entropy.
Selection as in section 5.

**Judge.** The resolution-cost classifier (section 4b, input 128, seed 0) labels generated
images. Its accuracy on each rare class's samples is reported for every generator run, and
decides the D4 fallback (A1).

**Out of scope.** Synthetic images combined with weighting or oversampling.

**Cost.** 3 N × 2 generators × 10 replicates (A4) = 60 generator runs;
3 N × 3 arms × 3 configurations × 10 replicates = 270 classifier runs.

## 7. Metrics and decision rules (frozen 7 November 2026)

**Test evaluation.** For each arm, N and replicate, only the configuration selected on
validation is evaluated on test, once (D5). The prediction for every test image is saved,
for the bootstrap below. The reference condition is defined in A2.

**Primary endpoint.** The conclusion rests on one number: the rare-class F1, mean of the
test F1 of the two rare classes (D3). For a class, F1 = 2TP / (2TP + FP + FN), where TP are
its images predicted as it, FN its images predicted as another class, and FP images of
other classes predicted as it. F1 counts both missed rare cells and other cells wrongly
called rare, so an arm cannot raise it just by predicting the rare classes more often.

**Secondary endpoints.** Macro-F1, which checks that the other classes are not harmed;
balanced accuracy; MCC; rare-class recall; recall and F1 of each class; and the fraction of
the gap closed, (arm − baseline) / (reference − baseline), on the means over replicates.

**Effect.** For each synthetic arm, N and replicate r,
Δ_r = endpoint(arm, r) − endpoint(baseline, r), both trained on the same subset
(section 3). The effect is Δ̄, the mean of Δ_r over the R replicates.

**Uncertainty.** Two sources are counted.
- Training randomness (subset, initialisation, batch order): s², the variance of Δ_r over
  replicates.
- Test sampling: every replicate is scored on the same 845 test images, so s² does not
  include it. v_test is the variance of Δ̄ over 10,000 bootstrap resamples of the test
  images. Each resample draws test images with replacement within each class, and the same
  resample is applied to every replicate and every arm.

SE = √(s²/R + v_test). The 95% interval is Δ̄ ± t · SE, where t is the 97.5% quantile of
Student's t distribution with R − 1 degrees of freedom.

**Test.** T = Δ̄ / SE, compared with Student's t with R − 1 degrees of freedom, two-sided.
The family is 6 tests: each synthetic arm against the baseline, at each N. Holm correction
at level 0.05: sort the p-values p(1) ≤ … ≤ p(6), reject the i-th as long as
p(i) ≤ 0.05 / (7 − i), and stop at the first that fails. This keeps the probability of any
false rejection in the family at or below 0.05.

**Decision.** For each synthetic arm and each N, on the primary endpoint, with δ = 0.02
(about one test image of each rare class, since 1/61 ≈ 0.016):
- **helped:** significant after correction, Δ̄ > 0;
- **hurt:** significant after correction, Δ̄ < 0;
- **no effect larger than δ:** not significant, and the 95% interval lies inside (−δ, δ);
- **inconclusive:** every other case.

**Exploratory.** Secondary endpoints and the other comparisons (cGAN against cVAE, each arm
against the reference) are reported with 95% intervals, without tests.

**Power.** With R = 10 (A4), the strictest Holm threshold (0.05/6, two-sided, 9 degrees of
freedom) is |T| > 3.4. Leaving out v_test, the smallest detectable effect is therefore
about 3.4 / √10 ≈ 1.1 times the standard deviation of Δ_r. v_test does not shrink with R,
so it sets a floor under the detectable effect.

## 8. Memorisation checks (frozen 7 November 2026)

**Distance.** φ(x) is the 512-dimensional output of the global pooling layer of an
ImageNet-pretrained ResNet-18 (`IMAGENET1K_V1`, never trained on this data), with the
128×128 input resized to 224×224 and ImageNet normalisation. The distance between two
images is d(x, x') = min over the 8 transforms T of section 6 of ‖φ(Tx) − φ(x')‖₂, so a
rotated or flipped copy counts as a copy.

### 8.1 Train–test near-duplicates

Cells cropped from the same smear can be in both train and test (section 1). For every test
image, the distance to its nearest cleaned training image is computed. Before any test
evaluation, the 50 closest pairs are inspected by eye, and pairs showing the same cell are
listed in `near_duplicates.txt`. Test is not modified (D1); the endpoints of section 7 are
also reported without the listed images.

### 8.2 Members and non-members

A membership attack tries to tell whether an image was in a generator's training set. It is
evaluated on images whose answer is known, in the rare classes only, for each replicate and
each N:
- **members:** the N images of the class kept by `data.subsample`;
- **non-members:** the images of the same class removed by the same call.

Membership is decided by a random permutation, so a score that does not use the trained
model separates the two groups no better than chance (expected AUC 0.5). Many published
attacks are invalid because their members and non-members differ in distribution (Das,
Zhang and Tramèr 2024; Dubiński et al. 2024); random assignment rules this out. It is
checked with a score that uses no model, the mean pixel intensity, whose AUC is reported.

The majority classes keep all their training images, so they have no non-members and are
not attacked.

### 8.3 Attack scores

Each attack gives an image x of label y a score s(x); a higher score means "predicted
member".
- **Nearest synthetic image** (cVAE and cGAN). Uses only the saved synthetic set, which is
  what someone publishing synthetic data releases. s(x) = − min over the 2000 synthetic
  images g of class y of d(x, g).
- **Discriminator** (cGAN). Uses the weights. A discriminator that has overfitted scores
  its own training images higher (Hayes et al. 2019). s(x) = mean over the 8 transforms T
  of the final discriminator's output for (Tx, y), without DiffAugment.
- **ELBO** (cVAE). Uses the weights. The loss is lower on training images. s(x) = − the
  training loss of (x, y), averaged over 16 latent draws and the 8 transforms.

**Calibration.** Some cells score high whether or not they were trained on. To remove this,
an image's score is compared with the scores it receives from models that did not see it
(Watson et al. 2022; offline LiRA, Carlini et al. 2022).
1. Within each run, the scores of the members and non-members of a class are standardised
   to mean 0 and standard deviation 1.
2. For an image x in replicate r, the calibrated score is s̃_r(x) = s_r(x) − (mean of
   s_r'(x) over the other replicates r' at the same N in which x is a non-member).

Images with fewer than 2 such replicates are left out of the calibrated attack, and their
number is reported. The subsets of different replicates are drawn independently, so this
does not depend on membership in replicate r. Calibrated scores are the primary analysis;
uncalibrated scores are also reported.

### 8.4 Measures and tests

For each arm, N and attack, with the two rare classes pooled:
- the AUC, the probability that a random member scores higher than a random non-member,
  per replicate, and its mean over replicates;
- the true positive rate at a false positive rate of 1%, replicates pooled: the fraction of
  members found when 1% of non-members are wrongly flagged.

**Test.** One-sided permutation test. Null hypothesis: member and non-member scores are
interchangeable. Statistic: mean AUC over replicates. The p-value comes from 10,000
permutations of the membership labels within each replicate and class. The family is 12
tests (2 arms × 3 N × 2 attacks), Holm correction at 0.05. The model-free check of 8.2 is
reported, not tested.

The effect of section 7 and the attack AUC are reported side by side for each arm and N, so
the trade-off between utility and privacy can be read directly.

### 8.5 Data copying

Meehan et al. (2020), global version, for each arm, N, rare class and replicate: the
distances from synthetic images to their nearest member are compared with the distances
from non-members to their nearest member, by a Mann–Whitney test summarised by its Z
statistic. Z < 0: synthetic images are closer to training images than unseen real images
are (copying). Z > 0: they are farther (underfitting). Descriptive, no correction. The 16
synthetic images closest to a member are shown next to that member.

### 8.6 Saved during generator training

The rest of section 8 is computed after training. The following cannot be recovered
afterwards, so it is saved for every run:
- the list of training images;
- the final weights (generator and discriminator; encoder and decoder);
- the weights at 25%, 50% and 75% of the budget;
- the 2000 synthetic images per rare class;
- every epoch, for the cGAN: the mean discriminator output on the epoch's training
  batches, on the rare-class training images and on the validation set; for the cVAE: the
  training loss, and the loss on the rare-class training images and on the validation set.

### 8.7 Exploratory

8.3 and 8.4 are repeated on the checkpoints at 25%, 50% and 75% of the budget. No tests,
no decision.

---

## Amendments

### A1 (7 November 2026). D4 fallback checked on the pilot only

As written, D4 could be triggered by any generator at any N: if the cGAN at N = 50 produced
basophils the judge does not recognise, every arm would move to 64×64. But a generator that
fails at small N is a result about that N, not evidence about resolution. On the uncut
pilot, the rule does test whether 128×128 can be learned at all.

The D4 fallback is checked once, on the pilot samples of each generator, by the judge of
section 6 ("baseline classifier" in D4 means this judge). On subsampled runs the judge's
accuracy is reported but triggers nothing. If the rule triggers, the judge and the 128
condition of 4b are rerun at 64×64 before any subsampled run.

### A2 (7 November 2026). Reference condition

The uncut reference condition (section 3) is the 128 condition of 4b, seeds 0–4. Its
classifier, data and seeds are exactly what the reference requires, so it is not retrained.

### A3 (7 November 2026). Editorial

The paragraph "Validation is not subsampled" of section 2 was rewritten without changing
its content. A blank line was added before the heading of section 4, and the dashes in the
headings of sections 1 to 4 were replaced by parentheses.

### A4 (7 November 2026). Ten replicates

With 5 replicates, the test of section 7 has 4 degrees of freedom and the strictest Holm
threshold is |T| > 4.9, so only effects above about 2.2 times the standard deviation of
Δ_r can be detected. With 10 replicates the threshold is 3.4 and the detectable effect
about 1.1 times that standard deviation: twice as sensitive. Run time is not a constraint.

Section 3 is changed: replicate r ∈ {0, 1, …, 9}, each with subset seed r, generator seed r
and classifier seed r. The reference condition keeps classifier seeds 0–4 (A2); it enters
no test.
