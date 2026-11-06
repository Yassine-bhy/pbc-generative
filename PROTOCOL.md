# PROTOCOL — PBC generative augmentation study

This file freezes experimental decisions **before** the experiments they govern are run.
Its git history is the timestamp: a decision committed before a result cannot have been
chosen because of that result. Sections are added as each step is reached; a section
marked *not yet frozen* has no binding content.

Changes after freezing are allowed only as dated amendments at the bottom, each with
its reason. Nothing above an amendment is edited.

---

## 1. Data — frozen 6 November 2026

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

## 2. Induced imbalance — frozen 6 November 2026

**Levels.** Both rare classes (D3) are cut to the same N ∈ {50, 150, 400}. The other six
classes keep all their training images. The uncut training set is a reference condition.
Against the largest class (neutrophil, 2,802) the ratios are about 56:1, 19:1 and 7:1.

**Procedure.** For replicate r, each rare class is randomly permuted with seed r and its
first N images are kept (`data.subsample`). The subsets are therefore nested: at a given
seed, the 50 images are contained in the 150, which are contained in the 400. Changing N
changes only the number of images, never which images are present.

**Rare classes in code.** Computed from the cleaned training labels by the D3 rule
(`data.rare_classes`), never hardcoded.

**Validation is not subsampled** (D1). Consequence, reported: model selection uses a
validation set in which the rare classes are not rare, which a genuine low-data
practitioner would not have. This applies equally to every arm.

## 3. Seeds and replicates — frozen 6 November 2026

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

**Budget.** 3 levels × 2 generators × 5 replicates = 30 generator runs;
3 levels × 3 arms × 5 replicates = 45 classifier runs, plus reference runs.

## 4. Downstream classifier — frozen 6 November 2026

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

## 5. Baseline arm — *not yet frozen*

Imbalance remedies included; tuning budget (must equal that of the synthetic arms).

## 6. Generative arms — *not yet frozen*

Models, training budget, number of synthetic images added per rare class.

## 7. Metrics and decision rules — *not yet frozen*

Macro-F1, per-class recall, balanced accuracy, MCC. Effect size counted as "helped",
statistical test, multiple-comparison correction.

## 8. Generalisation and memorisation checks — *not yet frozen*

Near-duplicate leakage check (train ↔ test, feature space); nearest-neighbour analysis of
synthetic images; membership inference per arm.

---

## Amendments

_None._
